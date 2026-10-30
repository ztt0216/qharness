"""检索评测：字面 / 语义 / 混合三种方式在真实运行记录上的对比，阈值按这里的结果设定（`qharness eval-retrieval`）。

两组任务，标准答案都来自已有的运行记录，不需要人工标注：
1. 研报检索：用归并后的观点去检索它所在研报的正文切片，出处（章节 + 页码）对上的切片算相关。
   看相关切片能不能排进前 2 / 前 5（Recall@k）和第一个相关切片的名次（MRR）。
2. 同一因子识别：候选假设两两配对，因子身份（公式 + 实际方向 + 字段 + 窗口，orchestrator.factor_key）
   相同算同一个因子。看相似度分数能不能把同一个因子和不同因子分开（AUC），以及召回 90% 同一因子时的阈值。
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from . import retrieval as R

ROOT = Path(__file__).resolve().parent.parent


def _report_cases(runs: Path):
    from .report_sop import RISK_SECTIONS, clean_and_chunk

    for d in sorted(runs.glob("*/")):
        rep, tr = d / "report.md", d / "trace.jsonl"
        if not rep.exists() or not tr.exists():
            continue
        merge = [json.loads(l) for l in tr.read_text(encoding="utf-8").splitlines() if '"sop.merge"' in l]
        if not merge:
            continue
        chunks = [c for c in clean_and_chunk(rep.read_text(encoding="utf-8"))
                  if len(c.text) >= 15 and not any(x in c.section for x in RISK_SECTIONS)]
        texts = [c.text for c in chunks]
        for v in merge[-1]["viewpoints"]:
            cites = {(c["section"], c["page"]) for c in v.get("citations", [])}
            rel = {i for i, c in enumerate(chunks) if (c.section, c.page) in cites}
            if rel and len(chunks) > 5:
                yield d.name, v["claim"], texts, rel


def eval_report_retrieval(runs: Path) -> dict:
    out = {}
    cases = list(_report_cases(runs))
    for mode in ("lexical", "semantic", "hybrid"):
        r2 = r5 = mrr = 0.0
        for _, q, texts, rel in cases:
            order = R.rank(q, texts, how=mode)
            r2 += len(rel & set(order[:2])) / min(2, len(rel))
            r5 += len(rel & set(order[:5])) / min(5, len(rel))
            mrr += 1.0 / (1 + next(i for i, x in enumerate(order) if x in rel))
        n = max(1, len(cases))
        out[mode] = {"recall@2": round(r2 / n, 3), "recall@5": round(r5 / n, 3), "mrr": round(mrr / n, 3)}
    out["n_queries"] = len(cases)
    out["n_reports"] = len({c[0] for c in cases})
    out["avg_chunks"] = round(float(np.mean([len(c[2]) for c in cases])), 1) if cases else 0
    return out


PARAPHRASE = """把下面每条判断改写成意思相同、但措辞明显不同的说法（像另一位研究员复述），
尽量不沿用原句的词组；数字和方向不能变。只输出 JSON：{{"items": ["改写1", "改写2", ...]}}，顺序和条数与输入一致。

{items}"""


def paraphrased_cases(runs: Path, llm, n: int = 80, batch: int = 20, seed: int = 0) -> list:
    """改写过措辞的查询：检验同一个意思换了说法还能不能找到（真实流程里主题找正文、假设找定义都是这种情况）。"""
    from .llm import extract_json

    cases = list(_report_cases(runs))
    rng = np.random.default_rng(seed)
    pick = [cases[i] for i in sorted(rng.choice(len(cases), size=min(n, len(cases)), replace=False))]
    out = []
    for i in range(0, len(pick), batch):
        part = pick[i:i + batch]
        items = "\n".join(f"{k + 1}. {c[1]}" for k, c in enumerate(part))
        msgs = [{"role": "system", "content": "严格按要求输出 JSON。"},
                {"role": "user", "content": PARAPHRASE.format(items=items)}]
        new = extract_json(llm.chat("eval.paraphrase", msgs, {})).get("items", [])
        out += [(c[0], q, c[2], c[3]) for c, q in zip(part, new) if isinstance(q, str) and q.strip()]
    return out


def eval_cases(cases: list) -> dict:
    out = {}
    for mode in ("lexical", "semantic", "hybrid"):
        r2 = mrr = 0.0
        for _, q, texts, rel in cases:
            order = R.rank(q, texts, how=mode)
            r2 += len(rel & set(order[:2])) / min(2, len(rel))
            mrr += 1.0 / (1 + next(i for i, x in enumerate(order) if x in rel))
        out[mode] = {"recall@2": round(r2 / max(1, len(cases)), 3), "mrr": round(mrr / max(1, len(cases)), 3)}
    out["n_queries"] = len(cases)
    return out


def _auc(pos: list[float], neg: list[float]) -> float:
    p, n = np.array(pos), np.array(neg)
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def eval_factor_identity(runs: Path) -> dict:
    from .orchestrator import factor_key, hypothesis_text

    hyps, seen = [], set()
    for f in sorted(glob.glob(str(runs / "*" / "trace.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if '"candidate.start"' in line:
                h = json.loads(line)["hypothesis"]
                t = hypothesis_text(h)
                if t not in seen:
                    seen.add(t)
                    hyps.append((t, json.dumps(factor_key(h), ensure_ascii=False)))
    out = {"n_hypotheses": len(hyps)}
    pairs = [(a, b) for i, a in enumerate(hyps) for b in hyps[i + 1:]]
    scorers = {"lexical": R.lexical, "semantic": R.semantic, "hybrid": R.score}
    for name, fn in scorers.items():
        pos = [fn(a[0], b[0]) for a, b in pairs if a[1] == b[1]]
        neg = [fn(a[0], b[0]) for a, b in pairs if a[1] != b[1]]
        if not pos or not neg:
            continue
        thr90 = float(np.quantile(pos, 0.1))  # 召回 90% 同一因子的阈值
        out[name] = {"auc": round(_auc(pos, neg), 3), "threshold_recall90": round(thr90, 3),
                     "false_positive_rate_at_threshold": round(float(np.mean(np.array(neg) >= thr90)), 3),
                     "pos_median": round(float(np.median(pos)), 3), "neg_median": round(float(np.median(neg)), 3)}
    out["n_same_pairs"] = sum(a[1] == b[1] for a, b in pairs)
    out["n_diff_pairs"] = len(pairs) - out["n_same_pairs"]
    return out


def run(runs: Path = ROOT / "runs", out: Path | None = None, llm=None) -> dict:
    if not R.enabled():
        raise SystemExit(f"向量模型不可用：{R.status()['error']}")
    res = {"model": R.MODEL, "report_retrieval": eval_report_retrieval(runs), "factor_identity": eval_factor_identity(runs)}
    if llm is not None:
        cases = paraphrased_cases(runs, llm)
        res["report_retrieval_paraphrased"] = eval_cases(cases)
        res["report_retrieval_paraphrased"]["examples"] = [{"query": q, "original_chunk": t[min(r)][:80]}
                                                           for _, q, t, r in cases[:3]]
    if out:
        Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    import sys

    from .llm import make_llm

    llm = None if "--no-llm" in sys.argv else make_llm("anthropic")
    print(json.dumps(run(out=ROOT / "runs" / "retrieval_eval.json", llm=llm), ensure_ascii=False, indent=2))
