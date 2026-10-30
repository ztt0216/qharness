"""研究层评测集：答案已知的因子，检验研究层（规则 + 模型）的判断质量。

每个用例给定因子代码（不经过实验层），在真实行情上算出指标，交给 ResearchAgent 判定，检查：
- verdict 是否落在预期集合里（规则决定，用来防回归）
- 模型诊断是否抓住了要点（关键词，衡量模型这一半做得好不好）
- 修改建议是否合理（该翻方向时翻、不该翻时规则拦住）
结果写 JSON，可以在改提示词 / 换模型前后对比分数。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .agents import ResearchAgent
from .evaluate import Evaluator
from .llm import LLM
from .sandbox import execute_factor


@dataclass
class Case:
    name: str
    hypothesis: dict
    code: str
    expect_verdict: set[str]
    expect_keywords: list[list[str]] = field(default_factory=list)  # 每组至少命中一个
    expect_flip: bool | None = None  # True：应建议翻转方向；False：不应翻转（或被规则拦下）
    note: str = ""


def _h(name, calc, window, direction, fields=("close",)):
    return {"name": name, "fields": list(fields), "calculation": calc, "window": window, "direction": direction,
            "rationale": "", "condition": ""}


CASES = [
    Case("重复因子：5日反转",
         _h("5日反转", "-(close/close.shift(5)-1)", 5, 1),
         "def factor(df):\n    g = df['close'].groupby(level='instrument')\n    return -(df['close'] / g.shift(5) - 1)",
         {"reject"}, [["反转", "reversal", "重复", "相关"]], None,
         "与因子库里的 reversal_5d 完全一样，应因高相关被淘汰，诊断应指出重复"),
    Case("噪声因子",
         _h("价格尾数", "sin(close*1000)", 1, 1),
         "import numpy as np\ndef factor(df):\n    return np.sin(df['close'] * 1000 + df['volume'] % 97)",
         {"reject", "refine"}, [["噪声", "随机", "无", "不显著", "不稳定", "弱"]], False,
         "没有经济含义，不应被接受；即使某段 IC 偶然为负也不应靠翻转方向修"),
    Case("市值代理",
         _h("成交额规模", "过去20日平均成交额的对数取负", 20, 1, ("amount",)),
         "import numpy as np\ndef factor(df):\n    a = df['amount'].groupby(level='instrument').rolling(20).mean().droplevel(0)\n"
         "    return -np.log(a.reindex(df.index))",
         {"reject", "refine"}, [["市值", "规模", "小盘", "中性化", "流动性"]], None,
         "本质是小市值 / 低流动性暴露，中性化后应明显变弱，诊断应提到市值或流动性"),
    Case("方向写反的信号：收益偏度",
         _h("20日收益偏度", "过去20日日收益率的偏度，偏度越高越好", 20, 1),
         "def factor(df):\n    r = df['close'].groupby(level='instrument').pct_change()\n"
         "    return r.groupby(level='instrument').rolling(20).skew().droplevel(0).reindex(df.index)",
         {"refine", "reject"}, [["方向", "反向", "取负", "翻转", "反"]], True,
         "彩票型偏好：高偏度股票后续收益更低。2023-01~2026-01 中性化后训练、验证两段 RankIC 都为负"
         "（训练段三年都为负），假设方向写反了，应建议翻转"),
]


def run_eval(llm: LLM, ev: Evaluator, cases: list[Case] | None = None) -> dict:
    cases = list(cases or CASES)
    rows, scores = [], []
    for c in cases:
        res = execute_factor(c.code, ev.df)
        if not res.ok:
            rows.append({"case": c.name, "error": f"{res.error_type}: {res.message[:200]}"})
            scores.append(0.0)
            continue
        metrics = ev.evaluate(res.values)
        v = ResearchAgent(llm).judge(c.hypothesis, metrics, can_refine=True)
        text = v.diagnosis + " " + " ".join(str(x) for x in (v.rule_details.get("findings") or []))
        checks = {"verdict": v.verdict in c.expect_verdict}
        for i, group in enumerate(c.expect_keywords):
            checks[f"keywords_{i}"] = any(k.lower() in text.lower() for k in group)
        flipped = bool(v.revision and v.revision.get("direction") not in (None, c.hypothesis["direction"]))
        if c.expect_flip is not None:
            checks["flip"] = flipped == c.expect_flip
        score = sum(checks.values()) / len(checks)
        scores.append(score)
        rows.append({"case": c.name, "note": c.note, "verdict": v.verdict, "failed": v.failed,
                     "revision": v.revision, "revision_check": v.rule_details.get("revision_check"),
                     "diagnosis": v.diagnosis, "checks": checks, "score": round(score, 3),
                     "valid_rank_ic": metrics["valid"].get("rank_ic"), "train_rank_ic": metrics["train"].get("rank_ic"),
                     "rank_ic_raw_valid": metrics["valid"].get("rank_ic_raw")})
    return {"summary": {"n_cases": len(cases), "mean_score": round(sum(scores) / len(scores), 3),
                        "by_case": {r["case"]: r.get("score", 0.0) for r in rows}},
            "cases": rows}
