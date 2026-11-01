"""从 runs/<run_id>/ 的 trace.jsonl 和 llm_calls.jsonl 统计 Agent 指标：`qharness agent-stats`。

- 编码：每一轮实现（候选 + round）第一次执行就通过的比例、修复次数、最终是否跑通、报错类型
- 研究：候选的最终结论、经验检索命中率、被提前拦下的候选
- 研报 SOP：观点因为找不到原文依据被丢掉的比例、假设因为和原文不符被丢掉的比例
- 成本：按角色的调用次数、token、延迟（早期运行没有记 llm_calls，只统计有记录的；不换算金额，价格会变）

局限：这些是开发过程中的运行记录，不是对照实验；同一份样例研报跑过多次，不能代表没见过的研报。
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def _names(x) -> list:
    return x if isinstance(x, list) else []


def collect(runs: Path) -> dict:
    run_dirs = sorted(p for p in Path(runs).iterdir() if p.is_dir() and (p / "trace.jsonl").exists())
    rounds: list[list[dict]] = []
    errors, status, sources, reports = Counter(), Counter(), Counter(), Counter()
    retrieve = skipped = duplicates = gave_up = cand_start = 0
    hits = raw = dropped = kept = faithful_dropped = 0
    role = defaultdict(lambda: {"calls": 0, "input": 0, "cached": 0, "output": 0, "latency": []})
    per_run = []
    for d in run_dirs:
        args = json.loads((d / "args.json").read_text(encoding="utf-8")) if (d / "args.json").exists() else {}
        kind = "研报" if args.get("report") else "手写假设" if args.get("hypothesis") else "未记录"
        sources[kind] += 1
        if args.get("report"):
            reports[Path(args["report"]).stem] += 1
        cur: dict = {}
        for e in _jsonl(d / "trace.jsonl"):
            n = e["node"]
            if n == "experiment.execute":
                # 早期运行没有 cand 字段，不能按（候选, round）分组：fix=0 就是一次新实现的开始
                c = e.get("cand")
                if e.get("fix") and c in cur:
                    cur[c].append(e)
                else:
                    cur[c] = [e]
                    rounds.append(cur[c])
                if not e.get("ok"):
                    errors[e.get("error_type") or "未知"] += 1
            elif n == "candidate.start":
                cand_start += 1
            elif n == "candidate.end":
                status[e.get("status") or "未知"] += 1
            elif n == "experiment.retrieve":
                retrieve += 1
                hits += bool(e.get("n"))
            elif n == "research.skip_rejected":
                skipped += 1
            elif n == "experiment.duplicate":
                duplicates += 1
            elif n == "experiment.give_up":
                gave_up += 1
            elif n == "sop.viewpoints":
                raw += e.get("n_raw") or 0
                dropped += len(_names(e.get("dropped_ungrounded")))
            elif n == "sop.faithful":
                kept += len(_names(e.get("kept")))
                faithful_dropped += len(_names(e.get("dropped")))
        calls = _jsonl(d / "llm_calls.jsonl")
        if calls:
            tot = {"run": d.name, "kind": kind, "calls": len(calls), "tokens": 0, "latency": 0.0}
            for c in calls:
                u, r = c.get("usage") or {}, role[c.get("role") or "未知"]
                i, h, o = u.get("input_tokens", 0), u.get("cache_read_input_tokens", 0), u.get("output_tokens", 0)
                r["calls"] += 1
                r["input"] += i
                r["cached"] += h
                r["output"] += o
                r["latency"].append(c.get("latency_s") or 0)
                tot["tokens"] += i + h + o
                tot["latency"] += c.get("latency_s") or 0
            per_run.append(tot)
    n = len(rounds)
    first = sum(bool(v[0].get("ok")) for v in rounds)
    final = sum(any(x.get("ok") for x in v) for v in rounds)
    fixes = [len(v) - 1 for v in rounds]
    repaired = [v for v in rounds if not v[0].get("ok")]
    out_roles = {}
    for k, r in role.items():
        out_roles[k] = {"calls": r["calls"], "input_tokens": r["input"], "cached_tokens": r["cached"],
                        "output_tokens": r["output"], "latency_median_s": round(median(r["latency"]), 2),
                        "latency_max_s": round(max(r["latency"]), 2)}

    def med(rows, key):
        return round(median(x[key] for x in rows), 1) if rows else None

    by_kind = {k: {"runs": len(rows), "calls_median": med(rows, "calls"), "tokens_median": med(rows, "tokens"),
                   "llm_seconds_median": med(rows, "latency")}
               for k in {x["kind"] for x in per_run} for rows in [[x for x in per_run if x["kind"] == k]]}
    return {
        "runs": len(run_dirs), "sources": dict(sources), "reports": dict(reports),
        "coding": {"rounds": n, "first_pass": first, "first_pass_rate": _rate(first, n),
                   "final_ok": final, "final_ok_rate": _rate(final, n),
                   "needed_fix": len(repaired), "fixed": sum(any(x.get("ok") for x in v) for v in repaired),
                   "fixes_total": sum(fixes), "fixes_max": max(fixes, default=0), "gave_up": gave_up,
                   "errors": dict(errors)},
        "research": {"candidates": cand_start, "status": dict(status), "skipped_as_rejected": skipped,
                     "duplicates": duplicates, "retrieve": retrieve, "retrieve_hits": hits,
                     "retrieve_hit_rate": _rate(hits, retrieve)},
        "sop": {"viewpoints_raw": raw, "ungrounded_dropped": dropped, "ungrounded_rate": _rate(dropped, raw),
                "hypotheses_kept": kept, "unfaithful_dropped": faithful_dropped,
                "unfaithful_rate": _rate(faithful_dropped, kept + faithful_dropped)},
        "llm": {"runs_with_record": len(per_run), "by_role": out_roles, "by_source": by_kind},
    }


def write(runs: Path) -> dict:
    s = collect(runs)
    (Path(runs) / "agent_stats.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    return s
