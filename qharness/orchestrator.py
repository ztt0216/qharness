"""模块二：实验编排。

    generate → execute/backtest ─(代码错)→ fix → execute ...      内环：修代码（实验层反馈），≤ max_fix
                     │
                     └─(跑通)→ research judge ─ accept → 测试段 → 模块三
                                              ├ refine → 改假设 → generate   外环：修假设（研究层反馈），≤ max_refine
                                              └ reject → 记录淘汰方向

用 LangGraph StateGraph 实现：每个方框是一个节点，内外两个循环是条件边；修复次数、调整轮数、总步数上限都在路由函数里。

    check_rejected ─(同一因子已淘汰)→ END
          └→ apply_correction → generate → execute ─(失败, 未超限)→ fix → execute
                                               ├(失败, 超限)→ give_up → END
                                               └(成功)→ dedup ─(与已评估因子重复)→ END
                                                          └→ judge ─ accept → END
                                                                   ├ reject → END
                                                                   └ refine ─→ generate / execute（只翻方向）/ step_limit → END
"""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import REVISION_KEYS, ExperimentAgent, ResearchAgent, Thresholds, Verdict
from .checkpoint import run_graph, thread_status
from .evaluate import Evaluator
from .experience import Experience, ExperienceStore
from .llm import LLM
from .trace import Tracer


@dataclass
class Limits:
    max_fix: int = 3
    max_refine: int = 3
    max_steps: int = 30


@dataclass
class CandidateResult:
    hypothesis: dict
    status: str  # accepted | rejected | failed | skipped
    reason: str = ""
    code: str = ""
    metrics: dict | None = None
    test_metrics: dict | None = None
    history: list[dict] = field(default_factory=list)  # 每一轮假设与判定


class CandidateState(TypedDict):
    """全部是可序列化的普通类型，检查点直接存进 SQLite。Agent 只存对话历史，节点里按历史重建；
    因子值（几十万个数）不进状态，需要时按代码重算（同一进程内有缓存）。"""
    hypothesis: dict          # 原始假设（记经验用）
    h: dict                   # 当前假设（研究层调整后的版本）
    note: str                 # 研究层给实验层的修改说明（已过滤指标）
    flip_only: bool           # 研究层只翻了方向：不重新生成代码
    notes: list[str]
    prior_trials: int
    corrected: bool           # 是否套用了历史纠正
    coder_state: dict               # 实验层私有上下文：对话历史（任务、tool_use、tool_result）和待回传的工具结果
    researcher_history: list[dict]  # 研究层私有上下文：只有结构化指标与判定
    steps: int
    refine: int
    fixes: int
    code: str
    pending_fix: Any          # [last_error, fixed_code] 或 None
    exec_ok: bool
    exec: dict                # {ok, error_type, message, checks}
    report: dict | None       # {code, exec, metrics}
    dedup_code: str           # 加入本次去重列表的因子代码（续跑时据此重建去重列表）
    verdict: str
    last_verdict: dict | None
    history: list[dict]
    status: str
    reason: str
    result_code: str
    metrics: dict | None
    test_metrics: dict | None


def hypothesis_text(h: dict) -> str:
    return f"{h.get('name', '')} {h.get('calculation', '')} 字段{','.join(h.get('fields', []))} 窗口{h.get('window')}"


def retrieve_experiences(store: ExperienceStore, h: dict, error_type: str | None = None) -> list[dict]:
    q = hypothesis_text(h)
    hits = store.search(q, kind="success", k=1) + store.search(q, kind="fix", k=2)
    if error_type:
        hits += store.search(q, kind="fix", error_type=error_type, k=2)
    seen, out = set(), []
    for s, e in sorted(hits, key=lambda x: -x[0]):
        if e.kind == "success" and not e.content.get("direction_by_harness"):
            continue  # 旧口径的成功实现里方向是模型自己处理的，照抄会把方向算两遍
        key = (e.kind, e.task, e.error_type, e.ts)
        if key not in seen:
            seen.add(key)
            out.append({"kind": e.kind, "task": e.task, "error_type": e.error_type, "content": e.content, "score": s})
    return out[:3]


class FactorHarness:
    def __init__(self, llm: LLM, evaluator: Evaluator, store: ExperienceStore, tracer: Tracer,
                 limits: Limits = Limits(), thresholds: Thresholds = Thresholds(), checkpointer=None,
                 interrupt_before: list[str] | None = None):
        self.llm, self.ev, self.store, self.tr, self.limits = llm, evaluator, store, tracer, limits
        self.thresholds = thresholds
        self._evaluated: list[tuple[str, object]] = []  # 本次运行已评估过的因子（名字, 预处理后的宽表）
        self._values_cache: dict[str, object] = {}
        self._dedup_codes: set[str] = set()  # 已进去重列表的因子代码（同名假设很常见，不能按名字判断）
        self._dedup_lock = threading.Lock()  # 候选并行：两个相同因子同时到去重这一步，只能放过一个
        self.checkpointer = checkpointer
        self.interrupt_before = interrupt_before or []  # 例如 ["judge"]：判定前停下，人工看过结果后再续跑
        self._graph = self._build_graph()

    def _duplicate_of(self, values, name: str, threshold: float = 0.98) -> tuple[str, float] | None:
        """按因子值去重：研报里同一个因子常被不同观点各写一遍，措辞不同、公式写法不同，只能看算出来的值。
        取符号无关的截面相关（同一因子换个方向也算重复，方向交给研究层判断）。"""
        import numpy as np

        from .data import segment_dates
        from .evaluate import _row_corr

        dates = segment_dates(self.ev.df, self.ev.split.valid)
        full = self.ev._prep(values)
        cur = full.loc[dates]
        with self._dedup_lock:
            for other, prev in self._evaluated:
                c = float(np.nanmean(_row_corr(cur.values, prev.loc[dates, cur.columns].values)))
                if abs(c) >= threshold:
                    return other, round(c, 4)
            self._evaluated.append((name, full))
        return None

    def _experiences(self, h: dict, error_type: str | None = None) -> list[dict]:
        return retrieve_experiences(self.store, h, error_type)

    def _same_rejected(self, h: dict) -> Experience | None:
        """只跳过算出来是同一个因子的假设。文本相似不够：公式里共用 rolling(20).max() 的两个因子可以完全不同；
        方向相反的变体也不能跳过（原因子因显著为负被淘汰时，反过来用恰好可能有效）。"""
        key = factor_key(h)
        for _, e in self.store.search(hypothesis_text(h), kind="rejected", k=5, threshold=0.3):
            if key in (e.content.get("factor_keys") or [e.content.get("factor_key")]):
                return e
        return None

    def _correction(self, h: dict) -> Experience | None:
        """找同一假设的历史纠正记录：文本高度相似，且方向、窗口、字段一致。
        calculation 是模型写的自由文本，同一个算法每次措辞都可能不同，不要求逐字相等。"""
        for _, e in self.store.search(hypothesis_text(h), kind="corrected", k=3, threshold=0.85):
            if _identity(e.content.get("from", {})) == _identity(h):
                return e
        return None

    # ------------------------------------------------------------------ LangGraph
    # 状态里只放层间传递的结构化字段；两个 Agent 各自持有自己的对话历史（上下文隔离），作为对象放进状态。

    def _build_graph(self):
        g = StateGraph(CandidateState)
        g.add_node("check_rejected", self._n_check_rejected)
        g.add_node("apply_correction", self._n_apply_correction)
        g.add_node("generate", self._n_generate)
        g.add_node("execute", self._n_execute)
        g.add_node("fix", self._n_fix)
        g.add_node("give_up", self._n_give_up)
        g.add_node("dedup", self._n_dedup)
        g.add_node("judge", self._n_judge)
        g.add_node("accept", self._n_accept)
        g.add_node("reject", self._n_reject)
        g.add_node("refine", self._n_refine)
        g.add_node("step_limit", self._n_step_limit)

        g.add_edge(START, "check_rejected")
        g.add_conditional_edges("check_rejected", lambda s: END if s["status"] == "skipped" else "apply_correction",
                                [END, "apply_correction"])
        g.add_edge("apply_correction", "generate")
        g.add_edge("generate", "execute")
        g.add_conditional_edges("execute", self._route_execute, ["dedup", "fix", "give_up"])  # 内环：修代码
        g.add_edge("fix", "execute")
        g.add_edge("give_up", END)
        g.add_conditional_edges("dedup", lambda s: END if s["status"] == "skipped" else "judge", [END, "judge"])
        g.add_conditional_edges("judge", lambda s: s["verdict"], ["accept", "reject", "refine"])
        g.add_edge("accept", END)
        g.add_edge("reject", END)
        g.add_conditional_edges("refine", self._route_refine, ["generate", "execute", "step_limit"])  # 外环：修假设
        g.add_edge("step_limit", END)
        return g.compile(checkpointer=self.checkpointer, interrupt_before=self.interrupt_before or None)

    # 节点里用的辅助：按历史重建 Agent；因子值按代码取（缓存 / 重算）
    def _coder(self, s) -> ExperimentAgent:
        return ExperimentAgent(self.llm, self.ev).load(s["coder_state"])

    def _researcher(self, s) -> ResearchAgent:
        a = ResearchAgent(self.llm, self.thresholds)
        a.history = list(s["researcher_history"])
        return a

    def _values(self, code: str):
        if code not in self._values_cache:
            from .sandbox import execute_factor
            self._values_cache[code] = execute_factor(code, self.ev.df).values
        return self._values_cache[code]

    def _route_refine(self, s: "CandidateState") -> str:
        if s["steps"] >= self.limits.max_steps:
            return "step_limit"
        # 只翻了方向：方向由 Harness 施加，原代码不用改，直接重新执行
        return "execute" if s.get("flip_only") else "generate"

    def _route_execute(self, s: "CandidateState") -> str:
        if s["exec_ok"]:
            return "dedup"
        if s["exec"].get("error_type") == "SANDBOX":  # 环境问题，让模型改代码没有意义
            return "give_up"
        if s["fixes"] >= self.limits.max_fix or s["steps"] >= self.limits.max_steps:
            return "give_up"
        return "fix"

    def _n_check_rejected(self, s):
        same = self._same_rejected(s["h"])
        if same:
            reason = f"与已淘汰的因子相同（{same.task}）：{same.content.get('reason', '')}"
            self.tr.log("research.skip_rejected", hypothesis=s["h"], reason=reason)
            return {"status": "skipped", "reason": reason}
        return {}

    def _n_apply_correction(self, s):
        """同一假设以前被研究层纠正过并最终通过：直接从纠正后的版本开始，不再重走一遍错误方向。
        研究层照常判定，纠正在新数据上不成立时会再次调整。"""
        corr = self._correction(s["h"])
        if not corr:
            return {}
        frm, to = corr.content["from"], corr.content["to"]
        self.tr.log("experience.apply_correction", task=corr.task, frm=frm, to=to, from_run=corr.run_id)
        return {"h": {**s["h"], **{k: v for k, v in to.items() if frm.get(k) != v}},  # 只套用被改过的字段
                "note": corr.content.get("note", ""), "corrected": True,
                # 之前为找到这个版本已经试过的次数也要计入多重检验
                "prior_trials": len(corr.content.get("diagnoses", []))}

    def _n_generate(self, s):
        task = {"hypothesis": s["h"], "revision_note": s["note"], "experiences": self._experiences(s["h"])}
        self.tr.log("experiment.retrieve", n=len(task["experiences"]),
                    kinds=[(e["kind"], e["error_type"], e["score"]) for e in task["experiences"]])
        coder = self._coder(s)
        code = coder.implement(task)
        self.tr.log("experiment.submit", via=coder.last_tool_used or "text", round=s["refine"], fix=0)
        return {"code": code, "fixes": 0, "pending_fix": None, "coder_state": coder.state()}

    def _n_execute(self, s):
        """s["code"] 是实验层写的原始实现；按当前假设的方向包一层再执行（之后的评估、去重、测试段、登记都用包过的代码）。"""
        from .sandbox import with_direction

        coder = self._coder(s)
        code = with_direction(s["code"], s["h"].get("direction"))
        report, res = coder.run(code)
        if res.ok:
            self._values_cache[code] = res.values
        self.tr.log("experiment.execute", round=s["refine"], fix=s["fixes"], ok=res.ok, error_type=res.error_type,
                    message=res.message[:300], checks=res.checks, code=s["code"],
                    direction=s["h"].get("direction"))
        return {"steps": s["steps"] + 1, "exec_ok": res.ok, "coder_state": coder.state(),
                "exec": {"ok": res.ok, "error_type": res.error_type, "message": res.message, "checks": res.checks},
                "report": {"code": report.code, "exec": report.exec, "metrics": report.metrics}}

    def _n_fix(self, s):
        ex = s["exec"]
        last = {"code": s["code"], "error_type": ex["error_type"], "message": ex["message"]}
        task = {"hypothesis": s["h"], "revision_note": s["note"], "last_error": last,
                "experiences": self._experiences(s["h"], ex["error_type"])}
        coder = self._coder(s)
        code = coder.implement(task)
        self.tr.log("experiment.submit", via=coder.last_tool_used or "text", round=s["refine"],
                    fix=s["fixes"] + 1)
        if code == last["code"]:
            self.tr.log("experiment.fix_noop")
        return {"code": code, "fixes": s["fixes"] + 1, "pending_fix": [last, code],  # 修复成功后再落经验
                "coder_state": coder.state()}

    def _n_give_up(self, s):
        self.tr.log("experiment.give_up", error_type=s["exec"]["error_type"])
        return {"status": "failed", "reason": f"修复 {s['fixes']} 次仍失败：{s['exec']['error_type']}",
                "result_code": s["code"]}

    def _n_dedup(self, s):
        report, upd = s["report"], {}
        if s["refine"] == 0 and not s["corrected"]:
            dup = self._duplicate_of(self._values(report["code"]), s["h"].get("name", ""))
            if dup:
                self.tr.log("experiment.duplicate", of=dup[0], corr=dup[1])
                return {"status": "skipped", "result_code": report["code"],
                        "reason": f"与本次已评估的「{dup[0]}」是同一个因子（截面相关 {dup[1]:+.3f}）"}
            upd["dedup_code"] = report["code"]
            self._dedup_codes.add(report["code"])
        if s["fixes"] and s["pending_fix"]:
            last, fixed = s["pending_fix"]
            r = self.store.add(Experience("fix", hypothesis_text(s["h"]),
                                          {"error_type": last["error_type"], "error": last["message"][:400],
                                           "before": last["code"], "after": fixed},
                                          error_type=last["error_type"], run_id=self.tr.run_id))
            self.tr.log("experience.add", kind="fix", error_type=last["error_type"], result=r)
        return upd

    def _backtest_tool(self, code: str):
        """研究层的回测工具：没有 Qlib 数据时不提供。"""
        from . import qlib_bridge as qb
        from .strategy import validation_backtest

        if not qb.available():
            return None
        return lambda topk, rebalance: validation_backtest(self.ev, self._values(code), topk, rebalance)

    def _n_judge(self, s):
        """研究层判定：只看结构化结果，看不到代码。"""
        h, report, refine = s["h"], s["report"], s["refine"]
        n_trials = s["prior_trials"] + refine + 1
        researcher = self._researcher(s)
        v = researcher.judge(h, report["metrics"], can_refine=refine < self.limits.max_refine, n_trials=n_trials,
                             backtest=self._backtest_tool(report["code"]))
        for b in v.rule_details.get("backtests") or []:
            self.tr.log("research.backtest", round=refine, **b)
        entry = {"round": refine, "hypothesis": dict(h), "metrics": report["metrics"],
                 "verdict": v.verdict, "failed": v.failed, "diagnosis": v.diagnosis,
                 "revision": v.revision, "fixes": s["fixes"], "first_try_ok": s["fixes"] == 0,
                 "n_trials": n_trials, "revision_check": v.rule_details.get("revision_check")}
        self.tr.log("research.judge", round=refine, verdict=v.verdict, failed=v.failed, n_trials=n_trials,
                    t_valid_required=v.rule_details.get("t_valid_required"),
                    revision_check=v.rule_details.get("revision_check"),
                    diagnosis=v.diagnosis, revision=v.revision, note=v.note,
                    valid_rank_ic=report["metrics"]["valid"].get("rank_ic"))
        return {"verdict": v.verdict, "last_verdict": asdict(v), "history": s["history"] + [entry],
                "researcher_history": researcher.history}

    def _n_accept(self, s):
        hypothesis, h, report = s["hypothesis"], s["h"], s["report"]
        frm, to = _revisable(hypothesis), _revisable(h)
        if s["refine"] and frm != to:
            r = self.store.add(Experience("corrected", hypothesis_text(hypothesis),
                                          {"from": frm, "to": to, "note": "；".join(n for n in s["notes"] if n),
                                           "diagnoses": [x["diagnosis"] for x in s["history"][:-1]]},
                                          run_id=self.tr.run_id))
            self.tr.log("experience.add", kind="corrected", frm=frm, to=to, result=r)
        test = self.ev.evaluate_test(self._values(report["code"]), key=report["code"])
        self.tr.log("final.test", test=test)
        # 经验里存实验层的原始实现（不含方向），下次参考时口径一致
        self.store.add(Experience("success", hypothesis_text(h), {"code": s["code"], "direction_by_harness": True},
                                  run_id=self.tr.run_id))
        return {"status": "accepted", "result_code": report["code"], "metrics": report["metrics"], "test_metrics": test}

    def _n_reject(self, s):
        v, report = Verdict(**s["last_verdict"]), s["report"]
        reason = f"{','.join(v.failed)}：{v.diagnosis}"
        self.store.add(Experience("rejected", hypothesis_text(s["hypothesis"]),
                                  # 原始版本和调整后的最终版本都记下：原始假设经过调整仍被淘汰，下次同样跳过
                                  {"reason": reason, "factor_key": factor_key(s["h"]),
                                   "factor_keys": [factor_key(s["hypothesis"]), factor_key(s["h"])]},
                                  run_id=self.tr.run_id))
        self.tr.log("experience.add", kind="rejected")
        return {"status": "rejected", "reason": reason, "result_code": report["code"], "metrics": report["metrics"]}

    def _n_refine(self, s):
        """只把修改后的假设和一句说明交回实验层。"""
        v = s["last_verdict"]
        changed = {k for k, x in v["revision"].items() if s["h"].get(k) != x}
        upd = {"refine": s["refine"] + 1, "h": {**s["h"], **v["revision"]}, "note": v["note"],
               "notes": s["notes"] + [v["note"]], "flip_only": changed == {"direction"}}
        if upd["flip_only"]:
            upd.update(fixes=0, pending_fix=None)
            self.tr.log("research.flip_only", direction=v["revision"].get("direction"))
        return upd

    def _n_step_limit(self, s):
        return {"status": "failed", "reason": "达到总步数上限"}

    def run_candidate(self, hypothesis: dict, thread_id: str | None = None) -> CandidateResult:
        h = dict(hypothesis)
        state: CandidateState = {
            "hypothesis": dict(hypothesis), "h": h, "note": "", "flip_only": False, "notes": [], "prior_trials": 0, "corrected": False,
            "coder_state": {}, "researcher_history": [],        # 两层各自的上下文，互不可见
            "steps": 0, "refine": 0, "fixes": 0, "code": "", "pending_fix": None, "exec_ok": False,
            "exec": {}, "report": None, "dedup_code": "", "verdict": "", "last_verdict": None, "history": [],
            "status": "failed", "reason": "", "result_code": "", "metrics": None, "test_metrics": None,
        }
        # 每轮修复 2 个节点、每轮调整 5 个节点，按总步数给足递归上限；真正的兜底是 max_steps
        out = run_graph(self._graph, state, thread_id, 12 * self.limits.max_steps + 20, self.tr, h.get("name", ""))
        if out.get("dedup_code") and out["dedup_code"] not in self._dedup_codes:
            # 从检查点恢复的已完成候选：把它加回本次运行的去重列表
            self._duplicate_of(self._values(out["dedup_code"]), h.get("name", ""))
            self._dedup_codes.add(out["dedup_code"])
        return CandidateResult(h, out["status"], out["reason"], out["result_code"], out["metrics"],
                               out["test_metrics"], out["history"])

    @staticmethod
    def thread_id(i: int, h: dict) -> str:
        """候选的 thread_id：序号 + 假设内容的哈希。续跑时假设列表变了，不会接到别的候选的检查点上。"""
        return f"cand-{i}-" + hashlib.sha1(json.dumps(h, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:10]

    def run(self, hypotheses: list[dict], workers: int = 1) -> list[CandidateResult]:
        """workers > 1：候选之间并行（线程池；时间主要花在等模型返回，执行因子在子进程里）。
        代价：同一批里的候选看不到彼此刚沉淀的经验；去重按谁先到去重节点算。结果按输入顺序返回。"""
        def one(i, h):
            tid = self.thread_id(i, h) if self.checkpointer else None
            with self.tr.bind(f"cand-{i}") if workers > 1 else _null():
                self.tr.log("candidate.start", hypothesis=h, thread=tid,
                            checkpoint=thread_status(self._graph, tid) if tid else None)
                r = self.run_candidate(h, tid)
                self.tr.log("candidate.end", status=r.status, reason=r.reason)
            return r

        if workers <= 1:
            return [one(i, h) for i, h in enumerate(hypotheses)]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cand") as pool:
            return list(pool.map(one, range(len(hypotheses)), hypotheses))


def _null():
    from contextlib import nullcontext
    return nullcontext()


def factor_key(h: dict) -> list:
    """因子身份：去掉空白和外层取负后的公式 + 实际方向（direction × 公式符号）+ 字段 + 窗口。"""
    calc = re.sub(r"\s+", "", str(h.get("calculation", ""))).lower()
    sign = 1
    while True:
        m = re.fullmatch(r"-\((.*)\)", calc)
        if m and _balanced(m.group(1)):
            calc, sign = m.group(1), -sign
        elif calc.startswith("-") and not calc.startswith("--"):
            calc, sign = calc[1:], -sign
        else:
            break
    while calc.startswith("(") and calc.endswith(")") and _balanced(calc[1:-1]):
        calc = calc[1:-1]
    return [calc, sign * int(h.get("direction") or 1), sorted(h.get("fields") or []), h.get("window")]


def _balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth < 0:
            return False
    return depth == 0


def _identity(h: dict) -> tuple:
    return h.get("direction"), h.get("window"), tuple(sorted(h.get("fields") or []))


def _revisable(h: dict) -> dict:
    return {k: h.get(k) for k in REVISION_KEYS}


def results_json(results: list[CandidateResult]) -> str:
    return json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2, default=str)
