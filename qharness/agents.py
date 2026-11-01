"""实验层 / 研究层两个 Agent，各自持有独立上下文，不共享历史。

- ExperimentAgent：根据假设写因子代码、调用工具执行和回测、出错自己修。它的提示词里
  不出现任何评价指标和阈值，没有动机为了指标去「走捷径」。
- ResearchAgent：只看结构化回测结果（不看代码），按预注册规则判定，模型只负责诊断和
  提出对假设的修改（方向 / 窗口 / 计算方式），不能推翻规则判定。

层间只通过固定 schema 传递：研究层 → 实验层只有「修改后的假设 + 一句修改说明」。
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from statistics import NormalDist

from .data import describe_fields, factor_columns
from .evaluate import Evaluator
from .llm import LLM, extract_code, extract_json
from .sandbox import ExecResult, execute_factor

HYPOTHESIS_KEYS = ("name", "fields", "calculation", "window", "direction", "rationale", "condition")
CODER_KEYS = tuple(k for k in HYPOTHESIS_KEYS if k != "direction")  # 方向由 Harness 施加，实验层看不到
REVISION_KEYS = ("direction", "window", "calculation", "fields")
METRIC_WORDS = re.compile(r"\b(rank_?ic|ic|icir|rank_?icir|ir)\b|阈值|分层|换手|收益率?\s*[\d.]|相关性\s*[\d.]|\d\.\d{2,}", re.I)


def sanitize_note(note: str) -> str:
    """研究层的修改说明里去掉提到指标和数值的句子，只留对假设的描述。"""
    parts = re.split(r"(?<=[。；;,.，])", note or "")
    return "".join(p for p in parts if not METRIC_WORDS.search(p)).strip()


# ---------------------------------------------------------------- 实验层

CODER_SYSTEM = """你是因子开发工程师。根据因子假设写 Python 代码，{submit}
数据约定：
- 定义 def factor(df: pd.DataFrame) -> pd.Series
- df 的索引是 MultiIndex (datetime, instrument)，按 (datetime, instrument) 排序；列只有 {columns}
- 字段含义：{field_desc}
- 价格是后复权价；停牌日整行为 NaN；amount 单位元，volume 单位股；财务字段已按公告日对齐，直接用当天的值即可，不要再 shift
- 返回与 df.index 对齐的一列数（每天每只股票一个分数）
规则：
- 只能 import numpy / pandas / math
- 所有时序操作（rolling、shift、pct_change、diff 等）必须先 groupby(level='instrument')
- 注意：s.groupby(level='instrument').rolling(n).mean() 返回的索引会多一层 instrument，
  要写成 s.groupby(level='instrument').transform(lambda x: x.rolling(n).mean())，
  或 .rolling(n).mean().droplevel(0).reindex(df.index)；截面排名用 s.groupby(level='datetime').rank(pct=True)
- 只能用当天及以前的数据：不能 shift 负数、不能居中窗口、不能做全样本标准化或 bfill
- 不要处理因子方向：按 calculation 的描述原样实现（描述里写了取负就取负），方向由 Harness 统一施加"""


SUBMIT_TEXT = "只输出一个 ```python 代码块。"
SUBMIT_TOOL = ("写好后调用 run_factor 工具提交完整代码（参数 code），工具会在沙箱里执行并返回执行状态和口径检查结果；"
               "执行失败时根据工具返回的报错修改后重新调用。不要在正文里贴代码。")

# 实验层唯一的工具。返回值只有执行状态和口径检查（覆盖率、行序不变、未来函数截断），不含任何收益指标：
# 回测指标由 Harness 在工具这一侧计算，直接交给研究层，写代码的一层看不到评价标准。
RUN_FACTOR_TOOL = {
    "name": "run_factor",
    "description": "在沙箱里执行因子代码：静态检查（禁止的导入/未来函数写法）→ 运行 → 覆盖率与截面常数检查 → "
                   "换行序重算（检查是否按股票分组）→ 截断数据重算（检查未来函数）。返回执行状态、报错和检查结果。",
    "input_schema": {"type": "object", "required": ["code"],
                     "properties": {"code": {"type": "string", "description": "完整 Python 代码，定义 def factor(df) -> pd.Series"}}},
}

# 有 Qlib 数据时多给一个工具：因子写成 Qlib 表达式，由 Qlib 计算，沙箱的口径检查照常
SUBMIT_TOOL_QLIB = ("写好后调用工具提交：run_factor（Python 代码）或 run_qlib_factor（Qlib 表达式），"
                    "能用 Qlib 表达式写清楚的因子优先用表达式。工具返回执行状态和口径检查结果；"
                    "执行失败时根据报错修改后重新调用。不要在正文里贴代码。")
QLIB_HELP = """Qlib 表达式（run_qlib_factor）：
- 字段写成 {fields}；算子：Ref(x,n)（n 天前，n 必须 ≥ 0）、Mean/Std/Var/Skew/Sum/Max/Min/Med/Rank/Slope/Rsquare/Resi/WMA/EMA(x,n)、
  Corr/Cov(x,y,n)、Delta(x,n)、Abs/Sign/Log(x)、Power(x,p)、If(cond,a,b)、Greater/Less(x,y)，以及 + - * / > < 等
- 窗口算子都是按股票的时间序列计算（Rank 是时序分位，不是截面排名）；窗口不足时用已有数据
- 例：5日反转 -($close/Ref($close,5)-1)；20日波动 Std($close/Ref($close,1)-1,20)；量价相关 Corr($close,$volume,10)"""
RUN_QLIB_TOOL = {
    "name": "run_qlib_factor",
    "description": "用 Qlib 计算因子表达式，并做与 run_factor 相同的口径检查（覆盖率、截面常数、截断数据重算查未来函数）。"
                   "返回执行状态、报错和检查结果。",
    "input_schema": {"type": "object", "required": ["expression"],
                     "properties": {"expression": {"type": "string", "description": "Qlib 表达式，如 -($close/Ref($close,5)-1)"}}},
}


def _show_code(code: str) -> str:
    from .qlib_bridge import expr_of, is_expr

    return f"Qlib 表达式：`{expr_of(code)}`" if is_expr(code) else f"\n```python\n{code}```"


@dataclass
class ExperimentReport:
    code: str
    exec: dict
    metrics: dict | None = None


class ExperimentAgent:
    """use_tools=True（接口支持时默认）：模型通过原生 tool calling 调用 run_factor 提交代码，执行结果以 tool_result
    回传；循环和上限仍由编排图控制：每次 tool_use 对应图里的一次执行节点，tool_result 随下一条 user 消息回传。"""

    def __init__(self, llm: LLM, evaluator: Evaluator, use_tools: bool | None = None):
        self.llm, self.evaluator = llm, evaluator
        self.use_tools = getattr(llm, "supports_tools", False) if use_tools is None else use_tools
        self.history: list[dict] = []  # 本层私有上下文：只有任务、代码、执行报错
        self.prompts: list[str] = []
        self.pending_tool_id: str | None = None  # 上一次 tool_use 的 id，执行结果还没回传
        self.pending_result: dict | None = None  # 随下一条 user 消息回传的 tool_result
        self.last_tool_used = False

    def state(self) -> dict:
        return {"history": self.history, "pending_tool_id": self.pending_tool_id, "pending_result": self.pending_result}

    def load(self, st: dict | None) -> "ExperimentAgent":
        st = st or {}
        self.history = list(st.get("history") or [])
        self.pending_tool_id, self.pending_result = st.get("pending_tool_id"), st.get("pending_result")
        return self

    def build_prompt(self, task: dict) -> str:
        h = {k: task["hypothesis"].get(k) for k in CODER_KEYS}
        parts = [f"因子假设：{json.dumps(h, ensure_ascii=False)}"]
        if task.get("revision_note"):
            parts.append(f"研究员对假设的修改说明：{task['revision_note']}")
        for e in task.get("experiences", []):
            if e["kind"] == "success":
                parts.append(f"参考：相似任务的成功实现（{e['task']}）：{_show_code(e['content']['code'])}")
            elif e["kind"] == "fix":
                parts.append(f"参考：相似错误的修复案例 [{e['error_type']}] {e['content']['error'][:200]}\n"
                             f"修复后：{_show_code(e['content']['after'])}")
        if task.get("last_error"):
            le = task["last_error"]
            if self.use_tools and self.pending_result:  # 报错已在 tool_result 里回传
                parts.append(f"上次提交执行失败 [{le['error_type']}]。请先说明失败原因，再调用工具提交修正后的完整版本。")
            else:
                parts.append(f"上一版代码：{_show_code(le['code'])}\n执行失败 [{le['error_type']}]：{le['message'][:800]}\n"
                             "请先说明失败原因，再给出修正后的完整代码。")
        return "\n\n".join(parts)

    def implement(self, task: dict) -> str:
        prompt = self.build_prompt(task)
        self.prompts.append(prompt)
        columns = ", ".join(factor_columns(self.evaluator.df))
        if not self.use_tools:
            system = CODER_SYSTEM.format(columns=columns, submit=SUBMIT_TEXT,
                                         field_desc=describe_fields(factor_columns(self.evaluator.df)))
            msgs = [{"role": "system", "content": system}, *self.history[-4:], {"role": "user", "content": prompt}]
            reply = self.llm.chat("coder", msgs, task)
            self.history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
            return extract_code(reply)

        from . import qlib_bridge as qb

        tools, submit = [RUN_FACTOR_TOOL], SUBMIT_TOOL
        if qb.available():
            tools, submit = [RUN_FACTOR_TOOL, RUN_QLIB_TOOL], SUBMIT_TOOL_QLIB
        system = CODER_SYSTEM.format(columns=columns, submit=submit,
                                     field_desc=describe_fields(factor_columns(self.evaluator.df)))
        if len(tools) > 1:
            system += "\n" + QLIB_HELP.format(fields=", ".join("$" + c for c in factor_columns(self.evaluator.df)))
        # 上一次工具调用的结果必须放在下一条 user 消息的最前面（tool_use / tool_result 成对）
        content = ([self.pending_result] if self.pending_result else []) + [{"type": "text", "text": prompt}]
        self.pending_result = None
        msgs = self.history + [{"role": "user", "content": content}]
        blocks = self.llm.respond("coder", system, msgs, tools, task)
        self.history = msgs + [{"role": "assistant", "content": blocks}]
        names = {t["name"] for t in tools}
        call = next((b for b in blocks if b.get("type") == "tool_use" and b.get("name") in names), None)
        if call:
            self.pending_tool_id, self.last_tool_used = call["id"], call["name"]
            if call["name"] == "run_qlib_factor":
                return qb.PREFIX + str(call.get("input", {}).get("expression", ""))
            return str(call.get("input", {}).get("code", ""))
        self.pending_tool_id, self.last_tool_used = None, False  # 没调工具：退回从正文里取代码
        return extract_code("".join(b.get("text", "") for b in blocks))

    def run(self, code: str) -> tuple[ExperimentReport, ExecResult]:
        """执行工具：执行 + 口径检查（结果回传给本层）；回测指标交给编排器，不进本层上下文。"""
        res = execute_factor(code, self.evaluator.df)
        metrics = self.evaluator.evaluate(res.values) if res.ok else None
        if self.use_tools:
            if self.pending_tool_id:
                self.pending_result = {"type": "tool_result", "tool_use_id": self.pending_tool_id, "is_error": not res.ok,
                                       "content": json.dumps(tool_view(res), ensure_ascii=False, default=str)}
                self.pending_tool_id = None
        else:
            self.history.append({"role": "user", "content": f"执行结果：{'成功' if res.ok else res.error_type}"})
        return ExperimentReport(code, res.public(), metrics), res


def tool_view(res: ExecResult) -> dict:
    """run_factor 工具返回给实验层的内容：执行状态、报错、口径检查。"""
    out = {"ok": res.ok, "checks": res.checks}
    if not res.ok:
        out.update(error_type=res.error_type, message=res.message[:1500])
    return out


# ---------------------------------------------------------------- 研究层

@dataclass(frozen=True)
class Thresholds:
    """预注册判定标准：在看到任何结果之前就定好，判定以规则为准。

    验证段会被研究层反复查看（每调整一轮就看一次），等于在验证段上做多次检验。
    所以验证段的显著性门槛随试验次数 n 提高（Bonferroni）：t ≥ Φ⁻¹(1 - α / 2n)，n=1 时 1.96，n=4 时 2.50。
    """
    min_rank_ic: float = 0.02
    min_t_train: float = 2.0
    alpha_valid: float = 0.05
    min_positive_year_ratio: float = 0.6  # 训练段各年份 RankIC 为正的比例
    max_corr_with_library: float = 0.7
    max_turnover: float = 0.6

    def t_valid(self, n_trials: int) -> float:
        return round(NormalDist().inv_cdf(1 - self.alpha_valid / (2 * max(1, n_trials))), 3)


@dataclass
class Verdict:
    verdict: str  # accept | refine | reject
    failed: list[str]
    diagnosis: str
    revision: dict | None = None
    note: str = ""
    rule_details: dict = field(default_factory=dict)


RESEARCH_SYSTEM = """你是量化研究员，只根据结构化的回测结果诊断因子假设，不看代码。
指标说明：评估前因子已做市值+行业中性化（neutralized=true 时）；rank_ic_raw 是未中性化的 RankIC，
二者差很多说明因子主要是市值/行业暴露；rank_ic_t_nw 是考虑标签重叠的 t 值；rank_ic_by_year 是分年 RankIC；
corr_with_library 是与已有因子（reversal_5d 短期反转、momentum_20d 中期动量、low_vol_20d 低波动、
illiquidity_20d 非流动性、low_turnover_20d 低换手）的截面相关。
输出 JSON：
{"findings": [{"id": 1, "finding": "一条诊断", "evidence": "依据的指标名"}...],
 "diagnosis": "一段总结",
 "revision": null 或 {"addresses": [对应的 finding id], "direction"?: 1/-1, "window"?: int, "calculation"?: "...", "fields"?: [...], "note": "对假设怎么改的一句话"}}
要求：
- 每个修改都必须针对某条 finding（addresses 不能为空），说清楚为什么这样改能解决它；只是换参数碰运气不算。
- 训练段和验证段方向不一致时，问题是不稳定，不是方向反了，不要靠翻转方向修。
- 看不出有改进路径时 revision 给 null。
- note 只描述对假设的修改，不要提任何指标名和数值。"""


# 研究层的工具：候选在训练段和验证段方向都对时，由研究层决定组合参数，在验证段做策略回测。
# 只在验证段（不碰测试段）；结果写进诊断和 trace，是否通过仍由规则决定。写代码的一层拿不到这个工具。
RUN_BACKTEST_TOOL = {
    "name": "run_qlib_backtest",
    "description": "用 Qlib 回测（次日开盘成交、整手、双边费用、涨跌停和停牌不成交）对比基线和基线 + 候选因子，只在验证段。"
                   "返回两者的年化超额、信息比率、超额最大回撤和换手。",
    "input_schema": {"type": "object", "required": ["topk", "rebalance"],
                     "properties": {"topk": {"type": "integer", "description": "持仓只数，10 到 50"},
                                    "rebalance": {"type": "integer", "description": "调仓间隔（交易日），1 到 20"}}},
}
BACKTEST_SYSTEM = ("你是量化研究员。候选因子在训练段和验证段方向都对，现在设计验证段策略回测：结合换手和分组收益选择持仓只数和调仓间隔，"
                   "调用 run_qlib_backtest。第一次结果需要换参数确认稳健性时可以再调用一次，否则直接用一句话说明结论。")
MAX_BACKTESTS = 2


class ResearchAgent:
    def __init__(self, llm: LLM, thresholds: Thresholds = Thresholds()):
        self.llm, self.th = llm, thresholds
        self.history: list[dict] = []

    def rules(self, metrics: dict, n_trials: int = 1) -> dict:
        th, failed = self.th, []
        tr, va = metrics["train"], metrics["valid"]
        if not (tr.get("valid") and va.get("valid")):
            return {"failed": ["insufficient"], "passed": False}
        t_valid = th.t_valid(n_trials)
        for seg, m in (("train", tr), ("valid", va)):
            if m["rank_ic"] < th.min_rank_ic:
                failed.append(f"weak_{seg}")
            t = m.get("rank_ic_t_nw")
            if t is None or not t == t or t < (th.min_t_train if seg == "train" else t_valid):
                failed.append(f"insignificant_{seg}")
        years = list((tr.get("rank_ic_by_year") or {}).values())
        if len(years) >= 2 and sum(y > 0 for y in years) / len(years) < th.min_positive_year_ratio:
            failed.append("unstable_years")
        if max(tr["max_abs_corr_with_library"], va["max_abs_corr_with_library"]) > th.max_corr_with_library:
            failed.append("high_corr")
        if va["turnover"] > th.max_turnover:
            failed.append("high_turnover")
        return {"failed": failed, "passed": not failed, "n_trials": n_trials, "t_valid_required": t_valid,
                "thresholds": asdict(th)}

    @staticmethod
    def facts(metrics: dict) -> list[str]:
        """由代码算好的事实，直接给模型，避免它从一大段 JSON 里读错符号、看错哪个因子最相关。"""
        tr, va = metrics["train"], metrics["valid"]
        if not (tr.get("valid") and va.get("valid")):
            return ["样本不足，无法评估"]
        sign = lambda x: "正" if x > 0 else "负"  # noqa: E731
        out = [f"训练段 RankIC 为{sign(tr['rank_ic'])}（t={tr.get('rank_ic_t_nw')}），验证段为{sign(va['rank_ic'])}"
               f"（t={va.get('rank_ic_t_nw')}）：两段方向{'一致' if (tr['rank_ic'] > 0) == (va['rank_ic'] > 0) else '不一致'}"]
        if (tr["rank_ic"] < 0) and (va["rank_ic"] < 0):
            t = max(abs(tr.get("rank_ic_t_nw") or 0), abs(va.get("rank_ic_t_nw") or 0))
            out.append("两段 RankIC 都为负：按当前方向使用是反的，取反后两段都为正"
                       if t >= 2 else "两段 RankIC 都为负但都不显著：翻转方向也不会得到有效因子")
        years = tr.get("rank_ic_by_year") or {}
        if years:
            neg = [y for y, v in years.items() if v < 0]
            out.append(f"训练段分年 RankIC：{years}，为负的年份 {neg or '无'}")
        if va.get("rank_ic_raw") is not None:
            gap = abs(va["rank_ic_raw"]) - abs(va["rank_ic"])
            out.append(f"验证段中性化前 RankIC {va['rank_ic_raw']:+.4f}，中性化后 {va['rank_ic']:+.4f}"
                       + ("：中性化后明显变弱，主要是市值/行业暴露" if gap > 0.01 else ""))
        corr = {**(tr.get("corr_with_library") or {})}
        if corr:
            k = max(corr, key=lambda x: abs(corr[x]))
            out.append(f"与已有因子相关性最高的是 {k}（{corr[k]:+.2f}）")
        return out

    @staticmethod
    def check_revision(hypothesis: dict, rev: dict | None, findings: list, metrics: dict) -> tuple[dict | None, str]:
        """规则层面过滤掉站不住的修改：没有对应诊断的、训练验证方向不一致时翻转方向的、和原假设一样的。"""
        if not rev:
            return None, "no_revision"
        ids = {f.get("id") for f in findings if isinstance(f, dict)}
        if not rev.get("addresses") or not set(rev["addresses"]) & ids:
            return None, "revision_without_finding"
        clean = {k: rev[k] for k in REVISION_KEYS if k in rev}
        if not clean or all(hypothesis.get(k) == v for k, v in clean.items()):
            return None, "revision_noop"
        if clean.get("direction") is not None and clean["direction"] != hypothesis.get("direction"):
            tr, va = metrics["train"], metrics["valid"]
            if not (tr.get("rank_ic", 0) < 0 and va.get("rank_ic", 0) < 0):  # 两段都显示方向反了，才允许翻转
                return None, "direction_flip_unsupported"
            t = max(abs(tr.get("rank_ic_t_nw") or 0), abs(va.get("rank_ic_t_nw") or 0))
            if t < Thresholds().min_t_train:  # 两段都为负但都不显著：翻过来也是噪声
                return None, "direction_flip_insignificant"
        return clean, "ok"

    def design_backtests(self, hypothesis: dict, metrics: dict, runner) -> list[dict]:
        """研究层自己决定回测参数并调用工具，最多 MAX_BACKTESTS 次。模型或回测出错只记下来，不影响判定。"""
        tr, va = metrics["train"], metrics["valid"]
        if not (tr.get("valid") and va.get("valid") and tr.get("rank_ic", 0) > 0 and va.get("rank_ic", 0) > 0):
            return []
        msgs = [{"role": "user", "content": f"假设：{json.dumps({k: hypothesis.get(k) for k in HYPOTHESIS_KEYS}, ensure_ascii=False)}\n"
                                            f"关键事实：\n- " + "\n- ".join(self.facts(metrics))}]
        out: list[dict] = []
        try:
            for _ in range(MAX_BACKTESTS):
                blocks = self.llm.respond("researcher", BACKTEST_SYSTEM, msgs, [RUN_BACKTEST_TOOL], {"hypothesis": hypothesis})
                call = next((b for b in blocks if b.get("type") == "tool_use" and b.get("name") == RUN_BACKTEST_TOOL["name"]), None)
                if call is None:
                    break
                args = call.get("input") or {}
                topk = min(50, max(10, int(args.get("topk", 20))))
                rebalance = min(20, max(1, int(args.get("rebalance", 5))))
                res = runner(topk, rebalance)
                out.append(res)
                msgs += [{"role": "assistant", "content": blocks},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"],
                                                       "content": json.dumps(res, ensure_ascii=False, default=str)}]}]
        except NotImplementedError:
            return out
        except Exception as e:  # noqa: BLE001
            out.append({"error": f"{type(e).__name__}: {e}"})
        return out

    def judge(self, hypothesis: dict, metrics: dict, can_refine: bool, n_trials: int = 1, backtest=None) -> Verdict:
        rule = self.rules(metrics, n_trials)
        backtests = self.design_backtests(hypothesis, metrics, backtest) if backtest else []
        ctx = {"hypothesis": {k: hypothesis.get(k) for k in HYPOTHESIS_KEYS}, "metrics": metrics, "rule_verdict": rule}
        prompt = (f"假设：{json.dumps(ctx['hypothesis'], ensure_ascii=False)}\n"
                  f"关键事实（已由程序核对，以此为准）：\n- " + "\n- ".join(self.facts(metrics)) + "\n"
                  f"回测结果：{json.dumps(metrics, ensure_ascii=False)}\n"
                  f"规则判定：{json.dumps(rule, ensure_ascii=False)}"
                  + (f"\n验证段策略回测（只作诊断依据，不改变规则判定）：{json.dumps(backtests, ensure_ascii=False, default=str)}"
                     if backtests else ""))
        msgs = [{"role": "system", "content": RESEARCH_SYSTEM}, *self.history[-4:], {"role": "user", "content": prompt}]
        try:
            out = extract_json(self.llm.chat("researcher", msgs, ctx))
        except Exception as e:  # noqa: BLE001  模型异常时保守处理：只用规则
            out = {"diagnosis": f"研究层模型输出无法解析（{type(e).__name__}），仅按规则判定", "revision": None}
        self.history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": json.dumps(out, ensure_ascii=False)}]

        rev = out.get("revision") or None
        clean_rev, rev_check = self.check_revision(hypothesis, rev, out.get("findings") or [], metrics)
        rule["revision_check"] = rev_check
        rule["findings"] = out.get("findings") or []
        rule["backtests"] = backtests
        note = sanitize_note((rev or {}).get("note", ""))
        if rule["passed"]:
            return Verdict("accept", [], out.get("diagnosis", ""), None, "", rule)
        if "high_corr" in rule["failed"] or "insufficient" in rule["failed"]:
            return Verdict("reject", rule["failed"], out.get("diagnosis", ""), None, "", rule)
        if clean_rev and can_refine:
            return Verdict("refine", rule["failed"], out.get("diagnosis", ""), clean_rev, note, rule)
        return Verdict("reject", rule["failed"], out.get("diagnosis", ""), None, "", rule)
