"""实验层评测集（`qharness eval-coder`）：跑通不等于算对。每题有人工写的参考实现，模型写的代码全部用代码判分，分四层（思路借鉴 AlphaQT-Bench）：

- 执行：能跑、覆盖率够、截面不是常数
- 结构：换行序结果不变（时序操作按股票分组）
- 未来函数：截断数据后结果不变
- 功能：和参考实现的每日截面秩相关（排名决定选股，单调变换算同一个因子）

只判最终结果，不看步骤；参考实现不进提示词，功能层的结果也不反馈给模型。陷阱题的描述容易写成未来函数（比如「历史均值」写成全样本均值）。
只用验证段结束之前的数据，不碰测试段，所以不用考虑测试锁。

经验库开关对照：同一批题分 on / off 两组，每轮题序相同。on 从现有经验库的副本开始，同一轮内继续积累；每题的经验不能来自这道题本身。
成功实现只在功能层通过后才写入，相当于所有经验都经过了验证，是经验库效果的乐观估计（真实流程要研究层通过才写入）。
"""
from __future__ import annotations

import json
import random
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .agents import ExperimentAgent
from .data import factor_columns
from .evaluate import _row_corr, _row_rank, _wide
from .experience import Experience, ExperienceStore, similarity
from .orchestrator import Limits, hypothesis_text, retrieve_experiences
from .research_eval import _h
from .sandbox import ErrorType, execute_factor

MATCH_MEDIAN, MATCH_P10, MATCH_OVERLAP = 0.99, 0.95, 0.9
SAME_TASK = 0.7  # 经验和当前题相似到这个程度算同一题，不给模型看
LAYERS = ("execute", "structure", "causal", "functional")

_HEAD = """def factor(df):
    def roll(s, n, how):
        return s.groupby(level='instrument').transform(lambda x: x.rolling(n).agg(how))
    c, v = df['close'], df['volume']
    r = c / c.groupby(level='instrument').shift(1) - 1
"""


@dataclass
class Case:
    name: str
    level: str  # basic | medium | hard | trap
    hypothesis: dict
    reference: str
    note: str = ""


def _case(name, level, calc, body, window=1, fields=("close",), note=""):
    code = _HEAD + "".join(f"    {line}\n" for line in body.strip().splitlines())
    return Case(name, level, _h(name, calc, window, 1, fields), code, note)


_MOM20 = "mom = c / c.groupby(level='instrument').shift(20) - 1"

CASES = [
    # ---------------- 基础：单字段、单窗口
    _case("20日动量", "basic", "今日收盘价 / 20个交易日前的收盘价 - 1", f"{_MOM20}\nreturn mom", 20),
    _case("5日反转", "basic", "-(今日收盘价 / 5个交易日前的收盘价 - 1)", "return -(c / c.groupby(level='instrument').shift(5) - 1)", 5),
    _case("20日波动率", "basic", "过去20个交易日日收益率的标准差", "return roll(r, 20, 'std')", 20),
    _case("Amihud非流动性", "basic", "过去20个交易日 |日收益率| / 当日成交额 的平均值", "return roll(r.abs() / df['amount'], 20, 'mean')", 20, ("close", "amount")),
    _case("20日平均换手", "basic", "过去20个交易日换手率的平均值", "return roll(df['turnover'], 20, 'mean')", 20, ("turnover",)),
    _case("EP", "basic", "1 / pe_ttm（亏损股保留负值）", "return 1 / df['pe_ttm']", 1, ("pe_ttm",)),
    _case("10日振幅", "basic", "过去10个交易日 (最高价 - 最低价) / 收盘价 的平均值", "return roll((df['high'] - df['low']) / c, 10, 'mean')", 10, ("high", "low", "close")),
    _case("量比", "basic", "过去5个交易日平均成交量 / 过去60个交易日平均成交量", "return roll(v, 5, 'mean') / roll(v, 60, 'mean')", 60, ("volume",)),
    # ---------------- 中等：多字段组合、截面操作
    _case("量价相关", "medium", "过去20个交易日，日收益率和成交量日变化率的相关系数",
          "dv = v / v.groupby(level='instrument').shift(1) - 1\ndv = dv.replace([np.inf, -np.inf], np.nan)\n"
          "return (roll(r * dv, 20, 'mean') - roll(r, 20, 'mean') * roll(dv, 20, 'mean')) / (roll(r, 20, 'std') * roll(dv, 20, 'std'))",
          20, ("close", "volume")),
    _case("距60日高点", "medium", "今日收盘价 / 过去60个交易日最高收盘价（含今日） - 1", "return c / roll(c, 60, 'max') - 1", 60),
    _case("成交额减动量分位", "medium", "当日成交额的截面分位 - 20日动量（今日收盘/20日前收盘-1）的截面分位",
          f"{_MOM20}\nreturn df['amount'].groupby(level='datetime').rank(pct=True) - mom.groupby(level='datetime').rank(pct=True)",
          20, ("close", "amount")),
    _case("下行波动", "medium", "过去20个交易日 min(日收益率, 0) 的平方和，再开平方", "return roll(r.clip(upper=0) ** 2, 20, 'sum') ** 0.5", 20),
    _case("上影线", "medium", "过去20个交易日 (最高价 - max(开盘价, 收盘价)) / 收盘价 的平均值",
          "return roll((df['high'] - df[['open', 'close']].max(axis=1)) / c, 20, 'mean')", 20, ("open", "high", "close")),
    _case("最大日收益", "medium", "过去20个交易日日收益率的最大值", "return roll(r, 20, 'max')", 20),
    _case("收益偏度", "medium", "过去20个交易日日收益率的偏度", "return roll(r, 20, 'skew')", 20),
    _case("ROE变化", "medium", "当日 roe - 60个交易日前的 roe", "return df['roe'] - df['roe'].groupby(level='instrument').shift(60)", 60, ("roe",)),
    # ---------------- 困难：有状态、条件逻辑、回归
    _case("收益EMA", "hard", "日收益率的指数移动平均，span=20，adjust=False",
          "return r.groupby(level='instrument').transform(lambda x: x.ewm(span=20, adjust=False).mean())", 20),
    _case("连涨天数", "hard", "截至当日连续上涨（日收益率 > 0）的交易日数；当日没涨或停牌时为 0",
          "up = (r > 0).astype(float).where(c.notna())\nup = up.fillna(0.0)\n"
          "return up.groupby(level='instrument').transform(lambda x: x.groupby((x == 0).cumsum()).cumsum())", None),
    _case("市场Beta", "hard", "过去60个交易日，个股日收益率对市场收益率回归的斜率；市场收益率是当日全部股票日收益率的等权平均",
          "m = r.groupby(level='datetime').transform('mean').where(r.notna())\n"
          "return (roll(r * m, 60, 'mean') - roll(r, 60, 'mean') * roll(m, 60, 'mean')) / (roll(m * m, 60, 'mean') - roll(m, 60, 'mean') ** 2)",
          60),
    _case("特质波动", "hard", "过去60个交易日，个股日收益率对市场收益率（当日全部股票日收益率的等权平均）回归的残差标准差",
          "m = r.groupby(level='datetime').transform('mean').where(r.notna())\n"
          "cov = roll(r * m, 60, 'mean') - roll(r, 60, 'mean') * roll(m, 60, 'mean')\n"
          "vm = roll(m * m, 60, 'mean') - roll(m, 60, 'mean') ** 2\nvr = roll(r * r, 60, 'mean') - roll(r, 60, 'mean') ** 2\n"
          "return (vr - cov * cov / vm).clip(lower=0) ** 0.5", 60),
    _case("量加权收益", "hard", "过去20个交易日，日收益率按成交量加权的平均值：sum(日收益率 × 成交量) / sum(成交量)",
          "return roll(r * v, 20, 'sum') / roll(v, 20, 'sum')", 20, ("close", "volume")),
    _case("上涨日成交额占比", "hard", "过去20个交易日，上涨日（日收益率 > 0）的成交额合计 / 全部成交额合计",
          "a = df['amount']\nreturn roll(a.where(r > 0, 0.0).where(a.notna()), 20, 'sum') / roll(a, 20, 'sum')", 20, ("close", "amount")),
    _case("高波动反转", "hard", "m = 20日动量（今日收盘/20日前收盘-1），vol = 过去20个交易日日收益率标准差；vol 高于当日截面中位数时取 -m，否则取 m",
          f"{_MOM20}\nvol = roll(r, 20, 'std')\nmed = vol.groupby(level='datetime').transform('median')\nreturn (-mom).where(vol > med, mom)", 20),
    _case("时序分位", "hard", "今日收盘价在过去60个交易日收盘价（含今日）中的分位，0~1",
          "return c.groupby(level='instrument').transform(lambda x: x.rolling(60).rank(pct=True))", 60),
    # ---------------- 陷阱：直接写容易用到未来数据或串到别的股票
    _case("历史价格分位", "trap", "当日收盘价在该股从样本第一天到当日全部收盘价中的分位，0~1",
          "return c.groupby(level='instrument').transform(lambda x: x.expanding().rank(pct=True))", None,
          note="rank(pct=True) 直接算是全样本分位"),
    _case("相对历史均量", "trap", "当日成交量 / 该股从样本第一天到当日全部成交量的平均值",
          "return v / v.groupby(level='instrument').transform(lambda x: x.expanding().mean())", None, ("volume",),
          note="mean() 直接算是全样本均值"),
    _case("收益历史标准分", "trap", "当日收益率的标准分：(当日收益率 - 均值) / 标准差，均值和标准差用该股从样本第一天到当日（含当日）的全部日收益率计算",
          "g = r.groupby(level='instrument')\nreturn (r - g.transform(lambda x: x.expanding().mean())) / g.transform(lambda x: x.expanding().std())", None,
          note="mean()/std() 直接算是全样本标准化"),
    _case("停牌填充动量", "trap", "20日动量（今日收盘/20日前收盘-1），停牌日（close 缺失）的收盘价用停牌前最后一个收盘价代替",
          "f = c.groupby(level='instrument').ffill()\nreturn f / f.groupby(level='instrument').shift(20) - 1", 20,
          note="ffill 不分组会串到上一只股票，bfill 是未来函数"),
    _case("EP截面缩尾", "trap", "1 / pe_ttm，在每个交易日的截面上按 1% 和 99% 分位缩尾",
          "e = 1 / df['pe_ttm']\ng = e.groupby(level='datetime')\nreturn e.clip(g.transform(lambda x: x.quantile(0.01)), g.transform(lambda x: x.quantile(0.99)))",
          1, ("pe_ttm",), note="不按日分组是全样本分位"),
    _case("样本累计收益", "trap", "该股从样本第一个有效收盘价到当日的累计收益",
          "return c / c.groupby(level='instrument').transform('first') - 1", None,
          note="不分组会串到别的股票"),
]


def compare(values: pd.Series, reference: pd.Series) -> dict:
    """和参考实现比：每日截面秩相关的中位数、10% 分位，以及参考实现有值的地方候选也有值的比例。"""
    a, b = _wide(values.reindex(reference.index)), _wide(reference)
    corr = _row_corr(_row_rank(a).values, _row_rank(b).values)
    both, ref = int((a.notna().values & b.notna().values).sum()), int(b.notna().values.sum())
    ok = corr[~np.isnan(corr)]
    median, p10 = (float(np.median(ok)), float(np.quantile(ok, 0.1))) if len(ok) else (float("nan"), float("nan"))
    overlap = both / ref if ref else 0.0
    return {"median": round(median, 4), "p10": round(p10, 4), "overlap": round(overlap, 4),
            "match": bool(median >= MATCH_MEDIAN and p10 >= MATCH_P10 and overlap >= MATCH_OVERLAP)}


def failed_layer(error_type: str | None) -> str | None:
    if error_type is None:
        return None
    return {ErrorType.ORDER_DEPENDENT: "structure", ErrorType.LOOKAHEAD: "causal"}.get(error_type, "execute")


def tokens(calls: list[dict]) -> int:
    total = 0
    for c in calls:
        u = c.get("usage") or {}
        total += (u.get("input_tokens") or u.get("prompt_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) \
            + (u.get("output_tokens") or u.get("completion_tokens") or 0)
    return total


class _Data:
    """实验层只需要数据；这里不评估收益，所以不构建 Evaluator。"""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def evaluate(self, values):
        return None


def available_cases(df: pd.DataFrame, cases: list[Case] | None = None) -> list[Case]:
    cols = set(factor_columns(df))
    return [c for c in (cases or CASES) if set(c.hypothesis["fields"]) <= cols]


def run_trial(llm, data: _Data, case: Case, reference: pd.Series, store: ExperienceStore | None, limits: Limits) -> dict:
    """一次试验：和编排图的内环一样，生成 → 执行 → 失败就修，最多 max_fix 次。"""
    h, n_calls = case.hypothesis, len(llm.calls)
    agent, q = ExperimentAgent(llm, data), hypothesis_text(h)

    def experiences(error_type=None):
        if store is None:
            return []
        return [e for e in retrieve_experiences(store, h, error_type) if similarity(q, e["task"]) < SAME_TASK]

    out = {"case": case.name, "level": case.level, "attempts": [], "retrieved": []}
    task = {"hypothesis": h, "experiences": experiences()}
    last = res = None
    for fix in range(limits.max_fix + 1):
        out["retrieved"] += [[e["kind"], e["score"], e["task"][:60]] for e in task["experiences"]]
        code = agent.implement(task)
        report, res = agent.run(code)
        out["attempts"].append({"code": code, "error_type": res.error_type, "message": res.message[:500], "checks": res.checks,
                                "via": agent.last_tool_used or "text"})
        if res.ok:
            if last and store is not None:
                store.add(Experience("fix", q, {"error_type": last["error_type"], "error": last["message"][:400],
                                                "before": last["code"], "after": code}, error_type=last["error_type"]))
            break
        if res.error_type == ErrorType.SANDBOX:
            out["environment_error"] = res.message[:300]
            break
        last = {"code": code, "error_type": res.error_type, "message": res.message}
        if fix < limits.max_fix:
            task = {"hypothesis": h, "last_error": last, "experiences": experiences(res.error_type)}
    out["fixes"] = len(out["attempts"]) - 1
    out["first_try_ok"] = out["attempts"][0]["error_type"] is None
    out["first_error"] = out["attempts"][0]["error_type"]
    out["exec_ok"] = bool(res.ok)
    out["layer_failed"] = failed_layer(None if res.ok else res.error_type)
    if res.ok:
        out["match"] = compare(res.values, reference)
        if not out["match"]["match"]:
            out["layer_failed"] = "functional"
        elif store is not None:
            store.add(Experience("success", q, {"code": code, "direction_by_harness": True}))
    out["passed"] = out["layer_failed"] is None
    out["tokens"], out["llm_calls"] = tokens(llm.calls[n_calls:]), len(llm.calls) - n_calls
    return out


def _rate(xs) -> float | None:
    xs = list(xs)
    return round(sum(map(bool, xs)) / len(xs), 4) if xs else None


def _group(rows: list[dict]) -> dict:
    n = len(rows)
    executed = [r for r in rows if r["exec_ok"]]
    return {"trials": n,
            "pass": _rate(r["passed"] for r in rows),
            "first_try_exec": _rate(r["first_try_ok"] for r in rows),
            "final_exec": _rate(r["exec_ok"] for r in rows),
            "silent_error": _rate(r["layer_failed"] == "functional" for r in rows),  # 跑通了但算错
            "wrong_among_executed": _rate(r["layer_failed"] == "functional" for r in executed),
            "layer_failed": {k: sum(r["layer_failed"] == k for r in rows) for k in LAYERS},
            "mean_fixes": round(float(np.mean([r["fixes"] for r in rows])), 3) if rows else None,
            "mean_tokens": round(float(np.mean([r["tokens"] for r in rows]))) if rows else None}


def paired_diff(rows: list[dict], field: str, seed: int = 0, n_boot: int = 5000) -> dict | None:
    """on - off 的差：先按题取平均，再按题做配对 bootstrap，题是独立单位，同一题的多次试验不是。"""
    frame = pd.DataFrame(rows)
    if frame.empty or {"on", "off"} - set(frame["arm"]):
        return None
    by = frame.assign(v=frame[field].astype(float)).pivot_table(index="case", columns="arm", values="v", aggfunc="mean").dropna()
    d = (by["on"] - by["off"]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return {"cases": len(d), "off": round(float(by["off"].mean()), 4), "on": round(float(by["on"].mean()), 4),
            "diff": round(float(d.mean()), 4), "ci95": [round(float(np.quantile(boots, q)), 4) for q in (0.025, 0.975)]}


def summarize(rows: list[dict]) -> dict:
    good = [r for r in rows if "error" not in r and "environment_error" not in r]
    by_case = {}
    for case in sorted({r["case"] for r in good}):
        sub = [r for r in good if r["case"] == case]
        by_case[case] = {"level": sub[0]["level"],
                         **{arm: {"pass": _rate(r["passed"] for r in x), "all_pass": all(r["passed"] for r in x),
                                  "layer_failed": sorted({r["layer_failed"] for r in x if r["layer_failed"]})}
                            for arm in ("off", "on") if (x := [r for r in sub if r["arm"] == arm])}}
    return {"trials": len(rows), "excluded": len(rows) - len(good),
            "by_arm": {arm: _group([r for r in good if r["arm"] == arm]) for arm in sorted({r["arm"] for r in good})},
            "by_level": {f"{arm}/{lv}": _group([r for r in good if r["arm"] == arm and r["level"] == lv])
                         for arm in sorted({r["arm"] for r in good}) for lv in ("basic", "medium", "hard", "trap")},
            "experience_on_minus_off": {f: paired_diff(good, f) for f in ("passed", "first_try_ok", "fixes", "tokens")},
            "by_case": by_case}


def run_eval(make_llm_fn: Callable[[], object], df: pd.DataFrame, seed_store: str | Path, out_dir: str | Path, trials: int = 3,
             arms: tuple[str, ...] = ("off", "on"), cases: list[Case] | None = None, workers: int = 6) -> dict:
    """df 必须已经去掉测试段（data.research_data）。每个（组，轮）一个线程、一个 LLM 实例和一份经验库，互不影响。"""
    out = Path(out_dir)
    (out / "stores").mkdir(parents=True, exist_ok=True)
    cases, data, limits = available_cases(df, cases), _Data(df), Limits()
    references = {}
    for c in cases:
        res = execute_factor(c.reference, df)
        if not res.ok:
            raise RuntimeError(f"参考实现没通过沙箱（{c.name}）：{res.error_type} {res.message[:300]}")
        references[c.name] = res.values
    lock, rows = threading.Lock(), []
    log = out / "trials.jsonl"
    log.unlink(missing_ok=True)

    def sequence(arm: str, trial: int):
        store = None
        if arm == "on":
            path = out / "stores" / f"on_{trial}.jsonl"
            if Path(seed_store).exists():
                shutil.copy(seed_store, path)
            store = ExperienceStore(path)
        llm = make_llm_fn()
        order = list(cases)
        random.Random(trial).shuffle(order)  # 两组同一轮题序相同
        for c in order:
            try:
                row = run_trial(llm, data, c, references[c.name], store, limits)
            except Exception as e:  # noqa: BLE001  模型接口出错不算模型答错，单独统计
                row = {"case": c.name, "level": c.level, "error": f"{type(e).__name__}: {e}"[:300]}
            row.update(arm=arm, trial=trial)
            with lock:
                rows.append(row)
                with log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda a: sequence(*a), [(arm, t) for t in range(trials) for arm in arms]))
    report = summarize(rows)
    report["thresholds"] = {"median": MATCH_MEDIAN, "p10": MATCH_P10, "overlap": MATCH_OVERLAP}
    report["data_end"] = str(df.index.get_level_values("datetime").max().date())
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report
