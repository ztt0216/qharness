"""不用 mock：确定性部分在沪深 300 真实行情切片上测；涉及模型的部分（@pytest.mark.llm）调真实模型，
只断言流程和约束（结构化输出、出处可追溯、上下文隔离、规则优先、经验复用），不断言模型的具体判断。"""
from pathlib import Path

import pytest

import numpy as np
import pandas as pd

from qharness.agents import ExperimentAgent, ResearchAgent, Thresholds, sanitize_note
from qharness.evaluate import Evaluator, TestSetLocked as _Locked, newey_west_t
from qharness.registry import FactorRegistry, admit
from qharness.risk import neutralize
from qharness.strategy import backtest, combine, pick
from qharness.experience import Experience, ExperienceStore
from qharness.orchestrator import FactorHarness
from qharness.report_sop import clean_and_chunk, grounded, run_report_sop
from qharness.sandbox import ErrorType, execute_factor
from qharness.trace import Tracer

ROOT = Path(__file__).resolve().parent.parent
REPORT = (ROOT / "samples" / "report_volume_price.md").read_text(encoding="utf-8")
GOOD = """
import pandas as pd
def factor(df):
    g = df.groupby(level='instrument', group_keys=False)
    return -g.apply(lambda x: x['close'].rolling(10).corr(x['volume']))
"""
PV_HYP = {"name": "量价相关性_20日", "fields": ["close", "volume"], "calculation": "过去20日close与volume的相关系数",
          "window": 20, "direction": 1, "rationale": "量价共振的股票后续表现更好", "condition": ""}


# ---------------------------------------------------------------- 工具层（真实行情）

@pytest.mark.parametrize("code,etype", [
    ("def factor(df):\n    return -df['close'].rolling(10).corr(df['volume'])", ErrorType.ORDER_DEPENDENT),
    ("def factor(df):\n    return df['close'].groupby(level='instrument').shift(-1)", ErrorType.LOOKAHEAD),
    ("def factor(df):\n    r = df['close'].groupby(level='instrument').pct_change(5)\n    return (r - r.mean()) / r.std()",
     ErrorType.LOOKAHEAD),
    ("import os\ndef factor(df):\n    return df['close']", ErrorType.FORBIDDEN),
    ("def factor(df):\n    return open('x').read()", ErrorType.FORBIDDEN),
    ("def factor(df):\n    return df['close'] *", ErrorType.SYNTAX),
    ("def factor(df):\n    return df['nope']", ErrorType.RUNTIME),
    ("def factor(df):\n    return df['close'] * 0 + 1", ErrorType.BAD_OUTPUT),
])
def test_sandbox_rejects_common_mistakes(df, code, etype):
    r = execute_factor(code, df)
    assert not r.ok and r.error_type == etype, (r.error_type, r.message)


def test_bfill_lookahead_is_caught_dynamically(df):
    code = "def factor(df):\n    s = df['close'].groupby(level='instrument').pct_change(20)\n    return s.groupby(level='instrument').bfill()"
    assert execute_factor(code, df).error_type == ErrorType.LOOKAHEAD


def test_good_factor_runs_and_test_set_opens_once(df, ev):
    r = execute_factor(GOOD, df)
    assert r.ok, r.message
    m = ev.evaluate(r.values)
    assert m["train"]["valid"] and m["valid"]["valid"]
    assert -1 <= m["valid"]["rank_ic"] <= 1 and {"reversal_5d", "momentum_20d", "low_vol_20d", "illiquidity_20d"} <= set(m["valid"]["corr_with_library"])
    ev.evaluate_test(r.values, key=GOOD)
    with pytest.raises(_Locked):
        ev.evaluate_test(r.values, key=GOOD)


def test_experience_retrieval_filters_by_error_type(tmp_path):
    s = ExperienceStore(tmp_path / "e.jsonl")
    s.add(Experience("fix", "量价相关性 close volume", {"error": "顺序", "before": "a", "after": "b"}, "ORDER_DEPENDENT"))
    s.add(Experience("fix", "量价相关性 close volume", {"error": "未来", "before": "a", "after": "c"}, "LOOKAHEAD"))
    hits = s.search("量价相关性_10日 close volume", kind="fix", error_type="LOOKAHEAD")
    assert [h[1].error_type for h in hits] == ["LOOKAHEAD"]
    assert s.add(Experience("fix", "量价相关性 close volume", {"error": "未来", "before": "a", "after": "d"}, "LOOKAHEAD")) == "merged"
    assert len(ExperienceStore(tmp_path / "e.jsonl").items) == 2


def test_correction_matching(ev, tmp_path):
    store = ExperienceStore(tmp_path / "exp.jsonl")
    frm = {"direction": 1, "window": 20, "calculation": PV_HYP["calculation"], "fields": PV_HYP["fields"]}
    store.add(Experience("corrected", "量价相关性_20日 过去20日close与volume的相关系数 字段close,volume 窗口20",
                         {"from": frm, "to": {**frm, "direction": -1}}))
    harness = FactorHarness(None, ev, store, Tracer(tmp_path))
    assert harness._correction(PV_HYP) is not None
    assert harness._correction({**PV_HYP, "window": 10}) is None
    assert harness._correction({**PV_HYP, "direction": -1}) is None
    assert harness._correction({**PV_HYP, "calculation": "过去20日close与volume的相关系数(corr)"}) is not None


def _m(ric, t, corr=0.1, years=None):
    return {"valid": True, "rank_ic": ric, "rank_ic_t_nw": t, "max_abs_corr_with_library": corr, "turnover": 0.1,
            "rank_ic_by_year": years or {}}


def test_research_rules():
    bad = _m(0.001, 0.1, corr=0.9)
    r = ResearchAgent(None).rules({"train": bad, "valid": bad})
    assert not r["passed"] and {"weak_train", "weak_valid", "high_corr", "insignificant_valid"} <= set(r["failed"])
    good = _m(0.04, 2.3, years={"2023": 0.03, "2024": 0.05})
    assert ResearchAgent(None).rules({"train": good, "valid": good}, n_trials=1)["passed"]
    # 同样的结果，试到第 4 次时验证段门槛升到 2.50，不再通过
    r4 = ResearchAgent(None).rules({"train": good, "valid": good}, n_trials=4)
    assert r4["failed"] == ["insignificant_valid"] and r4["t_valid_required"] == Thresholds().t_valid(4) > 2.4
    flip_flop = _m(0.04, 2.3, years={"2023": 0.08, "2024": -0.01, "2025": -0.02})
    assert "unstable_years" in ResearchAgent(None).rules({"train": flip_flop, "valid": good})["failed"]


def test_revision_guard():
    h = {"direction": 1, "window": 20}
    findings = [{"id": 1, "finding": "x"}]
    mixed = {"train": {"rank_ic": -0.02}, "valid": {"rank_ic": 0.01}}
    both_neg = {"train": {"rank_ic": -0.03, "rank_ic_t_nw": -2.5}, "valid": {"rank_ic": -0.02, "rank_ic_t_nw": -1.2}}
    both_neg_noise = {"train": {"rank_ic": -0.003, "rank_ic_t_nw": -0.6}, "valid": {"rank_ic": -0.01, "rank_ic_t_nw": -1.9}}
    chk = ResearchAgent.check_revision
    assert chk(h, {"direction": -1, "addresses": [1]}, findings, mixed)[1] == "direction_flip_unsupported"
    assert chk(h, {"direction": -1, "addresses": [1]}, findings, both_neg)[1] == "ok"
    assert chk(h, {"direction": -1, "addresses": [1]}, findings, both_neg_noise)[1] == "direction_flip_insignificant"
    assert chk(h, {"window": 60}, findings, mixed)[1] == "revision_without_finding"
    assert chk(h, {"window": 20, "addresses": [1]}, findings, mixed)[1] == "revision_noop"


def test_newey_west_accounts_for_overlap():
    rng = np.random.default_rng(0)
    e = rng.standard_normal(2000)
    overlapped = np.convolve(e, np.ones(5), mode="valid") / 5 + 0.05  # 5 日重叠，强序列相关
    naive = overlapped.mean() / overlapped.std() * np.sqrt(len(overlapped))
    assert 0 < newey_west_t(overlapped, lags=5) < naive * 0.7


def test_neutralize_removes_size_and_industry(df, risk):
    size = risk["log_mcap"].unstack("instrument")
    out = neutralize(size, risk)
    lm = size.reindex(columns=out.columns)
    c = [np.corrcoef(out.loc[d].dropna(), lm.loc[d, out.loc[d].dropna().index])[0, 1]
         for d in out.index[-50:] if out.loc[d].notna().sum() > 30]
    assert c and np.nanmax(np.abs(c)) < 0.2  # 市值本身中性化后与市值几乎无关


def test_neutralized_evaluation_reports_raw_ic(ev_risk):
    m = ev_risk.evaluate(ev_risk.raw_library["momentum_20d"])
    assert m["neutralized"] and m["valid"]["rank_ic_raw"] is not None
    assert {"rank_ic_t_nw", "rank_ic_by_year"} <= set(m["train"])
    assert m["valid"]["corr_with_library"]["momentum_20d"] > 0.99


def test_pick_respects_tradability():
    s = pd.Series({"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}, dtype=float)
    can_buy = pd.Series({"A": False, "B": True, "C": True, "D": True, "E": True})  # A 涨停买不进
    can_sell = pd.Series({"A": True, "B": True, "C": True, "D": True, "E": False})  # E 跌停卖不出
    assert set(pick(s, {"E"}, 3, can_buy, can_sell)) == {"E", "B", "C"}


def test_backtest_with_tradability(ev_risk):
    from qharness.data import segment_dates
    score = combine(ev_risk, list(ev_risk.raw_library.values()))
    dates = segment_dates(ev_risk.df, ev_risk.split.test)
    a = backtest(ev_risk.df, score, dates, topk=10)
    b = backtest(ev_risk.df, score, dates, topk=10, risk=ev_risk.risk)
    assert a["blocked_trades"] <= b["blocked_trades"] and b["n_days"] == a["n_days"]


def test_registry_admission(tmp_path):
    ok, _ = admit({}, {"test": {"valid": True, "rank_ic": 0.03, "rank_ic_t_nw": 1.2}})
    bad, why = admit({}, {"test": {"valid": True, "rank_ic": -0.01, "rank_ic_t_nw": -0.5}})
    assert ok and not bad and "测试段不成立" in why
    reg = FactorRegistry(tmp_path / "f")
    reg.add("量价/相关 20日", GOOD, {}, {}, {}, "run1")
    assert [x["name"] for x in reg.items()] == ["量价/相关 20日"]


def test_sandbox_kills_runaway_code(df):
    code = "def factor(df):\n    while True:\n        pass"
    r = execute_factor(code, df, timeout_s=2)
    assert not r.ok and r.error_type == ErrorType.TIMEOUT


def test_sanitize_note_drops_metric_sentences():
    note = sanitize_note("RankIC 为 -0.03，方向反了；改为反向使用")
    assert "RankIC" not in note and "0.03" not in note and "反向使用" in note


def test_chunking_and_grounding():
    chunks = clean_and_chunk(REPORT)
    assert not any(c.section == "前言" or "免责" in c.section or "仅供参考" in c.text for c in chunks)
    core = next(c for c in chunks if c.section == "核心观点")
    assert grounded("过去20个交易日收盘价与成交量相关性较高的个股，后续往往表现更好", core.text)
    assert not grounded("量价共振的股票后续表现更好", "日期：2026-03-15")


# ---------------------------------------------------------------- 真实模型

@pytest.mark.llm
def test_report_sop_real_model(llm, tmp_path):
    out = run_report_sop(REPORT, llm, Tracer(tmp_path))
    assert out["viewpoints"], "没有抽出观点"
    pages = {c["page"] for v in out["viewpoints"] for c in v["citations"]}
    assert pages <= {1, 3, 5}  # 出处只能落在有正文观点的页
    assert any(len(v["citations"]) >= 2 for v in out["viewpoints"])  # 核心观点页和正文页的同一观点被归并
    for h in out["hypotheses"]:
        assert {"name", "fields", "calculation", "window", "direction"} <= set(h)
        assert h["direction"] in (1, -1) and int(h["window"]) > 0


@pytest.mark.llm
def test_research_rules_override_real_model(llm):
    bad = {"valid": True, "rank_ic": 0.001, "rank_icir": 0.01, "max_abs_corr_with_library": 0.95, "turnover": 0.1,
           "ic": 0.001, "icir": 0.01, "long_short": 0.0, "group_monotonicity": 0.1, "rank_ic_t_nw": 0.1}
    v = ResearchAgent(llm).judge(PV_HYP, {"train": bad, "valid": bad}, can_refine=True)
    assert v.verdict == "reject"  # 与因子库高度相关：不管模型怎么说都淘汰


@pytest.mark.llm
def test_end_to_end_real_model(llm, df, tmp_path):
    """两次运行，只断言流程约束，不断言因子有没有效：
    - 写代码的一层提示词里没有指标和阈值
    - 第一次被淘汰的方向，第二次直接跳过；被纠正过的假设，第二次从纠正后的版本开始
    """
    store_path = tmp_path / "exp.jsonl"
    prompts = []
    orig = ExperimentAgent.implement

    def spy(self, task):
        code = orig(self, task)
        prompts.append(self.prompts[-1])
        return code

    def once():
        tr = Tracer(tmp_path)
        res = FactorHarness(llm, Evaluator(df), ExperienceStore(store_path), tr).run([PV_HYP])
        return res[0], tr

    ExperimentAgent.implement = spy
    try:
        r1, _ = once()
        r2, tr2 = once()
    finally:
        ExperimentAgent.implement = orig

    assert prompts
    for p in prompts:
        for word in ("rank_ic", "rankic", "icir", "min_rank", "threshold", "阈值", "long_short", "turnover"):
            assert word not in p.lower(), (word, p)

    assert r1.status in ("accepted", "rejected", "failed")
    if r1.status == "rejected":
        assert r2.status == "skipped"
    if r1.status == "accepted" and len(r1.history) > 1:
        applied = [e for e in tr2.events if e["node"] == "experience.apply_correction"]
        assert applied and r2.history[0]["hypothesis"]["direction"] == r1.history[-1]["hypothesis"]["direction"]


def test_factor_key_same_factor_only():
    from qharness.orchestrator import factor_key
    mom = {"calculation": "close / close.shift(20) - 1", "direction": 1, "fields": ["close"], "window": 20}
    rev = {**mom, "calculation": "-(close / close.shift(20) - 1)"}
    assert factor_key(rev) != factor_key(mom)  # 方向相反：不是同一个因子
    assert factor_key({**rev, "direction": -1}) == factor_key(mom)  # 公式取负 + 方向取负 = 同一个因子
    pos = {"calculation": "(close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min())",
           "direction": -1, "fields": ["close", "high", "low"], "window": 20}
    rng = {**pos, "calculation": "((high.rolling(20).max() - low.rolling(20).min()) / close.shift(20))"}
    assert factor_key(pos) != factor_key(rng)


def test_duplicate_by_values(ev, tmp_path):
    harness = FactorHarness(None, ev, ExperienceStore(tmp_path / "e.jsonl"), Tracer(tmp_path))
    r = execute_factor("def factor(df):\n    g = df['close'].groupby(level='instrument')\n    return df['close'] / g.shift(20) - 1", ev.df)
    assert harness._duplicate_of(r.values, "动量") is None
    dup = harness._duplicate_of(-r.values, "反转")  # 同一因子取负：也是重复
    assert dup and dup[0] == "动量" and dup[1] < -0.98


def test_duplicate_check_is_atomic_across_threads(ev, tmp_path):
    """候选并行：同一个因子同时到去重这一步，只能放过一个，其余都判重复。"""
    from concurrent.futures import ThreadPoolExecutor

    harness = FactorHarness(None, ev, ExperienceStore(tmp_path / "e.jsonl"), Tracer(tmp_path))
    r = execute_factor("def factor(df):\n    g = df['close'].groupby(level='instrument')\n    return df['close'] / g.shift(20) - 1", ev.df)
    with ThreadPoolExecutor(4) as pool:
        out = list(pool.map(lambda i: harness._duplicate_of(r.values, f"动量{i}"), range(4)))
    assert sum(o is None for o in out) == 1 and len(harness._evaluated) == 1


def test_tracer_tags_events_per_thread(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    tr = Tracer(tmp_path)

    def work(i):
        with tr.bind(f"cand-{i}"):
            for _ in range(50):
                tr.log("x", i=i)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(work, range(4)))
    lines = [__import__("json").loads(l) for l in tr.file.open(encoding="utf-8")]
    assert len(lines) == 200 and sorted(e["step"] for e in lines) == list(range(200))  # 编号不重不漏
    assert all(e["cand"] == f"cand-{e['i']}" for e in lines)


def test_pdf_formula_and_table_parsing():
    from qharness.report_pdf import pages_to_text
    page = "\n".join([
        "金融工程专题报告",
        "1.2 因子的构建",
        "我们构建了如下因子：",
        "买入浮亏占比因子（HCVOL）：",
        "∑𝑁𝑣𝑜𝑙𝑢𝑚𝑒 ∗𝐼",
        "𝑖 𝑏𝑢𝑦𝑖 𝑝𝑏𝑢𝑦𝑖>𝑐𝑙𝑜𝑠𝑒",
        "𝐻𝐶𝑉𝑂𝐿 =",
        "𝑡𝑜𝑡𝑎𝑙_𝑣𝑜𝑙𝑢𝑚𝑒",
        "图 2 展示的是根据每个节点市值因子𝑅𝑆𝑀𝐵,𝑡收益率拟合而成的净值曲线。",
        "图表3：因子IC指标（日频）",
        "因子 平均值 标准差 风险调整后IC T 统计量",
        "HCVOL 2.08% 6.99% 0.30 11.96",
        "LCVOL 2.19% 8.57% 0.26 10.27",
        "来源：Wind，国金证券研究所",
        "图表4：多空净值",
        "18",
        "16",
        "2016/1/4 2017/1/4",
        "来源：Wind，国金证券研究所",
        "特别声明：",
        "本报告版权归某证券所有",
    ])
    footer = "金融工程专题报告\n来源：Wind，国金证券研究所\n正文。"
    t = pages_to_text([page, footer, footer, footer], "r")
    assert "【公式】HCVOL：∑Nvolume ∗I ; i buyi pbuyi>close ; HCVOL = ; total_volume" in t
    assert "市值因子RSMB,t收益率" in t  # 正文里夹的数学字母留在正文
    assert "【表格】因子IC指标（日频）\n因子 | 平均值 | 标准差 | 风险调整后IC | T 统计量\nHCVOL | 2.08%" in t
    assert "多空净值" not in t and "2016/1/4" not in t  # 图整块去掉
    assert "版权" not in t and "金融工程专题报告" not in t  # 声明之后、页眉
    assert "表格 1 个，公式 1 个，去掉图 1 个" in t


def test_needs_intraday_guard():
    from qharness.report_sop import needs_intraday
    assert needs_intraday("低于收盘价卖出的成交量占比越高，次日买回动力越弱")
    assert needs_intraday("【公式】LCVOL：∑Nvolume ∗I ; LCVOL = ; i selli pselli<close ; total_volume")
    assert needs_intraday("尾盘30分钟成交额占全天比例")
    assert not needs_intraday("A 股 1 个月反转效应显著", "过去20日收益率取负")
    assert not needs_intraday("日内收益 close/open-1 越高，次日收益越低")


def test_checkpoint_resume_from_new_instance(tmp_path):
    """检查点机制本身：图在中途停下，换一个新编译的实例（相当于新进程）从断点续跑，已跑过的节点不重跑。"""
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    from qharness.checkpoint import open_checkpointer, run_graph, thread_status

    class S(TypedDict):
        visited: list[str]

    def build(stop_before=None):
        g = StateGraph(S)
        for n in ("a", "b", "c"):
            g.add_node(n, lambda s, n=n: {"visited": s["visited"] + [n]})
        g.add_edge(START, "a"); g.add_edge("a", "b"); g.add_edge("b", "c"); g.add_edge("c", END)
        return g.compile(checkpointer=open_checkpointer(tmp_path / "ck.sqlite"), interrupt_before=stop_before)

    first = run_graph(build(["c"]), {"visited": []}, "t1", 20)
    assert first["visited"] == ["a", "b"] and thread_status(build(), "t1") == "running"
    tr = Tracer(tmp_path)
    second = run_graph(build(), {"visited": []}, "t1", 20, tr)
    assert second["visited"] == ["a", "b", "c"] and thread_status(build(), "t1") == "done"
    assert [e["node"] for e in tr.events] == ["checkpoint.resume"]
    assert run_graph(build(), {"visited": []}, "t1", 20, tr)["visited"] == ["a", "b", "c"]  # 已完成：直接取结果
    assert tr.events[-1]["node"] == "checkpoint.restored"


def test_test_set_lock_survives_restart(df, tmp_path):
    log = str(tmp_path / "test_opened.txt")
    r = execute_factor(GOOD, df)
    Evaluator(df, test_log=log).evaluate_test(r.values, key=GOOD)
    with pytest.raises(_Locked):  # 新实例（相当于续跑的新进程）也不能再开同一个因子的测试段
        Evaluator(df, test_log=log).evaluate_test(r.values, key=GOOD)


@pytest.mark.llm
def test_candidate_resumes_at_judge_real_model(llm, df, tmp_path):
    """真实模型：判定前停下；新的 FactorHarness（新的 Agent、新的 Evaluator）从检查点续跑，不重新生成和执行代码。"""
    from qharness.checkpoint import open_checkpointer

    ck = tmp_path / "ck.sqlite"
    store = ExperienceStore(tmp_path / "exp.jsonl")
    log = str(tmp_path / "test_opened.txt")
    tr1 = Tracer(tmp_path, "run1")
    FactorHarness(llm, Evaluator(df, test_log=log), store, tr1, checkpointer=open_checkpointer(ck),
                  interrupt_before=["judge"]).run([PV_HYP])
    assert any(e["node"] == "experiment.execute" and e["ok"] for e in tr1.events)
    assert not any(e["node"] == "research.judge" for e in tr1.events)

    tr2 = Tracer(tmp_path, "run1")  # 同一个运行目录，接着写
    r2 = FactorHarness(llm, Evaluator(df, test_log=log), ExperienceStore(tmp_path / "exp.jsonl"), tr2,
                       checkpointer=open_checkpointer(ck)).run([PV_HYP])[0]
    nodes = [e["node"] for e in tr2.events]
    assert "checkpoint.resume" in nodes and nodes.index("checkpoint.resume") < nodes.index("research.judge")
    assert not any(e["node"] == "experiment.execute" and e["round"] == 0 for e in tr2.events)  # 第 0 轮没重跑
    assert r2.status in ("accepted", "rejected", "failed") and r2.history and r2.history[0]["round"] == 0


@pytest.mark.llm
def test_experiment_agent_native_tool_calling(llm, df):
    """实验层通过 run_factor 工具提交代码；工具结果以 tool_result 回传，内容里没有任何收益指标。"""
    agent = ExperimentAgent(llm, Evaluator(df))
    assert agent.use_tools
    code = agent.implement({"hypothesis": PV_HYP, "revision_note": "", "experiences": []})
    assert agent.last_tool_used and "def factor" in code
    report, res = agent.run(code)
    assert report.metrics is None or "valid" in report.metrics  # 指标交给编排器
    tr = agent.pending_result
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == agent.history[-1]["content"][-1]["id"]
    for word in ("rank_ic", "icir", "long_short", "max_turnover", "阈值"):
        assert word not in tr["content"].lower()
    # 第二轮：tool_result 放在下一条 user 消息最前面，接口接受这种消息序列
    agent.implement({"hypothesis": {**PV_HYP, "window": 10}, "revision_note": "窗口改为 10 日", "experiences": []})
    first_user_block = agent.history[-2]["content"][0]
    assert first_user_block["type"] == "tool_result"


def _paper_market(df):
    """切片里没有不复权价：用 后复权价 × 0.5 当不复权价，再在第 12 天给 A 股票做一次 10 送 10（复权系数减半）。"""
    from qharness.paper import Market
    close = df["close"].unstack("instrument")
    raw = close * 0.5
    a = close.columns[0]
    raw.loc[raw.index[12]:, a] = raw.loc[raw.index[12]:, a] / 2
    return Market(df, raw), a


def test_paper_fills_next_open_with_lots_and_fees(df, tmp_path):
    from qharness.paper import Account, Costs, PaperBroker, place_order
    m, _ = _paper_market(df)
    d0, d1, d2 = m.dates[5], m.dates[6], m.dates[7]
    picks = [c for c in m.close.columns if pd.notna(m.close.at[d1, c]) and pd.notna(m.open.at[d1, c])][1:6]
    acct = Account(cash=1_000_000, initial=1_000_000, last_date=str(d0.date()))
    b = PaperBroker(m, Costs())
    place_order(acct, d0, picks)
    b.settle(acct, until=d1)
    buys = [t for t in b.trades if t["status"] == "filled"]
    assert {t["date"] for t in buys} == {str(d1.date())}  # 下单日之后的第一个交易日成交
    assert all(t["shares"] % 100 == 0 and t["price"] >= t["open"] and t["fee"] >= 5 for t in buys)
    assert acct.cash >= 0 and len(acct.positions) == len(picks)
    # 不调仓的日子：净值变化 = 持仓市值变化
    v1 = {k: s * m.close.at[d1, k] for k, s in acct.positions.items()}
    b.settle(acct, until=d2)
    v2 = {k: s * m.close.at[d2, k] for k, s in acct.positions.items()}
    assert abs((acct.nav[-1]["nav"] - acct.nav[-2]["nav"]) - (sum(v2.values()) - sum(v1.values()))) < 0.05
    acct.save(tmp_path / "acct.json")
    place_order(acct, d2, picks[:2])
    acct.save(tmp_path / "acct.json")
    back = Account.load(tmp_path / "acct.json")
    assert back.positions == acct.positions and back.pending["weights"] == {p: 0.5 for p in picks[:2]}


def test_paper_blocks_halted_and_limit_up(df):
    from qharness.paper import Account, PaperBroker, place_order
    m, _ = _paper_market(df)
    d0, d1 = m.dates[5], m.dates[6]
    ok = [c for c in m.close.columns if pd.notna(m.open.at[d1, c])]
    halted, limit = ok[1], ok[2]
    m.open.at[d1, halted] = float("nan")                 # 停牌
    m.open.at[d1, limit] = m.up.at[d1, limit]            # 开盘即涨停
    acct = Account(cash=1_000_000, initial=1_000_000, last_date=str(d0.date()))
    b = PaperBroker(m)
    place_order(acct, d0, [halted, limit, ok[3]])
    b.settle(acct, until=d1)
    reasons = {t["instrument"]: t["reason"] for t in b.trades if t["status"] == "blocked"}
    assert reasons == {halted: "停牌", limit: "开盘涨停"} and set(acct.positions) == {ok[3]}


def test_paper_corporate_action_keeps_value(df):
    from qharness.paper import Account, Costs, PaperBroker, place_order
    m, a = _paper_market(df)
    d0 = m.dates[9]
    acct = Account(cash=1_000_000, initial=1_000_000, last_date=str(d0.date()))
    b = PaperBroker(m, Costs(0, 0, 0, 0))
    place_order(acct, d0, [a])
    b.settle(acct, until=m.dates[11])
    shares_before = acct.positions[a]
    b.settle(acct, until=m.dates[12])                    # 送转日：不复权价减半
    assert abs(acct.positions[a] / shares_before - 2) < 1e-9
    hfq = df["close"].unstack("instrument")[a]
    nav_ret = acct.nav[-1]["nav"] / acct.nav[-2]["nav"] - 1
    stock_ret = hfq.iloc[12] / hfq.iloc[11] - 1
    w = shares_before * m.close.at[m.dates[11], a] / acct.nav[-2]["nav"]
    assert abs(nav_ret - w * stock_ret) < 1e-6          # 净值跟着后复权收益走，没有因为价格减半而「亏一半」


@pytest.mark.llm
def test_parallel_candidates_real_model(llm, ev, tmp_path):
    """真实模型、3 个候选并行：结果按输入顺序返回；同一个因子写了两遍（措辞不同），只评估一次；trace 能按候选拆开。"""
    hyps = [
        {"name": "20日动量", "fields": ["close"], "calculation": "过去20个交易日的收益率 close/close.shift(20)-1",
         "window": 20, "direction": 1, "rationale": "", "condition": ""},
        {"name": "20日价格变化率", "fields": ["close"], "calculation": "收盘价相对20个交易日前收盘价的涨跌幅",
         "window": 20, "direction": 1, "rationale": "", "condition": ""},
        {"name": "20日振幅", "fields": ["high", "low", "close"],
         "calculation": "过去20日最高价减最低价除以收盘价", "window": 20, "direction": -1, "rationale": "", "condition": ""},
    ]
    tr = Tracer(tmp_path)
    harness = FactorHarness(llm, ev, ExperienceStore(tmp_path / "e.jsonl"), tr)
    results = harness.run(hyps, workers=3)
    assert [r.hypothesis["name"] for r in results] == [h["name"] for h in hyps]
    dup = [r for r in results[:2] if r.status == "skipped" and "同一个因子" in r.reason]
    assert len(dup) == 1, [(r.status, r.reason) for r in results]
    starts = [e for e in tr.events if e["node"] == "candidate.start"]
    assert sorted(e["cand"] for e in starts) == ["cand-0", "cand-1", "cand-2"]


def test_sandbox_reports_startup_failure_not_timeout(tmp_path):
    """调用方脚本没有 __main__ 保护：spawn 的子进程启动就崩。要立刻报 SANDBOX（环境问题），不能等到超时报 TIMEOUT。"""
    import subprocess
    import sys
    import time

    script = tmp_path / "noguard.py"
    script.write_text(
        "import pandas as pd\nfrom qharness.sandbox import execute_factor\n"
        "idx = pd.MultiIndex.from_product([pd.date_range('2024-01-01', periods=30), ['A', 'B', 'C']],"
        " names=['datetime', 'instrument'])\n"
        "df = pd.DataFrame({c: range(1, 91) for c in ['open', 'high', 'low', 'close', 'volume', 'amount']}, index=idx, dtype=float)\n"
        "r = execute_factor('def factor(df):\\n    return df[\"close\"]', df, timeout_s=5)\n"
        "print('RESULT', r.error_type)\n", encoding="utf-8")
    t = time.time()
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120,
                         cwd=Path(__file__).resolve().parents[1],
                         env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
    assert "RESULT SANDBOX" in out.stdout, out.stdout + out.stderr[-500:]
    assert time.time() - t < 30


@pytest.fixture
def qlib_env(df, risk, tmp_path_factory, monkeypatch):
    """把测试行情切片转成 Qlib 数据目录，并让沙箱（含子进程）用它。"""
    from qharness import qlib_bridge as qb

    uri = tmp_path_factory.mktemp("qlib")
    qb.dump(df, None, risk, uri)
    monkeypatch.setenv(qb.ENV_URI, str(uri))
    return uri


def test_qlib_expression_matches_pandas(df, qlib_env):
    """同一个因子：Qlib 表达式算出来的值和 pandas 写法一致；沙箱的口径检查照常跑。"""
    r = execute_factor("QLIB:-($close/Ref($close,5)-1)", df)
    assert r.ok and r.checks["lookahead"] == "pass", r.message
    p = -(df["close"] / df["close"].groupby(level="instrument").shift(5) - 1)
    m = r.values.notna() & p.notna()
    assert m.mean() > 0.9 and float((r.values[m] - p[m]).abs().max()) < 1e-5


def test_qlib_expression_static_rules(df, qlib_env):
    assert execute_factor("QLIB:Ref($close,-1)/$close", df).error_type == ErrorType.LOOKAHEAD
    assert execute_factor("QLIB:__import__('os')", df).error_type == ErrorType.FORBIDDEN
    assert execute_factor("QLIB:Mean($vwap,5)", df).error_type == ErrorType.FORBIDDEN  # 不在行情字段里


def test_qlib_rejects_stale_data(df, risk, tmp_path, monkeypatch):
    """Qlib 目录和当前行情不是同一份（比如 update 后没重新 dump）：报错，不静默算出另一份数。"""
    from qharness import qlib_bridge as qb

    qb.dump(df.assign(close=df["close"] * 1.01), None, risk, tmp_path / "q")
    monkeypatch.setenv(qb.ENV_URI, str(tmp_path / "q"))
    r = execute_factor("QLIB:Mean($close,5)", df)
    assert not r.ok and "不一致" in r.message


def test_qlib_backtest_runs(ev_risk, qlib_env):
    from qharness import qlib_bridge as qb
    from qharness.data import segment_dates
    from qharness.evaluate import baseline_library

    score = combine(ev_risk, list(baseline_library(ev_risk.df, ev_risk.risk).values()))
    d = segment_dates(ev_risk.df, ev_risk.split.test)
    out = qb.qlib_backtest(score, d[0], d[-1], topk=5)
    assert out["n_days"] > 10 and 0 < out["avg_daily_turnover"] < 1


@pytest.mark.llm
def test_experiment_agent_uses_qlib_tool_real_model(llm, df, qlib_env):
    """真实模型：有 Qlib 数据时，实验层通过 run_qlib_factor 提交表达式，沙箱执行通过。"""
    agent = ExperimentAgent(llm, Evaluator(df))
    h = {"name": "20日动量", "fields": ["close"], "calculation": "过去20个交易日收益率", "window": 20, "direction": 1}
    code = agent.implement({"hypothesis": h, "experiences": []})
    assert agent.last_tool_used == "run_qlib_factor" and code.startswith("QLIB:"), (agent.last_tool_used, code)
    report, res = agent.run(code)
    assert res.ok, res.message
    assert agent.pending_result["type"] == "tool_result"


@pytest.mark.parametrize("raw", [
    "def factor(df):\n    return df['close'].groupby(level='instrument').pct_change(fill_method=None)",
    "QLIB:$close/Ref($close,1)-1",
])
def test_direction_applied_by_harness(df, raw, request):
    """方向由 Harness 施加：direction=-1 的结果正好是原始实现取负；direction=1 原样不动。"""
    from qharness.sandbox import with_direction

    if raw.startswith("QLIB:"):
        request.getfixturevalue("qlib_env")
    assert with_direction(raw, 1) == raw
    a, b = execute_factor(raw, df), execute_factor(with_direction(raw, -1), df)
    assert a.ok and b.ok, b.message
    m = a.values.notna()
    assert float((a.values[m] + b.values[m]).abs().max()) < 1e-9


def test_coder_does_not_see_direction(df):
    agent = ExperimentAgent(None, Evaluator(df), use_tools=False)
    p = agent.build_prompt({"hypothesis": {"name": "x", "calculation": "c", "direction": -1}, "experiences": []})
    assert "direction" not in p


PPTX = Path(__file__).resolve().parents[1] / "samples" / "report_volume_price.pptx"  # Keynote 导出的真实 pptx


def test_pptx_parsing_keynote_export():
    from qharness.report_ppt import pptx_to_text
    from qharness.report_sop import clean_and_chunk

    text = pptx_to_text(PPTX)
    assert text.startswith("# 量价共振视角下的个股择时")
    assert text.index("## 核心观点") < text.index("## 量价关系的市场含义") < text.index("## 短期动量")  # 放映顺序
    assert "演讲者备注（与本页正文同等对待）：统计显示" in text  # 演讲者备注并入该页
    chunks = clean_and_chunk(text)
    assert {c.section for c in chunks} >= {"核心观点", "量价关系的市场含义", "风险提示"}
    assert not any("免责" in c.section for c in chunks)
    assert chunks[0].date == "2026-03-15" and chunks[0].page == 1


def test_pptx_table_and_summary_fallback(tmp_path):
    """在真实 pptx 的一页里插入表格、去掉「核心观点」标题：表格转成【表格】，第一张有正文的页被标成摘要。"""
    import zipfile

    from qharness.report_ppt import pptx_to_text

    tbl = ('<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="99" name="t"/><p:cNvGraphicFramePr/><p:nvPr/>'
           '</p:nvGraphicFramePr><p:xfrm><a:off x="0" y="5000000"/><a:ext cx="1" cy="1"/></p:xfrm>'
           '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table"><a:tbl>'
           + "".join("<a:tr h=\"1\">" + "".join(f"<a:tc><a:txBody><a:p><a:r><a:t>{c}</a:t></a:r></a:p></a:txBody></a:tc>"
                                              for c in row) + "</a:tr>"
                     for row in (["分组", "年化收益"], ["Top", "12.3%"], ["Bottom", "-4.1%"]))
           + "</a:tbl></a:graphicData></a:graphic></p:graphicFrame>")
    out = tmp_path / "x.pptx"
    with zipfile.ZipFile(PPTX) as zin, zipfile.ZipFile(out, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "ppt/slides/slide4.xml":
                data = data.decode().replace("</p:spTree>", tbl + "</p:spTree>").encode()
            if item.filename == "ppt/slides/slide2.xml":
                data = data.decode().replace("核心观点", "研究结论").encode()
            zout.writestr(item, data)
    text = pptx_to_text(out)
    assert "【表格】短期动量\n分组 | 年化收益\nTop | 12.3%\nBottom | -4.1%" in text
    assert "## 摘要：研究结论" in text


def test_qlib_model_train_alpha158(df, risk, qlib_env, tmp_path):
    """Alpha158 + 梯度提升树：跑通 Qlib 的 workflow（记录预测和 IC），按 Harness 口径评估；同一配置测试段只开一次。"""
    from qharness.qlib_model import train

    ev = Evaluator(df, risk=risk, test_log=str(tmp_path / "opened.txt"))
    kw = {"step": 10, "max_iter": 30, "patience": 1}
    extra = {"5日反转": "-1*($close/Ref($close,5)-1)"}
    s = train(ev, tmp_path / "run", model="hgb", topk=5, extra=extra, model_kw=kw)
    assert s["n_features"] == 159 and s["extra_features"] == ["5日反转"]
    assert s["test"]["n_days"] > 10 and s["test"]["rank_ic"] is not None
    assert set(s["backtest_by_phase"]) == {"baseline", "model"} and len(s["backtest_by_phase"]["model"]["all"]) == 5
    assert (tmp_path / "run" / "pred.pkl").exists() and (tmp_path / "run" / "summary.json").exists()
    with pytest.raises(_Locked):  # 同一配置再训一次：测试段已经开过
        train(Evaluator(df, risk=risk, test_log=str(tmp_path / "opened.txt")), tmp_path / "run2", model="hgb",
              topk=5, extra=extra, model_kw=kw)


def test_registry_exprs_keep_registered_direction():
    from qharness.qlib_model import registry_exprs

    items = [{"name": "a", "code": "QLIB:-1*(Mean($volume,5))", "hypothesis": {"direction": -1}},
             {"name": "b", "code": "def factor(df):\n    return df['close']", "hypothesis": {}}]
    assert registry_exprs(items) == {"a": "-1*(Mean($volume,5))"}  # 登记代码已含方向，不再翻一次；Python 因子跳过


def test_walk_forward_folds_have_embargo():
    """滚动切分：训练 → 验证 → 预测依次排开，中间各隔 EMBARGO 天；预测段首尾相接、不重叠。"""
    from qharness.data import EMBARGO
    from qharness.qlib_model import folds

    cal = pd.bdate_range("2023-01-02", periods=925)
    fs = folds(cal, step=63, valid_days=120, min_train=250)
    pos = lambda s: cal.get_loc(pd.Timestamp(s))  # noqa: E731
    for f in fs:
        assert pos(f["valid"][0]) - pos(f["train"][1]) == EMBARGO + 1
        assert pos(f["test"][0]) - pos(f["valid"][1]) == EMBARGO + 1
    for a, b in zip(fs, fs[1:]):
        assert pos(b["test"][0]) == pos(a["test"][1]) + 1
    assert fs[-1]["test"][1] == str(cal[-1].date())
    rolling = folds(cal, step=63, valid_days=120, min_train=250, train_days=250)
    assert all(pos(f["train"][1]) - pos(f["train"][0]) == 249 for f in rolling)


def test_walk_forward_runs_and_locks_test(df, risk, qlib_env, tmp_path):
    from qharness.qlib_model import walk_forward

    kw = dict(model="hgb", topk=5, model_kw={"step": 10, "max_iter": 20, "patience": 1}, step=60, valid_days=40,
              min_train=120)
    s = walk_forward(Evaluator(df, risk=risk, test_log=str(tmp_path / "o.txt")), tmp_path / "wf", **kw)
    assert s["n_folds"] >= 3 and len(s["folds"]) == s["n_folds"]
    assert s["oos"]["model"]["n_days"] > 100 and "test" in s["by_segment"]
    with pytest.raises(_Locked):
        walk_forward(Evaluator(df, risk=risk, test_log=str(tmp_path / "o.txt")), tmp_path / "wf2", **kw)


def test_web_api_reads_runs_and_guards_paths(tmp_path):
    """前端后端：读运行记录；运行编号、研报路径、上传格式都做校验。"""
    import json as _json
    from fastapi.testclient import TestClient
    from qharness.web import create_app

    rid = "20261031-000000-abcdef"
    d = tmp_path / "runs" / rid
    d.mkdir(parents=True)
    (d / "args.json").write_text(_json.dumps({"report": "samples/r.md"}), encoding="utf-8")
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "r.md").write_text("# 标题\n<!-- page: 1 -->\n## 核心观点\n正文", encoding="utf-8")
    events = [{"node": "run.start", "ts": 1.0, "step": 0, "model": "m"},
              {"node": "sop.topic", "ts": 2.0, "step": 1, "output": {"theme": "量价"}},
              {"node": "candidate.start", "ts": 3.0, "step": 2, "hypothesis": {"name": "a"}},
              {"node": "candidate.end", "ts": 4.0, "step": 3, "status": "rejected"}]
    (d / "trace.jsonl").write_text("\n".join(_json.dumps(e, ensure_ascii=False) for e in events) + "\n{\"node\": ",
                                   encoding="utf-8")  # 最后一行写到一半：跳过，不报错
    (d / "results.json").write_text(_json.dumps([{"hypothesis": {"name": "a"}, "status": "rejected"}]), encoding="utf-8")
    (d / "summary.md").write_text("# x", encoding="utf-8")

    c = TestClient(create_app(tmp_path))
    runs = c.get("/api/runs").json()
    assert runs[0]["id"] == rid and runs[0]["theme"] == "量价" and runs[0]["counts"] == {"rejected": 1}
    assert runs[0]["state"] == "finished" and runs[0]["report"] == "r.md"
    detail = c.get(f"/api/runs/{rid}").json()
    assert detail["report"].startswith("# 标题") and len(detail["trace"]) == 4
    assert c.get("/api/runs/..%2F..%2Fetc").status_code == 404
    assert c.get("/api/runs/20261031-000000-zzzzzz").status_code == 404
    assert c.post("/api/uploads", files={"file": ("x.exe", b"MZ")}).status_code == 400
    ok = c.post("/api/uploads", files={"file": ("研报 1.md", "正文".encode())}).json()
    assert ok["value"].startswith("uploads/") and (tmp_path / "runs" / ok["value"]).exists()
    assert c.post("/api/jobs/run", json={"report": "../qharness/cli.py"}).status_code == 400
    assert c.get("/api/paper").json()["account"] is None


def test_report_fetch_title_filter():
    from qharness.report_fetch import title_ok

    assert title_ok("Alpha 掘金系列之五：基于日频量价的筹码因子")[0]
    assert title_ok("北交所选股：小市值第一性原理×五因子模型")[0]
    assert not title_ok("宏观量化经济指数周报：出口保持韧性")[0]       # 周报、宏观
    assert not title_ok("基于 LSTM 的量化择时与动态仓位管理")[0]          # 择时
    assert not title_ok("月度市场策略：为有源头活水来")[0]                # 没有因子相关词


def test_report_fetch_pipeline_offline(tmp_path, monkeypatch):
    """列表 → 标题筛选 → 下载 → 程序预筛（日内数据）→ 模型预筛（程序复核）；全部离线，网络和 PDF 解析用假的。"""
    import io
    import json as _json
    from qharness import report_fetch as rf
    from qharness import report_pdf

    rows = [{"infoCode": "A1", "title": "量价因子研究：日频换手率的反转效应", "orgSName": "某证券", "publishDate": "2026-10-01 00:00:00", "attachPages": 20},
            {"infoCode": "A2", "title": "逐笔成交视角下的选股因子", "orgSName": "某证券", "publishDate": "2026-10-02 00:00:00", "attachPages": 18},
            {"infoCode": "A3", "title": "多因子选股：分析师预期因子", "orgSName": "某证券", "publishDate": "2026-10-03 00:00:00", "attachPages": 15},
            {"infoCode": "B1", "title": "宏观量化经济指数周报", "orgSName": "某证券", "publishDate": "2026-10-04 00:00:00", "attachPages": 9}]
    pages = {1: {"data": rows[:2], "TotalPage": 2}, 2: {"data": rows[2:], "TotalPage": 2}}
    import re
    fetch = lambda url: pages[int(re.search(r"pageNo=(\d+)", url).group(1))]  # noqa: E731
    idx = rf.ReportIndex(tmp_path)
    c = rf.collect(idx, days=30, end="2026-10-30", fetch=fetch)
    assert c == {"seen": 4, "new": 3, "begin": "2026-09-30", "end": "2026-10-30"}
    assert set(idx.items) == {"A1", "A2", "A3"} and idx.items["A1"]["status"] == "listed"
    assert rf.collect(idx, days=30, end="2026-10-30", fetch=fetch)["new"] == 0  # 已在索引里的不重复

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    for code in ("A1", "A2", "A3"):
        rf.download(idx, code, opener=lambda req, timeout: Resp(b"%PDF-1.4 fake"))
    assert idx.items["A1"]["status"] == "downloaded" and (tmp_path / "A1.pdf").exists()
    with pytest.raises(ValueError):
        idx.update("X", url="u", title="t")
        rf.download(idx, "X", opener=lambda req, timeout: Resp(b"<html>"))  # 不是 PDF

    texts = {"A1": "本文用日频换手率构建反转因子。", "A2": "逐笔 委托 主动买 分钟 " * 5, "A3": "分析师一致预期上调因子。"}
    monkeypatch.setattr(report_pdf, "pdf_to_text", lambda p: texts[Path(p).stem])

    class LLM:
        def __init__(self): self.n = 0
        def chat(self, role, msgs, ctx):
            self.n += 1
            if "日频换手率构建" in msgs[-1]["content"]:
                return _json.dumps({"type": "截面选股因子", "factors": [{"name": "换手反转", "data": "日频行情"}], "reason": "日线可算"})
            return _json.dumps({"type": "截面选股因子", "factors": [{"name": "预期上调", "data": "分析师预期"}], "reason": "要预期数据"})
    llm = LLM()
    assert rf.screen(idx, "A1", llm)["status"] == "queued"
    a2 = rf.screen(idx, "A2", llm)
    assert a2["status"] == "screened_out" and a2["stage"] == "program"   # 日内标记太多，不调用模型
    a3 = rf.screen(idx, "A3", llm)
    assert a3["status"] == "screened_out" and "分析师预期" in a3["reason"]
    assert llm.n == 2
    assert [x["code"] for x in rf.ReportIndex(tmp_path).by_status("queued")] == ["A1"]  # 落盘，重开还在


def test_web_api_reports_joins_runs(tmp_path):
    """研报来源页：读索引；从页面发起的研究按研报文件名对上运行记录。"""
    import json as _json
    from fastapi.testclient import TestClient
    from qharness.web import create_app

    auto = tmp_path / "data" / "reports" / "auto"
    auto.mkdir(parents=True)
    (auto / "A1.pdf").write_bytes(b"%PDF")
    (auto / "index.json").write_text(_json.dumps({
        "A1": {"code": "A1", "title": "量价因子", "date": "2026-10-01", "status": "queued", "path": "A1.pdf"},
        "A2": {"code": "A2", "title": "逐笔因子", "date": "2026-10-02", "status": "screened_out"}}), encoding="utf-8")
    d = tmp_path / "runs" / "20261031-000000-abcdef"
    d.mkdir(parents=True)
    (d / "args.json").write_text(_json.dumps({"report": str(auto / "A1.pdf")}), encoding="utf-8")
    c = TestClient(create_app(tmp_path))
    r = c.get("/api/reports").json()
    assert [x["code"] for x in r["items"]] == ["A2", "A1"] and r["counts"] == {"queued": 1, "screened_out": 1}
    assert r["items"][1]["runs"][0]["id"] == "20261031-000000-abcdef" and r["items"][0]["runs"] == []
    assert c.post("/api/jobs/run", json={"report": "auto/../../x.pdf"}).status_code == 400


def test_report_inbox_scan_dedup_and_screen(tmp_path):
    """收件夹：新文件进索引并预筛；同内容改名不重复；刚写入的先等；其他格式忽略；原文件不动。"""
    import json as _json
    import os
    import time as _time
    from qharness import report_fetch as rf

    inbox, idx = tmp_path / "inbox", rf.ReportIndex(tmp_path / "auto")
    inbox.mkdir()
    old = _time.time() - 60
    (inbox / "量价研报.md").write_text("<!-- page: 1 -->\n## 核心观点\n日频换手率构建的反转因子。", encoding="utf-8")
    (inbox / "同一篇改了名.md").write_text("<!-- page: 1 -->\n## 核心观点\n日频换手率构建的反转因子。", encoding="utf-8")
    (inbox / "说明.docx").write_bytes(b"PK")
    (inbox / ".DS_Store").write_bytes(b"x")
    for f in inbox.iterdir():
        os.utime(f, (old, old))
    (inbox / "还在下载.pdf").write_bytes(b"%PDF")  # 刚写入：这次跳过

    class LLM:
        def chat(self, role, msgs, ctx):
            return _json.dumps({"type": "截面选股因子", "factors": [{"name": "换手反转", "data": "日频行情"}], "reason": "日线可算"})

    r = rf.run_inbox(LLM(), 0, idx, inbox, log=lambda m: None)
    assert len(r["scan"]["added"]) == 1 and r["scan"]["skipped"] == 1 and r["scan"]["waiting"] == 1
    code = r["scan"]["added"][0]
    rec = idx.items[code]
    assert code.startswith("LOCAL-") and rec["source"] == "inbox" and rec["status"] == "queued"
    assert (tmp_path / "auto" / rec["path"]).exists() and (tmp_path / "auto" / f"{code}.text.md").exists()
    assert sorted(f.name for f in inbox.iterdir()) == sorted([".DS_Store", "同一篇改了名.md", "说明.docx", "还在下载.pdf", "量价研报.md"])
    os.utime(inbox / "还在下载.pdf", (old, old))
    r2 = rf.scan_inbox(idx, inbox)  # 再扫：之前的不重复，刚才没写完的这次进来
    assert len(r2["added"]) == 1 and r2["skipped"] == 2


def test_retrieval_modes(monkeypatch):
    """默认混合检索，向量模型不可用时退回字面；字面模式与改动前排序一致；混合模式下语义通道能召回换了说法的同一观点。"""
    from qharness import retrieval as R
    from qharness.experience import similarity

    q = "过去20日收盘价与成交量相关性高的个股后续表现更好"
    texts = ["凝血因子产品快速放量", "收盘价与成交量的20日相关系数越高，个股未来收益越高", "短期动量延续"]
    monkeypatch.setenv("QH_RETRIEVAL", "lexical")
    assert R.mode() == "lexical"
    assert R.rank(q, texts) == sorted(range(3), key=lambda i: -similarity(q, texts[i]))
    monkeypatch.delenv("QH_RETRIEVAL")
    assert R.mode() == ("hybrid" if R.enabled() else "lexical")
    monkeypatch.setenv("QH_RETRIEVAL", "nonsense")
    with pytest.raises(ValueError):
        R.mode()
    monkeypatch.setenv("QH_RETRIEVAL", "hybrid")
    monkeypatch.setenv("QH_EMBED", "off")
    assert R.mode() == "lexical" and R.rank(q, texts) == R.rank(q, texts, how="lexical")
    monkeypatch.delenv("QH_EMBED")
    if not R.enabled():
        pytest.skip("没有本地向量模型（python -m qharness.retrieval --download）")
    assert R.rank(q, texts, how="semantic")[0] == 1 and R.rank(q, texts, how="hybrid")[0] == 1
    a, b = "低波动股票长期跑赢", "波动率小的个股未来收益更高"
    assert similarity(a, b) < 0.3 and R.maybe_same(a, b)            # 字面过不了线，语义通道召回
    monkeypatch.setenv("QH_RETRIEVAL", "lexical")
    assert not R.maybe_same(a, b)


def test_monitor_deactivates_decayed_factor(df, risk, tmp_path, monkeypatch):
    """衰减监控：最近窗口里方向反了且显著的登记因子被停用（写回登记表，带原因和历史）；基础因子只报告。"""
    from qharness.monitor import run_monitor
    from qharness.data import make_label

    reg = FactorRegistry(tmp_path / "f")
    reg.add("失效因子", GOOD, {"name": "失效因子"}, {}, {"test": {"rank_ic": 0.05, "valid": True}}, "r1")
    reg.add("正常因子", GOOD, {"name": "正常因子"}, {}, {"test": {"rank_ic": 0.05, "valid": True}}, "r2")
    noise = np.random.default_rng(0).normal(size=len(df))
    lab = make_label(df)
    monkeypatch.setattr(FactorRegistry, "compute", lambda self, d: {
        "失效因子": -lab + 0.02 * noise,           # 和未来收益反着走（只在测试里直接给值，不过沙箱）
        "正常因子": lab + 0.02 * noise})
    res = run_monitor(df, risk, reg, tmp_path / "mon", log=lambda m: None)
    st = {r["name"]: r for r in res["factors"]}
    assert st["失效因子"]["status"] == "deactivate" and st["正常因子"]["status"] == "ok"
    assert res["deactivated"] == ["失效因子"] and {r["kind"] for r in res["factors"]} == {"registered", "baseline"}
    assert [x["name"] for x in reg.items()] == ["正常因子"]
    x = [x for x in reg.items(active_only=False) if x["name"] == "失效因子"][0]
    assert x["active"] is False and x["history"][-1]["action"] == "deactivate" and "方向反了" in x["history"][-1]["reason"]
    assert (tmp_path / "mon" / "latest.json").exists()
    reg.set_active("失效因子", True, reason="人工确认")
    assert len(reg.items()) == 2


def test_daily_job_idempotent_and_resumes(tmp_path):
    """每日任务：周末、收盘前跳过；处理过当天就不再联网；没有新行情记 no_new_data；中途失败下次从监控接着做。"""
    from datetime import datetime
    from qharness.daily_job import BEIJING, history, run_daily_job

    cache = tmp_path / "data"
    cache.mkdir()
    f = cache / "sh.600000.csv"
    f.write_text("date,code,close\n2026-10-30,sh.600000,1\n", encoding="utf-8")
    calls = []

    def update(c):
        calls.append("update")
        if update.publish:
            f.write_text(f.read_text() + f"{update.publish},sh.600000,1\n")
        return {"appended": int(bool(update.publish)), "no_new_data": int(not update.publish), "failed": []}
    update.publish = None
    daily = {"fail": False}

    def run_daily():
        calls.append("daily")
        if daily["fail"]:
            raise RuntimeError("撮合出错")
        return {"nav": 1.0}
    steps = {"update": update, "monitor": lambda: calls.append("monitor") or {"deactivated": []}, "daily": run_daily}
    job = lambda h, d="2026-11-02": run_daily_job(tmp_path, cache, tmp_path / "f", tmp_path / "p", steps=steps,  # noqa: E731
                                                 now=datetime.fromisoformat(f"{d}T{h}").replace(tzinfo=BEIJING))
    assert job("18:00", "2026-10-31")["note"] == "周末" and job("15:00")["outcome"] == "skipped" and not calls
    assert job("17:30")["outcome"] == "no_new_data" and calls == ["update"]           # 数据源还没发布
    update.publish, daily["fail"] = "2026-11-02", True
    assert job("18:30")["outcome"] == "failed" and calls[-3:] == ["update", "monitor", "daily"]
    daily["fail"], calls[:] = False, []
    r = job("19:30")
    assert r["outcome"] == "done" and calls == ["monitor", "daily"]                   # 行情已更新：不重复更新
    calls.clear()
    assert job("20:30")["outcome"] == "skipped" and not calls                         # 当天已处理：不联网
    assert [h["outcome"] for h in history(tmp_path / "runs")] == ["no_new_data", "failed", "done"]


def test_daily_job_time_limit(tmp_path):
    """某一步卡住（比如网络请求一直不返回）：到总时限记为失败退出，不会一直挂着挡住之后的定时触发。"""
    import time as _time
    from datetime import datetime
    from qharness.daily_job import BEIJING, run_daily_job

    cache = tmp_path / "data"
    cache.mkdir()
    (cache / "sh.600000.csv").write_text("date,code,close\n2026-10-30,sh.600000,1\n", encoding="utf-8")
    steps = {"update": lambda c: _time.sleep(30), "monitor": lambda: {}, "daily": lambda: None}
    t = _time.time()
    r = run_daily_job(tmp_path, cache, tmp_path / "f", tmp_path / "p", steps=steps, max_seconds=1,
                      now=datetime(2026, 11, 2, 18, 0, tzinfo=BEIJING))
    assert r["outcome"] == "failed" and "超过 1 秒" in r["error"] and _time.time() - t < 5


def test_fundamentals_point_in_time(tmp_path):
    """财务按公告日对齐：公告日之后第一个交易日才生效；同日公布两期取报告期新的；晚公布的旧报告期更正不回退。"""
    from qharness.fundamentals import load_fundamentals

    cal = pd.bdate_range("2025-04-25", "2025-05-09")  # 4-30 周三公布年报和一季报，5-01~05 放假（这里用工作日近似）
    panel = pd.DataFrame({"close": 1.0}, index=pd.MultiIndex.from_product([cal, ["SH600000"]], names=["datetime", "instrument"]))
    (tmp_path / "ref" / "fin").mkdir(parents=True)
    (tmp_path / "ref" / "val").mkdir(parents=True)
    pd.DataFrame([
        {"year": 2024, "quarter": 3, "pubDate": "2024-10-30", "statDate": "2024-09-30", "roe": 0.05, "yoy_ni": 0.1},
        {"year": 2024, "quarter": 4, "pubDate": "2025-04-30", "statDate": "2024-12-31", "roe": 0.08, "yoy_ni": 0.2},
        {"year": 2025, "quarter": 1, "pubDate": "2025-04-30", "statDate": "2025-03-31", "roe": 0.02, "yoy_ni": 0.3},
        {"year": 2024, "quarter": 2, "pubDate": "2025-05-06", "statDate": "2024-06-30", "roe": 0.99, "yoy_ni": 0.9},  # 更正旧报告期
        {"year": 2025, "quarter": 2, "pubDate": "", "statDate": "", "roe": None, "yoy_ni": None},                    # 还没公布
    ]).to_csv(tmp_path / "ref" / "fin" / "sh.600000.csv", index=False)
    pd.DataFrame({"date": cal.strftime("%Y-%m-%d"), "pe_ttm": range(len(cal)), "pb_mrq": 1.0, "ps_ttm": 1.0,
                  "pcf_ttm": -2.0}).to_csv(tmp_path / "ref" / "val" / "sh.600000.csv", index=False)
    f = load_fundamentals(panel, tmp_path).xs("SH600000", level="instrument")
    assert f.loc["2025-04-30", "roe"] == 0.05            # 公告当天还不能用
    assert f.loc["2025-05-01", "roe"] == 0.02            # 次日生效，同日两期取报告期新的（一季报）
    assert f.loc["2025-05-09", "roe"] == 0.02            # 5-06 的旧报告期更正不回退
    assert f.loc["2025-05-02", "pe_ttm"] == list(cal).index(pd.Timestamp("2025-05-02"))  # 估值按日直接对上
    assert list(f.columns[:4]) == ["pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]


def test_fundamental_fields_reach_factor_code_and_prompts(df):
    """有财务列时：因子代码能用、字段说明写进提示词、缺失数据说明不再说没有财务；预筛按本地数据判断能否检验。"""
    from qharness.data import describe_fields, factor_columns, missing_data_note
    from qharness.report_fetch import decide

    d = df.copy()
    d["pe_ttm"] = d["close"] / 3 + 5
    d["roe"] = 0.1
    cols = factor_columns(d)
    assert "pe_ttm" in cols and "roe" in cols
    r = execute_factor("def factor(df):\n    return 1 / df['pe_ttm']", d)
    assert r.ok, r.message
    assert "按公告日之后第一个交易日生效" in describe_fields(cols) and "财务（ROE" not in missing_data_note(cols)
    assert "财务（ROE" in missing_data_note(factor_columns(df))
    out = {"type": "截面选股因子", "factors": [{"name": "EP", "data": "估值"}]}
    assert not decide(out, ("日频行情",))[0] and decide(out, ("日频行情", "估值", "季报财务", "财务"))[0]


def test_daily_job_fundamentals_failure_does_not_block(tmp_path):
    """财务和估值更新失败只记警告，照常监控和结算。"""
    from datetime import datetime
    from qharness.daily_job import BEIJING, run_daily_job

    cache = tmp_path / "data"
    cache.mkdir()
    f = cache / "sh.600000.csv"
    f.write_text("date,code,close\n2026-10-30,sh.600000,1\n", encoding="utf-8")

    def update(c):
        f.write_text(f.read_text() + "2026-11-02,sh.600000,1\n")
        return {"appended": 1, "no_new_data": 0, "failed": []}

    def fundamentals(c):
        raise ConnectionError("baostock 断开")
    steps = {"update": update, "fundamentals": fundamentals, "monitor": lambda: {}, "daily": lambda: None}
    r = run_daily_job(tmp_path, cache, tmp_path / "f", tmp_path / "p", steps=steps,
                      now=datetime(2026, 11, 2, 18, 0, tzinfo=BEIJING))
    assert r["outcome"] == "done" and r["steps"] == ["update", "monitor", "daily"] and "baostock 断开" in r["warnings"][0]


def test_test_lock_by_values_blocks_transforms_and_flips(df, tmp_path):
    """测试段锁按因子值认身份：单调变换、取反都算同一个因子；跨 Evaluator 实例（跨运行）生效；数据范围不同算新测试。"""
    log = str(tmp_path / "opened.txt")
    r = execute_factor(GOOD, df)
    Evaluator(df, test_log=log).evaluate_test(r.values)
    for variant in (2 * r.values + 3, -r.values, r.values.rank()):
        with pytest.raises(_Locked):
            Evaluator(df, test_log=log).evaluate_test(variant)
    short = df[df.index.get_level_values("datetime") < df.index.get_level_values("datetime").unique()[-60]]
    Evaluator(short, test_log=log).evaluate_test(r.values.reindex(short.index))  # 测试段不同：允许


def test_registered_factors_join_correlation_library(df):
    """已登记因子进入相关性比较：和它高度相关的新候选会被看出来。"""
    r = execute_factor(GOOD, df)
    plain = Evaluator(df).evaluate(r.values * 1.5)["valid"]["max_abs_corr_with_library"]
    withreg = Evaluator(df, extra_library={"已登记": r.values}).evaluate(r.values * 1.5)["valid"]
    assert withreg["corr_with_library"]["已登记"] > 0.99 and withreg["max_abs_corr_with_library"] > plain
