"""命令行入口。

  python -m qharness demo --env-file .env         用示例研报跑完整流程（真实模型 + 真实行情）
  python -m qharness run --report 研报.md          研报 → 假设 → 实验循环 → 策略验证 → 通过的因子登记
  python -m qharness run --hypothesis h.json      跳过模块一，直接验证假设（JSON 列表）
  python -m qharness update                       每日增量更新行情、市值、最新成分
  python -m qharness daily                        因子库 + 登记因子 → 下一交易日目标持仓与买卖清单
  python -m qharness factors                      查看登记因子
  python -m qharness eval-research --env-file .env  研究层评测集
  python -m qharness experience                   查看经验库
  python -m qharness mcp --role coder             MCP 服务（stdio）：沙箱、经验检索、验证段回测
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .data import factor_columns, load_csv_panel
from .evaluate import Evaluator
from .experience import ExperienceStore
from .llm import load_env_file, make_llm
from .checkpoint import open_checkpointer
from .market import DEFAULT_CACHE, load_hs300
from .orchestrator import FactorHarness, results_json
from .registry import FactorRegistry, admit
from .report_sop import run_report_sop
from .sandbox import execute_factor
from .strategy import target_portfolio, validate_strategy
from .trace import Tracer

ROOT = Path(__file__).resolve().parent.parent


def _fmt_seg(m: dict | None) -> str:
    if not m or not m.get("valid"):
        return "—"
    raw = f"（未中性化 {m['rank_ic_raw']:+.4f}）" if m.get("rank_ic_raw") is not None else ""
    return (f"RankIC {m['rank_ic']:+.4f}{raw} · t(NW) {m.get('rank_ic_t_nw', float('nan')):+.2f} · "
            f"分年 {m.get('rank_ic_by_year', {})} · 多空 {m['long_short']:+.4f} · "
            f"换手 {m['turnover']:.3f} · 与因子库最大相关 {m['max_abs_corr_with_library']:.3f}")


def write_summary(tr: Tracer, sop: dict | None, results, strategy: dict | None,
                  registered: dict | None = None, neutralized: bool = False) -> Path:
    L = [f"# 运行报告 {tr.run_id}", "",
         f"评估口径：{'市值 + 申万一级行业中性化后' if neutralized else '未中性化（缺市值/行业数据）'}的 RankIC；"
         "显著性用 Newey-West t；验证段 t 门槛随试验次数提高。", ""]
    if sop:
        L += ["## 模块一 研报 → 假设", f"- 主题：{sop['topic'].get('theme', '')}", "- 观点（归并后）："]
        L += [f"  - {v['claim']}  出处：{', '.join(f'{c['section']} p{c['page']}' for c in v['citations'])}"
              for v in sop["viewpoints"]]
        L += ["- 因子假设："] + [f"  - {h['name']}｜{h['calculation']}｜窗口 {h['window']}｜方向 {h['direction']:+d}"
                                f"｜出处 {', '.join(f'p{c['page']}' for c in h.get('citations', [])) or '—'}"
                                for h in sop["hypotheses"]]
        if sop.get("unfaithful"):
            L += ["- 忠实性核查剔除（研报没有直接提到）："] + [f"  - {u['name']}：{u['reason']}" for u in sop["unfaithful"]]
        if sop.get("not_testable"):
            L += ["- 不可检验的观点："] + [f"  - {n.get('claim')}：{n.get('reason')}"
                                       f"（出处 {', '.join(f'p{c['page']}' for c in n.get('citations', [])) or '—'}）"
                                       for n in sop["not_testable"]]
        L.append("")
    L.append("## 模块二 因子实验")
    for r in results:
        L += [f"### {r.hypothesis.get('name')} → **{r.status}**", ""]
        if r.reason:
            L.append(f"原因：{r.reason}\n")
        for i, h in enumerate(r.history):
            L += [f"- 第 {i + 1} 轮（第 {h.get('n_trials', i + 1)} 次试验）：方向 {h['hypothesis']['direction']:+d}，"
                  f"窗口 {h['hypothesis']['window']}，修复 {h['fixes']} 次 → **{h['verdict']}** "
                  f"{('未通过：' + ','.join(h['failed'])) if h['failed'] else ''}"
                  f"{'；修改建议被规则驳回：' + h['revision_check'] if h.get('revision_check') not in (None, 'ok', 'no_revision') else ''}",
                  f"  - 训练段：{_fmt_seg(h['metrics']['train'])}",
                  f"  - 验证段：{_fmt_seg(h['metrics']['valid'])}",
                  f"  - 研究层诊断：{h['diagnosis']}"]
            if h["revision"]:
                L.append(f"  - 修改：{h['revision']}")
        if r.test_metrics:
            L.append(f"- 测试段（只开放一次）：{_fmt_seg(r.test_metrics['test'])}")
        if registered and r.hypothesis.get("name") in registered:
            L.append(f"- 登记：{registered[r.hypothesis['name']]}")
        if r.code:
            L += ["", "```python", r.code.rstrip(), "```"]
        L.append("")
    if strategy:
        L += ["## 模块三 策略验证（测试段）", f"区间：{strategy['segment']}", "",
              f"基线因子：{', '.join(strategy.get('baseline_factors', []))}", "",
              "| 组合 | 年化超额 | 超额夏普 | 超额最大回撤 | 每次调仓换手（双边，买+卖） | 受限成交 |",
              "|---|---|---|---|---|---|"]
        for k, name in (("baseline", "基线"), ("baseline_plus_new", "基线 + 新因子")):
            if k in strategy:
                s = strategy[k]
                L.append(f"| {name} | {s['annual_excess']:+.2%} | {s['excess_sharpe']:+.2f} | "
                         f"{s['max_drawdown_excess']:.2%} | {s['avg_turnover_per_rebalance']:.2f} | "
                         f"{s.get('blocked_trades', 0)} |")
        L += ["", "受限成交：成交日停牌或涨停买不进、停牌或跌停卖不出的股票次数。"]
    L += ["", "> 样本期短（约 3.8 年）、未处理 ST 的 5% 涨跌幅，结果只用于检验研究流程，不构成投资建议。"]
    return tr.write("summary.md", "\n".join(L) + "\n")


RESUMABLE_ARGS = ("report", "hypothesis", "llm", "csv", "data", "experience", "registry", "no_strategy")


def build(args):
    if args.env_file:
        print(f"从 env 文件读取：{', '.join(load_env_file(args.env_file))}")
    tr = Tracer(args.runs, args.resume or getattr(args, "run_id", None))
    if args.resume:  # 续跑：沿用原运行的参数（研报、数据、经验库等），保证接上的是同一个任务
        saved = json.loads((tr.dir / "args.json").read_text(encoding="utf-8"))
        for k in RESUMABLE_ARGS:
            setattr(args, k, saved.get(k))
    else:
        tr.write("args.json", json.dumps({k: getattr(args, k) for k in RESUMABLE_ARGS}, ensure_ascii=False, indent=2))
    llm = make_llm(args.llm)
    llm.calls_file = tr.dir / "llm_calls.jsonl"
    df, risk = (load_csv_panel(args.csv), None) if args.csv else load_hs300(Path(args.data))
    # 测试段锁用全局文件：跨运行生效（原来每次运行一个文件，新开一次运行同一个因子就能再看测试段）；
    # 相关性比较带上已登记因子
    registered = FactorRegistry(Path(args.registry)).compute(df) if not args.csv else {}
    ev = Evaluator(df, risk=risk, test_log=str(Path(args.runs) / "test_opened.txt"), extra_library=registered)
    store = ExperienceStore(args.experience)
    ckpt = None if args.no_checkpoint else open_checkpointer(tr.dir / "checkpoints.sqlite")
    return llm, ev, store, tr, ckpt


def cmd_run(args) -> dict:
    llm, ev, store, tr, ckpt = build(args)
    try:
        return _run(args, llm, ev, store, tr, ckpt)
    except (Exception, KeyboardInterrupt):
        if ckpt is not None:
            print(f"\n运行中断。已完成的节点都在检查点里，续跑：\n"
                  f"  python -m qharness run --resume {tr.run_id}" + (f" --env-file {args.env_file}" if args.env_file else ""))
        raise
    finally:
        _save_llm_calls(tr, llm)


def _save_llm_calls(tr: Tracer, llm):
    """llm_calls.jsonl 是每次调用当场追加的（续跑时接在后面）；这里汇总成 llm_calls.json 方便查看。"""
    f = tr.dir / "llm_calls.jsonl"
    calls = [json.loads(l) for l in f.open(encoding="utf-8")] if f.exists() else list(llm.calls)
    tr.write("llm_calls.json", json.dumps(calls, ensure_ascii=False, indent=2))


def _run(args, llm, ev, store, tr, ckpt) -> dict:
    tr.log("run.resume" if args.resume else "run.start", llm=llm.name, model=llm.model, data=args.csv or args.data,
           n_rows=len(ev.df), split={k: [str(d.date()) for d in v] for k, v in vars(ev.split).items()})
    sop = None
    if args.report:
        if args.resume and (tr.dir / "report.md").exists():
            text = (tr.dir / "report.md").read_text(encoding="utf-8")
        elif args.report.lower().endswith(".pdf"):
            from .report_pdf import pdf_to_text
            text = pdf_to_text(args.report)
            tr.write("report.md", text)  # 转换结果留档，切片和出处都基于它
        elif args.report.lower().endswith(".pptx"):
            from .report_ppt import pptx_to_text
            text = pptx_to_text(args.report)
            tr.write("report.md", text)  # 转换结果留档，切片和出处都基于它
        else:
            text = Path(args.report).read_text(encoding="utf-8")
        sop = run_report_sop(text, llm, tr, fields=factor_columns(ev.df), neutralized=ev.neutralized,
                             checkpointer=ckpt, thread_id="report_sop" if ckpt else None)
        hyps = sop["hypotheses"]
    else:
        hyps = json.loads(Path(args.hypothesis).read_text(encoding="utf-8"))
    results = FactorHarness(llm, ev, store, tr, checkpointer=ckpt).run(hyps, workers=args.workers)
    accepted = {r.hypothesis["name"]: execute_factor(r.code, ev.df).values for r in results if r.status == "accepted"}
    reg = FactorRegistry(Path(args.registry))
    registered_before = reg.compute(ev.df)
    strategy = validate_strategy(ev, accepted, registered=registered_before) if not args.no_strategy else None
    reg_status = {}
    for r in results:
        if r.status != "accepted":
            continue
        ok, why = admit(r.metrics, r.test_metrics)
        if ok:
            reg.add(r.hypothesis["name"], r.code, r.hypothesis, r.metrics, r.test_metrics, tr.run_id)
        reg_status[r.hypothesis["name"]] = "已登记" if ok else f"未登记：{why}"
        tr.log("registry", name=r.hypothesis["name"], admitted=ok, reason=why)
    tr.write("results.json", results_json(results))
    if strategy:
        tr.write("strategy.json", json.dumps(strategy, ensure_ascii=False, indent=2))
    path = write_summary(tr, sop, results, strategy, reg_status, ev.neutralized)
    print(path.read_text(encoding="utf-8"))
    print(f"\ntrace: {tr.file}")
    return {"run_id": tr.run_id, "results": results, "strategy": strategy, "summary": path}


def cmd_daily(args) -> Path:
    """最新收盘后：因子库 + 登记因子等权合成 → 下一交易日目标持仓（只在最新成分里选）+ 相对上次持仓的买卖清单。"""
    import pandas as pd

    from .evaluate import baseline_library

    data = Path(args.data)
    df, risk = load_hs300(data, universe="live")
    ev = Evaluator(df, risk=risk)
    factors = {**baseline_library(df, risk), **FactorRegistry(Path(args.registry)).compute(df)}
    cur_file = data / "universe_current.csv"
    universe = ({c.replace(".", "").upper() for c in pd.read_csv(cur_file)["code"]} if cur_file.exists() else None)
    if universe is not None:
        have = set(df.index.get_level_values("instrument"))
        if universe - have:
            print(f"注意：最新成分里有 {len(universe - have)} 只本地没有行情（先运行 qharness update）：{sorted(universe - have)[:10]}")
    else:
        print("注意：没有 universe_current.csv，按研究股票池选股（先运行 qharness update 刷新最新成分）")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    last = str(df.index.get_level_values("datetime").max().date())
    prev = sorted(p for p in out.glob("*.csv") if p.stem < last)  # 上一交易日及以前的持仓；同日重跑不算
    current = []
    if prev:
        p = pd.read_csv(prev[-1])
        current = p[p["weight"] > 0]["instrument"].tolist()
    acct = broker = None
    if args.paper:
        # 模拟账户：先把上次挂单在之后的第一个交易日开盘撮合、逐日记净值，再以账户实际持仓为当前持仓
        from .paper import Account, Market, PaperBroker
        from .risk import raw_close
        acct = Account.load(Path(args.paper), args.capital)
        broker = PaperBroker(Market(df, raw_close(df, data)), trades_log=Path(args.paper).with_suffix(".trades.csv"))
        if not acct.last_date:  # 新账户：从今天开始，不回补历史
            acct.last_date = str(broker.m.dates[-2].date())
        broker.settle(acct)
        current = sorted(acct.positions)
    tp = target_portfolio(ev, factors, args.topk, universe, current)
    path = out / f"{tp['date'].iloc[0]}.csv"
    tp.to_csv(path, index=False)
    print(f"因子：{', '.join(factors)}")
    print(tp.to_string(index=False))
    if acct is not None:
        dates = list(broker.m.dates)
        since = (len(dates) - 1 - dates.index(pd.Timestamp(acct.last_rebalance))) if acct.last_rebalance else None
        if since is None or since >= args.rebalance:
            from .paper import place_order
            place_order(acct, dates[-1], tp[tp.weight > 0]["instrument"].tolist())
            print(f"\n模拟账户：已挂单，下一个交易日开盘按目标权重调仓（{len(acct.pending['weights'])} 只）")
        else:
            print(f"\n模拟账户：距上次调仓 {since} 个交易日，未到调仓周期（{args.rebalance}），不挂单")
        acct.save(Path(args.paper))
        last = acct.nav[-1] if acct.nav else {"nav": acct.cash, "date": "-"}
        filled = [t for t in broker.trades if t["status"] == "filled"]
        print(f"模拟账户 {last['date']}：净值 {last['nav']:,.0f}（初始 {acct.initial:,.0f}），现金 {acct.cash:,.0f}，"
              f"持仓 {len(acct.positions)} 只；本次结算成交 {len(filled)} 笔，受限 {len(broker.trades) - len(filled)} 笔")
    print(f"\n已写入 {path}；买入 {int((tp.action == 'buy').sum())}，卖出 {int((tp.action == 'sell').sum())}，"
          f"持有 {int((tp.action == 'hold').sum())}")
    return path


def cmd_paper_sim(args) -> dict:
    """在历史数据上逐日回放模拟账户：d 日收盘按 d 日及以前的数据打分挂单，d+1 开盘撮合。"""
    import pandas as pd

    from .data import segment_dates
    from .evaluate import baseline_library
    from .paper import Account, Market, PaperBroker, nav_stats, place_order
    from .risk import raw_close
    from .strategy import backtest, combine, pick

    data = Path(args.data)
    df, risk = load_hs300(data)
    ev = Evaluator(df, risk=risk)
    factors = {**baseline_library(df, risk), **FactorRegistry(Path(args.registry)).compute(df)}
    score = combine(ev, list(factors.values()))  # 因子都只用当天及以前的数据（沙箱已检查），第 d 行不含未来信息
    dates = score.index
    start = pd.Timestamp(args.start) if args.start else segment_dates(df, ev.split.test)[0]
    end = pd.Timestamp(args.end) if args.end else dates[-1]
    days = [d for d in dates if start <= d <= end]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trades = out / "trades.csv"
    trades.unlink(missing_ok=True)
    acct = Account(cash=args.capital, initial=args.capital)
    acct.last_date = str(dates[dates.get_loc(days[0]) - 1].date())  # 从起始日开始结算，不回补之前的空仓净值
    broker = PaperBroker(Market(df, raw_close(df, data)), trades_log=trades)
    halted = df["close"].unstack("instrument").isna()
    for i, d in enumerate(days):
        broker.settle(acct, until=d)
        if i % args.rebalance == 0 and d != days[-1]:
            hold = pick(score.loc[d], set(acct.positions), args.topk, ~halted.loc[d], ~halted.loc[d])
            place_order(acct, d, hold)
    nav = pd.Series({pd.Timestamp(x["date"]): x["nav"] for x in acct.nav})
    close = df["close"].unstack("instrument")
    bench = close.ffill().pct_change(fill_method=None).loc[nav.index].mean(axis=1)
    stats = nav_stats(nav, bench)
    filled = [t for t in broker.trades if t["status"] == "filled"]
    stats.update(trades_filled=len(filled), trades_blocked=len(broker.trades) - len(filled),
                 fees=round(sum(t.get("fee", 0) for t in filled), 2), final_cash=round(acct.cash, 2))
    bt = backtest(df, score, pd.DatetimeIndex(days), args.topk, args.rebalance, risk=risk)  # 同区间回测（T+1 收盘成交口径）
    stats["backtest_same_period"] = bt
    pd.DataFrame(acct.nav).to_csv(out / "nav.csv", index=False)
    acct.save(out / "account.json")
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n净值 {out / 'nav.csv'}，成交 {trades}，账户 {out / 'account.json'}")
    return stats


DEFAULT_QLIB = ROOT / "data" / "qlib_cn"


def cmd_qlib_dump(data: Path, out: Path) -> Path:
    """行情缓存 → Qlib 数据目录（每日持仓用的 live 股票池，覆盖研究股票池）。"""
    from . import qlib_bridge as qb
    from .risk import raw_close

    df, risk = load_hs300(data, universe="live")
    path = qb.dump(df, raw_close(df, data), risk, out)
    print(f"Qlib 数据已写入 {path}：{df.index.get_level_values('instrument').nunique()} 只股票，"
          f"{df.index.get_level_values('datetime').nunique()} 个交易日")
    return path


def cmd_qlib_train(args) -> dict:
    """Alpha158 + 梯度提升树，按 Harness 的切分和测试段纪律评估，和 5 个基础因子的等权基线比较。"""
    from . import qlib_bridge as qb
    from .qlib_model import registry_exprs, train, walk_forward

    if not qb.available():
        raise SystemExit("没有 Qlib 数据目录：先运行 qharness qlib-dump")
    df, risk = load_hs300(Path(args.data))
    out = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    extra = registry_exprs(FactorRegistry(Path(args.registry)).items()) if args.with_registry else None
    ev = Evaluator(df, risk=risk, test_log=str(Path(args.out) / "test_opened.txt"))  # 同一配置的测试段只开一次
    if args.walk_forward:
        return _print_walk_forward(walk_forward(ev, out, model=args.model, extra=extra, topk=args.topk, seed=args.seed,
                                                step=args.step, valid_days=args.valid_days,
                                                min_train=args.min_train, train_days=args.train_days), out)
    s = train(ev, out, model=args.model, extra=extra, topk=args.topk, seed=args.seed)
    t, b = s["test"], s["baseline_test"]
    print(f"模型 {s['model']}，特征 {s['n_features']} 个，最优轮数 {s['best_iter']}；结果在 {out}")
    print(f"测试段（中性化）RankIC  模型 {t['rank_ic']}（t {t['rank_ic_t_nw']}）  基线 {b['rank_ic']}（t {b['rank_ic_t_nw']}）")
    for k in ("baseline", "model"):
        q = s["backtest_qlib"].get(k, {}) if isinstance(s["backtest_qlib"], dict) else {}
        ph = s["backtest_by_phase"][k]
        print(f"  {k:8s} 年化超额 自有回测 {s['backtest'][k]['annual_excess']:+.2%}（5 个调仓起点 "
              f"{ph['min']:+.1%} ~ {ph['max']:+.1%}）  Qlib 回测 {q.get('annual_excess', float('nan')):+.2%}")
    return s


def cmd_reports(args):
    from .report_fetch import INBOX, ReportIndex, research, run_inbox, run_pipeline

    if args.env_file:
        load_env_file(args.env_file)
    index = ReportIndex()
    if args.action == "fetch":
        llm = None if args.no_screen else make_llm(args.llm)
        return run_pipeline(args.days, args.download, args.research, llm, index)
    if args.action in ("scan", "watch"):
        llm = None if args.no_screen else make_llm(args.llm)
        inbox = Path(args.inbox) if args.inbox else INBOX
        if args.action == "scan":
            return run_inbox(llm, args.research, index, inbox)
        print(f"监视 {inbox}，每 {args.interval} 秒扫描一次（Ctrl-C 结束）")
        r = None
        try:
            while True:
                r = run_inbox(llm, args.research, ReportIndex(), inbox,
                              log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
                time.sleep(args.interval)
        except KeyboardInterrupt:
            return r
    if args.action == "list":
        for x in sorted(index.items.values(), key=lambda x: x.get("date", ""), reverse=True):
            print(f"{x.get('date')}  [{x.get('status')}]  {x.get('org')}《{x.get('title')}》  {x.get('reason', '')}")
        return index.items
    codes = [args.code] if args.code else [x["code"] for x in index.by_status("queued")[:args.top]]
    if not codes:
        print("没有通过预筛、等待研究的研报。先运行 qharness reports fetch")
    out = []
    for c in codes:
        if c not in index.items or not index.items[c].get("path"):
            raise SystemExit(f"索引里没有下载好的研报 {c}")
        r = research(index, c)
        print(f"《{r['title']}》：{r['reason']}（运行 {r['run_id']}）")
        out.append(r)
    return out


def _print_walk_forward(s: dict, out: Path) -> dict:
    f = lambda r: f"{r['rank_ic']:+.4f}（t {r['rank_ic_t_nw']:+.1f}）" if r.get("rank_ic") is not None else "—"  # noqa: E731
    pct = lambda v: f"{v:+.1%}" if v is not None else "—"  # noqa: E731
    print(f"滚动训练 {s['n_folds']} 折（每折预测 {s['step']} 天，训练窗口 {s['train_window']}），模型 {s['model']}；结果在 {out}")
    print("预测区间                  最优轮数  模型 RankIC         基线 RankIC         模型超额  基线超额")
    for r in s["folds"]:
        print(f"{r['predict'][0]}~{r['predict'][1]}  {r['best_iter']!s:>6}  {f(r['model']):18s}  {f(r['baseline']):18s}  "
              f"{pct(r['excess_model']):>7}  {pct(r['excess_baseline']):>7}")
    o = s["oos"]
    print(f"样本外整体 {o['dates'][0]}~{o['dates'][1]}：模型 {f(o['model'])}  基线 {f(o['baseline'])}")
    print(f"RankIC 为正的折 {s['folds_positive_rank_ic']}，跑赢基线的折 {s['folds_beat_baseline']}")
    for k in ("baseline", "model"):
        ph = s["backtest_by_phase"][k]
        print(f"  {k:8s} 样本外年化超额（5 个调仓起点）{ph['min']:+.1%} ~ {ph['max']:+.1%}，中位数 {ph['median']:+.1%}")
    return s


def _use_qlib(args):
    """有 Qlib 数据目录就启用（实验层多一个 run_qlib_factor 工具，策略验证多一份 Qlib 回测复核）。"""
    if getattr(args, "no_qlib", False) or getattr(args, "csv", None):  # 自带 CSV 行情和 Qlib 目录不是同一份数据
        os.environ.pop("QH_QLIB_URI", None)
        return
    if (DEFAULT_QLIB / "calendars" / "day.txt").exists() and not os.environ.get("QH_QLIB_URI"):
        os.environ["QH_QLIB_URI"] = str(DEFAULT_QLIB)


def main(argv=None):
    p = argparse.ArgumentParser(prog="qharness")
    sub = p.add_subparsers(dest="cmd", required=True)
    qd = sub.add_parser("qlib-dump", help="把行情缓存转成 Qlib 数据目录（update 之后会自动重做）")
    qd.add_argument("--data", default=str(DEFAULT_CACHE))
    qd.add_argument("--out", default=str(DEFAULT_QLIB))
    rp = sub.add_parser("reports", help="自动找研报：东方财富策略类研报 → 标题筛选 → 下载 → 预筛 → 研究")
    rp.add_argument("action", choices=["fetch", "scan", "watch", "list", "run"],
                    help="fetch：从东方财富搜集、下载并预筛；scan：扫描收件夹并预筛新文件；watch：持续监视收件夹；"
                         "list：查看索引；run：研究已通过预筛的研报")
    rp.add_argument("--inbox", help="收件夹（默认 data/reports/inbox）：自己下载的研报放进去")
    rp.add_argument("--interval", type=int, default=60, help="watch：每隔多少秒扫描一次")
    rp.add_argument("--days", type=int, default=30, help="fetch：往前找多少天")
    rp.add_argument("--download", type=int, default=10, help="fetch：最多下载多少篇")
    rp.add_argument("--research", type=int, default=0, help="fetch / scan / watch：预筛之后接着研究多少篇（默认 0，只筛不跑）")
    rp.add_argument("--top", type=int, default=1, help="run：研究最新的几篇")
    rp.add_argument("--code", help="run：指定研报编号")
    rp.add_argument("--no-screen", action="store_true", help="fetch：不调用模型预筛")
    rp.add_argument("--llm", choices=["anthropic", "openai"], default="anthropic")
    rp.add_argument("--env-file")
    sc6 = sub.add_parser("strategy-compare", help="已登记因子加入前后的策略对比：取最近若干个月（默认 6 个月）")
    sc6.add_argument("--data", default=str(DEFAULT_CACHE))
    sc6.add_argument("--registry", default="factors")
    sc6.add_argument("--months", type=int, default=6)
    sub.add_parser("agent-stats", help="从运行记录统计 Agent 指标：编码首次通过率、修复、经验命中、token 和延迟")
    er = sub.add_parser("eval-retrieval", help="检索评测：字面 / 语义 / 混合在真实运行记录上的对比")
    er.add_argument("--no-llm", action="store_true", help="不跑改写查询那组（要调用模型）")
    er.add_argument("--out", default=str(ROOT / "runs" / "retrieval_eval.json"))
    mc = sub.add_parser("mcp", help="MCP 服务（stdio）：沙箱、经验检索、验证段回测；不开放测试段和因子登记")
    mc.add_argument("--role", choices=["coder", "research"], default="coder", help="research 才有回测工具和完整经验")
    mc.add_argument("--data", default=str(DEFAULT_CACHE))
    mc.add_argument("--csv", help="真实行情 CSV（不启用 Qlib，也没有回测工具）")
    mc.add_argument("--registry", default=str(ROOT / "factors"))
    mc.add_argument("--experience", default=str(ROOT / "runs" / "experience.jsonl"))
    mc.add_argument("--runs", default=str(ROOT / "runs"))
    mc.add_argument("--no-qlib", action="store_true")
    sv = sub.add_parser("serve", help="本地前端：运行记录、审计轨迹、因子库、模型、模拟账户；可发起研究运行")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    qt = sub.add_parser("qlib-train", help="Qlib 模型训练：Alpha158 特征 + 梯度提升树，测试段评估和回测")
    qt.add_argument("--data", default=str(DEFAULT_CACHE))
    qt.add_argument("--registry", default=str(ROOT / "factors"))
    qt.add_argument("--with-registry", action="store_true", help="把因子库里 Qlib 表达式写的登记因子加进特征")
    qt.add_argument("--model", choices=["auto", "lgb", "hgb"], default="auto",
                    help="auto：LightGBM 能加载就用，否则用 sklearn 的直方图梯度提升树")
    qt.add_argument("--topk", type=int, default=20)
    qt.add_argument("--seed", type=int, default=0)
    qt.add_argument("--out", default=str(ROOT / "runs" / "qlib_model"))
    qt.add_argument("--walk-forward", action="store_true", help="滚动训练：每折重新训练，只预测下一段，样本外拼接评估")
    qt.add_argument("--step", type=int, default=63, help="滚动训练每折预测的天数（默认 63，约一个季度）")
    qt.add_argument("--valid-days", type=int, default=120, help="滚动训练每折的验证段天数（早停用）")
    qt.add_argument("--min-train", type=int, default=250, help="第一折训练段至少多少天")
    qt.add_argument("--train-days", type=int, help="训练窗口天数（不给则从头扩张）")
    for name in ("run", "demo"):
        s = sub.add_parser(name)
        s.add_argument("--report")
        s.add_argument("--hypothesis")
        s.add_argument("--llm", choices=["anthropic", "openai"], default="anthropic")
        s.add_argument("--env-file", help="从 .env 读取 ANTHROPIC_* / QH_LLM_* 配置")
        s.add_argument("--csv", help="真实行情 CSV：datetime,instrument,open,high,low,close,volume[,amount]")
        s.add_argument("--data", default=str(DEFAULT_CACHE), help="market.py 下载的行情缓存目录")
        s.add_argument("--runs", default=str(ROOT / "runs"))
        s.add_argument("--experience", default=str(ROOT / "runs" / "experience.jsonl"))
        s.add_argument("--no-strategy", action="store_true")
        s.add_argument("--registry", default=str(ROOT / "factors"))
        s.add_argument("--resume", metavar="RUN_ID", help="从检查点续跑中断的运行（沿用原运行的参数）")
        s.add_argument("--run-id", help="指定新运行的编号（前端发起运行时用来跟踪进度）")
        s.add_argument("--no-checkpoint", action="store_true", help="不写检查点")
        s.add_argument("--no-qlib", action="store_true", help="不启用 Qlib（即使有 Qlib 数据目录）")
        s.add_argument("--workers", type=int, default=1,
                       help="候选并行数（同一批候选看不到彼此刚沉淀的经验；默认 1 串行）")
    e = sub.add_parser("experience")
    e.add_argument("--experience", default=str(ROOT / "runs" / "experience.jsonl"))
    u = sub.add_parser("update")
    u.add_argument("--data", default=str(DEFAULT_CACHE))
    u.add_argument("--end")
    d = sub.add_parser("daily")
    d.add_argument("--data", default=str(DEFAULT_CACHE))
    d.add_argument("--registry", default=str(ROOT / "factors"))
    d.add_argument("--topk", type=int, default=20)
    d.add_argument("--out", default=str(ROOT / "portfolio"))
    d.add_argument("--paper", nargs="?", const=str(ROOT / "portfolio" / "account.json"),
                   help="同时更新模拟账户（默认 portfolio/account.json）：结算上次挂单、记净值、按周期挂新单")
    d.add_argument("--capital", type=float, default=1_000_000)
    d.add_argument("--rebalance", type=int, default=5, help="调仓周期（交易日）")
    ps = sub.add_parser("paper-sim", help="模拟账户历史回放：逐日挂单、次日开盘撮合，和回测对照")
    ps.add_argument("--data", default=str(DEFAULT_CACHE))
    ps.add_argument("--registry", default=str(ROOT / "factors"))
    ps.add_argument("--start", help="默认测试段起点")
    ps.add_argument("--end")
    ps.add_argument("--topk", type=int, default=20)
    ps.add_argument("--rebalance", type=int, default=5)
    ps.add_argument("--capital", type=float, default=1_000_000)
    ps.add_argument("--out", default=str(ROOT / "runs" / "paper_sim"))
    f = sub.add_parser("factors")
    f.add_argument("--registry", default=str(ROOT / "factors"))
    f.add_argument("--activate", metavar="名称", help="重新启用一个被停用的登记因子（人工确认后）")
    f.add_argument("--deactivate", metavar="名称", help="人工停用一个登记因子")
    f.add_argument("--reason", default="", help="启用 / 停用的原因，记进登记表")
    mo = sub.add_parser("monitor", help="因子衰减监控：最近 120 个有标签的交易日重新评估，失效的登记因子自动停用")
    mo.add_argument("--data", default=str(DEFAULT_CACHE))
    mo.add_argument("--registry", default=str(ROOT / "factors"))
    mo.add_argument("--dry-run", action="store_true", help="只报告，不停用")
    dj = sub.add_parser("daily-job", help="每日任务（给定时任务调用）：更新行情 → 衰减监控 → 每日持仓和模拟账户结算；幂等")
    dj.add_argument("--data", default=str(DEFAULT_CACHE))
    dj.add_argument("--registry", default=str(ROOT / "factors"))
    dj.add_argument("--out", default=str(ROOT / "portfolio"))
    dj.add_argument("--force", action="store_true", help="不管时间和是否处理过，强制跑一遍")
    sc = sub.add_parser("schedule", help="定时运行每日任务（macOS launchd）")
    sc.add_argument("action", choices=["install", "status", "uninstall"])
    sc.add_argument("--dry-run", action="store_true", help="install：只显示要写入的配置，不安装")
    r = sub.add_parser("eval-research")
    r.add_argument("--llm", choices=["anthropic", "openai"], default="anthropic")
    r.add_argument("--env-file")
    r.add_argument("--data", default=str(DEFAULT_CACHE))
    r.add_argument("--out", default=str(ROOT / "runs" / "research_eval.json"))
    args = p.parse_args(argv)
    _use_qlib(args)

    if args.cmd == "qlib-dump":
        return cmd_qlib_dump(Path(args.data), Path(args.out))
    if args.cmd == "strategy-compare":
        import pandas as pd

        from .strategy import segment_dates, validate_strategy
        df, risk = load_hs300(Path(args.data))
        ev = Evaluator(df, risk=risk)
        factors = FactorRegistry(Path(args.registry)).compute(df)
        all_d = segment_dates(df, ev.split.test)
        dates = all_d[all_d > all_d[-1] - pd.DateOffset(months=args.months)]
        res = validate_strategy(ev, factors, dates=dates)
        res["note"] = "落在已经打开过的测试段内，是同一段数据的子区间，不是新的独立证据"
        out = ROOT / "runs" / f"strategy_{args.months}m.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return res
    if args.cmd == "agent-stats":
        from .agent_stats import write
        print(json.dumps(write(ROOT / "runs"), ensure_ascii=False, indent=2))
        return
    if args.cmd == "eval-retrieval":
        from .retrieval_eval import run as eval_retrieval
        res = eval_retrieval(ROOT / "runs", Path(args.out), None if args.no_llm else make_llm("anthropic"))
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return res
    if args.cmd == "reports":
        return cmd_reports(args)
    if args.cmd == "mcp":
        from .mcp_server import Tools, serve
        df, risk = (load_csv_panel(args.csv), None) if args.csv else load_hs300(Path(args.data))
        registered = FactorRegistry(Path(args.registry)).compute(df) if not args.csv else {}
        return serve(Tools(df, risk, args.role, args.experience, args.runs, extra_library=registered))
    if args.cmd == "serve":
        from .web import serve
        return serve(args.host, args.port)
    if args.cmd == "qlib-train":
        return cmd_qlib_train(args)
    if args.cmd == "update":
        from .market import update
        print(update(Path(args.data), args.end))
        if (DEFAULT_QLIB / "calendars" / "day.txt").exists():  # Qlib 数据跟着行情一起刷新，否则因子计算会被一致性检查拦下
            cmd_qlib_dump(Path(args.data), DEFAULT_QLIB)
        return None
    if args.cmd == "daily":
        return cmd_daily(args)
    if args.cmd == "paper-sim":
        return cmd_paper_sim(args)
    if args.cmd == "monitor":
        from .monitor import run_monitor
        df, risk = load_hs300(Path(args.data))
        return run_monitor(df, risk, FactorRegistry(Path(args.registry)), ROOT / "runs" / "monitor",
                           apply=not args.dry_run)
    if args.cmd == "daily-job":
        from .daily_job import run_daily_job
        r = run_daily_job(ROOT, Path(args.data), Path(args.registry), Path(args.out), force=args.force)
        if r.get("outcome") == "skipped":
            print(f"跳过：{r['note']}")
        if r.get("outcome") == "failed":
            raise SystemExit(1)
        return r
    if args.cmd == "schedule":
        from . import schedule as sch
        if args.action == "install":
            r = sch.install(dry_run=args.dry_run)
            if args.dry_run:
                print(r["content"])
            else:
                print(f"已安装 {r['plist']}：{sch.describe(r['slots'])}（北京时间周一到周五 17:30、18:30、19:30）")
        elif args.action == "uninstall":
            r = sch.uninstall()
            print("已卸载" if r["removed"] else "本来就没有安装")
        else:
            r = sch.status()
            print(json.dumps(r, ensure_ascii=False, indent=2))
        return r
    if args.cmd == "factors" and (args.activate or args.deactivate):
        reg = FactorRegistry(Path(args.registry))
        x = reg.set_active(args.activate or args.deactivate, bool(args.activate), reason=args.reason or "人工操作")
        print(f"{x['name']}：{'已启用' if x['active'] else '已停用'}")
        return x
    if args.cmd == "factors":
        for x in FactorRegistry(Path(args.registry)).items(active_only=False):
            t = (x.get("test_metrics") or {}).get("test", {})
            print(f"[{'启用' if x.get('active', True) else '停用'}] {x['name']}  测试段 RankIC {t.get('rank_ic')}  "
                  f"t {t.get('rank_ic_t_nw')}  来自 {x['run_id']}")
        return None
    if args.cmd == "eval-research":
        from .research_eval import run_eval
        if args.env_file:
            load_env_file(args.env_file)
        df, risk = load_hs300(Path(args.data))
        report = run_eval(make_llm(args.llm), Evaluator(df, risk=risk))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return report
    if args.cmd == "experience":
        for x in ExperienceStore(args.experience).items:
            print(f"[{x.kind}{'/' + x.error_type if x.error_type else ''}] {x.task}")
        return None
    if args.cmd == "demo" and not (args.report or args.hypothesis):
        args.report = str(ROOT / "samples" / "report_volume_price.md")
    if not (args.report or args.hypothesis or args.resume):
        p.error("需要 --report、--hypothesis 或 --resume")
    return cmd_run(args)
