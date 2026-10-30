"""模块三：策略验证与每日持仓。基线（已有因子库的多因子 TopK 多头）vs 基线 + 新因子，样本外比较。

撮合口径：T 日收盘后算分，T+1 日收盘成交（A 股 T+1，保守处理），T+2 起计收益；
每 rebalance 天调仓一次，单边成本 cost；基准为股票池等权。
有风险数据时考虑可交易性：成交日停牌或涨停买不进，停牌或跌停卖不出（继续持有）。
停牌期间收益记 0，复牌日补上停牌期间的累计涨跌。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import segment_dates
from .evaluate import Evaluator


def combine(ev: Evaluator, factors: list[pd.Series]) -> pd.DataFrame:
    """等权合成：各因子先做和评估时一样的预处理（排名标准化，有风险数据时再中性化）。"""
    parts = [ev._prep(f) for f in factors]
    total = sum(p.fillna(0) for p in parts)
    n = sum(p.notna().astype(int) for p in parts)
    return (total / n).where(n * 2 > len(parts))  # 超过一半因子有值才打分


def _flags(risk: pd.DataFrame | None, name: str, like: pd.DataFrame) -> pd.DataFrame:
    if risk is None:
        return pd.DataFrame(False, index=like.index, columns=like.columns)
    return risk[name].unstack("instrument").reindex(index=like.index, columns=like.columns).fillna(False).astype(bool)


def pick(scores: pd.Series, current: set[str], topk: int, can_buy: pd.Series, can_sell: pd.Series) -> list[str]:
    """目标持仓：排名靠前的继续持有；卖不出的被迫留下；空位按排名补买得进的股票。"""
    ranked = scores.dropna().sort_values(ascending=False)
    desired = set(ranked.index[:topk])
    hold = [s for s in current if s in desired or not can_sell.get(s, True)]
    for s in ranked.index:
        if len(hold) >= topk:
            break
        if s not in hold and can_buy.get(s, False):
            hold.append(s)
    return hold


def backtest(df: pd.DataFrame, score: pd.DataFrame, dates: pd.DatetimeIndex, topk: int = 20,
             rebalance: int = 5, cost: float = 0.0015, risk: pd.DataFrame | None = None) -> dict:
    close = df["close"].unstack("instrument")
    ret = close.ffill().pct_change(fill_method=None)  # 停牌日 0，复牌日补上累计涨跌
    halted = close.isna()
    no_buy = halted | _flags(risk, "limit_up", close)
    no_sell = halted | _flags(risk, "limit_down", close)
    all_dates = ret.index
    w = pd.DataFrame(0.0, index=all_dates, columns=ret.columns)
    cur: list[str] = []
    turnovers, costs, blocked = [], pd.Series(0.0, index=all_dates), 0
    for i, d in enumerate(dates):
        pos = all_dates.get_loc(d)
        if i % rebalance == 0 and pos + 1 < len(all_dates):
            nxt = all_dates[pos + 1]  # 成交日
            s = score.loc[d]
            if s.notna().sum() >= topk:
                new = pick(s, set(cur), topk, ~no_buy.loc[nxt], ~no_sell.loc[nxt])
                desired = set(s.dropna().nlargest(topk).index)
                blocked += len(desired - set(new))
                old_w = pd.Series(0.0, index=ret.columns)
                old_w[cur] = 1.0 / len(cur) if cur else 0.0
                new_w = pd.Series(0.0, index=ret.columns)
                new_w[new] = 1.0 / len(new)
                to = float((new_w - old_w).abs().sum())
                turnovers.append(to)
                costs.iloc[pos + 1] += to * cost
                cur = new
        if cur:
            w.loc[d, cur] = 1.0 / len(cur)
    held = w.shift(2).loc[dates].fillna(0.0)
    port = (held * ret.loc[dates].fillna(0.0)).sum(axis=1) - costs.loc[dates]
    bench = ret.loc[dates].mean(axis=1)
    excess = (port - bench).iloc[2:]
    nav = (1 + excess).cumprod()
    mdd = float((nav / nav.cummax() - 1).min())
    ann = float(excess.mean() * 252)
    return {"annual_excess": round(ann, 4),
            "excess_sharpe": round(float(excess.mean() / excess.std() * np.sqrt(252)), 3) if excess.std() > 0 else 0.0,
            "max_drawdown_excess": round(mdd, 4),
            "avg_turnover_per_rebalance": round(float(np.mean(turnovers)) if turnovers else 0.0, 3),
            "blocked_trades": int(blocked),  # 因停牌 / 涨跌停没能按目标成交的股票次数
            "n_days": int(len(excess))}


def validate_strategy(ev: Evaluator, new_factors: dict[str, pd.Series], topk: int = 20,
                      registered: dict[str, pd.Series] | None = None) -> dict:
    """基线 = 因子库 + 已登记因子；对比加入本次通过的新因子后的效果（测试段）。"""
    lib = list(ev.raw_library.values()) + list((registered or {}).values())
    test_dates = segment_dates(ev.df, ev.split.test)
    base = backtest(ev.df, combine(ev, lib), test_dates, topk, risk=ev.risk)
    out = {"segment": f"{test_dates[0].date()} ~ {test_dates[-1].date()}", "baseline": base,
           "baseline_factors": list(ev.raw_library) + list(registered or {})}
    scores = {"baseline": combine(ev, lib)}
    if new_factors:
        scores["baseline_plus_new"] = combine(ev, lib + list(new_factors.values()))
        out["baseline_plus_new"] = backtest(ev.df, scores["baseline_plus_new"], test_dates, topk, risk=ev.risk)
    from . import qlib_bridge as qb

    if qb.available():  # 用 Qlib 的回测组件复核同一组信号（次日开盘成交、整手、双边费用、涨跌停停牌不成交）
        try:
            out["qlib"] = {k: qb.qlib_backtest(s, test_dates[0], test_dates[-1], topk) for k, s in scores.items()}
        except Exception as e:  # noqa: BLE001  复核失败不影响主流程，记下原因
            out["qlib"] = {"error": f"{type(e).__name__}: {e}"}
    return out


def target_portfolio(ev: Evaluator, factors: dict[str, pd.Series], topk: int = 20,
                     universe: set[str] | None = None, current: list[str] | None = None) -> pd.DataFrame:
    """最新交易日收盘后的目标持仓（下一交易日执行）。停牌股不买；今天涨停的股票标出来，明天可能买不进。"""
    score = combine(ev, list(factors.values()))
    last = score.index[-1]
    s = score.loc[last].dropna()
    if universe is not None:
        s = s[s.index.isin(universe)]
    halted = ev.df["close"].unstack("instrument").loc[last].isna()
    limit_up = _flags(ev.risk, "limit_up", score).loc[last]
    limit_dn = _flags(ev.risk, "limit_down", score).loc[last]
    can_buy = ~halted.reindex(s.index).fillna(True)
    can_sell = ~halted.reindex(s.index).fillna(False)
    hold = pick(s, set(current or []), topk, can_buy, can_sell)
    rank = s.rank(ascending=False)
    rows = []
    for inst in hold:
        rows.append({"date": str(last.date()), "instrument": inst, "weight": round(1.0 / len(hold), 6),
                     "score": round(float(s.get(inst, np.nan)), 4), "rank": int(rank.get(inst, -1)),
                     "industry": ev.risk.loc[(last, inst), "industry"] if ev.risk is not None else "",
                     "action": "hold" if inst in (current or []) else "buy",
                     "note": "今日涨停，明日可能买不进" if limit_up.get(inst, False) else ""})
    for inst in current or []:
        if inst not in hold:
            rows.append({"date": str(last.date()), "instrument": inst, "weight": 0.0,
                         "score": round(float(s.get(inst, np.nan)), 4), "rank": int(rank.get(inst, -1)),
                         "industry": ev.risk.loc[(last, inst), "industry"] if ev.risk is not None else "",
                         "action": "sell", "note": "今日跌停，明日可能卖不出" if limit_dn.get(inst, False) else ""})
    return pd.DataFrame(rows)
