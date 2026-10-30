"""模拟交易账户：T 日收盘后挂单，T+1 日按开盘价撮合的多头账户，状态落盘。

- 价格：成交和记市值都用不复权价（不复权开盘价 = 后复权开盘价 × 当天复权系数），按 100 股一手取整
- 限制：停牌不成交；开盘即涨停买不进、开盘即跌停卖不出
- 费用：佣金万 2.5（单笔最低 5 元），卖出印花税 0.05%，滑点单边 0.1%（买入价上浮、卖出价下浮）。
  合计单边约 0.13%~0.18%，与回测假设的单边 0.15% 同一量级；不加滑点时模拟盘会系统性好于回测
- 分红送转：复权系数变化时按比例调整持股数（相当于分红再投资）
- 状态：现金、持仓、挂单、每日净值、成交记录写在一个 JSON 里，每天接着上次的状态继续
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .risk import limit_pct

LOT = 100


@dataclass
class Costs:
    commission: float = 0.00025
    min_commission: float = 5.0
    stamp_duty: float = 0.0005  # 卖出
    slippage: float = 0.001     # 单边，作用在成交价上

    def fill_price(self, p: float, side: str) -> float:
        return round(p * (1 + self.slippage if side == "buy" else 1 - self.slippage), 2)

    def fee(self, value: float, side: str) -> float:
        c = max(self.min_commission, value * self.commission) if value > 0 else 0.0
        return c + (value * self.stamp_duty if side == "sell" else 0.0)


@dataclass
class Account:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)  # 股数
    last_date: str = ""                                          # 已结算到的交易日
    last_rebalance: str = ""
    pending: dict | None = None                                  # {"date": 下单日, "weights": {股票: 目标权重}}
    nav: list[dict] = field(default_factory=list)                # 每日 {date, nav, cash, n_positions}
    initial: float = 0.0

    @classmethod
    def load(cls, path: Path, capital: float = 1_000_000) -> "Account":
        path = Path(path)
        if path.exists():
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        return cls(cash=capital, initial=capital)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.rename(path)


class Market:
    """不复权开盘 / 收盘价、复权系数和涨跌停价（宽表，行是交易日，列是股票）。"""

    def __init__(self, df: pd.DataFrame, raw_close: pd.DataFrame):
        close = df["close"].unstack("instrument")
        self.dates = close.index
        raw = raw_close.reindex(index=close.index, columns=close.columns).where(close.notna())
        self.adj = (raw / close)                            # 不复权 / 后复权，分红送转日会变
        self.open = (df["open"].unstack("instrument").reindex_like(close) * self.adj).round(2)  # 最小价位 0.01 元
        self.close = raw
        lim = pd.Series({c: limit_pct(c) for c in close.columns})
        prev = raw.ffill().shift(1)
        self.up = (prev * (1 + lim)).round(2)
        self.down = (prev * (1 - lim)).round(2)

    def px(self, table: pd.DataFrame, d, inst) -> float:
        v = table.at[d, inst] if inst in table.columns else np.nan
        return float(v) if pd.notna(v) else np.nan


class PaperBroker:
    def __init__(self, market: Market, costs: Costs = Costs(), trades_log: Path | None = None):
        self.m, self.costs, self.trades_log = market, costs, trades_log
        self.trades: list[dict] = []

    def _log(self, t: dict):
        self.trades.append(t)
        if self.trades_log:
            new = not Path(self.trades_log).exists()
            pd.DataFrame([t]).to_csv(self.trades_log, mode="a", header=new, index=False)

    def value(self, acct: Account, d, table: pd.DataFrame | None = None) -> float:
        table = self.m.close if table is None else table
        v = 0.0
        for inst, sh in acct.positions.items():
            p = self.m.px(table, d, inst)
            if np.isnan(p):  # 停牌：用最近的收盘价
                s = self.m.close[inst].loc[:d].dropna() if inst in self.m.close.columns else pd.Series(dtype=float)
                p = float(s.iloc[-1]) if len(s) else 0.0
            v += sh * p
        return acct.cash + v

    def _corporate_actions(self, acct: Account, prev, d):
        """复权系数变了（分红、送转）：股数按 后复权价不变 的口径换算。"""
        if not prev:
            return
        for inst in list(acct.positions):
            a0 = self.m.adj[inst].loc[:prev].dropna() if inst in self.m.adj.columns else pd.Series(dtype=float)
            a1 = self.m.px(self.m.adj, d, inst)
            if len(a0) and not np.isnan(a1) and abs(a1 / a0.iloc[-1] - 1) > 0.002:  # 两个数据源取整带来的抖动不算
                acct.positions[inst] *= a0.iloc[-1] / a1

    def execute(self, acct: Account, d, weights: dict[str, float]):
        """在交易日 d 开盘按目标权重调仓：先卖后买，卖不出 / 买不进的跳过。"""
        nav = self.value(acct, d, self.m.open)
        target = {k: v for k, v in weights.items() if v > 0}
        # 卖出：不在目标里的清仓，超配的减到目标
        for inst, sh in sorted(acct.positions.items()):
            p, dn = self.m.px(self.m.open, d, inst), self.m.px(self.m.down, d, inst)
            want = nav * target.get(inst, 0.0) / p if p and not np.isnan(p) else sh
            qty = sh if inst not in target else np.floor((sh - want) / LOT) * LOT
            if qty <= 0:
                continue
            if np.isnan(p) or (not np.isnan(dn) and p <= dn + 0.001):
                self._log({"date": str(d.date()), "instrument": inst, "side": "sell", "shares": qty, "price": p,
                           "status": "blocked", "reason": "停牌" if np.isnan(p) else "开盘跌停"})
                continue
            value = qty * self.costs.fill_price(p, "sell")
            fee = self.costs.fee(value, "sell")
            acct.cash += value - fee
            acct.positions[inst] = sh - qty
            if acct.positions[inst] < 1e-6:
                del acct.positions[inst]
            self._log({"date": str(d.date()), "instrument": inst, "side": "sell", "shares": qty,
                       "price": self.costs.fill_price(p, "sell"), "open": p,
                       "value": round(value, 2), "fee": round(fee, 2), "status": "filled", "reason": ""})
        # 买入：按目标权重补足，整手；现金不够时按顺序买到用完
        for inst, w in sorted(target.items(), key=lambda x: -x[1]):
            p, up = self.m.px(self.m.open, d, inst), self.m.px(self.m.up, d, inst)
            have = acct.positions.get(inst, 0.0)
            if np.isnan(p):
                self._log({"date": str(d.date()), "instrument": inst, "side": "buy", "shares": 0, "price": p,
                           "status": "blocked", "reason": "停牌"})
                continue
            qty = np.floor((nav * w / p - have) / LOT) * LOT
            if qty <= 0:
                continue
            if not np.isnan(up) and p >= up - 0.001:
                self._log({"date": str(d.date()), "instrument": inst, "side": "buy", "shares": qty, "price": p,
                           "status": "blocked", "reason": "开盘涨停"})
                continue
            fp = self.costs.fill_price(p, "buy")
            while qty > 0 and qty * fp + self.costs.fee(qty * fp, "buy") > acct.cash:
                qty -= LOT
            if qty <= 0:
                continue
            value = qty * fp
            fee = self.costs.fee(value, "buy")
            acct.cash -= value + fee
            acct.positions[inst] = have + qty
            self._log({"date": str(d.date()), "instrument": inst, "side": "buy", "shares": qty, "price": fp, "open": p,
                       "value": round(value, 2), "fee": round(fee, 2), "status": "filled", "reason": ""})

    def settle(self, acct: Account, until=None) -> list:
        """从上次结算日之后逐个交易日：处理分红送转 → 挂单在下单日后第一个交易日开盘成交 → 收盘记净值。"""
        days = [d for d in self.m.dates if (not acct.last_date or d > pd.Timestamp(acct.last_date))
                and (until is None or d <= until)]
        prev = acct.last_date
        for d in days:
            self._corporate_actions(acct, prev, d)
            if acct.pending and d > pd.Timestamp(acct.pending["date"]):
                self.execute(acct, d, acct.pending["weights"])
                acct.last_rebalance = str(d.date())
                acct.pending = None
            acct.nav.append({"date": str(d.date()), "nav": round(self.value(acct, d), 2),
                             "cash": round(acct.cash, 2), "n_positions": len(acct.positions)})
            acct.last_date = prev = str(d.date())
        return days


def place_order(acct: Account, d, holdings: list[str]):
    """收盘后挂单：目标持仓等权，下一个交易日开盘成交。"""
    acct.pending = {"date": str(pd.Timestamp(d).date()), "weights": {h: 1.0 / len(holdings) for h in holdings}}


def nav_stats(nav: pd.Series, bench: pd.Series | None = None) -> dict:
    ret = nav.pct_change().dropna()
    out = {"start": str(nav.index[0].date()), "end": str(nav.index[-1].date()),
           "total_return": round(float(nav.iloc[-1] / nav.iloc[0] - 1), 4),
           "annual_return": round(float((1 + ret).prod() ** (252 / max(len(ret), 1)) - 1), 4),
           "max_drawdown": round(float((nav / nav.cummax() - 1).min()), 4)}
    if bench is not None:
        ex = (ret - bench.reindex(ret.index).fillna(0)).dropna()
        enav = (1 + ex).cumprod()
        out.update(annual_excess=round(float(ex.mean() * 252), 4),
                   excess_max_drawdown=round(float((enav / enav.cummax() - 1).min()), 4))
    return out
