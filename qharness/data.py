"""行情面板、标签与样本切分（真实行情见 market.py）。

面板格式与 Qlib 一致：MultiIndex (datetime, instrument)，按 (datetime, instrument) 排序。
因子代码只能拿到 FIELDS 里的列；标签和切分只在 Harness 内部使用。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume", "amount"]
OPTIONAL_FIELDS = ["turnover"]  # 换手率（成交量 / 流通股本），有市值数据时才有


def factor_columns(df: pd.DataFrame) -> list[str]:
    """因子代码能拿到的列。"""
    return FIELDS + [c for c in OPTIONAL_FIELDS if c in df.columns]
LABEL_HORIZON = 5  # 标签：T+1 收盘买入，持有 5 天
EMBARGO = LABEL_HORIZON + 1  # 每段末尾丢掉的天数，防止标签跨段


def load_csv_panel(path: str) -> pd.DataFrame:
    """CSV 列：datetime, instrument, open, high, low, close, volume[, amount]。"""
    df = pd.read_csv(path, parse_dates=["datetime"])
    if "amount" not in df:
        df["amount"] = df["volume"] * (df["open"] + df["close"]) / 2
    df = df.set_index(["datetime", "instrument"]).sort_index()
    return df[factor_columns(df)]


def make_label(df: pd.DataFrame) -> pd.Series:
    """Ref($close, -(1+H)) / Ref($close, -1) - 1，按股票计算。"""
    g = df["close"].groupby(level="instrument")
    return (g.shift(-(1 + LABEL_HORIZON)) / g.shift(-1) - 1).rename("label")


@dataclass(frozen=True)
class Split:
    train: tuple[pd.Timestamp, pd.Timestamp]
    valid: tuple[pd.Timestamp, pd.Timestamp]
    test: tuple[pd.Timestamp, pd.Timestamp]


def make_split(df: pd.DataFrame, ratios=(0.6, 0.2, 0.2)) -> Split:
    dates = df.index.get_level_values("datetime").unique().sort_values()
    n = len(dates)
    a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    return Split((dates[0], dates[a - 1]), (dates[a], dates[b - 1]), (dates[b], dates[-1]))


def segment_dates(df: pd.DataFrame, seg: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DatetimeIndex:
    """段内日期，去掉末尾 EMBARGO 天（标签会看到下一段的价格）。"""
    dates = df.index.get_level_values("datetime").unique().sort_values()
    d = dates[(dates >= seg[0]) & (dates <= seg[1])]
    return d[:-EMBARGO] if len(d) > EMBARGO else d[:0]
