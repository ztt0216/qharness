"""行情面板、标签与样本切分（真实行情见 market.py）。

面板格式与 Qlib 一致：MultiIndex (datetime, instrument)，按 (datetime, instrument) 排序。
因子代码只能拿到 FIELDS 里的列；标签和切分只在 Harness 内部使用。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume", "amount"]
VAL_FIELDS = ["pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]
FIN_FIELDS = ["roe", "np_margin", "gp_margin", "eps_ttm", "net_profit", "yoy_ni", "yoy_equity", "yoy_asset", "yoy_pni"]
FUNDAMENTAL_FIELDS = VAL_FIELDS + FIN_FIELDS
# 换手率（成交量 / 流通股本），有市值数据时才有；估值和财务（fundamentals.py，按公告日对齐），下载过才有
OPTIONAL_FIELDS = ["turnover"] + FUNDAMENTAL_FIELDS
FIELD_DESC = {
    "open": "开盘价（后复权）", "high": "最高价（后复权）", "low": "最低价（后复权）", "close": "收盘价（后复权）",
    "volume": "成交量（股）", "amount": "成交额（元）", "turnover": "日换手率",
    "pe_ttm": "市盈率 TTM（亏损为负）", "pb_mrq": "市净率 MRQ", "ps_ttm": "市销率 TTM", "pcf_ttm": "市现率 TTM（经营现金流，可为负）",
    "roe": "净资产收益率（报告期累计）", "np_margin": "销售净利率", "gp_margin": "销售毛利率（银行等为空）",
    "eps_ttm": "每股收益 TTM", "net_profit": "净利润（元，报告期累计）", "yoy_ni": "净利润同比增速",
    "yoy_equity": "净资产同比增速", "yoy_asset": "总资产同比增速", "yoy_pni": "归母净利润同比增速",
}


def describe_fields(fields: list[str]) -> str:
    """给模型看的字段说明；财务字段注明按公告日对齐、季度更新。"""
    base = [f"{f}（{FIELD_DESC.get(f, f)}）" for f in fields if f not in FUNDAMENTAL_FIELDS]
    val = [f"{f}（{FIELD_DESC[f]}）" for f in fields if f in VAL_FIELDS]
    fin = [f"{f}（{FIELD_DESC[f]}）" for f in fields if f in FIN_FIELDS]
    out = "日频行情：" + "、".join(base)
    if val:
        out += "；每日估值：" + "、".join(val)
    if fin:
        out += "；季报财务（按公告日之后第一个交易日生效，季度之间保持不变）：" + "、".join(fin)
    return out


def missing_data_note(fields: list[str]) -> str:
    """还没有的数据，提示词里用来判断哪些观点不可检验。"""
    have_fin = any(f in FIN_FIELDS for f in fields)
    have_val = any(f in VAL_FIELDS for f in fields)
    lack = []
    if not have_val:
        lack.append("估值（PE/PB 等）")
    if not have_fin:
        lack.append("财务（ROE、利润率、盈利增速等）")
    lack += ["资产负债表和现金流量表细项", "分析师预期", "宏观", "资金流", "指数数据", "日内分钟、逐笔成交、委托单数据"]
    return "没有" + "、".join(lack)


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
