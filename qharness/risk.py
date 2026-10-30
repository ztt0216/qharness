"""风险与交易约束数据：流通市值、申万一级行业（按时点）、停牌与涨跌停。

只给 Harness 用（中性化、因子库、回测撮合），因子代码拿不到这些列，避免把行业/市值暴露当成 alpha。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

UNKNOWN = "未知"


def _inst(code: str) -> str:
    return code.replace(".", "").upper()


def limit_pct(instrument: str) -> float:
    """涨跌幅限制：创业板(300/301)、科创板(688) 20%，其余 10%。ST 的 5% 没有数据，未处理。"""
    c = instrument[2:]
    return 0.2 if c.startswith(("300", "301", "688")) else 0.1


def load_risk(panel: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """返回与 panel 同索引的 DataFrame：log_mcap, industry, halted, limit_up, limit_down。缺数据时对应列为空。"""
    cache = Path(cache)
    dates = panel.index.get_level_values("datetime").unique().sort_values()
    insts = panel.index.get_level_values("instrument").unique()
    close = panel["close"].unstack("instrument").reindex(index=dates, columns=insts)

    raw = pd.DataFrame(np.nan, index=dates, columns=insts)
    shares = raw.copy()
    cap_dir = cache / "ref" / "cap"
    for f in sorted(cap_dir.glob("*.csv")) if cap_dir.exists() else []:
        inst = _inst(f.stem)
        if inst not in raw.columns:
            continue
        d = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        raw[inst] = d["close_raw"].reindex(dates)
        shares[inst] = d["float_shares"].reindex(dates)
    shares = shares.ffill()
    raw = raw.where(close.notna())  # 停牌日不算市值
    mcap = raw * shares
    log_mcap = np.log(mcap.where(mcap > 0))

    ind = pd.DataFrame(UNKNOWN, index=dates, columns=insts, dtype=object)
    ind_file = cache / "ref" / "industry.csv"
    if ind_file.exists():
        t = pd.read_csv(ind_file, dtype=str).dropna(subset=["sw_l1"])
        t["date"] = pd.to_datetime(t["date"])
        for code, g in t.groupby("code"):
            inst = _inst(code)
            if inst not in ind.columns:
                continue
            s = g.sort_values("date").drop_duplicates("date", keep="last").set_index("date")["sw_l1"]
            # 按变更日期向后填充；最早一条记录之前用最早的分类（申万 2021 版发布前没有记录，属于轻微回填）
            ind[inst] = s.reindex(s.index.union(dates)).ffill().bfill().reindex(dates)

    # 涨跌停：优先用不复权价，涨停价 = round(前收 × (1 + 限幅), 2)；没有不复权价时用复权收益率近似
    lim = pd.Series({c: limit_pct(c) for c in insts})
    prev = raw.shift(1)
    up_px, dn_px = (prev * (1 + lim)).round(2), (prev * (1 - lim)).round(2)
    ret = close.pct_change(fill_method=None)
    limit_up = (raw >= up_px - 0.001).where(raw.notna() & prev.notna(), ret >= lim - 0.002)
    limit_dn = (raw <= dn_px + 0.001).where(raw.notna() & prev.notna(), ret <= -lim + 0.002)

    out = pd.DataFrame({
        "log_mcap": log_mcap.stack(future_stack=True),
        "industry": ind.stack(future_stack=True),
        "halted": close.isna().stack(future_stack=True),
        "limit_up": limit_up.fillna(False).astype(bool).stack(future_stack=True),
        "limit_down": limit_dn.fillna(False).astype(bool).stack(future_stack=True),
    })
    out.index = out.index.set_names(["datetime", "instrument"])
    return out.reindex(panel.index)


def raw_close(panel: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """不复权收盘价宽表（模拟交易按真实价格成交和记市值）。"""
    dates = panel.index.get_level_values("datetime").unique().sort_values()
    insts = panel.index.get_level_values("instrument").unique()
    out = pd.DataFrame(np.nan, index=dates, columns=insts)
    cap_dir = Path(cache) / "ref" / "cap"
    for f in sorted(cap_dir.glob("*.csv")) if cap_dir.exists() else []:
        inst = _inst(f.stem)
        if inst in out.columns:
            out[inst] = pd.read_csv(f, parse_dates=["date"]).set_index("date")["close_raw"].reindex(dates)
    return out


def turnover_rate(panel: pd.DataFrame, cache: Path) -> pd.Series | None:
    """换手率 = 成交量 / 流通股本，作为因子代码可用的可选字段。"""
    cap_dir = Path(cache) / "ref" / "cap"
    if not cap_dir.exists() or not any(cap_dir.glob("*.csv")):
        return None
    parts = []
    for f in sorted(cap_dir.glob("*.csv")):
        d = pd.read_csv(f, parse_dates=["date"])
        d["instrument"] = _inst(f.stem)
        parts.append(d.rename(columns={"date": "datetime"}).set_index(["datetime", "instrument"])["float_shares"])
    sh = pd.concat(parts).sort_index()
    sh = sh[~sh.index.duplicated()].reindex(panel.index)
    sh = sh.groupby(level="instrument").ffill()
    return (panel["volume"] / sh).rename("turnover")


def neutralize(fw: pd.DataFrame, risk: pd.DataFrame | None, min_n: int = 30) -> pd.DataFrame:
    """逐日截面：因子转成排名标准分，对 [市值, 行业哑变量] 回归取残差。没有风险数据时只做排名标准化。"""
    z = fw.rank(axis=1, pct=True)
    z = z.sub(z.mean(axis=1), axis=0).div(z.std(axis=1), axis=0)
    if risk is None:
        return z
    lm = risk["log_mcap"].unstack("instrument").reindex(index=fw.index, columns=fw.columns)
    ind = risk["industry"].unstack("instrument").reindex(index=fw.index, columns=fw.columns)
    res = np.full(fw.shape, np.nan)
    zv, lv, iv = z.to_numpy(), lm.to_numpy(dtype=float), ind.to_numpy()
    for i in range(len(fw.index)):
        y = zv[i]
        m = ~np.isnan(y)
        if m.sum() < min_n:
            continue
        cols = [np.ones(m.sum())]
        size = lv[i, m]
        if np.isfinite(size).sum() >= min_n:
            size = np.where(np.isfinite(size), size, np.nanmedian(size))
            cols.append((size - size.mean()) / (size.std() or 1.0))
        labels = iv[i, m]
        for lab in pd.unique(labels):
            if lab in (None, UNKNOWN) or (isinstance(lab, float) and np.isnan(lab)):
                continue
            d = (labels == lab).astype(float)
            if 2 <= d.sum() < len(d):
                cols.append(d)
        X = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
        res[i, m] = y[m] - X @ beta  # 行业哑变量含截距会共线，lstsq 取最小范数解，残差不受影响
    return pd.DataFrame(res, index=fw.index, columns=fw.columns)
