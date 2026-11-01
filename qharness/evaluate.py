"""因子评估工具：IC / RankIC / ICIR / Newey-West t / 分年 IC / 分层 / 换手 / 与因子库相关性。

- 有风险数据（risk.load_risk）时，因子和因子库都先做市值 + 行业中性化再评估，
  防止「其实是在押小盘 / 押某个行业」的因子混过去；同时给出未中性化的 RankIC 作对照。
- 标签是 5 日收益，相邻两天高度重叠，ICIR 会高估稳定性，所以显著性用 Newey-West t（滞后 = 标签期）。
- 分段评估，每段末尾去掉 EMBARGO 天。测试段由 Evaluator 加锁，只能在研究层判定有效后打开一次。
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LABEL_HORIZON, Split, make_label, make_split, segment_dates
from .risk import neutralize


def _wide(s: pd.Series) -> pd.DataFrame:
    return s.unstack("instrument")


def _row_corr(a: np.ndarray, b: np.ndarray, min_n: int = 20) -> np.ndarray:
    mask = ~(np.isnan(a) | np.isnan(b))
    out = np.full(a.shape[0], np.nan)
    for i in range(a.shape[0]):
        m = mask[i]
        if m.sum() < min_n:
            continue
        x, y = a[i, m], b[i, m]
        if x.std() == 0 or y.std() == 0:
            continue
        out[i] = np.corrcoef(x, y)[0, 1]
    return out


def _row_rank(w: pd.DataFrame) -> pd.DataFrame:
    return w.rank(axis=1, pct=True)


def _roll(s: pd.Series, n: int, how: str) -> pd.Series:
    r = s.groupby(level="instrument").rolling(n, min_periods=max(5, n // 2))
    return getattr(r, how)().droplevel(0).reindex(s.index)


def baseline_library(df: pd.DataFrame, risk: pd.DataFrame | None = None) -> dict[str, pd.Series]:
    """已有因子库（基线策略在用的常见因子），用于判断新因子有没有增量，以及策略对照。"""
    g = df["close"].groupby(level="instrument")
    ret = g.pct_change(fill_method=None)
    lib = {
        "reversal_5d": -(df["close"] / g.shift(5) - 1),
        "momentum_20d": df["close"] / g.shift(20) - 1,
        "low_vol_20d": -_roll(ret, 20, "std"),
        "illiquidity_20d": _roll(ret.abs() / df["amount"] * 1e9, 20, "mean"),
    }
    if "turnover" in df:
        lib["low_turnover_20d"] = -_roll(df["turnover"], 20, "mean")
    return {k: v.rename(k) for k, v in lib.items()}


def newey_west_t(x: np.ndarray, lags: int = LABEL_HORIZON) -> float:
    """均值的 Newey-West t 统计量（Bartlett 核），处理重叠标签带来的序列相关。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < lags + 2:
        return float("nan")
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else float("nan")


class TestSetLocked(RuntimeError):
    pass


class Evaluator:
    def __init__(self, df: pd.DataFrame, split: Split | None = None, n_groups: int = 5,
                 risk: pd.DataFrame | None = None, test_log: str | None = None,
                 extra_library: dict[str, pd.Series] | None = None):
        self.df = df.sort_index()
        self.split = split or make_split(self.df)
        self.label_w = _wide(make_label(self.df))
        self.n_groups = n_groups
        self.risk = risk
        self.neutralized = risk is not None
        self.raw_library = baseline_library(self.df, risk)
        self.raw_extra = extra_library or {}
        # 相关性比较的因子库：基础因子 + 已登记因子（extra_library）。不加登记因子的话，和已登记的 EP 高度相关的 BP
        # 也能过「和因子库相关不超过 0.7」这一关，组合会在价值上重复加仓
        self.library = {k: self._prep(v) for k, v in {**self.raw_library, **(extra_library or {})}.items()}
        # 测试段锁：传了 test_log 就落盘，进程中断后续跑也不能把同一个因子的测试段再开一次
        self._test_log = test_log
        self._test_opened: set[str] = set()
        self._test_lock = threading.Lock()
        if test_log and Path(test_log).exists():
            self._test_opened = set(Path(test_log).read_text(encoding="utf-8").split())

    def _prep(self, factor: pd.Series) -> pd.DataFrame:
        """宽表 + 排名标准化；有风险数据时再做市值行业中性化。"""
        return neutralize(_wide(factor.reindex(self.df.index)), self.risk)

    def _segment_metrics(self, fw: pd.DataFrame, dates: pd.DatetimeIndex, raw: pd.DataFrame | None = None) -> dict:
        f = fw.loc[dates]
        y = self.label_w.loc[dates, f.columns]
        ic = _row_corr(f.values, y.values)
        ric_all = _row_corr(_row_rank(f).values, _row_rank(y).values)
        ic, ric = ic[~np.isnan(ic)], ric_all[~np.isnan(ric_all)]
        if len(ric) < 10:
            return {"n_days": int(len(ric)), "valid": False}

        # 分层：按因子分成 n 组，每组未来收益均值；多空 = 最高组 - 最低组
        q = _row_rank(f)
        grp = np.ceil(q * self.n_groups).clip(1, self.n_groups)
        group_ret = {}
        for k in range(1, self.n_groups + 1):
            group_ret[k] = float(np.nanmean(np.where(grp.values == k, y.values, np.nan)))
        rets = [group_ret[k] for k in range(1, self.n_groups + 1)]
        monotonic = float(pd.Series(rets).rank().corr(pd.Series(range(len(rets)), dtype=float)))

        # 换手：1 - 相邻两日截面排名自相关
        r = _row_rank(f).values
        auto = _row_corr(r[1:], r[:-1])
        turnover = float(1 - np.nanmean(auto)) if np.isfinite(auto).any() else float("nan")

        years = pd.Series(ric_all, index=f.index).dropna()
        by_year = {str(k): round(float(v), 4) for k, v in years.groupby(years.index.year).mean().items()}
        raw_ric = None
        if raw is not None:
            rr = _row_corr(_row_rank(raw.loc[dates]).values, _row_rank(y).values)
            raw_ric = round(float(np.nanmean(rr)), 4)

        corr = {}
        for name, lw in self.library.items():
            c = _row_corr(_row_rank(f).values, _row_rank(lw.loc[dates, f.columns]).values)
            corr[name] = round(float(np.nanmean(c)), 4)

        return {
            "valid": True,
            "n_days": int(len(ric)),
            "ic": round(float(ic.mean()), 4),
            "rank_ic": round(float(ric.mean()), 4),
            "icir": round(float(ic.mean() / ic.std()), 4) if ic.std() > 0 else 0.0,
            "rank_icir": round(float(ric.mean() / ric.std()), 4) if ric.std() > 0 else 0.0,
            "rank_ic_t_nw": round(newey_west_t(ric_all), 2),
            "rank_ic_positive_ratio": round(float((ric > 0).mean()), 4),
            "rank_ic_by_year": by_year,
            "rank_ic_raw": raw_ric,  # 未中性化（有风险数据时才有意义）
            "group_returns": {f"G{k}": round(v, 5) for k, v in group_ret.items()},
            "long_short": round(rets[-1] - rets[0], 5),
            "group_monotonicity": round(monotonic, 4),
            "turnover": round(turnover, 4),
            "corr_with_library": corr,
            "max_abs_corr_with_library": round(max(abs(v) for v in corr.values()), 4) if corr else 0.0,
        }

    def evaluate(self, factor: pd.Series) -> dict:
        """训练段 + 验证段。研究层只能看到这里的结果。"""
        fw, raw = self._prep(factor), (_wide(factor.reindex(self.df.index)) if self.neutralized else None)
        return {
            "neutralized": self.neutralized,
            "train": self._segment_metrics(fw, segment_dates(self.df, self.split.train), raw),
            "valid": self._segment_metrics(fw, segment_dates(self.df, self.split.valid), raw),
        }

    def _value_keys(self, factor: pd.Series) -> list[str]:
        """按因子值认身份：测试段每天的截面分位数（保留 4 位）+ 数据范围。分位数对单调变换不变（2×EP、EP+常数是同一个因子），
        取反后的分位数是 1 − 原值，一起锁上：看过测试段之后不能翻个方向再开一次。换了数据范围（比如扩大股票池）算新的测试。"""
        d = segment_dates(self.df, self.split.test)
        w = _wide(factor.reindex(self.df.index)).loc[d]
        r = ((w.rank(axis=1) - 0.5).div(w.count(axis=1), axis=0)).round(4)  # (名次 − 0.5) / n：取反后正好是 1 − 原值
        sig = f"{d[0].date()}|{d[-1].date()}|{w.shape[1]}|".encode()
        pos = r.fillna(-1).to_numpy().tobytes()
        neg = (1 - r).round(4).fillna(-1).to_numpy().tobytes()
        return [hashlib.sha1(sig + b"v" + pos).hexdigest(), hashlib.sha1(sig + b"v" + neg).hexdigest()]

    def evaluate_test(self, factor: pd.Series, key: str | None = None) -> dict:
        """测试段对每个因子只开放一次，结果不回流到研究循环。身份按因子值认（见 _value_keys），
        另外给了 key（比如模型配置）也一起记；锁记在 test_log 里（研究运行用全局的 runs/test_opened.txt，跨运行生效）。"""
        keys = self._value_keys(factor) + ([hashlib.sha1(key.encode("utf-8")).hexdigest()] if key else [])
        with self._test_lock:  # 候选并行：查和记要一起做
            if any(k in self._test_opened for k in keys):
                raise TestSetLocked("这个因子（或它的单调变换、取反）的测试段已经评估过一次，不能重复使用")
            self._test_opened.update(keys)
            if self._test_log:
                with open(self._test_log, "a", encoding="utf-8") as f:
                    f.write("\n".join(keys) + "\n")
        fw, raw = self._prep(factor), (_wide(factor.reindex(self.df.index)) if self.neutralized else None)
        return {"test": self._segment_metrics(fw, segment_dates(self.df, self.split.test), raw)}
