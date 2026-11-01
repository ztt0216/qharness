"""财务和估值数据（baostock），按公告日对齐成每日面板：因子在 T 日只能用到 T 日之前已经公布的数据。

- 估值（每日）：pe_ttm、pb_mrq、ps_ttm、pcf_ttm，来自 baostock 日线的估值字段（不复权）。ref/val/<代码>.csv
- 财务（季报）：roe、np_margin、gp_margin、eps_ttm、net_profit、yoy_ni、yoy_equity、yoy_asset、yoy_pni，
  来自 baostock 的盈利能力和成长能力接口，每期带公告日 pubDate。ref/fin/<代码>.csv
- 对齐：季报在**公告日之后的第一个交易日**起生效（公告常在收盘后发布，当天就用算前视），一直沿用到下一次公告。
  同一天公布两期（比如年报和一季报一起发）时取报告期更新的那期。不按报告期对齐：一季报的报告期是 3 月 31 日，
  4 月底才公布，按报告期用就是未来数据。
- 局限：baostock 返回的是最新口径，事后更正过的数据会覆盖原值（没有「当时公布的原始值」）。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .market import DEFAULT_CACHE

from .data import FIELD_DESC, FIN_FIELDS, FUNDAMENTAL_FIELDS, VAL_FIELDS  # noqa: E402,F401

_PROFIT = {"roeAvg": "roe", "npMargin": "np_margin", "gpMargin": "gp_margin", "epsTTM": "eps_ttm", "netProfit": "net_profit"}
_GROWTH = {"YOYNI": "yoy_ni", "YOYEquity": "yoy_equity", "YOYAsset": "yoy_asset", "YOYPNI": "yoy_pni"}


def _rows(rs) -> list[list[str]]:
    out = []
    while rs.error_code == "0" and rs.next():
        out.append(rs.get_row_data())
    return out


class _Session:
    """baostock 的连接没有超时：实测一次连接被对方重置后卡了一个多小时。这里设 socket 超时，
    超时或断开就重新登录再试，最多 retries 次。"""

    def __init__(self, timeout: float = 20.0, retries: int = 3, log=print):
        import socket

        import baostock as bs

        self.bs, self.retries, self.log = bs, retries, log
        self._prev = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        self._login()

    def _login(self):
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):  # baostock 登录登出会打印，刷屏
            self.bs.login()

    def call(self, fn: str, *args, **kw) -> list[list[str]]:
        last = None
        for attempt in range(self.retries):
            try:
                rs = getattr(self.bs, fn)(*args, **kw)
                if rs.error_code != "0" and "网络" not in rs.error_msg and "接收" not in rs.error_msg:
                    return []  # 业务错误（比如没有这一期）：不重试
                if rs.error_code == "0":
                    return _rows(rs)
                last = rs.error_msg
            except Exception as e:  # noqa: BLE001  超时、连接重置
                last = f"{type(e).__name__}: {e}"
            time.sleep(1 + attempt)
            try:
                self.bs.logout()
            except Exception:  # noqa: BLE001
                pass
            self._login()
        raise ConnectionError(f"{fn} 重试 {self.retries} 次仍失败：{last}")

    def relogin(self):
        try:
            self.bs.logout()
        except Exception:  # noqa: BLE001
            pass
        self._login()

    def close(self):
        import socket

        try:
            self.bs.logout()
        except Exception:  # noqa: BLE001
            pass
        socket.setdefaulttimeout(self._prev)


def _codes(cache: Path) -> list[str]:
    codes = set(pd.read_csv(cache / "universe.csv")["code"])
    if (cache / "universe_current.csv").exists():
        codes |= set(pd.read_csv(cache / "universe_current.csv")["code"])
    return sorted(codes)


def quarters(start: str, end: str) -> list[tuple[int, int]]:
    """报告期（季末）落在 [start, end] 里的季度。"""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return [(y, q) for y in range(s.year, e.year + 1) for q in range(1, 5)
            if s <= pd.Timestamp(y, 3 * q, 1) + pd.offsets.MonthEnd(0) <= e]


class _StockTimeout(Exception):
    pass


class _deadline:
    """单只股票的硬时限（主线程里用 SIGALRM）：socket 超时之外再兜一层，实测有一次卡了 43 分钟没触发超时。"""

    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        import signal
        import threading

        self.on = threading.current_thread() is threading.main_thread()
        if self.on:
            def _raise(signum, frame):
                raise _StockTimeout(f"超过 {self.seconds} 秒")
            self.old = signal.signal(signal.SIGALRM, _raise)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        import signal

        if self.on:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.old)
        return False


def download_valuation(cache: Path, codes: list[str], start: str, end: str, log=print) -> dict:
    d = cache / "ref" / "val"
    d.mkdir(parents=True, exist_ok=True)
    stats = {"done": 0, "appended": 0, "failed": []}
    sess = _Session(log=log)
    try:
        for i, code in enumerate(codes):
            f = d / f"{code}.csv"
            old = pd.read_csv(f, dtype={"date": str}) if f.exists() else None
            s = start if old is None or old.empty else str((pd.Timestamp(old["date"].iloc[-1]) + pd.Timedelta(days=1)).date())
            if s > end:
                stats["done"] += 1
                continue
            try:
                with _deadline(120):
                    rows = sess.call("query_history_k_data_plus", code, "date,peTTM,pbMRQ,psTTM,pcfNcfTTM", start_date=s,
                                     end_date=end, frequency="d", adjustflag="3")
            except (Exception, _StockTimeout) as e:  # noqa: BLE001
                stats["failed"].append(code)
                log(f"  {code} 估值下载失败：{type(e).__name__} {e}")
                sess.relogin()
                continue
            new = pd.DataFrame(rows, columns=["date"] + VAL_FIELDS)
            out = new if old is None else pd.concat([old, new], ignore_index=True)
            out.to_csv(f, index=False)
            stats["appended"] += len(new)
            stats["done"] += 1
            if (i + 1) % 50 == 0:
                log(f"  估值 {i + 1}/{len(codes)}")
    finally:
        sess.close()
    return stats


def download_fin(cache: Path, codes: list[str], qs: list[tuple[int, int]], refresh_last: int = 0, log=print) -> dict:
    """每只股票一个文件，记下已下载的季度；中断后再跑只补没下载的季度。refresh_last：最近几个季度强制重下（更新用）。"""
    d = cache / "ref" / "fin"
    d.mkdir(parents=True, exist_ok=True)
    stats = {"done": 0, "requests": 0, "failed": []}
    redo = set(qs[-refresh_last:]) if refresh_last else set()
    sess = _Session(log=log)
    try:
        for i, code in enumerate(codes):
            f = d / f"{code}.csv"
            old = pd.read_csv(f, dtype={"pubDate": str, "statDate": str}) if f.exists() else pd.DataFrame()
            have = set(zip(old["year"], old["quarter"])) if not old.empty else set()
            todo = [q for q in qs if q not in have or q in redo]
            rows = []
            try:
                with _deadline(180):
                    for y, q in todo:
                        p = sess.call("query_profit_data", code=code, year=y, quarter=q)
                        g = sess.call("query_growth_data", code=code, year=y, quarter=q)
                        stats["requests"] += 2
                        rec = {"year": y, "quarter": q, "pubDate": "", "statDate": ""}
                        if p:  # code,pubDate,statDate,roeAvg,npMargin,gpMargin,netProfit,epsTTM,MBRevenue,totalShare,liqaShare
                            r = p[0]
                            rec.update(pubDate=r[1], statDate=r[2], roe=r[3], np_margin=r[4], gp_margin=r[5],
                                       net_profit=r[6], eps_ttm=r[7])
                        if g:  # code,pubDate,statDate,YOYEquity,YOYAsset,YOYNI,YOYEPSBasic,YOYPNI
                            r = g[0]
                            rec.update(pubDate=rec["pubDate"] or r[1], statDate=rec["statDate"] or r[2],
                                       yoy_equity=r[3], yoy_asset=r[4], yoy_ni=r[5], yoy_pni=r[7])
                        rows.append(rec)  # 没有数据的季度也记一行（pubDate 为空），下次不再重复请求
            except (Exception, _StockTimeout) as e:  # noqa: BLE001  已下载的季度照样保存，下次只补没下载的
                stats["failed"].append(code)
                log(f"  {code} 财务下载中断：{type(e).__name__} {e}（已保存 {len(rows)} 个季度）")
                sess.relogin()
            if rows:
                new = pd.DataFrame(rows)
                keep = old[~old.apply(lambda r: (r["year"], r["quarter"]) in {(x["year"], x["quarter"]) for x in rows},
                                      axis=1)] if not old.empty else old
                pd.concat([keep, new], ignore_index=True).sort_values(["year", "quarter"]).to_csv(f, index=False)
            stats["done"] += 1
            if (i + 1) % 20 == 0:
                log(f"  财务 {i + 1}/{len(codes)}（请求 {stats['requests']} 次，失败 {len(stats['failed'])}）")
    finally:
        sess.close()
    return stats


def _inst(code: str) -> str:
    return code.replace(".", "").upper()


def load_fundamentals(panel: pd.DataFrame, cache: Path = DEFAULT_CACHE) -> pd.DataFrame | None:
    """按 panel 的 (datetime, instrument) 返回估值和财务字段；没有下载过就返回 None。"""
    cache = Path(cache)
    vdir, fdir = cache / "ref" / "val", cache / "ref" / "fin"
    if not vdir.exists() and not fdir.exists():
        return None
    cal = panel.index.get_level_values("datetime").unique().sort_values()
    insts = panel.index.get_level_values("instrument").unique()
    out = {}
    for f in sorted(vdir.glob("*.csv")) if vdir.exists() else []:
        inst = _inst(f.stem)
        if inst not in insts:
            continue
        v = pd.read_csv(f, parse_dates=["date"]).set_index("date").reindex(cal)
        out[inst] = v[VAL_FIELDS].apply(pd.to_numeric, errors="coerce")
    for f in sorted(fdir.glob("*.csv")) if fdir.exists() else []:
        inst = _inst(f.stem)
        if inst not in insts:
            continue
        r = pd.read_csv(f, dtype={"pubDate": str, "statDate": str})
        r = r[r["pubDate"].fillna("").str.len() > 0].copy()
        aligned = pd.DataFrame(np.nan, index=cal, columns=FIN_FIELDS)
        if not r.empty:
            r["pub"] = pd.to_datetime(r["pubDate"])
            r["stat"] = pd.to_datetime(r["statDate"])
            # 生效日：公告日之后的第一个交易日
            pos = cal.searchsorted(r["pub"].to_numpy(), side="right")
            r = r[pos < len(cal)].assign(eff=cal[pos[pos < len(cal)]])
            r = r.sort_values(["eff", "stat"]).drop_duplicates("eff", keep="last")  # 同一天生效取报告期更新的
            r = r[r["stat"] == r["stat"].cummax()]  # 晚公布的旧报告期（更正）不回退成旧数据
            vals = r.set_index("eff")[[c for c in FIN_FIELDS if c in r]].apply(pd.to_numeric, errors="coerce")
            aligned = vals.reindex(cal, method="ffill").reindex(columns=FIN_FIELDS)
        base = out.get(inst)
        out[inst] = aligned if base is None else base.join(aligned)
    if not out:
        return None
    wide = pd.concat(out, names=["instrument", "datetime"]).swaplevel().sort_index()
    return wide.reindex(panel.index).reindex(columns=[c for c in FUNDAMENTAL_FIELDS if c in wide.columns])


def update(cache: Path = DEFAULT_CACHE, end: str | None = None, fin_refresh_days: int = 7, log=print) -> dict:
    """每日任务调用：估值追加到最新；财务每隔 fin_refresh_days 天重下最近 2 个季度（新公告、更正）。"""
    cache = Path(cache)
    end = end or time.strftime("%Y-%m-%d")
    if not (cache / "ref" / "val").exists():
        return {"skipped": "还没有下载过财务和估值数据（python -m qharness.fundamentals --download）"}
    codes = _codes(cache)
    res = {"valuation": download_valuation(cache, codes, "2023-01-01", end, log=log)}
    stamp = cache / "ref" / "fin" / ".refreshed"
    if not stamp.exists() or time.time() - stamp.stat().st_mtime > fin_refresh_days * 86400:
        qs = quarters("2022-07-01", end)
        res["fin"] = download_fin(cache, codes, qs, refresh_last=2, log=log)
        stamp.touch()
    return res


def main():
    p = argparse.ArgumentParser(description="下载财务和估值数据（baostock）")
    p.add_argument("--cache", default=str(DEFAULT_CACHE))
    p.add_argument("--start", default="2023-01-01", help="估值起始日；财务从起始日前两个季度开始（保证起始日有可用的最新季报）")
    p.add_argument("--end", default=time.strftime("%Y-%m-%d"))
    p.add_argument("--only", choices=["val", "fin"])
    a = p.parse_args()
    cache = Path(a.cache)
    codes = _codes(cache)
    print(f"{len(codes)} 只股票", flush=True)
    if a.only != "fin":
        print("估值：", download_valuation(cache, codes, a.start, a.end, log=lambda m: print(m, flush=True)), flush=True)
    if a.only != "val":
        qs = quarters(str((pd.Timestamp(a.start) - pd.DateOffset(months=6)).date()), a.end)
        print(f"财务：{qs[0][0]}Q{qs[0][1]} ~ {qs[-1][0]}Q{qs[-1][1]}，{len(qs)} 个季度", flush=True)
        print("财务：", download_fin(cache, codes, qs, log=lambda m: print(m, flush=True)), flush=True)


if __name__ == "__main__":
    main()
