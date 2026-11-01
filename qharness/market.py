"""真实 A 股日线：沪深 300 成分股，后复权，按股票缓存到本地，可断点续传。

两个数据源写成同一种格式（source 列标明来源）：
- baostock：股票池（query_hs300_stocks）和日线；并发过高会被封 IP（"黑名单用户"）。
- akshare（新浪 stock_zh_a_daily）：补齐 baostock 没下到的股票。成交量单位是股、成交额是元，与 baostock
  一致；后复权的基准价不同，但同一只股票的收益率一致（verify_sources 检查）。东财接口同样会断连接，没用。

- 股票池取**起始日**的沪深 300 成分（含之后退市、调出的股票），避免用今天的成分回看历史带来的幸存者偏差。
- 后复权（adjustflag=1）：价格序列不会因为之后的分红送转被改写，收益率正确。
- 停牌日（tradestatus=0）价格和成交量置为 NaN。

  python -m qharness.market --start 2023-01-01 --end 2026-10-29
  python -m qharness.market --source akshare      # 补齐缺失的股票
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .data import FIELDS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "data" / "hs300"
K_FIELDS = "date,code,open,high,low,close,volume,amount,tradestatus,isST"


def _rows(rs) -> list[list[str]]:
    out = []
    while rs.error_code == "0" and rs.next():
        out.append(rs.get_row_data())
    if rs.error_code != "0":
        raise RuntimeError(f"baostock: {rs.error_code} {rs.error_msg}")
    return out


def download_hs300(start: str, end: str, cache: Path = DEFAULT_CACHE, retries: int = 3,
                  shard: int = 0, n_shards: int = 1) -> Path:
    import baostock as bs

    cache.mkdir(parents=True, exist_ok=True)
    bs.login()
    try:
        uni_file = cache / "universe.csv"
        if not uni_file.exists():
            uni = pd.DataFrame(_rows(bs.query_hs300_stocks(date=start)), columns=["updateDate", "code", "code_name"])
            uni.to_csv(uni_file, index=False)
        codes = pd.read_csv(uni_file)["code"].tolist()[shard::n_shards]
        (cache / "meta.txt").write_text(f"start={start}\nend={end}\nadjustflag=1(后复权)\n", encoding="utf-8")
        for i, code in enumerate(codes):
            f = cache / f"{code}.csv"
            if f.exists():
                continue
            for attempt in range(retries):
                try:
                    rows = _rows(bs.query_history_k_data_plus(code, K_FIELDS, start_date=start, end_date=end,
                                                              frequency="d", adjustflag="1"))
                    tmp = f.with_suffix(".part")
                    pd.DataFrame(rows, columns=K_FIELDS.split(",")).to_csv(tmp, index=False)
                    tmp.rename(f)  # 写完再改名，中断不会留下半截文件
                    break
                except Exception as e:  # noqa: BLE001  网络抖动重试，重新登录
                    print(f"{code} 第 {attempt + 1} 次失败：{e}", flush=True)
                    time.sleep(3)
                    bs.login()
            print(f"[{i + 1}/{len(codes)}] {code}", flush=True)
    finally:
        bs.logout()
    return cache


def _ak_hist(code: str, start: str, end: str, retries: int = 4):
    """接口偶尔断连接，退避重试；全部失败返回 None。"""
    import akshare as ak

    for attempt in range(retries):
        try:
            return ak.stock_zh_a_daily(symbol=code.replace(".", ""), adjust="hfq",
                                       start_date=start.replace("-", ""), end_date=end.replace("-", ""))
        except Exception as e:  # noqa: BLE001
            print(f"{code} 第 {attempt + 1} 次失败：{type(e).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def download_akshare(start: str, end: str, cache: Path = DEFAULT_CACHE, pause: float = 1.5,
                     codes: list[str] | None = None) -> list[str]:
    """按 universe.csv（或指定代码）补齐缓存里缺的股票，串行 + 间隔，避免再被限流。返回没拿到数据的代码。"""
    codes = codes if codes is not None else pd.read_csv(cache / "universe.csv")["code"].tolist()
    todo = [c for c in codes if not (cache / f"{c}.csv").exists()]
    missing = []
    for i, code in enumerate(todo):
        f = cache / f"{code}.csv"
        d = _ak_hist(code, start, end)
        if d is None or d.empty:
            missing.append(code)  # 区间内已退市等
            print(f"[{i + 1}/{len(todo)}] {code} 无数据", flush=True)
            continue
        out = d[["open", "high", "low", "close", "volume", "amount"]].copy()
        out.insert(0, "code", code)
        out.insert(0, "date", d["date"].astype(str))
        out["tradestatus"], out["isST"], out["source"] = 1, "", "akshare_sina"  # 停牌日新浪不返回，加载时补成 NaN
        tmp = f.with_suffix(".part")
        out.to_csv(tmp, index=False)
        tmp.rename(f)
        print(f"[{i + 1}/{len(todo)}] {code}", flush=True)
        time.sleep(pause)
    (cache / "akshare_missing.txt").write_text("\n".join(missing), encoding="utf-8")
    return missing


SW_CODE = "008003"  # 申银万国行业分类标准
FALLBACK_CODE = "008004"  # 新财富行业分类：一级行业名与申万基本一致，部分股票巨潮没有申万记录时用它


def download_industry(cache: Path = DEFAULT_CACHE, pause: float = 1.0, codes: list[str] | None = None) -> Path:
    """巨潮资讯的行业变更历史，取申万一级（行业门类），带变更日期，可按时点还原；没有申万记录时用新财富分类。"""
    import akshare as ak

    out = cache / "ref" / "industry.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out, dtype=str) if out.exists() else pd.DataFrame(columns=["code", "date", "sw_l1"])
    codes = codes if codes is not None else pd.read_csv(cache / "universe.csv")["code"].tolist()
    codes = [c for c in codes if c not in set(done["code"])]
    rows = []
    for i, code in enumerate(codes):
        for attempt in range(3):
            try:
                d = ak.stock_industry_change_cninfo(symbol=code.split(".")[1], start_date="20000101",
                                                    end_date=time.strftime("%Y%m%d"))
                sw = d[d["分类标准编码"] == SW_CODE]
                sw = sw if not sw.empty else d[d["分类标准编码"] == FALLBACK_CODE]
                rows += [{"code": code, "date": str(r["变更日期"])[:10], "sw_l1": r["行业门类"]} for _, r in sw.iterrows()]
                break
            except Exception as e:  # noqa: BLE001
                print(f"{code} 行业 第 {attempt + 1} 次失败：{type(e).__name__}", flush=True)
                time.sleep(5 * (attempt + 1))
        if (i + 1) % 20 == 0 or i + 1 == len(codes):  # 分批落盘，可断点续传
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(out, index=False)
            rows = []
            print(f"[行业 {i + 1}/{len(codes)}]", flush=True)
        time.sleep(pause)
    return out


def download_cap(start: str, end: str, cache: Path = DEFAULT_CACHE, pause: float = 1.0,
                 codes: list[str] | None = None) -> Path:
    """新浪不复权日线里的收盘价和流通股本 → 流通市值；不复权价格也用于判断涨跌停。"""
    import akshare as ak

    d_out = cache / "ref" / "cap"
    d_out.mkdir(parents=True, exist_ok=True)
    codes = codes if codes is not None else pd.read_csv(cache / "universe.csv")["code"].tolist()
    codes = [c for c in codes if not (d_out / f"{c}.csv").exists()]
    for i, code in enumerate(codes):
        for attempt in range(3):
            try:
                d = ak.stock_zh_a_daily(symbol=code.replace(".", ""), adjust="",
                                        start_date=start.replace("-", ""), end_date=end.replace("-", ""))
                break
            except Exception as e:  # noqa: BLE001
                d = None
                print(f"{code} 市值 第 {attempt + 1} 次失败：{type(e).__name__}", flush=True)
                time.sleep(5 * (attempt + 1))
        if d is not None and not d.empty:
            out = pd.DataFrame({"date": d["date"].astype(str), "close_raw": d["close"],
                                "float_shares": d["outstanding_share"]})
            tmp = d_out / f"{code}.part"
            out.to_csv(tmp, index=False)
            tmp.rename(d_out / f"{code}.csv")
        print(f"[市值 {i + 1}/{len(codes)}] {code}", flush=True)
        time.sleep(pause)
    return d_out


def current_constituents(cache: Path = DEFAULT_CACHE) -> list[str]:
    """中证指数官网的最新沪深 300 成分，写到 universe_current.csv（每日持仓只在这里面选）。"""
    import akshare as ak

    d = ak.index_stock_cons_csindex(symbol="000300")
    ex = d["交易所"].map(lambda x: "sh" if "上海" in x else "sz")
    codes = (ex + "." + d["成分券代码"].astype(str).str.zfill(6)).tolist()
    pd.DataFrame({"code": codes, "name": d["成分券名称"], "date": d["日期"].astype(str)}).to_csv(
        cache / "universe_current.csv", index=False)
    return codes


def update(cache: Path = DEFAULT_CACHE, end: str | None = None, pause: float = 1.0, overlap: int = 10,
           net_timeout: float = 30.0) -> dict:
    """net_timeout：每次网络读写的超时秒数。akshare 内部发请求不带超时，服务器不回数据时会一直等
    （实测一次卡了 7 个多小时）；这里设全局默认超时，超时的股票记为失败，接着处理下一只。"""
    import socket

    prev = socket.getdefaulttimeout()
    socket.setdefaulttimeout(net_timeout)
    try:
        return _update(cache, end, pause, overlap)
    finally:
        socket.setdefaulttimeout(prev)


def _update(cache: Path, end: str | None, pause: float, overlap: int) -> dict:
    """每日增量更新：刷新成分 → 新成分股补全历史 → 已有股票追加最近几天（新浪后复权）。

    缓存里的后复权价可能来自 baostock，基准和新浪不同：用重叠区间最后一天的收盘价比值把新数据缩放后拼接，
    收益率不受影响。不复权价和流通股本直接追加。"""
    end = end or time.strftime("%Y-%m-%d")
    start = dict(l.split("=", 1) for l in (cache / "meta.txt").read_text(encoding="utf-8").split() if "=" in l)["start"]
    current = current_constituents(cache)
    known = set(pd.read_csv(cache / "universe.csv")["code"])
    new = [c for c in current if not (cache / f"{c}.csv").exists()]
    if new:
        print(f"新成分股 {len(new)} 只，下载完整历史", flush=True)
        download_akshare(start, end, cache, codes=new)
        download_cap(start, end, cache, codes=new)
        download_industry(cache, codes=new)
    stats = {"appended": 0, "up_to_date": 0, "no_new_data": 0, "failed": [], "new": new}
    codes = sorted(known | set(current))
    for i, code in enumerate(codes):
        if i and i % 50 == 0:
            print(f"  已处理 {i}/{len(codes)} 只（追加 {stats['appended']}，失败 {len(stats['failed'])}）", flush=True)
        f = cache / f"{code}.csv"
        if not f.exists():
            continue
        old = pd.read_csv(f, dtype={"date": str})
        last = old["date"].max()
        if last >= end:
            stats["up_to_date"] += 1
            continue
        frm = (pd.Timestamp(last) - pd.Timedelta(days=overlap * 2)).strftime("%Y-%m-%d")
        d = _ak_hist(code, frm, end)
        if d is None or d.empty:
            stats["failed"].append(code)
            continue
        d["date"] = d["date"].astype(str)
        both = sorted(set(d["date"]) & set(old["date"]))
        if not both:
            stats["failed"].append(code)
            continue
        k = float(old.set_index("date").loc[both[-1], "close"]) / float(d.set_index("date").loc[both[-1], "close"])
        add = d[d["date"] > last]
        if not add.empty:
            rows = pd.DataFrame({"date": add["date"], "code": code, "open": add["open"] * k, "high": add["high"] * k,
                                 "low": add["low"] * k, "close": add["close"] * k, "volume": add["volume"],
                                 "amount": add["amount"], "tradestatus": 1, "isST": "", "source": "akshare_sina_update"})
            out = pd.concat([old, rows], ignore_index=True)
            tmp = f.with_suffix(".part")
            out.to_csv(tmp, index=False)
            tmp.rename(f)
            stats["appended"] += 1
        else:  # 数据源还没发布今天的数据（新浪一般收盘后 1~2 小时才更新）
            stats["no_new_data"] += 1
        cf = cache / "ref" / "cap" / f"{code}.csv"
        if cf.exists():
            import akshare as ak
            try:
                r = ak.stock_zh_a_daily(symbol=code.replace(".", ""), adjust="", start_date=frm.replace("-", ""),
                                        end_date=end.replace("-", ""))
                c_old = pd.read_csv(cf, dtype={"date": str})
                r = r[r["date"].astype(str) > c_old["date"].max()]
                if not r.empty:
                    pd.concat([c_old, pd.DataFrame({"date": r["date"].astype(str), "close_raw": r["close"],
                                                    "float_shares": r["outstanding_share"]})]).to_csv(cf, index=False)
            except Exception as e:  # noqa: BLE001
                print(f"{code} 市值更新失败：{type(e).__name__}", flush=True)
        print(f"[更新 {i + 1}/{len(codes)}] {code}", flush=True)
        time.sleep(pause)
    (cache / "meta.txt").write_text(f"start={start}\nend={end}\nadjustflag=1(后复权)\n", encoding="utf-8")
    return stats


def verify_sources(codes: list[str], start: str, end: str, cache: Path = DEFAULT_CACHE) -> pd.DataFrame:
    """同一只股票两个来源的日收益率和成交量对比（应几乎一致）。"""
    rows = []
    for code in codes:
        b = pd.read_csv(cache / f"{code}.csv").set_index("date")
        a = _ak_hist(code, start, end)
        a = a.set_index(a["date"].astype(str))
        idx = b.index.intersection(a.index)
        rb, ra = b.loc[idx, "close"].pct_change(), a.loc[idx, "close"].pct_change()
        rows.append({"code": code, "days": len(idx), "ret_max_abs_diff": float((rb - ra).abs().max()),
                     "volume_ratio_median": float((b.loc[idx, "volume"] / a.loc[idx, "volume"]).median())})
        time.sleep(1.5)
    return pd.DataFrame(rows)


def load_hs300_panel(cache: Path = DEFAULT_CACHE, min_days: int = 60, universe: str = "research") -> pd.DataFrame:
    """universe="research"：只用起始日成分（universe.csv），研究和回测用，避免幸存者偏差；
    universe="live"：起始日成分 + 最新成分（universe_current.csv），每日持仓用。"""
    cache = Path(cache)
    codes = set(pd.read_csv(cache / "universe.csv")["code"]) if (cache / "universe.csv").exists() else None
    if universe == "live" and codes is not None and (cache / "universe_current.csv").exists():
        codes |= set(pd.read_csv(cache / "universe_current.csv")["code"])
    files = sorted(p for p in cache.glob("s[hz].*.csv") if codes is None or p.stem in codes)
    if not files:
        raise FileNotFoundError(f"{cache} 里没有行情缓存，先运行 python -m qharness.market")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        if len(d) < min_days:
            continue
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["date"])
    df["instrument"] = df["code"].str.replace(".", "", regex=False).str.upper()
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    halted = (df["tradestatus"] != 1) | (df["volume"] <= 0)
    df.loc[halted, ["open", "high", "low", "close", "volume", "amount"]] = float("nan")
    df = df.set_index(["datetime", "instrument"])[FIELDS].sort_index()
    # 补齐成 (交易日 × 股票) 的完整面板：上市前、退市后、停牌都是 NaN
    full = pd.MultiIndex.from_product([df.index.levels[0], df.index.levels[1]], names=["datetime", "instrument"])
    return df.reindex(full)


def load_hs300(cache: Path = DEFAULT_CACHE, universe: str = "research") -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """行情面板（有市值数据时附带 turnover 列，下载过财务数据时附带估值和财务列）+ 风险数据（市值、行业、停牌、涨跌停）。"""
    from .risk import load_risk, turnover_rate

    df = load_hs300_panel(cache, universe=universe)
    to = turnover_rate(df, cache)
    if to is not None:
        df["turnover"] = to
    from .fundamentals import load_fundamentals
    fund = load_fundamentals(df, cache)  # 估值和财务（按公告日对齐），下载过才有
    if fund is not None:
        df = df.join(fund)
    has_ref = (Path(cache) / "ref").exists()
    return df, (load_risk(df, cache) if has_ref else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=time.strftime("%Y-%m-%d"))
    p.add_argument("--cache", default=str(DEFAULT_CACHE))
    p.add_argument("--workers", type=int, default=1, help="并行进程数（每个进程一个分片）；baostock 并发太高会封 IP")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--source", choices=["baostock", "akshare", "industry", "cap"], default="baostock")
    p.add_argument("--update", action="store_true", help="每日增量更新（新浪），并刷新最新成分")
    a = p.parse_args()
    if a.update:
        print(update(Path(a.cache)))
        return
    if a.source == "industry":
        download_industry(Path(a.cache))
        return
    if a.source == "cap":
        download_cap(a.start, a.end, Path(a.cache))
        return
    if a.source == "akshare":
        missing = download_akshare(a.start, a.end, Path(a.cache))
        print("无数据：", missing)
        return
    if a.shard is not None or a.workers == 1:
        download_hs300(a.start, a.end, Path(a.cache), shard=a.shard or 0, n_shards=a.workers)
        return
    download_hs300(a.start, a.end, Path(a.cache), n_shards=10**9)  # 只取股票池
    import subprocess
    import sys
    procs = [subprocess.Popen([sys.executable, "-m", "qharness.market", "--start", a.start, "--end", a.end,
                               "--cache", a.cache, "--workers", str(a.workers), "--shard", str(i)])
             for i in range(a.workers)]
    sys.exit(max(p.wait() for p in procs))


if __name__ == "__main__":
    main()
