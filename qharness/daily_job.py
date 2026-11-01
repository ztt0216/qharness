"""每日任务（`qharness daily-job`）：更新行情 → 更新财务和估值 → 因子衰减监控 → 每日持仓和模拟账户结算，给定时任务调用。

- 按北京时间判断：周末不跑；收盘前（16:00 前）不跑
- 幂等：当天已经处理完就直接退出（不联网），所以一晚上可以安排多次，先到先做
- 数据源还没发布当天行情（或节假日）：记为 no_new_data，等下一次
- 行情更新了但后面的步骤没跑完（比如中途崩溃）：下一次从监控接着做，不重复更新
- 每次的结果追加到 runs/daily/status.jsonl，日志在 runs/daily/<日期>.log
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BEIJING = ZoneInfo("Asia/Shanghai")


def beijing_now() -> datetime:
    return datetime.now(BEIJING)


def cache_last_date(cache: Path, sample: int = 30) -> date | None:
    """行情缓存的最新交易日（取前若干只股票文件最后一行的最大值）。"""
    last = None
    for f in sorted(cache.glob("s[hz].*.csv"))[:sample]:
        with f.open("rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 400))
            tail = fh.read().decode("utf-8", "ignore").strip().splitlines()[-1]
        try:
            d = date.fromisoformat(tail.split(",")[0])
        except ValueError:
            continue
        last = d if last is None or d > last else last
    return last


def _status_log(runs: Path) -> Path:
    p = runs / "daily"
    p.mkdir(parents=True, exist_ok=True)
    return p / "status.jsonl"


def history(runs: Path, n: int = 30) -> list[dict]:
    p = _status_log(runs)
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
    return rows[-n:]


def _done(runs: Path, d: date) -> bool:
    return any(r.get("date") == str(d) and r.get("outcome") == "done" for r in history(runs, 200))


class JobTimeout(Exception):
    pass


def run_daily_job(root: Path, cache: Path, registry: Path, portfolio: Path, force: bool = False,
                  now: datetime | None = None, steps: dict | None = None, max_seconds: int = 7200) -> dict:
    """steps 可替换各步（测试用）：update(cache) -> stats、monitor()、daily() -> nav。
    max_seconds：整个任务的时限（默认 2 小时），超时记为失败退出，不挡住之后的定时触发
    （launchd 在上一次还没结束时不会启动新的一次）。正常一次全量更新约 7 秒一只、380 只约 45 分钟。"""
    import signal
    import threading

    use_alarm = threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGALRM")
    if use_alarm:
        def _timeout(signum, frame):
            raise JobTimeout(f"每日任务超过 {max_seconds} 秒没有结束")
        old = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(max_seconds)
    try:
        return _run(root, cache, registry, portfolio, force, now, steps)
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


def _run(root: Path, cache: Path, registry: Path, portfolio: Path, force: bool, now: datetime | None,
         steps: dict | None) -> dict:
    runs = root / "runs"
    now = now or beijing_now()
    rec = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "beijing": now.strftime("%Y-%m-%d %H:%M"), "steps": []}
    logf = runs / "daily" / f"{now.date()}.log"
    _status_log(runs)

    def log(msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with logf.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def finish(outcome: str, **kw) -> dict:
        rec.update(outcome=outcome, ended=time.strftime("%Y-%m-%d %H:%M:%S"), **kw)
        with _status_log(runs).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        log(f"结束：{outcome}" + (f"（{kw.get('note')}）" if kw.get("note") else ""))
        return rec

    steps = steps or {}
    if not force and now.weekday() >= 5:
        return {"outcome": "skipped", "note": "周末"}  # 不写状态，免得刷屏
    if not force and now.hour < 16:
        return {"outcome": "skipped", "note": "北京时间 16:00 前，还没收盘结算"}
    last = cache_last_date(cache)
    if not force and last == now.date() and _done(runs, now.date()):
        return {"outcome": "skipped", "note": f"{now.date()} 已经处理过"}

    log(f"开始：北京时间 {rec['beijing']}，行情缓存最新 {last}")
    try:
        if last != now.date() or force:
            if "update" in steps:
                stats = steps["update"](cache)
            else:
                from .market import update
                stats = update(cache)
            rec["steps"].append("update")
            new_last = cache_last_date(cache)
            log(f"行情更新：追加 {stats.get('appended')} 只，数据源未发布 {stats.get('no_new_data')} 只，"
                f"失败 {len(stats.get('failed') or [])} 只；最新交易日 {new_last}")
            if new_last is None or (last is not None and new_last <= last):
                return finish("no_new_data", date=str(now.date()),
                              note="数据源还没有当天行情（或今天休市），等下一次")
            last = new_last
            if "fundamentals" in steps or "update" not in steps:
                try:  # 估值追加、财务每周重下最近两个季度；失败不挡结算，财务晚一天更新问题不大
                    if "fundamentals" in steps:
                        fs = steps["fundamentals"](cache)
                    else:
                        from .fundamentals import update as update_fundamentals
                        fs = update_fundamentals(cache, log=log)
                    rec["steps"].append("fundamentals")
                    log(f"财务和估值更新：{fs if 'skipped' in fs else {k: {kk: (len(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()} for k, v in fs.items()}}")
                except Exception as e:  # noqa: BLE001
                    rec.setdefault("warnings", []).append(f"财务和估值更新失败：{type(e).__name__}: {e}")
                    log(f"警告：财务和估值更新失败（{type(e).__name__}: {e}），继续用已有数据")
            if "update" not in steps:
                qlib = root / "data" / "qlib_cn"
                if (qlib / "calendars" / "day.txt").exists():
                    from .cli import cmd_qlib_dump
                    cmd_qlib_dump(cache, qlib)
                    rec["steps"].append("qlib_dump")
        if "monitor" in steps:
            mon = steps["monitor"]()
        else:
            from .market import load_hs300
            from .monitor import run_monitor
            from .registry import FactorRegistry
            df, risk = load_hs300(cache)
            log("因子衰减监控：")
            mon = run_monitor(df, risk, FactorRegistry(registry), runs / "monitor", log=log)
        rec["steps"].append("monitor")
        if "daily" in steps:
            nav = steps["daily"]()
        else:
            from .cli import cmd_daily
            ns = argparse.Namespace(data=str(cache), registry=str(registry), topk=20, out=str(portfolio),
                                    paper=str(portfolio / "account.json"), capital=1_000_000, rebalance=5)
            cmd_daily(ns)
            acct = json.loads((portfolio / "account.json").read_text(encoding="utf-8"))
            nav = acct["nav"][-1] if acct.get("nav") else None
        rec["steps"].append("daily")
        return finish("done", date=str(last), nav=nav, deactivated=mon.get("deactivated", []))
    except (Exception, JobTimeout) as e:  # noqa: BLE001  记下来，下一次定时会再试
        log(traceback.format_exc())
        return finish("failed", date=str(now.date()), error=f"{type(e).__name__}: {e}")
