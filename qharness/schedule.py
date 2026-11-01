"""定时运行每日任务（macOS launchd，用户级 LaunchAgent，不需要管理员权限）：`qharness schedule install / status / uninstall`。

- 北京时间周一到周五 08:30、09:00（盘前算组合、挂单）和 17:30、18:30、19:30（收盘后更新数据）各触发一次（换算成本机时间），每日任务本身幂等，先到先做；
  新浪一般收盘后 1~2 小时才发布当天行情，所以从 17:30 开始
- 换算用安装时的时差：本机有夏令时的话，切换后触发时间按北京时间会差 1 小时，仍在收盘之后；也可以切换后重新安装
- 电脑睡眠时错过的触发，launchd 会在唤醒后补跑一次；任务用 caffeinate -i 包着，跑起来之后不会因为空闲而睡过去
  （合上盖子仍会睡眠：要每天自动跑，电脑需要开着、接电源）
- 日志：runs/daily/launchd.log（进程输出）、runs/daily/<日期>.log（每日任务）
"""
from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
LABEL = "com.qharness.daily"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
BEIJING_TIMES = ((8, 30), (9, 0), (17, 30), (18, 30), (19, 30))


def local_slots(times=BEIJING_TIMES, now: datetime | None = None) -> list[dict]:
    """北京时间周一到周五的触发时刻 → 本机时区的 launchd StartCalendarInterval（Weekday：0 周日，1 周一……）。"""
    bj = ZoneInfo("Asia/Shanghai")
    now = (now or datetime.now(bj)).astimezone(bj)
    monday = (now - timedelta(days=now.weekday())).replace(second=0, microsecond=0)
    out = []
    for wd in range(5):
        for h, m in times:
            t = (monday + timedelta(days=wd)).replace(hour=h, minute=m).astimezone()  # 本机时区
            out.append({"Weekday": (t.weekday() + 1) % 7, "Hour": t.hour, "Minute": t.minute})
    return out


def plist_dict(root: Path = ROOT) -> dict:
    logdir = root / "runs" / "daily"
    return {
        "Label": LABEL,
        # caffeinate -i：任务运行期间不让系统空闲睡眠（实测这台 Mac 空闲 1 分钟就睡，任务只能在维护唤醒的几十秒里一点点跑）；
        # 进程结束就失效，不改系统设置；合上盖子的睡眠挡不住
        "ProgramArguments": ["/usr/bin/caffeinate", "-i", sys.executable, "-m", "qharness", "daily-job"],
        "WorkingDirectory": str(root),
        "StartCalendarInterval": local_slots(),
        "StandardOutPath": str(logdir / "launchd.log"),
        "StandardErrorPath": str(logdir / "launchd.log"),
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1", "MLFLOW_DISABLE_AGENT_HINT": "1"},
        "ProcessType": "Background",
    }


def _domain() -> str:
    return f"gui/{os.getuid()}"


def install(dry_run: bool = False) -> dict:
    d = plist_dict()
    data = plistlib.dumps(d)
    if dry_run:
        return {"plist": str(PLIST), "content": data.decode(), "installed": False}
    (ROOT / "runs" / "daily").mkdir(parents=True, exist_ok=True)
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["launchctl", "bootout", _domain(), str(PLIST)], capture_output=True)  # 先卸掉旧的（没有就忽略）
    PLIST.write_bytes(data)
    r = subprocess.run(["launchctl", "bootstrap", _domain(), str(PLIST)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"launchctl bootstrap 失败：{r.stderr.strip()}")
    return {"plist": str(PLIST), "installed": True, "slots": d["StartCalendarInterval"]}


def uninstall() -> dict:
    r = subprocess.run(["launchctl", "bootout", _domain(), str(PLIST)], capture_output=True, text=True)
    existed = PLIST.exists()
    if existed:
        PLIST.unlink()
    return {"removed": existed, "unloaded": r.returncode == 0}


def status() -> dict:
    r = subprocess.run(["launchctl", "print", f"{_domain()}/{LABEL}"], capture_output=True, text=True)
    info = {"installed": PLIST.exists(), "loaded": r.returncode == 0, "plist": str(PLIST)}
    if PLIST.exists():
        d = plistlib.loads(PLIST.read_bytes())
        info["slots"] = d.get("StartCalendarInterval")
        info["working_directory"] = d.get("WorkingDirectory")
        info["points_here"] = d.get("WorkingDirectory") == str(ROOT)
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith(("state =", "runs =", "last exit code =")):
                k, v = line.split("=", 1)
                info[k.strip().replace(" ", "_")] = v.strip()
    return info


def describe(slots: list[dict]) -> str:
    names = "日一二三四五六"
    by_time: dict[tuple, list[int]] = {}
    for s in slots:
        by_time.setdefault((s["Hour"], s["Minute"]), []).append(s["Weekday"])
    return "；".join(f"本机周{'、'.join(names[w] for w in sorted(set(ws)))} {h:02d}:{m:02d}"
                    for (h, m), ws in sorted(by_time.items()))
