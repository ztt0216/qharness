"""本地前端的后端（`qharness serve`）：读运行记录、因子库、模型结果和模拟账户；发起研究运行和每日任务。

- 只监听 127.0.0.1；接口只读项目目录里的文件，路径参数都校验过（运行编号、样例名不能带路径）
- 发起的任务是子进程跑 `python -m qharness ...`，命令只能是下面 JOBS 里的几种，日志写到 runs/jobs/<任务>.log
- 模型训练不提供按钮：同一配置的测试段只能开一次，从页面反复点会撞上测试段锁
- 前端是 web/ 下的 React 应用，`npm run build` 之后由这里一并托管
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
RUN_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
MODEL_ID = re.compile(r"^\d{8}_\d{6}$")
REPORT_EXT = {".pdf", ".pptx", ".md", ".txt"}
MAX_UPLOAD = 30 * 1024 * 1024


class Paths:
    def __init__(self, root: Path = ROOT):
        self.root = root
        self.runs = root / "runs"
        self.samples = root / "samples"
        self.uploads = self.runs / "uploads"
        self.jobs = self.runs / "jobs"
        self.factors = root / "factors"
        self.portfolio = root / "portfolio"
        self.data = root / "data" / "hs300"
        self.models = self.runs / "qlib_model"
        self.web = root / "web" / "dist"
        self.auto = root / "data" / "reports" / "auto"  # 自动搜集的研报（report_fetch.py）
        self.inbox = root / "data" / "reports" / "inbox"  # 收件夹：自己下载的研报放这里


def _json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:  # 正在写的最后一行
            pass
    return out


def _csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    df = pd.read_csv(p)
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _tokens(calls: list[dict]) -> dict:
    tin = sum((c.get("usage") or {}).get("input_tokens", 0) + (c.get("usage") or {}).get("cache_read_input_tokens", 0)
              for c in calls)
    tout = sum((c.get("usage") or {}).get("output_tokens", 0) for c in calls)
    return {"calls": len(calls), "input_tokens": tin, "output_tokens": tout,
            "latency_s": round(sum(c.get("latency_s", 0) for c in calls), 1)}


# ------------------------------------------------------------------ 任务

class Jobs:
    """子进程任务。kind → 依次执行的 qharness 命令。"""

    def __init__(self, paths: Paths):
        self.p = paths
        self.items: dict[str, dict] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, commands: list[list[str]], meta: dict | None = None) -> dict:
        self.p.jobs.mkdir(parents=True, exist_ok=True)
        jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        job = {"id": jid, "kind": kind, "status": "running", "started": time.time(), "ended": None,
               "returncode": None, "commands": [" ".join(c) for c in commands], **(meta or {})}
        with self._lock:
            if any(j["status"] == "running" and j["kind"] == kind for j in self.items.values()):
                raise HTTPException(409, f"已有一个「{kind}」任务在运行，等它结束再发起")
            self.items[jid] = job
        threading.Thread(target=self._run, args=(job, commands), daemon=True).start()
        return job

    def _run(self, job: dict, commands: list[list[str]]):
        log = self.p.jobs / f"{job['id']}.log"
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "MLFLOW_DISABLE_AGENT_HINT": "1"}
        code = 0
        with log.open("w", encoding="utf-8") as f:
            for cmd in commands:
                f.write(f"$ qharness {' '.join(cmd)}\n")
                f.flush()
                proc = subprocess.Popen([sys.executable, "-m", "qharness", *cmd], cwd=self.p.root, env=env,
                                        stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                job["pid"] = proc.pid
                code = proc.wait()
                if code != 0:
                    break
        job.update(status="done" if code == 0 else "failed", returncode=code, ended=time.time())

    def view(self, jid: str, tail: int = 200) -> dict:
        job = self.items.get(jid)
        if job is None:
            raise HTTPException(404, "没有这个任务")
        log = self.p.jobs / f"{jid}.log"
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
        return {**job, "log": lines[-tail:]}


# ------------------------------------------------------------------ 数据读取

def run_summary(d: Path, running: set[str]) -> dict:
    args = _json(d / "args.json", {}) or {}
    results = _json(d / "results.json")
    trace = _jsonl(d / "trace.jsonl")
    calls = _json(d / "llm_calls.json") or _jsonl(d / "llm_calls.jsonl")
    statuses = [r.get("status") for r in results] if isinstance(results, list) else []
    start = next((e for e in trace if e.get("node") == "run.start"), {})
    topic = next((e.get("output", {}) for e in trace if e.get("node") == "sop.topic"), {})
    state = "running" if d.name in running else ("finished" if (d / "summary.md").exists() else "interrupted")
    report = args.get("report") or ""
    return {"id": d.name, "state": state, "report": Path(report).name if report else None,
            "has_report": bool(report) or any(e.get("node") == "sop.chunk" for e in trace),  # 早期运行没存 args.json
            "hypothesis_file": Path(args["hypothesis"]).name if args.get("hypothesis") else None,
            "theme": topic.get("theme"), "model": start.get("model"),
            "started": trace[0]["ts"] if trace else None, "ended": trace[-1]["ts"] if trace else None,
            "n_candidates": len(statuses) or sum(e.get("node") == "candidate.start" for e in trace),
            "counts": {s: statuses.count(s) for s in set(statuses)}, "llm": _tokens(calls)}


def _report_text(d: Path, args: dict, root: Path) -> str | None:
    if (d / "report.md").exists():
        return (d / "report.md").read_text(encoding="utf-8")
    rp = args.get("report")
    if rp:
        p = Path(rp) if Path(rp).is_absolute() else root / rp
        if p.suffix.lower() in {".md", ".txt"} and p.exists():
            return p.read_text(encoding="utf-8")
    return None


class Names:
    """股票代码 → 名称（最新成分名单 + 历史成分名单）。"""

    def __init__(self, data: Path):
        self.data, self._map, self._mtime = data, {}, None

    def get(self) -> dict[str, str]:
        files = [self.data / "universe.csv", self.data / "universe_current.csv"]
        mt = tuple(f.stat().st_mtime for f in files if f.exists())
        if mt != self._mtime:
            m = {}
            for f in files:
                if f.exists():
                    df = pd.read_csv(f)
                    col = "name" if "name" in df else "code_name" if "code_name" in df else None
                    if col:
                        m.update({str(c).replace(".", "").upper(): n for c, n in zip(df["code"], df[col])})
            self._map, self._mtime = m, mt
        return self._map


def _last_prices(data: Path, insts: list[str]) -> dict[str, dict]:
    """持仓股票最近两天的不复权收盘价（账户按不复权价记账；来自 ref/cap/<代码>.csv 的 close_raw）。"""
    out = {}
    for inst in insts:
        f = data / "ref" / "cap" / f"{inst[:2].lower()}.{inst[2:]}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, usecols=["date", "close_raw"]).dropna().tail(2)
        if len(df):
            out[inst] = {"date": str(df["date"].iloc[-1]), "close": float(df["close_raw"].iloc[-1]),
                         "prev_close": float(df["close_raw"].iloc[-2]) if len(df) > 1 else None}
    return out


# ------------------------------------------------------------------ 应用

class FetchReports(BaseModel):
    days: int = 30
    download: int = 10


class StartRun(BaseModel):
    report: str  # samples/ 下的文件名，或上传后返回的 uploads/ 路径
    workers: int = 1
    no_strategy: bool = False


def create_app(root: Path = ROOT) -> FastAPI:
    P = Paths(root)
    jobs = Jobs(P)
    names = Names(P.data)
    app = FastAPI(title="qharness")

    def running_runs() -> set[str]:
        return {j["run_id"] for j in jobs.items.values() if j["status"] == "running" and j.get("run_id")}

    def run_dir(rid: str) -> Path:
        if not RUN_ID.match(rid) or not (P.runs / rid).is_dir():
            raise HTTPException(404, "没有这次运行")
        return P.runs / rid

    @app.get("/api/runs")
    def list_runs():
        dirs = sorted((d for d in P.runs.glob("*") if d.is_dir() and RUN_ID.match(d.name)), reverse=True)
        r = running_runs()
        return [run_summary(d, r) for d in dirs]

    @app.get("/api/runs/{rid}")
    def get_run(rid: str):
        d = run_dir(rid)
        args = _json(d / "args.json", {}) or {}
        calls = _json(d / "llm_calls.json") or _jsonl(d / "llm_calls.jsonl")
        by_role: dict[str, list] = {}
        for c in calls:
            by_role.setdefault(c.get("role", "?"), []).append(c)
        return {"summary": run_summary(d, running_runs()),
                "args": {k: (Path(v).name if isinstance(v, str) and k in {"report", "hypothesis", "csv"} else v)
                         for k, v in args.items() if k not in {"data", "experience", "registry"}},
                "report": _report_text(d, args, P.root), "trace": _jsonl(d / "trace.jsonl"),
                "results": _json(d / "results.json"), "strategy": _json(d / "strategy.json"),
                "llm_by_role": {k: _tokens(v) for k, v in by_role.items()}}

    @app.get("/api/factors")
    def factors():
        items = [_json(p) for p in sorted(P.factors.glob("*.json"))] if P.factors.exists() else []
        return {"registered": [x for x in items if x],
                "baseline": [{"name": "reversal_5d", "desc": "5 日反转：-(close / close.shift(5) - 1)"},
                             {"name": "momentum_20d", "desc": "20 日动量：close / close.shift(20) - 1"},
                             {"name": "low_vol_20d", "desc": "20 日低波动：-日收益率 20 日标准差"},
                             {"name": "illiquidity_20d", "desc": "20 日非流动性：|日收益率| / 成交额 的 20 日均值（Amihud）"},
                             {"name": "low_turnover_20d", "desc": "20 日低换手：-换手率 20 日均值"}]}

    @app.get("/api/models")
    def models():
        out = []
        for d in sorted((x for x in P.models.glob("*") if x.is_dir() and MODEL_ID.match(x.name)), reverse=True):
            s = _json(d / "summary.json")
            if s:
                out.append({"id": d.name, "kind": "walk_forward" if "folds" in s else "single", **s})
        opened = (P.models / "test_opened.txt")
        return {"items": out, "test_opened": len(opened.read_text().split()) if opened.exists() else 0}

    @app.get("/api/paper")
    def paper():
        acct = _json(P.portfolio / "account.json")
        trades = _csv(P.portfolio / "account.trades.csv")
        targets = sorted(P.portfolio.glob("????-??-??.csv"))
        nm = names.get()
        positions = []
        if acct:
            px = _last_prices(P.data, list(acct["positions"]))
            nav = acct["nav"][-1]["nav"] if acct.get("nav") else acct.get("initial")
            cost: dict[str, list[float]] = {}
            for t in trades:
                if t.get("status") == "filled" and t.get("side") == "buy":
                    c = cost.setdefault(t["instrument"], [0.0, 0.0])
                    c[0] += t["value"] + t["fee"]
                    c[1] += t["shares"]
            for inst, sh in sorted(acct["positions"].items()):
                p = px.get(inst, {})
                val = sh * p["close"] if p.get("close") else None
                c = cost.get(inst)
                positions.append({"instrument": inst, "name": nm.get(inst), "shares": sh, "close": p.get("close"),
                                  "date": p.get("date"),
                                  "day_change": (p["close"] / p["prev_close"] - 1) if p.get("prev_close") else None,
                                  "value": val, "weight": (val / nav) if val and nav else None,
                                  "avg_cost": (c[0] / c[1]) if c and c[1] else None,
                                  "pnl": (val - sh * c[0] / c[1]) if (val and c and c[1]) else None})
        target = _csv(targets[-1]) if targets else []
        for r in target:
            r["name"] = nm.get(r["instrument"])
        for t in trades:
            t["name"] = nm.get(t["instrument"])
        sim = P.runs / "paper_sim"
        return {"account": acct, "positions": positions, "trades": trades[::-1],
                "target": {"date": targets[-1].stem if targets else None, "rows": target},
                "replay": {"nav": _csv(sim / "nav.csv"), "stats": _json(sim / "stats.json")}}

    @app.get("/api/samples")
    def samples():
        files = [f for f in sorted(P.samples.glob("*")) if f.suffix.lower() in REPORT_EXT]
        ups = sorted(P.uploads.glob("*"), reverse=True) if P.uploads.exists() else []
        return {"samples": [{"value": f.name, "label": f.name, "size": f.stat().st_size} for f in files],
                "uploads": [{"value": f"uploads/{f.name}", "label": f.name.split("_", 2)[-1], "size": f.stat().st_size}
                            for f in ups if f.suffix.lower() in REPORT_EXT]}

    @app.post("/api/uploads")
    async def upload(file: UploadFile = File(...)):
        name = Path(file.filename or "report").name
        ext = Path(name).suffix.lower()
        if ext not in REPORT_EXT:
            raise HTTPException(400, f"只支持 {'、'.join(sorted(REPORT_EXT))} 格式的研报")
        data = await file.read()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(400, f"文件超过 {MAX_UPLOAD // 1024 // 1024} MB")
        P.uploads.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^\w.\-一-鿿]+", "_", name)
        dest = P.uploads / f"{time.strftime('%Y%m%d%H%M%S')}_{safe}"
        dest.write_bytes(data)
        return {"value": f"uploads/{dest.name}", "label": safe, "size": len(data)}

    def _report_path(value: str) -> Path:
        if value.startswith("uploads/"):
            p = P.uploads / Path(value).name
        elif value.startswith("auto/"):
            p = P.auto / Path(value).name
        else:
            p = P.samples / Path(value).name
        if not p.exists() or p.suffix.lower() not in REPORT_EXT:
            raise HTTPException(400, "找不到这份研报")
        return p

    @app.post("/api/jobs/run")
    def start_run(req: StartRun):
        p = _report_path(req.report)
        rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        # 子进程在项目根目录下运行，研报用相对路径（日志里不出现本机绝对路径）
        cmd = ["run", "--report", str(p.relative_to(P.root)), "--run-id", rid,
               "--workers", str(max(1, min(4, req.workers)))]
        if req.no_strategy:
            cmd.append("--no-strategy")
        return jobs.start("research", [cmd], {"run_id": rid})

    @app.get("/api/reports")
    def reports():
        idx = _json(P.auto / "index.json", {}) or {}
        runs_by_report: dict[str, list[dict]] = {}  # 从页面发起的研究不经过索引：按研报文件名对上运行记录
        r = running_runs()
        for d in sorted((x for x in P.runs.glob("*") if x.is_dir() and RUN_ID.match(x.name)), reverse=True):
            rep = (_json(d / "args.json", {}) or {}).get("report")
            if rep and Path(rep).parent.name == "auto":
                runs_by_report.setdefault(Path(rep).name, []).append(run_summary(d, r))
        items = []
        for x in sorted(idx.values(), key=lambda x: (x.get("date", ""), x.get("code", "")), reverse=True):
            items.append({**{k: v for k, v in x.items() if k != "screen"}, "screen": x.get("screen"),
                          "runs": runs_by_report.get(x.get("path") or "", [])})
        counts: dict[str, int] = {}
        for x in items:
            counts[x.get("status")] = counts.get(x.get("status"), 0) + 1
        pending = [f.name for f in sorted(P.inbox.glob("*")) if f.is_file() and f.suffix.lower() in REPORT_EXT
                   and not f.name.startswith(".")] if P.inbox.exists() else []
        return {"items": items, "counts": counts, "inbox": str(P.inbox), "inbox_files": len(pending)}

    @app.post("/api/jobs/reports")
    def start_fetch(req: FetchReports):
        return jobs.start("reports", [["reports", "fetch", "--days", str(max(1, min(365, req.days))),
                                       "--download", str(max(1, min(50, req.download)))]])

    @app.post("/api/jobs/scan")
    def start_scan():
        P.inbox.mkdir(parents=True, exist_ok=True)
        return jobs.start("reports", [["reports", "scan", "--inbox", str(P.inbox)]])

    @app.post("/api/jobs/daily")
    def start_daily():
        return jobs.start("daily", [["update"], ["daily", "--paper"]])

    @app.get("/api/jobs")
    def list_jobs():
        return sorted(jobs.items.values(), key=lambda j: j["started"], reverse=True)

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        return jobs.view(jid)

    if P.web.exists():
        app.mount("/assets", StaticFiles(directory=P.web / "assets"), name="assets")

        @app.get("/{path:path}")
        def spa(path: str):  # 前端路由都回到 index.html
            f = P.web / path
            if path and f.is_file() and P.web in f.resolve().parents:
                return FileResponse(f)
            return FileResponse(P.web / "index.html")

    return app


def serve(host: str = "127.0.0.1", port: int = 8765):
    import uvicorn

    print(f"打开 http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
