"""因子代码执行工具（实验层调用）。

约定：代码里定义 `def factor(df: pd.DataFrame) -> pd.Series`，df 只含行情字段，
返回与 df.index 对齐的一列分数。

检查顺序：静态检查 → 运行 → 输出检查 → 行序不变检验 → 未来函数截断检验。
任何一步失败都返回结构化错误（error_type + message），供修复循环和经验检索使用。
隔离：默认在独立子进程（spawn）里跑，带 CPU 时间和内存上限、整体超时后直接杀进程；
子进程崩溃或卡死不影响 Harness。设置 QH_SANDBOX_INPROC=1 可退回进程内执行（调试用）。
这仍不是对抗恶意代码的安全沙箱（没有文件系统 / 网络隔离），生产环境应再套容器。
"""
from __future__ import annotations

import ast
import math
import multiprocessing as mp
import os
import time
import signal
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import FIELDS, factor_columns

ALLOWED_IMPORTS = {"numpy", "pandas", "math"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals",
                   "vars", "getattr", "setattr", "delattr", "input", "breakpoint"}
SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                 for k in ["abs", "min", "max", "sum", "len", "range", "enumerate", "zip", "float",
                           "int", "bool", "list", "dict", "tuple", "set", "round", "isinstance",
                           "ValueError", "Exception", "print", "sorted", "map", "filter", "any", "all"]}


class ErrorType:
    SYNTAX = "SYNTAX"
    FORBIDDEN = "FORBIDDEN"
    LOOKAHEAD = "LOOKAHEAD"
    RUNTIME = "RUNTIME"
    TIMEOUT = "TIMEOUT"
    BAD_OUTPUT = "BAD_OUTPUT"
    ORDER_DEPENDENT = "ORDER_DEPENDENT"
    SANDBOX = "SANDBOX"  # 环境问题（子进程没能启动），不是因子代码的错，不进修复循环


@dataclass
class ExecResult:
    ok: bool
    values: pd.Series | None = None
    error_type: str | None = None
    message: str = ""
    checks: dict = field(default_factory=dict)

    def public(self) -> dict:
        """传给其他层的结构化结果（不含因子值本身）。"""
        return {"ok": self.ok, "error_type": self.error_type, "message": self.message[:1500],
                "checks": self.checks}


def _fail(t: str, msg: str, checks=None) -> ExecResult:
    return ExecResult(False, None, t, msg, checks or {})


def static_check(code: str, fields: list[str] | None = None) -> ExecResult | None:
    from . import qlib_bridge as qb

    if qb.is_expr(code):  # Qlib 表达式：算子白名单 + Ref 不能取负偏移
        err = qb.check_expr(qb.expr_of(code), fields or FIELDS)
        return _fail(ErrorType.LOOKAHEAD if err and "未来" in err else ErrorType.FORBIDDEN, err) if err else None
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return _fail(ErrorType.SYNTAX, f"SyntaxError line {e.lineno}: {e.msg}")
    has_factor = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "factor":
            has_factor = True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            bad = [m for m in mods if m.split(".")[0] not in ALLOWED_IMPORTS]
            if bad:
                return _fail(ErrorType.FORBIDDEN, f"不允许导入 {bad}，只能用 {sorted(ALLOWED_IMPORTS)}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return _fail(ErrorType.FORBIDDEN, f"不允许使用 {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return _fail(ErrorType.FORBIDDEN, f"不允许访问双下划线属性 {node.attr}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            low = node.value.lower()
            if low in {"label", "target"} or low.startswith("future"):
                return _fail(ErrorType.LOOKAHEAD, f"代码引用了字符串 {node.value!r}，因子只能用行情字段 {FIELDS}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "shift":
            args = list(node.args) + [k.value for k in node.keywords if k.arg in ("periods",)]
            for a in args[:1]:
                v = a
                if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub) and isinstance(v.operand, ast.Constant):
                    return _fail(ErrorType.LOOKAHEAD, f"shift(-{v.operand.value}) 使用了未来数据（第 {node.lineno} 行）")
                if isinstance(v, ast.Constant) and isinstance(v.value, (int, float)) and v.value < 0:
                    return _fail(ErrorType.LOOKAHEAD, f"shift({v.value}) 使用了未来数据（第 {node.lineno} 行）")
        if isinstance(node, ast.keyword) and node.arg == "center" and isinstance(node.value, ast.Constant) and node.value.value:
            return _fail(ErrorType.LOOKAHEAD, "rolling(center=True) 使用了未来数据")
    if not has_factor:
        return _fail(ErrorType.BAD_OUTPUT, "代码里没有定义 factor(df) 函数")
    return None


class _Timeout(Exception):
    pass


def _run(code: str, df: pd.DataFrame, timeout_s: int) -> pd.Series:
    from . import qlib_bridge as qb

    if qb.is_expr(code):
        return qb.compute(qb.expr_of(code), df)
    ns = {"__builtins__": SAFE_BUILTINS, "np": np, "pd": pd, "math": math}

    def _import(name, *a, **k):
        if name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError(name)
        return __import__(name, *a, **k)

    ns["__builtins__"] = {**SAFE_BUILTINS, "__import__": _import}

    def _handler(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)
    try:
        exec(compile(code, "<factor>", "exec"), ns)
        out = ns["factor"](df.copy())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    if isinstance(out, pd.DataFrame) and out.shape[1] == 1:
        out = out.iloc[:, 0]
    if not isinstance(out, pd.Series):
        raise TypeError(f"factor 必须返回 pd.Series，实际返回 {type(out).__name__}")
    return out


def _where(code: str, e: BaseException) -> str:
    """报错落在因子代码哪一行：只给因子代码里的帧（Harness 内部的帧对修复没有帮助）。"""
    lines = code.splitlines()
    frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "<factor>"]
    if not frames:
        return ""
    f = frames[-1]
    src = lines[f.lineno - 1].strip() if 0 < f.lineno <= len(lines) else ""
    return f"出错位置：因子代码第 {f.lineno} 行：{src}"


def _align(out: pd.Series, index: pd.MultiIndex) -> pd.Series:
    if out.index.nlevels != 2:
        raise ValueError(f"返回值索引层数为 {out.index.nlevels}，应为 (datetime, instrument) 两层；"
                         "groupby 后请用 group_keys=False 或 droplevel")
    out = out.copy()
    names = list(out.index.names)
    if set(names) != {"datetime", "instrument"}:
        out.index = out.index.set_names(["datetime", "instrument"])
    elif names != ["datetime", "instrument"]:
        out = out.reorder_levels(["datetime", "instrument"])
    out = out[~out.index.duplicated()]
    return out.reindex(index).astype(float)


def _child_main(conn, code, df, timeout_s, min_coverage, tol, mem_mb):
    conn.send("started")  # 子进程已正常启动：之后再退出就是因子代码的问题
    try:
        conn.send(_child(code, df, timeout_s, min_coverage, tol, mem_mb))
    except BaseException as e:  # noqa: BLE001
        conn.send(_fail(ErrorType.RUNTIME, f"{type(e).__name__}: {e}", {"static": "pass"}))
    finally:
        conn.close()


def _child(code, df, timeout_s, min_coverage, tol, mem_mb):
    try:
        import resource
        cpu = timeout_s * 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
        if mem_mb:
            lim = mem_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
            except (ValueError, OSError):  # macOS 不支持 RLIMIT_AS
                pass
    except ImportError:  # Windows
        pass
    return _execute(code, df, timeout_s, min_coverage, tol)


def with_direction(raw: str, direction) -> str:
    """方向由 Harness 统一施加，不交给模型：实验层按 calculation 原样实现，这里按假设的 direction 包一层。
    （模型自己处理方向时会漏负号，比如 direction=-1 的偏度因子直接写成 Skew(...)。）"""
    if int(direction or 1) >= 0:
        return raw
    from . import qlib_bridge as qb

    if qb.is_expr(raw):
        return f"{qb.PREFIX}-1*({qb.expr_of(raw)})"
    return (f"{raw.rstrip()}\n\n\n# --- Harness 按假设方向取负（direction=-1），不是实验层写的 ---\n"
            "_qh_raw_factor = factor\n\n\ndef factor(df):\n    return -_qh_raw_factor(df)\n")


def execute_factor(code: str, df: pd.DataFrame, timeout_s: int = 60,
                   min_coverage: float = 0.5, tol: float = 1e-6, isolate: bool | None = None,
                   mem_mb: int = 4096) -> ExecResult:
    """对外入口。静态检查在本进程做（快、无副作用），运行相关的检查放进子进程。"""
    df = df[factor_columns(df)].sort_index()
    bad = static_check(code, list(df.columns))
    if bad:
        return bad
    if isolate is None:
        isolate = os.environ.get("QH_SANDBOX_INPROC") != "1"
    if not isolate:
        return _execute(code, df, timeout_s, min_coverage, tol)
    # 单个 Process + 管道（不用 Pool：Pool 的工作进程启动就崩时会不停重建，表现成「超时」，把环境问题误报成代码太慢）
    ctx = mp.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child_main, args=(send, code, df, timeout_s, min_coverage, tol, mem_mb), daemon=True)
    limit = timeout_s * 5 + 30  # 本体 + 行序 + 两次截断，共 4 次运行
    started, t0 = False, time.time()
    try:
        p.start()
        send.close()
        while time.time() - t0 < limit:
            if recv.poll(0.5):
                try:
                    msg = recv.recv()
                except EOFError:
                    break
                if msg == "started":
                    started = True
                    continue
                return msg
            if not p.is_alive() and not recv.poll(0):
                break
        else:
            return _fail(ErrorType.TIMEOUT, f"子进程超过 {limit}s 未返回，已终止", {"static": "pass"})
        p.join(1)
        code_ = p.exitcode
        if not started:
            return _fail(ErrorType.SANDBOX, f"沙箱子进程没能启动（exitcode={code_}）。常见原因：调用方脚本缺少 "
                         "if __name__ == '__main__' 保护（spawn 会重新导入主模块）", {"static": "pass"})
        sig = -code_ if code_ and code_ < 0 else None
        why = f"被信号 {sig} 终止（CPU / 内存上限）" if sig else f"exitcode={code_}"
        return _fail(ErrorType.RUNTIME, f"子进程异常退出：{why}", {"static": "pass"})
    finally:
        if p.is_alive():
            p.kill()
        recv.close()


def _execute(code: str, df: pd.DataFrame, timeout_s: int, min_coverage: float, tol: float) -> ExecResult:
    bad = static_check(code, list(df.columns))
    if bad:
        return bad
    checks = {"static": "pass"}
    try:
        raw = _run(code, df, timeout_s)
        vals = _align(raw, df.index)
    except _Timeout:
        return _fail(ErrorType.TIMEOUT, f"运行超过 {timeout_s}s", checks)
    except Exception as e:  # noqa: BLE001  因子代码的任何错误都要回传给修复循环
        return _fail(ErrorType.RUNTIME, f"{type(e).__name__}: {e}\n{_where(code, e)}", checks)

    vals = vals.replace([np.inf, -np.inf], np.nan)
    coverage = float(vals.notna().mean())
    checks["coverage"] = round(coverage, 4)
    if coverage < min_coverage:
        return _fail(ErrorType.BAD_OUTPUT, f"有效值覆盖率 {coverage:.1%} 低于 {min_coverage:.0%}", checks)
    if vals.groupby(level="datetime").std().fillna(0).median() == 0:
        return _fail(ErrorType.BAD_OUTPUT, "因子在截面上是常数，无法区分股票", checks)

    # 行序不变：换成 (instrument, datetime) 排序重新计算，结果必须一致；否则多半是没按股票分组
    try:
        alt = _align(_run(code, df.sort_index(level=["instrument", "datetime"]), timeout_s), df.index)
        alt = alt.replace([np.inf, -np.inf], np.nan)
    except Exception as e:  # noqa: BLE001
        return _fail(ErrorType.ORDER_DEPENDENT, f"换行序后运行失败：{type(e).__name__}: {e}", checks)
    diff = _max_diff(vals, alt)
    checks["order_invariance_max_diff"] = diff
    if diff > tol:
        return _fail(ErrorType.ORDER_DEPENDENT,
                     f"换一种行排序后结果变化（最大差 {diff:.3g}）。数据按 (datetime, instrument) 排列，"
                     "rolling/shift/pct_change 等时序操作必须先 groupby(level='instrument')", checks)

    # 未来函数截断检验：分别截到很早（落在窗口预热期，抓 bfill 一类）和 70% 处重新计算，
    # 截断日之前的值必须不变
    dates = df.index.get_level_values("datetime").unique().sort_values()
    diffs = []
    for frac, early in ((0.025, True), (0.7, False)):
        cut = dates[max(5, int(len(dates) * frac))]
        try:
            part = _align(_run(code, df.loc[:cut], timeout_s), df.loc[:cut].index)
            part = part.replace([np.inf, -np.inf], np.nan)
        except Exception as e:  # noqa: BLE001
            if early:  # 数据太短跑不了很正常，不作为证据
                continue
            return _fail(ErrorType.LOOKAHEAD, f"截断数据后运行失败：{type(e).__name__}: {e}", checks)
        diff = _max_diff(vals.loc[:cut], part)
        diffs.append(diff)
        if diff > tol:
            checks["lookahead_max_diff"] = diff
            return _fail(ErrorType.LOOKAHEAD,
                         f"去掉 {cut.date()} 之后的数据后，之前的因子值变了（最大差 {diff:.3g}）：用到了未来信息，"
                         "常见原因是全样本标准化、bfill、居中窗口、负向 shift", checks)
    checks["lookahead_max_diff"] = max(diffs) if diffs else None
    checks["lookahead"] = "pass"
    return ExecResult(True, vals.rename("factor"), None, "", checks)


def _max_diff(a: pd.Series, b: pd.Series) -> float:
    a, b = a.align(b, join="left")
    nan_mismatch = (a.isna() != b.isna()).mean()
    if nan_mismatch > 0.001:
        return float("inf")
    d = (a - b).abs()
    scale = max(1.0, float(a.abs().quantile(0.99)) if a.notna().any() else 1.0)
    return float(d.max() / scale) if d.notna().any() else 0.0
