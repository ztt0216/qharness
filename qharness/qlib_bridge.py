"""Qlib 接入：行情转成 Qlib 的二进制格式；因子可以写成 Qlib 表达式，由 Qlib 计算；策略用 Qlib 的回测组件复核。

- 数据：`dump()` 把面板写成 provider_uri 目录（calendars/day.txt、instruments/*.txt、features/<股票>/<字段>.day.bin）。
  价格是后复权价，`$factor` = 后复权价 / 不复权价（Qlib 的约定，回测按不复权价整手成交）；
  `$vwap` = 成交额 / 成交量 × 复权因子；另存 `$limit_up` / `$limit_down`（当天涨跌停或停牌时为 1，与 strategy.backtest 同口径），给回测的成交限制用
- 因子：实验层提交 `QLIB:<表达式>`（如 `QLIB:-($close/Ref($close,5)-1)`），沙箱的静态检查、覆盖率、
  截断重算照常执行；表达式只允许白名单里的算子和行情字段，`Ref` 的偏移不能为负（未来数据）
- 回测：`qlib_backtest()` 用 Qlib 的 TopkDropoutStrategy + SimulatorExecutor，次日开盘价成交、整手、
  双边费用、涨跌停和停牌不成交，基准是股票池等权，用来和 strategy.backtest 互相核对

Qlib 的 `D.features` 不是线程安全的，进程内的计算串行（加锁）；沙箱子进程各自初始化。
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd

PREFIX = "QLIB:"
ENV_URI = "QH_QLIB_URI"

# 允许的 Qlib 算子（都只用当天及以前的数据；Ref 的偏移另查）
OPS = {"Ref", "Mean", "Std", "Var", "Skew", "Kurt", "Max", "Min", "IdxMax", "IdxMin", "Med", "Mad", "Rank", "Count",
       "Delta", "Slope", "Rsquare", "Resi", "WMA", "EMA", "Sum", "Corr", "Cov", "Abs", "Sign", "Log", "Power", "If",
       "Gt", "Lt", "Ge", "Le", "Eq", "Ne", "And", "Or", "Not", "Greater", "Less", "Quantile"}
# 窗口参数的位置。Qlib 里 N=0 是全样本累积，Min(x,0) 这样的写法能跑通但结果不对（模型常把它当成逐元素取小）
WINDOW_ARG = {**dict.fromkeys(("Mean", "Std", "Var", "Skew", "Kurt", "Max", "Min", "IdxMax", "IdxMin", "Med", "Mad", "Rank", "Count",
                               "Delta", "Slope", "Rsquare", "Resi", "WMA", "EMA", "Sum", "Quantile"), 1),
              "Corr": 2, "Cov": 2}
TOKEN = re.compile(r"\s*(?:(\$[a-z_]+)|([A-Za-z_][A-Za-z0-9_]*)|(\d+\.?\d*(?:e-?\d+)?)|(==|!=|>=|<=|[-+*/(),<>&|~]))")
FIELD_LIMIT_UP, FIELD_LIMIT_DOWN = "limit_up", "limit_down"

_lock = threading.Lock()
_inited: str | None = None


def is_expr(code: str) -> bool:
    return code.lstrip().startswith(PREFIX)


def expr_of(code: str) -> str:
    return code.lstrip()[len(PREFIX):].strip()


# ------------------------------------------------------------------ 数据

def dump(df: pd.DataFrame, raw_close: pd.DataFrame | None, risk: pd.DataFrame | None, out: str | Path,
         market: str = "csi300") -> Path:
    """面板（datetime, instrument）→ Qlib provider 目录。重复调用会整体覆盖。"""
    out = Path(out)
    df = df.sort_index()
    cal = df.index.get_level_values("datetime").unique().sort_values()
    (out / "calendars").mkdir(parents=True, exist_ok=True)
    (out / "instruments").mkdir(parents=True, exist_ok=True)
    (out / "calendars" / "day.txt").write_text("\n".join(d.strftime("%Y-%m-%d") for d in cal) + "\n")

    wide = {c: df[c].unstack("instrument").reindex(cal) for c in df.columns}
    close = wide["close"]
    if raw_close is not None:
        raw = raw_close.reindex(index=cal, columns=close.columns)
        wide["factor"] = (close / raw).where(raw > 0)
    if "amount" in wide and "volume" in wide:
        # 成交均价（Alpha158 的 VWAP0 要用）：成交额 / 成交量是不复权价（实测与不复权收盘价之比中位数 1.000），乘复权因子
        vwap = (wide["amount"] / wide["volume"]).where(wide["volume"] > 0)
        wide["vwap"] = vwap * wide["factor"] if "factor" in wide else vwap
    if risk is not None:
        halted = close.isna()
        for name in (FIELD_LIMIT_UP, FIELD_LIMIT_DOWN):
            flag = risk[name].astype(float).unstack("instrument").reindex(index=cal, columns=close.columns).fillna(0)
            wide[name] = (flag.astype(bool) | halted).astype(float)

    rows = []
    for inst in close.columns:
        valid = close[inst].notna().to_numpy()
        if not valid.any():
            continue
        i0, i1 = int(np.argmax(valid)), len(cal) - 1 - int(np.argmax(valid[::-1]))
        d = out / "features" / inst.lower()
        d.mkdir(parents=True, exist_ok=True)
        for f, w in wide.items():
            v = w[inst].to_numpy(dtype=float)[i0:i1 + 1]
            np.r_[float(i0), v].astype("<f4").tofile(d / f"{f.lower()}.day.bin")
        rows.append(f"{inst}\t{cal[i0]:%Y-%m-%d}\t{cal[i1]:%Y-%m-%d}")
    text = "\n".join(rows) + "\n"
    for m in {"all", market}:
        (out / "instruments" / f"{m}.txt").write_text(text)
    return out


def init(uri: str | Path | None = None):
    """同一进程只初始化一次；uri 默认取环境变量 QH_QLIB_URI（沙箱子进程靠它继承）。"""
    global _inited
    uri = str(uri or os.environ.get(ENV_URI) or "")
    if not uri:
        raise RuntimeError(f"没有 Qlib 数据目录：先运行 qharness qlib-dump，或设置 {ENV_URI}")
    if _inited == uri:
        return
    import logging

    import qlib
    from qlib.constant import REG_CN

    qlib.init(provider_uri=uri, region=REG_CN, expression_cache=None, dataset_cache=None, kernels=1,
              logging_level=logging.WARNING)
    os.environ[ENV_URI] = uri
    _inited = uri


def available() -> bool:
    uri = os.environ.get(ENV_URI)
    return bool(uri) and (Path(uri) / "calendars" / "day.txt").exists()


# ------------------------------------------------------------------ 表达式

def check_expr(expr: str, fields: list[str]) -> str | None:
    """返回错误说明；合法返回 None。只做白名单和未来数据检查，具体语法交给 Qlib 解析。"""
    if not expr:
        return "表达式为空"
    pos, depth = 0, 0
    while pos < len(expr):
        m = TOKEN.match(expr, pos)
        if not m or m.end() == pos:
            return f"无法识别的字符：{expr[pos:pos + 10]!r}"
        field, name, _, op = m.groups()
        if field and field[1:] not in fields:
            return f"字段 {field} 不可用，只能用 {', '.join('$' + f for f in fields)}"
        if name and name not in OPS:
            return f"算子 {name} 不在允许列表里（允许：{', '.join(sorted(OPS))}）"
        if op == "(":
            depth += 1
        elif op == ")":
            depth -= 1
            if depth < 0:
                return "括号不匹配"
        pos = m.end()
    if depth:
        return "括号不匹配"
    for m in re.finditer(r"\bRef\s*\(", expr):
        args = _top_args(expr, m.end())
        if len(args) == 2:
            try:
                if float(args[1]) < 0:
                    return f"Ref 的偏移 {args[1].strip()} 为负：用到了未来数据"
            except ValueError:
                return "Ref 的偏移必须是常数"
    for m in re.finditer(rf"\b({'|'.join(WINDOW_ARG)})\s*\(", expr):
        name, args = m.group(1), _top_args(expr, m.end())
        if len(args) <= WINDOW_ARG[name]:
            continue  # 参数个数不对，交给 Qlib 报错
        window = args[WINDOW_ARG[name]].strip()
        try:
            bad = float(window) <= 0 if name == "EMA" else float(window) < 1  # EMA 的 N 可以是 0~1 的平滑系数
        except ValueError:
            bad = True
        if bad:
            hint = "；逐元素取两者中较小 / 较大的值用 Less(x,y) / Greater(x,y)，返回的是值不是布尔" if name in ("Min", "Max") else ""
            return f"{name} 的窗口 {window} 不合法：必须是正数常数{hint}；要用全部历史就改用 run_factor"
    return None


def _top_args(s: str, start: int) -> list[str]:
    depth, args, cur = 0, [], ""
    for ch in s[start:]:
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                break
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur)
            cur = ""
        else:
            cur += ch
    return args + [cur]


def normalize(expr: str) -> str:
    """Qlib 的表达式对象不支持一元负号（`-($close)` 会报错）：改写成 `-1*`。"""
    toks = [m.group(1) or m.group(2) or m.group(3) or m.group(4) for m in TOKEN.finditer(expr)]
    out, prev = [], None
    for i, tok in enumerate(toks):
        unary = tok == "-" and (prev is None or prev in {"(", ",", "+", "-", "*", "/", "<", ">", "==", "!=", ">=", "<=", "&", "|"})
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        out.append(("-" if nxt[:1].isdigit() else "-1*") if unary else tok)  # 负的常数（-1）保持原样
        prev = tok
    return "".join(out)


def compute(expr: str, df: pd.DataFrame) -> pd.Series:
    """在 df 覆盖的股票和日期上用 Qlib 计算表达式，返回 (datetime, instrument) 索引的一列。"""
    from qlib.data import D

    init()
    dates = df.index.get_level_values("datetime")
    insts = sorted(df.index.get_level_values("instrument").unique())
    with _lock:
        out = D.features(insts, [normalize(expr), "$close"], start_time=dates.min(), end_time=dates.max(), freq="day")
    out = out.swaplevel().sort_index()
    out.index = out.index.set_names(["datetime", "instrument"])
    # Qlib 读的是自己的数据目录：和当前行情对不上（比如 update 之后没重新 dump）就报错，不能静默算出另一份数
    close = df["close"].reindex(out.index)
    m = close.notna() & out.iloc[:, 1].notna()
    if m.any() and float(((out.iloc[:, 1][m] - close[m]).abs() / close[m]).max()) > 1e-4:
        raise RuntimeError("Qlib 数据目录和当前行情不一致：先运行 qharness qlib-dump 重新生成")
    return out.iloc[:, 0]


# ------------------------------------------------------------------ 回测

def qlib_backtest(score: pd.DataFrame, start, end, topk: int = 20, n_drop: int | None = None, rebalance: int = 5,
                  account: float = 1e7, open_cost: float = 0.00125, close_cost: float = 0.00175,
                  min_cost: float = 5.0, deal_price: str = "open",
                  risk_degree: float = 0.99) -> dict:
    """score：宽表（行是交易日，列是股票），d 日的分数 d+1 日开盘成交（Qlib 的信号默认按下一根 bar 执行）。
    每 rebalance 天调仓一次：调仓日的分数前向填充到下一个调仓日，期间目标不变、不交易。
    （不能把非调仓日的信号置空：Qlib 按「上一根 bar 内最后一个信号」取数，置空后调仓日的下一天会再换一次仓。）"""
    from qlib.backtest import backtest as qbt
    from qlib.contrib.evaluate import risk_analysis
    from qlib.contrib.strategy import TopkDropoutStrategy

    init()
    sig = score.loc[start:end]
    sig = sig.iloc[::rebalance].reindex(sig.index, method="ffill")
    signal = sig.stack().rename("score")
    signal.index = signal.index.set_names(["datetime", "instrument"])
    bench = _bench(score.columns.tolist(), start, end)
    strategy = TopkDropoutStrategy(signal=signal.to_frame(), topk=topk, n_drop=n_drop or topk, only_tradable=True,
                                   risk_degree=risk_degree)  # Qlib 默认只用 95% 资金；这里接近满仓，和 strategy.backtest 可比
    executor = {"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}}
    with _lock:
        pm, _ = qbt(start_time=start, end_time=end, strategy=strategy, executor=executor, benchmark=bench,
                    account=account,
                    exchange_kwargs={"freq": "day", "deal_price": deal_price, "open_cost": open_cost,
                                     "close_cost": close_cost, "min_cost": min_cost, "trade_unit": 100,
                                     "limit_threshold": (f"${FIELD_LIMIT_UP} > 0.5", f"${FIELD_LIMIT_DOWN} > 0.5")})
    report = pm["1day"][0]
    excess = (report["return"] - report["bench"] - report["cost"]).dropna()
    ra = risk_analysis(excess, freq="day", mode="sum")["risk"]
    enav = (1 + excess).cumprod()
    return {"annual_excess": round(float(ra["annualized_return"]), 4),
            "information_ratio": round(float(ra["information_ratio"]), 3),
            "max_drawdown_excess": round(float((enav / enav.cummax() - 1).min()), 4),
            "avg_daily_turnover": round(float(report["turnover"].mean()), 4),
            "n_days": int(len(excess)), "engine": "qlib.backtest(TopkDropoutStrategy, SimulatorExecutor)"}


def _bench(insts: list[str], start, end) -> pd.Series:
    """股票池等权日收益（与 strategy.backtest 的基准一致；停牌日不计）。"""
    from qlib.data import D

    with _lock:
        r = D.features(insts, ["$close/Ref($close,1)-1"], start_time=start, end_time=end, freq="day")
    return r.iloc[:, 0].groupby(level="datetime").mean().fillna(0.0)
