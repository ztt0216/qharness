"""因子衰减监控：每天收盘后，在最近一段已经有标签的日子上重新评估每个因子，失效的登记因子自动停用。

规则和研究层一样预先定好，不看结果再调：
- 窗口：最近 120 个交易日（标签要用到 T+6 的价格，最近 6 天还没有标签，不算）
- 停用：登记因子在窗口内中性化 RankIC ≤ 0，且 Newey-West t ≤ −1（方向反了，而且不是偶然）
- 警告（不停用）：最近 60 天 RankIC 转负；或窗口 RankIC 不到登记时测试段的一半
- 基础因子只报告，不停用（它们是基线，组合里一直在用）

停用写回登记表（active=false，记下日期、原因和当时的指标），每日持仓从下一次起就不再用它；
重新启用要人工确认：`qharness factors --activate 名称`。每次的结果存到 runs/monitor/<日期>.json。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .evaluate import Evaluator, baseline_library
from .registry import FactorRegistry


@dataclass(frozen=True)
class MonitorRules:
    window: int = 120
    short_window: int = 60
    deactivate_ic: float = 0.0
    deactivate_t: float = -1.0
    warn_fraction: float = 0.5


def labeled_dates(ev: Evaluator, n: int) -> pd.DatetimeIndex:
    """最近 n 个已经有标签的交易日。"""
    have = ev.label_w.notna().sum(axis=1)
    d = have[have >= 20].index
    return d[-n:]


def _metrics(ev: Evaluator, fw, dates) -> dict:
    m = ev._segment_metrics(fw, dates)
    return {k: m.get(k) for k in ("valid", "n_days", "rank_ic", "rank_ic_t_nw", "rank_ic_positive_ratio")}


def check(ev: Evaluator, factors: dict[str, pd.Series], registered: dict[str, dict],
          rules: MonitorRules = MonitorRules()) -> list[dict]:
    long_d, short_d = labeled_dates(ev, rules.window), labeled_dates(ev, rules.short_window)
    out = []
    for name, values in factors.items():
        fw = ev._prep(values)
        lm, sm = _metrics(ev, fw, long_d), _metrics(ev, fw, short_d)
        reg = registered.get(name)
        row = {"name": name, "kind": "registered" if reg else "baseline",
               "window": [str(long_d[0].date()), str(long_d[-1].date())] if len(long_d) else None,
               "long": lm, "short": sm, "status": "ok", "reasons": []}
        if not lm.get("valid"):
            row.update(status="unknown", reasons=["窗口内有效天数不足"])
            out.append(row)
            continue
        ic, t = lm["rank_ic"], lm["rank_ic_t_nw"]
        if reg and ic <= rules.deactivate_ic and t <= rules.deactivate_t:
            row["status"] = "deactivate"
            row["reasons"].append(f"最近 {rules.window} 天 RankIC {ic:+.4f}、t {t:+.2f}，方向反了且不是偶然")
        else:
            if sm.get("valid") and sm["rank_ic"] < 0:
                row["reasons"].append(f"最近 {rules.short_window} 天 RankIC 转负（{sm['rank_ic']:+.4f}）")
            test_ic = ((reg or {}).get("test_metrics") or {}).get("test", {}).get("rank_ic")
            if test_ic and ic < rules.warn_fraction * test_ic:
                row["reasons"].append(f"最近 {rules.window} 天 RankIC {ic:+.4f}，不到登记时测试段（{test_ic:+.4f}）的一半")
            if row["reasons"]:
                row["status"] = "warn"
        out.append(row)
    return out


def run_monitor(df: pd.DataFrame, risk, registry: FactorRegistry, out_dir: Path, apply: bool = True,
                rules: MonitorRules = MonitorRules(), log=print) -> dict:
    ev = Evaluator(df, risk=risk)
    reg_items = {x["name"]: x for x in registry.items()}
    factors = {**baseline_library(df, risk), **registry.compute(df)}
    rows = check(ev, factors, reg_items, rules)
    deactivated = []
    for r in rows:
        if r["status"] == "deactivate" and apply:
            registry.set_active(r["name"], False, reason="；".join(r["reasons"]), health=r)
            deactivated.append(r["name"])
    last = str(df.index.get_level_values("datetime").max().date())
    res = {"date": last, "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rules": asdict(rules),
           "factors": rows, "deactivated": deactivated, "applied": apply}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(res, ensure_ascii=False, indent=2, default=str)
    (out_dir / f"{last}.json").write_text(text, encoding="utf-8")
    (out_dir / "latest.json").write_text(text, encoding="utf-8")
    for r in rows:
        lm = r["long"]
        ic = f"{lm['rank_ic']:+.4f}（t {lm['rank_ic_t_nw']:+.2f}）" if lm.get("valid") else "—"
        log(f"  [{r['status']}] {r['name']}（{'登记' if r['kind'] == 'registered' else '基础'}）最近 {rules.window} 天 RankIC {ic}"
            + (f"：{'；'.join(r['reasons'])}" if r["reasons"] else ""))
    if deactivated:
        log(f"  已停用：{'、'.join(deactivated)}（重新启用：qharness factors --activate 名称）")
    return res
