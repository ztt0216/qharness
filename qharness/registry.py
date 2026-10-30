"""因子登记表：研究流程产出的、可以进实盘组合的因子。

登记条件（预注册，写死在这里）：研究层判定通过，且测试段（只开放一次）RankIC 与验证段同号、t 值不为负。
每个因子一个 JSON（代码 + 假设 + 各段指标 + 来源 run_id），每日持仓直接读这里，不依赖某一次运行目录。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd

from .sandbox import execute_factor


def admit(metrics: dict, test_metrics: dict | None) -> tuple[bool, str]:
    t = (test_metrics or {}).get("test") or {}
    if not t.get("valid"):
        return False, "测试段数据不足"
    if t["rank_ic"] <= 0 or (t.get("rank_ic_t_nw") or 0) < 0:
        return False, f"测试段不成立（RankIC {t['rank_ic']:+.4f}，t {t.get('rank_ic_t_nw')}）"
    return True, "通过"


class FactorRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def slug(name: str) -> str:
        return re.sub(r"[^\w一-鿿]+", "_", name).strip("_")[:60] or "factor"

    def add(self, name: str, code: str, hypothesis: dict, metrics: dict, test_metrics: dict, run_id: str) -> Path:
        path = self.root / f"{self.slug(name)}.json"
        path.write_text(json.dumps({"name": name, "code": code, "hypothesis": hypothesis, "metrics": metrics,
                                    "test_metrics": test_metrics, "run_id": run_id,
                                    "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"), "active": True},
                                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    def items(self, active_only: bool = True) -> list[dict]:
        out = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.root.glob("*.json"))]
        return [x for x in out if x.get("active", True) or not active_only]

    def compute(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """在最新数据上重新跑一遍登记因子（同样过沙箱检查）；跑不通的跳过并报出来。"""
        out = {}
        for x in self.items():
            r = execute_factor(x["code"], df)
            if r.ok:
                out[x["name"]] = r.values
            else:
                print(f"登记因子 {x['name']} 在当前数据上运行失败：{r.error_type} {r.message[:200]}")
        return out
