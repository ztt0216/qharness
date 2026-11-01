"""MCP 服务（`qharness mcp`，stdio，只在本机用）：把沙箱、经验检索和验证段回测开放给外部 MCP 客户端。

内部的 LangGraph 流程继续直接调用函数，这里只是对外的入口，研究纪律不放宽：
- 没有打开测试段和登记因子的工具；因子代码只拿到验证段结束之前的数据，报错信息也带不出测试段
- 分层：coder 看不到收益指标，经验只有成功实现和修复案例（淘汰原因和纠正诊断里可能有指标）；
  回测工具只给 research，每个因子最多回测 MAX_BACKTESTS 次，和研究层内部的上限一致，
  按因子值认身份并落盘：改写法、翻方向、重启服务都不能多跑
- 每次调用记进 runs/mcp/<启动编号>/trace.jsonl；这里的结果不进入判定，也不算入多重检验的试验次数
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from . import qlib_bridge as qb
from .agents import MAX_BACKTESTS, RUN_FACTOR_TOOL, RUN_QLIB_TOOL, tool_view
from .data import make_split
from .evaluate import Evaluator
from .experience import ExperienceStore
from .sandbox import execute_factor, with_direction
from .strategy import validation_backtest
from .trace import Tracer

CODER_KINDS = ("success", "fix")
RESEARCH_KINDS = ("success", "fix", "rejected", "corrected")


def research_data(df: pd.DataFrame) -> pd.DataFrame:
    """验证段结束之前的数据：对外的工具都只在这上面算。"""
    return df.loc[df.index.get_level_values("datetime") <= make_split(df.sort_index()).valid[1]]


class Tools:
    def __init__(self, df: pd.DataFrame, risk: pd.DataFrame | None, role: str, experience: str | Path, runs: str | Path,
                 extra_library: dict[str, pd.Series] | None = None):
        self.df, self.risk, self.extra, self.role = df.sort_index(), risk, extra_library, role
        self.visible = research_data(self.df)
        self.experience = Path(experience)
        self.root = Path(runs) / "mcp"
        self.tr = Tracer(self.root)
        self._ev: Evaluator | None = None
        self._lock = threading.Lock()

    def _execute(self, code: str, tool: str) -> dict:
        res = execute_factor(code, self.visible)
        self.tr.log(f"mcp.{tool}", role=self.role, ok=res.ok, error_type=res.error_type)
        return tool_view(res)

    def run_factor(self, code: str) -> dict:
        return self._execute(code, "run_factor")

    def run_qlib_factor(self, expression: str) -> dict:
        return self._execute(qb.PREFIX + expression, "run_qlib_factor")

    def search_experience(self, query: str, kind: str | None = None, error_type: str | None = None, k: int = 3) -> dict:
        kinds = RESEARCH_KINDS if self.role == "research" else CODER_KINDS
        if kind is not None and kind not in kinds:
            raise ToolError(f"{self.role} 只能检索这些经验：{'、'.join(kinds)}")
        hits = []
        for score, e in ExperienceStore(self.experience).search(query, error_type=error_type, k=len(kinds) * k):
            if e.kind in kinds and (kind is None or e.kind == kind):
                hits.append({"score": score, "kind": e.kind, "task": e.task, "error_type": e.error_type, "content": e.content})
        self.tr.log("mcp.search_experience", role=self.role, hits=len(hits[:k]))
        return {"hits": hits[:k]}

    def _evaluator(self) -> Evaluator:
        with self._lock:  # 第一次回测才构建：中性化基线要花时间，只用沙箱的客户端不必等
            if self._ev is None:
                self._ev = Evaluator(self.df, risk=self.risk, extra_library=self.extra)
            return self._ev

    def _backtests_used(self, keys: list[str]) -> int:
        used = 0
        for f in self.root.glob("*/trace.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                ev = json.loads(line)
                used += ev.get("node") == "mcp.run_validation_backtest" and bool(set(ev.get("factor_keys", [])) & set(keys))
        return used

    def run_validation_backtest(self, code: str, topk: int, rebalance: int, direction: int = 1) -> dict:
        res = execute_factor(with_direction(code, direction), self.visible)
        if not res.ok:
            return tool_view(res)
        ev = self._evaluator()
        keys = ev._value_keys(res.values, ev.split.valid)
        with self._lock:  # 查次数和记次数要一起做；失败的回测也算一次
            if self._backtests_used(keys) >= MAX_BACKTESTS:
                raise ToolError(f"这个因子（或它的单调变换、取反）已经回测过 {MAX_BACKTESTS} 次，不能继续调参")
            self.tr.log("mcp.run_validation_backtest", role=self.role, factor_keys=keys, topk=topk, rebalance=rebalance)
        return {"ok": True, "checks": res.checks, **validation_backtest(ev, res.values, topk, rebalance)}


def build_server(tools: Tools):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("qharness", instructions="A 股因子研究工具。只开放验证段结束之前的数据；不能打开测试段，也不能登记因子。")

    @server.tool(description=RUN_FACTOR_TOOL["description"])
    def run_factor(code: Annotated[str, Field(description="完整 Python 代码，定义 def factor(df) -> pd.Series")]) -> dict[str, Any]:
        return tools.run_factor(code)

    if qb.available():
        @server.tool(description=RUN_QLIB_TOOL["description"])
        def run_qlib_factor(expression: Annotated[str, Field(description="Qlib 表达式，如 -($close/Ref($close,5)-1)")]) -> dict[str, Any]:
            return tools.run_qlib_factor(expression)

    @server.tool(description="检索经验库（字面相似度）。coder 能查成功实现和修复案例，research 还能查被淘汰的方向和纠正记录。")
    def search_experience(query: Annotated[str, Field(description="假设描述或报错信息")],
                          kind: Annotated[Literal["success", "fix", "rejected", "corrected"] | None, Field(description="经验类型")] = None,
                          error_type: Annotated[str | None, Field(description="只查这类错误的修复案例")] = None,
                          k: Annotated[int, Field(ge=1, le=10, description="返回条数")] = 3) -> dict[str, Any]:
        return tools.search_experience(query, kind, error_type, k)

    if tools.role == "research" and qb.available():
        @server.tool(description=f"用 Qlib 回测对比基线和基线 + 候选因子，只在验证段，每个因子最多 {MAX_BACKTESTS} 次。"
                                 "返回两者的年化超额、信息比率、超额最大回撤和换手。结果不进入判定。")
        def run_validation_backtest(code: Annotated[str, Field(description="因子代码，或 QLIB: 开头的 Qlib 表达式")],
                                    topk: Annotated[int, Field(ge=10, le=50, description="持仓只数")],
                                    rebalance: Annotated[int, Field(ge=1, le=20, description="调仓间隔（交易日）")],
                                    direction: Annotated[Literal[1, -1], Field(description="因子方向")] = 1) -> dict[str, Any]:
            return tools.run_validation_backtest(code, topk, rebalance, direction)

    return server


def serve(tools: Tools) -> None:
    build_server(tools).run("stdio")
