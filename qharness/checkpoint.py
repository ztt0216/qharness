"""LangGraph 检查点：每个节点执行完同步写入 SQLite，进程中断后用同一个 thread_id 从断点继续。

一次运行一个库（runs/<run_id>/checkpoints.sqlite）；研报 SOP 一个 thread，每个候选因子一个 thread。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def open_checkpointer(path: str | Path):
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def run_graph(app, init: dict, thread_id: str | None, recursion_limit: int, tracer=None, label: str = "") -> dict:
    """没有检查点：直接跑。有检查点：已跑完的直接取最终状态；跑到一半的从断点继续；没跑过的从头跑。"""
    if thread_id is None or app.checkpointer is None:
        return app.invoke(init, {"recursion_limit": recursion_limit})
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    st = app.get_state(cfg)
    if st.values and not st.next:
        if tracer:
            tracer.log("checkpoint.restored", thread=thread_id, label=label)
        return st.values
    if st.values:
        if tracer:
            tracer.log("checkpoint.resume", thread=thread_id, label=label, next=list(st.next))
        return app.invoke(None, cfg, durability="sync")
    return app.invoke(init, cfg, durability="sync")


def thread_status(app, thread_id: str) -> str:
    """none | running | done"""
    st = app.get_state({"configurable": {"thread_id": thread_id}})
    return "none" if not st.values else ("running" if st.next else "done")
