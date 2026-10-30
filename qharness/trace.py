"""run_id 级 trace：每个节点做了什么、输入输出摘要、报错、耗时。经验从这里沉淀。"""
from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class Tracer:
    def __init__(self, root: str | Path, run_id: str | None = None):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.dir = Path(root) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "trace.jsonl"
        self.events: list[dict] = []
        self._offset = sum(1 for _ in self.file.open(encoding="utf-8")) if self.file.exists() else 0  # 续跑时接着编号
        self._lock = threading.Lock()
        self._local = threading.local()

    @contextmanager
    def bind(self, cand: str):
        """候选并行时，本线程记的事件都带上 cand 字段，trace 里能按候选拆开看。"""
        self._local.cand = cand
        try:
            yield
        finally:
            self._local.cand = None

    def log(self, node: str, **data):
        cand = getattr(self._local, "cand", None)
        with self._lock:
            ev = {"run_id": self.run_id, "step": self._offset + len(self.events), "ts": round(time.time(), 3),
                  "node": node, **({"cand": cand} if cand else {}), **data}
            self.events.append(ev)
            with self.file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        return ev

    def write(self, name: str, text: str) -> Path:
        p = self.dir / name
        p.write_text(text, encoding="utf-8")
        return p
