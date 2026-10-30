"""经验库（参考 RD-Agent Co-STEER 的知识库）。

四类经验：
- success：成功实现（假设 + 代码）
- fix：修复案例（错误类型 + 报错 + 修前/修后代码）
- rejected：被研究层淘汰的方向（假设 + 原因），避免换个说法再试
- corrected：被研究层纠正后通过的假设（原版 → 纠正后的方向/窗口/算法），下次同一假设直接从纠正后开始

检索：文本相似度（英文词 + 中文二元组的余弦）→ 按类型/错误类型过滤 → 阈值 → top-k。
写入时与同类经验相似度过高则合并（只保留最新），防止越积越乱。
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path


def tokenize(text: str) -> Counter:
    text = text.lower()
    toks = re.findall(r"[a-z_][a-z0-9_]+|\d+", text)
    for seg in re.findall(r"[一-鿿]+", text):
        toks += [seg[i:i + 2] for i in range(max(1, len(seg) - 1))]
    return Counter(toks)


def similarity(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    dot = sum(ta[k] * tb[k] for k in ta.keys() & tb.keys())
    return dot / (math.sqrt(sum(v * v for v in ta.values())) * math.sqrt(sum(v * v for v in tb.values())))


@dataclass
class Experience:
    kind: str  # success | fix | rejected | corrected
    task: str  # 假设描述，检索主键
    content: dict  # success: {code}; fix: {error_type, error, before, after}; rejected: {reason}; corrected: {from, to, note}
    error_type: str | None = None
    tags: list[str] = field(default_factory=list)
    run_id: str = ""
    ts: float = field(default_factory=time.time)

    def key_text(self) -> str:
        extra = self.content.get("error", "") if self.kind == "fix" else ""
        return f"{self.task} {extra}"


class ExperienceStore:
    def __init__(self, path: str | Path, merge_threshold: float = 0.92):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.merge_threshold = merge_threshold
        self.items: list[Experience] = []
        self._lock = threading.RLock()  # 候选并行时多个线程同时写
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.items.append(Experience(**json.loads(line)))

    def _flush(self):
        self.path.write_text("".join(json.dumps(asdict(e), ensure_ascii=False) + "\n" for e in self.items),
                             encoding="utf-8")

    def add(self, exp: Experience) -> str:
        """返回 added / merged。"""
        with self._lock:
            for i, old in enumerate(self.items):
                if old.kind == exp.kind and old.error_type == exp.error_type \
                        and similarity(old.key_text(), exp.key_text()) >= self.merge_threshold:
                    self.items[i] = exp
                    self._flush()
                    return "merged"
            self.items.append(exp)
            self._flush()
            return "added"

    def search(self, query: str, kind: str | None = None, error_type: str | None = None,
               k: int = 3, threshold: float = 0.15) -> list[tuple[float, Experience]]:
        hits = []
        with self._lock:
            items = list(self.items)
        for e in items:
            if kind and e.kind != kind:
                continue
            if error_type and e.error_type != error_type:
                continue
            s = similarity(query, e.key_text())
            if error_type and e.error_type == error_type:
                s += 0.2  # 同类错误的修复经验即使任务不同也值得参考
            if s >= threshold:
                hits.append((round(s, 3), e))
        hits.sort(key=lambda x: -x[0])
        return hits[:k]
