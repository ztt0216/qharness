"""检索：语义向量（bge-small-zh-v1.5，本地 ONNX）+ 字面相似度（英文词 + 中文二元组余弦），混合排序。

- 语义：同一个意思换了说法也能找到（「量价共振」↔「价格和成交量同向」）
- 字面：数字和专有名词要靠它（「5 日反转」和「20 日反转」在语义向量里几乎一样）
- 混合排序用 RRF（倒数排名融合），两种分数不在一个尺度上，按名次合并不需要换算
- 模型没装或 QH_EMBED=off 时退回纯字面，流程照常跑
- 默认用字面（QH_RETRIEVAL=lexical）：在本项目的运行记录上，字面检索的召回比语义和混合都好或持平（retrieval_eval.py）；
  QH_RETRIEVAL=hybrid 打开混合检索。经验检索、论据核对固定用字面（要的就是字面一致，数字不能混）

模型放在 data/models/（不纳入版本管理），首次使用时由 `python -m qharness.retrieval --download` 下载。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from .experience import similarity as lexical

ROOT = Path(__file__).resolve().parent.parent
MODEL = "BAAI/bge-small-zh-v1.5"
MODEL_DIR = ROOT / "data" / "models"
RRF_K = 60

_lock = threading.RLock()  # embed() 持锁时会调 _load()，必须可重入
_model = None
_failed: str | None = None


MODES = ("lexical", "hybrid", "semantic")
# 观点归并初筛的语义通道阈值。同义改写实测 0.67~0.77，无关观点也常到 0.63~0.66，区分度有限：
# 一篇 47 条观点的研报上，0.7 会额外多 119 次模型确认（字面初筛才 47 次），0.8 只多 8 次但几乎召回不到改写，取 0.75（多 34 次）
SEMANTIC_SAME = 0.75


def mode() -> str:
    """研报检索方式（QH_RETRIEVAL）。默认字面：在本项目的运行记录上实测字面最好（见 README「检索」）。"""
    m = os.environ.get("QH_RETRIEVAL", "lexical").lower()
    if m not in MODES:
        raise ValueError(f"QH_RETRIEVAL 只能是 {'、'.join(MODES)}")
    return m if m == "lexical" or enabled() else "lexical"


def maybe_same(a: str, b: str, lexical_min: float = 0.3) -> bool:
    """观点归并初筛（之后由模型确认）：字面相似度过线；混合 / 语义模式下语义相似度过线也算，多召回换了说法的同一观点。"""
    if lexical(a, b) >= lexical_min:
        return True
    return mode() != "lexical" and semantic(a, b) >= SEMANTIC_SAME


def enabled() -> bool:
    if os.environ.get("QH_EMBED", "").lower() in {"off", "0", "false"}:
        return False
    return _load() is not None


def _load():
    global _model, _failed
    if _model is not None or _failed is not None:
        return _model
    with _lock:
        if _model is None and _failed is None:
            try:
                from fastembed import TextEmbedding

                if not any(MODEL_DIR.glob("*bge-small-zh*")):
                    raise FileNotFoundError(f"没有向量模型：先运行 python -m qharness.retrieval --download")
                _model = TextEmbedding(MODEL, cache_dir=str(MODEL_DIR), local_files_only=True)
            except Exception as e:  # noqa: BLE001  没装或没下载：退回字面相似度
                _failed = f"{type(e).__name__}: {e}"
    return _model


_cached: dict[str, np.ndarray] = {}
CACHE_MAX = 50000


def embed(texts: list[str]) -> np.ndarray:
    """单位向量（bge 输出已归一化），带缓存：同一段文字只算一次。"""
    model = _load()
    if model is None:
        raise RuntimeError(f"向量模型不可用：{_failed}")
    with _lock:
        todo = [t for t in dict.fromkeys(texts) if t not in _cached]
        if todo:
            if len(_cached) + len(todo) > CACHE_MAX:
                _cached.clear()
            for t, v in zip(todo, model.embed([t[:1000] for t in todo])):
                _cached[t] = v
        return np.array([_cached[t] for t in texts]) if texts else np.zeros((0, 512))


def semantic(a: str, b: str) -> float:
    va, vb = embed([a, b])
    return float(va @ vb)


def rank(query: str, texts: list[str], how: str | None = None) -> list[int]:
    """返回 texts 的下标，按与 query 的相关度从高到低。how：hybrid / semantic / lexical，默认取 mode()。"""
    if not texts:
        return []
    how = how or mode()
    lex = [lexical(query, t) for t in texts]
    lex_order = sorted(range(len(texts)), key=lambda i: -lex[i])
    if how == "lexical" or not enabled():
        return lex_order
    v = embed([query] + list(texts))
    sem = v[1:] @ v[0]
    sem_order = sorted(range(len(texts)), key=lambda i: -sem[i])
    if how == "semantic":
        return sem_order
    score = np.zeros(len(texts))
    for order in (lex_order, sem_order):
        for r, i in enumerate(order):
            score[i] += 1.0 / (RRF_K + r + 1)
    return sorted(range(len(texts)), key=lambda i: -score[i])


def score(a: str, b: str) -> float:
    """用于阈值判断的单个相关度：语义可用时取语义和字面的平均（各自 0~1），否则只用字面。
    阈值按 `evaluate()` 的实测结果设定。"""
    lx = lexical(a, b)
    if not enabled():
        return lx
    return 0.5 * (semantic(a, b) + lx)


def status() -> dict:
    return {"enabled": enabled(), "model": MODEL, "dir": str(MODEL_DIR), "error": _failed}


def download():
    from fastembed import TextEmbedding

    TextEmbedding(MODEL, cache_dir=str(MODEL_DIR))
    print(f"已下载 {MODEL} 到 {MODEL_DIR}")


if __name__ == "__main__":
    import sys

    if "--download" in sys.argv:
        download()
    print(status())
