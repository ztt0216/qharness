"""研报 PPT（.pptx）→ 带页码和章节标题的文本（report_sop 的输入格式，和 report_pdf 的输出同一格式）。

.pptx 是 zip 包里的一组 XML，直接用标准库解析，不依赖 python-pptx：
- 页序：按 presentation.xml 里的幻灯片列表（文件名编号不一定等于放映顺序）
- 每页：标题占位符 → 章节标题（`## 标题`）；其余文本框按版面从上到下、从左到右排列，每个段落一行
- 表格 → 【表格】结构化输出；图表、图片没有文字，只记一个数
- 演讲者备注：常写着页面上没有的论证，附在该页末尾（标成「演讲者备注（与本页正文同等对待）」）
- 摘要定位：标题里没有「核心观点 / 摘要 / 投资要点」时，把第一张有正文的页标成摘要，供主题定位节点使用
"""
from __future__ import annotations

import posixpath
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
R_ID = f"{{{NS['r']}}}id"
SUMMARY = ("核心观点", "摘要", "投资要点")
TITLE_TYPES = {"title", "ctrTitle"}
SKIP_PH = {"sldNum", "dt", "ftr"}  # 页码、日期、页脚
NOTES_LABEL = "演讲者备注（与本页正文同等对待）："


def _rels(z: zipfile.ZipFile, part: str) -> dict[str, str]:
    """某个部件的关系表：rId → 目标部件的包内路径。"""
    d, name = posixpath.split(part)
    path = posixpath.join(d, "_rels", name + ".rels")
    if path not in z.namelist():
        return {}
    out = {}
    for r in ET.fromstring(z.read(path)).findall("rel:Relationship", NS):
        out[r.get("Id")] = posixpath.normpath(posixpath.join(d, r.get("Target")))
    return out


def _slide_parts(z: zipfile.ZipFile) -> list[str]:
    rels = _rels(z, "ppt/presentation.xml")
    root = ET.fromstring(z.read("ppt/presentation.xml"))
    parts = [rels[s.get(R_ID)] for s in root.iterfind("p:sldIdLst/p:sldId", NS) if s.get(R_ID) in rels]
    if parts:
        return parts
    num = lambda n: int(re.search(r"(\d+)\.xml$", n).group(1))  # noqa: E731
    return sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)), key=num)


def _para_texts(el) -> list[str]:
    lines = []
    for p in el.iterfind(".//a:p", NS):
        t = "".join(x.text or "" for x in p.iterfind(".//a:t", NS)).strip()
        if t:
            lines.append(t)
    return lines


def _pos(el) -> tuple[int, int]:
    off = el.find(".//a:off", NS)
    return (int(off.get("y", 0)), int(off.get("x", 0))) if off is not None else (0, 0)


def _ph_type(sp) -> str | None:
    ph = sp.find("./p:nvSpPr/p:nvPr/p:ph", NS)
    if ph is None:
        return None
    return ph.get("type", "body")


def _table(frame) -> list[list[str]]:
    rows = []
    for tr in frame.iterfind(".//a:tbl/a:tr", NS):
        rows.append([" ".join(_para_texts(tc)) for tc in tr.iterfind("a:tc", NS)])
    return rows


def parse_slide(xml: bytes) -> dict:
    """一页 → {title, blocks: [(y, x, 文本)], tables, n_charts, n_pictures}。"""
    root = ET.fromstring(xml)
    title, blocks, tables, charts, pics = "", [], [], 0, 0
    tree = root.find("p:cSld/p:spTree", NS)
    if tree is None:
        return {"title": "", "blocks": [], "tables": [], "n_charts": 0, "n_pictures": 0}
    for el in tree.iter():
        tag = el.tag.split("}")[1]
        if tag == "sp":
            kind = _ph_type(el)
            if kind in SKIP_PH:
                continue
            lines = _para_texts(el)
            if not lines:
                continue
            if kind in TITLE_TYPES and not title:
                title = " ".join(lines)
            else:
                blocks.append((*_pos(el), "\n".join(lines)))
        elif tag == "graphicFrame":
            uri = (el.find(".//a:graphicData", NS).get("uri", "") if el.find(".//a:graphicData", NS) is not None else "")
            if uri.endswith("/table"):
                rows = _table(el)
                if rows:
                    tables.append((*_pos(el), rows))
            elif uri.endswith("/chart"):
                charts += 1
        elif tag == "pic":
            pics += 1
    return {"title": title, "blocks": sorted(blocks), "tables": sorted(tables, key=lambda t: t[:2]),
            "n_charts": charts, "n_pictures": pics}


def _notes(z: zipfile.ZipFile, part: str) -> list[str]:
    for target in _rels(z, part).values():
        if "notesSlides/" in target and target in z.namelist():
            root = ET.fromstring(z.read(target))
            lines = []
            for sp in root.iter(f"{{{NS['p']}}}sp"):
                if _ph_type(sp) in {"body", None}:
                    lines += _para_texts(sp)
            return lines
    return []


def pptx_to_text(path: str | Path) -> str:
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        parts = _slide_parts(z)
        slides = [(parse_slide(z.read(p)), _notes(z, p)) for p in parts]
    has_summary = any(any(k in s["title"] for k in SUMMARY) for s, _ in slides)
    out, stats = [], {"tables": 0, "charts": 0, "pictures": 0, "notes": 0}
    for i, (s, notes) in enumerate(slides, 1):
        body = [b[2] for b in s["blocks"]]
        title = s["title"] or (body.pop(0).split("\n")[0] if body and len(body[0]) <= 30 else f"第 {i} 页")
        if not has_summary and sum(len(b) for b in body) >= 60:
            title, has_summary = f"摘要：{title}", True  # 第一张有正文的页当摘要
        out += [f"<!-- page: {i} -->", f"## {title}", ""]
        for b in body:
            out += [b, ""]
        for _, _, rows in s["tables"]:
            out += [f"【表格】{title}", *(" | ".join(r) for r in rows), ""]
            stats["tables"] += 1
        if notes:
            # 只写「备注：」时模型会把这段当成非正文跳过（实测同一段落 0/2 次被抽取）；注明同等对待后能正常抽取
            out += [NOTES_LABEL + "\n".join(notes), ""]
            stats["notes"] += 1
        stats["charts"] += s["n_charts"]
        stats["pictures"] += s["n_pictures"]
    meta = (f"<!-- 解析：{len(slides)} 页，表格 {stats['tables']} 个，备注 {stats['notes']} 页，"
            f"图表 {stats['charts']} 个、图片 {stats['pictures']} 张（无文字，未解析） -->")
    doc_title = (slides[0][0]["title"] if slides and slides[0][0]["title"] else path.stem)  # 封面标题即研报标题
    return f"# {doc_title}\n{meta}\n\n" + re.sub(r"\n{3,}", "\n\n", "\n".join(out)) + "\n"
