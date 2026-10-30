"""券商研报 PDF → 带页码和章节标题的文本（report_sop 的输入格式）。

券商 PDF 的常见噪声和处理方式：
- 每页页眉 / 页脚（日期 + 报告名 + 页码）：在多数页上重复出现的行，去掉数字后比对
- 目录页（标题 + 一串点 + 页码）：用来识别正文里的章节标题，本身不进正文
- 图表块（「图表 N：标题」…「来源：」）：数字行占多数的是表格，转成【表格】结构化输出；
  其余是图（坐标轴刻度、图例），整块去掉
- 公式：字母是 Unicode 数学斜体（𝐻𝐶𝑉𝑂𝐿），归一成普通字母；分数被排版拆成几行，
  连续的公式行收拢成一个【公式】块，按原顺序保留各部分，并标出公式名
- 分析师联系方式、证书编号；披露声明 / 免责声明之后整段跳过
- 换行打断句子：按句末标点重新拼成段落
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

TOC_LINE = re.compile(r"^\s*(.+?)\s*\.{4,}\s*\d+\s*$")
NUMERIC = re.compile(r"^[\s\d.,%()（）\-+/:：年月日QH−]*$")
NOISE = re.compile(r"@|证书编号|^\s*\(?\d{3,4}\)?\s*\d{6,}|具备证券投资咨询|股份有限公司\s*$")
CAPTION = re.compile(r"^\s*图表\s*\d+\s*[.:：、]?\s*(.+)$")
SOURCE = re.compile(r"^\s*(资料来源|数据来源|来源)\s*[:：]")
STOP = re.compile(r"^\s*(特别声明|披露声明|免责声明|风险提示及免责声明|评级体系说明|分析师声明|重要声明|法律声明)\s*[:：]?\s*$")
SENT_END = re.compile(r"[。！？；：:]\s*$")
CJK = re.compile(r"[一-鿿]")
NUM_TOKEN = re.compile(r"^[-−+]?[\d,]*\.?\d+%?$|^[-−]$")
FORMULA_NAME = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*)$")
MATH_OPS = set("∑∏√∗≤≥")


def demath(s: str) -> str:
    """数学字母区段（U+1D400–U+1D7FF）归一成普通字母数字；其他字符（包括中文标点）不动。"""
    return "".join(unicodedata.normalize("NFKC", ch) if 0x1D400 <= ord(ch) <= 0x1D7FF else ch for ch in s)


def _has_math(raw: str) -> bool:
    return any(0x1D400 <= ord(ch) <= 0x1D7FF or ch in MATH_OPS for ch in raw)


def _norm(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", "", line))


def pdf_pages(path: str | Path) -> list[str]:
    """pdfplumber 的行顺序更接近版面（公式、表格行不会串到一起）；没装时退回 pypdf。"""
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            return [p.extract_text() or "" for p in pdf.pages]
    except ImportError:
        from pypdf import PdfReader

        return [p.extract_text() or "" for p in PdfReader(str(path)).pages]


def _is_table_row(line: str) -> bool:
    toks = line.split()
    if len(toks) < 3:
        return False
    nums = sum(bool(NUM_TOKEN.match(t)) for t in toks[1:])
    return nums >= max(2, (len(toks) - 1) // 2)


def _render_table(title: str, lines: list[str]) -> str | None:
    """图表块里数字行占多数才算表格；第一行非数字行当表头。"""
    rows = [l for l in lines if not NUMERIC.match(l)]
    data = [l for l in rows if _is_table_row(l)]
    if len(data) < 2 or len(data) < len(rows) / 2:
        return None
    header = next((l for l in rows if not _is_table_row(l)), "").split()
    ncols = max(len(l.split()) for l in data)
    while len(header) > ncols:  # 「风险调整后IC T 统计量」这类列名里带空格：把短碎片并回相邻列名
        i = next((k for k, h in enumerate(header[:-1]) if h.isascii() and len(h) <= 2), len(header) - 2)
        header[i:i + 2] = [header[i] + " " + header[i + 1]]
    out = [f"【表格】{title}"]
    if header:
        out.append(" | ".join(header))
    out += [" | ".join(l.split()) for l in data]
    return "\n".join(out)


def _render_formula(lines: list[str]) -> str:
    name = next((m.group(1) for m in (FORMULA_NAME.match(l) for l in lines) if m), "")
    parts = " ; ".join(re.sub(r"\s+", " ", l).strip() for l in lines)
    return f"【公式】{name + '：' if name else ''}{parts}（PDF 排版把分子、下标、分母拆成了多行，按原顺序列出）"


def pdf_to_text(path: str | Path) -> str:
    return pages_to_text(pdf_pages(path), Path(path).stem)


def pages_to_text(page_texts: list[str], name: str = "report") -> str:
    pages = [p.splitlines() for p in page_texts]
    n = len(pages)

    # 页眉页脚：同一行（数字归一）出现在一半以上页面
    freq = Counter(_norm(l) for lines in pages for l in set(lines) if l.strip())
    # 「来源：Wind…」也会在多数页出现，但它是图表块的结束标记，不能当页眉去掉；纯数字行也不算页眉
    repeated = {k for k, c in freq.items() if n >= 4 and c >= n / 2 and len(k) > 2
                and re.search(r"[A-Za-z\u4e00-\u9fff]", k) and not SOURCE.match(k)}

    # 目录：抽出章节标题
    headings = set()
    toc_pages = set()
    for i, lines in enumerate(pages):
        hits = [m.group(1) for m in (TOC_LINE.match(l) for l in lines) if m]
        if len(hits) >= 3:
            toc_pages.add(i)
            headings |= {re.sub(r"\s+", "", h) for h in hits if not h.strip().startswith("图表")}

    out: list[str] = []
    stats = {"tables": 0, "figures": 0, "formulas": 0}

    def emit(block: str):
        out.extend(["", block, ""])

    for i, lines in enumerate(pages):
        if i in toc_pages:
            continue
        out.append(f"<!-- page: {i + 1} -->")
        if i == 0:
            out.append("## 摘要")  # 券商研报首页是摘要栏，供主题定位节点使用
        para: list[str] = []
        caption: tuple[str, list[str]] | None = None
        formula: list[str] = []
        stopped = False

        def flush_para():
            if para:
                out.append("".join(para))
                out.append("")
                para.clear()

        def flush_formula():
            if formula and not any(ch in l for l in formula for ch in "=∑∏√"):
                para.extend(formula)  # 没有等号和求和号：是正文里拆出来的下标（如 SMB,t），放回正文
                formula.clear()
            if formula:
                flush_para()
                emit(_render_formula(formula))
                stats["formulas"] += 1
                formula.clear()

        for raw in lines:
            line = demath(raw.strip())
            if not line or _norm(line) in repeated:
                continue
            if STOP.match(line):
                stopped = True
                break

            # 图表块：从标题收集到「来源」
            if caption is not None:
                if SOURCE.match(line):
                    table = _render_table(*caption)
                    if table:
                        flush_para()
                        emit(table)
                        stats["tables"] += 1
                    else:
                        stats["figures"] += 1
                    caption = None
                elif len(caption[1]) > 60:  # 没等到「来源」：不是图表块，按正文处理
                    para.extend(caption[1])
                    caption = None
                else:
                    caption[1].append(line)
                continue
            m = CAPTION.match(line)
            if m:
                flush_formula()
                caption = (m.group(1).strip(), [])
                continue
            if SOURCE.match(line):
                continue

            # 公式：含数学字母或运算符的行；公式块里紧跟的短行（如「−1」「close」）也算
            # 正文里夹着数学字母（如「市值因子 RSMB,t 收益率」）的行仍是正文
            if (_has_math(raw) and len(CJK.findall(line)) <= 3) or (formula and len(line) <= 16 and not CJK.search(line)):
                formula.append(line)
                continue
            flush_formula()

            compact = re.sub(r"\s+", "", line)
            if compact in headings or re.match(r"^\d\.\d+(\.\d+)?\s*\S{2,30}$", line):
                flush_para()
                out.append(f"## {line}")
                continue
            if NUMERIC.match(line) or NOISE.search(line):
                continue
            para.append(line)
            if SENT_END.search(line):
                flush_para()
        flush_formula()
        if caption is not None:
            para.extend(caption[1])
        flush_para()
        if stopped:
            break  # 声明之后都是合规文本
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(out))
    meta = f"<!-- 解析：表格 {stats['tables']} 个，公式 {stats['formulas']} 个，去掉图 {stats['figures']} 个 -->"
    return f"# {name}\n{meta}\n\n{body}\n"
