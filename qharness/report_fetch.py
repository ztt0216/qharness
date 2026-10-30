"""自动找研报：东方财富研报中心的策略类研报（金工研报归在这一类）→ 标题筛选 → 下载 PDF → 预筛 → 送进研究流程。

完整跑一篇研报要 2~7 万 token，而公开研报里能用日线验证的个股因子研报很少（风格、择时、逐笔数据的居多），
所以重点是先便宜地筛掉跑不了的：
1. 标题：要有因子 / 选股 / 量价等词，排除周报、基金、转债、宏观、择时等
2. 程序：全文里逐笔、分钟、委托等日内数据标记出现太多（因子定义要用日内数据）→ 不可检验
3. 模型（一次调用，只看前几页）：是不是截面选股因子研报；列出的因子各需要什么数据；
   程序再复核：至少有一个因子只用日频行情就能算，才送进研究流程

状态记在 data/reports/auto/index.json（每篇一条）：listed → downloaded → screened_out / queued → researched。
PDF 只在本地使用，不纳入版本管理。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from datetime import date, timedelta
from pathlib import Path

from .report_sop import INTRADAY

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "data" / "reports" / "auto"
LIST_URL = "https://reportapi.eastmoney.com/report/list"
PDF_URL = "https://pdf.dfcfw.com/pdf/H3_{code}_1.pdf"
HEADERS = {"Referer": "https://data.eastmoney.com/", "User-Agent": "Mozilla/5.0 (Macintosh) qharness-report-fetch"}

# 标题必须明确在讲因子或量化选股；「定价」「资金流」「反转」「Alpha」单独出现多是宏观、资金面、主题投资，实测会大量误入
TITLE_POS = re.compile(r"(?<!风险)因子|多因子|量化选股|选股(模型|体系|策略|因子)|量价(因子|特征)|alpha\s*(因子|掘金|挖掘)|"
                       r"异象|技术(指标|因子)选股|截面收益", re.I)
TITLE_NEG = re.compile(r"周报|日报|晨会|早报|早参|月报|周观察|周度|月度|跟踪|基金|ETF|LOF|转债|期货|期权|债券|利率|汇率|宏观|"
                       r"配置|择时|行业轮动|风格轮动|港股|美股|海外|大宗|商品|黄金|原油|REITs|ESG|业绩|财报|年报|中报|季报|"
                       r"北向|两融|IPO|新股|指数基金|产品|保险|点评", re.I)
DAILY_DATA = ("日频行情",)
INTRADAY_HITS_MAX = 8  # 全文日内数据标记超过这个数，判为因子依赖日内数据


def title_ok(title: str) -> tuple[bool, str]:
    if TITLE_NEG.search(title):
        return False, f"标题含「{TITLE_NEG.search(title).group(0)}」"
    if not TITLE_POS.search(title):
        return False, "标题里没有因子 / 选股 / 量价相关的词"
    return True, f"标题含「{TITLE_POS.search(title).group(0)}」"


def _get_json(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def search(begin: str, end: str, max_pages: int = 50, sleep: float = 0.3, fetch=_get_json):
    """逐页拉东方财富策略类研报列表（qType=2），按发布日期倒序。返回原始行。"""
    for n in range(1, max_pages + 1):
        q = {"industryCode": "*", "pageSize": 100, "industry": "*", "rating": "*", "ratingChange": "*",
             "beginTime": begin, "endTime": end, "pageNo": n, "fields": "", "qType": 2, "orgCode": "", "rcode": "",
             "code": "*"}
        d = fetch(f"{LIST_URL}?{urllib.parse.urlencode(q)}")
        rows = d.get("data") or []
        yield from rows
        if not rows or n >= int(d.get("TotalPage") or 0):
            return
        time.sleep(sleep)


class ReportIndex:
    def __init__(self, root: Path = DEFAULT_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "index.json"
        self.items: dict[str, dict] = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def update(self, code: str, **kw):
        self.items.setdefault(code, {"code": code}).update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        self.save()

    def by_status(self, *status: str) -> list[dict]:
        return sorted((x for x in self.items.values() if x.get("status") in status),
                      key=lambda x: x.get("date", ""), reverse=True)


def collect(index: ReportIndex, days: int = 30, end: str | None = None, max_pages: int = 50, fetch=_get_json) -> dict:
    """列表 → 标题筛选。新出现、标题通过的记为 listed；已经在索引里的不动。"""
    end_d = date.fromisoformat(end) if end else date.today()
    begin = str(end_d - timedelta(days=days))
    seen = new = 0
    for r in search(begin, str(end_d), max_pages, fetch=fetch):
        seen += 1
        code, title = r.get("infoCode"), r.get("title", "")
        if not code or code in index.items:
            continue
        ok, why = title_ok(title)
        if ok:
            index.update(code, title=title, org=r.get("orgSName"), date=(r.get("publishDate") or "")[:10],
                         pages=r.get("attachPages"), url=PDF_URL.format(code=code), status="listed", reason=why)
            new += 1
    return {"seen": seen, "new": new, "begin": begin, "end": str(end_d)}


def download(index: ReportIndex, code: str, max_mb: float = 30, opener=urllib.request.urlopen) -> Path:
    rec = index.items[code]
    dest = index.root / f"{code}.pdf"
    if not dest.exists():
        req = urllib.request.Request(rec["url"], headers=HEADERS)
        with opener(req, timeout=60) as r:
            data = r.read(int(max_mb * 1024 * 1024) + 1)
        if len(data) > max_mb * 1024 * 1024:
            raise ValueError(f"文件超过 {max_mb} MB")
        if not data.startswith(b"%PDF"):
            raise ValueError("下载到的不是 PDF")
        dest.write_bytes(data)
    import hashlib

    # 记内容指纹：同一篇研报再从收件夹放进来，按内容认出来，不重复研究
    index.update(code, status="downloaded", path=dest.name, size=dest.stat().st_size, source="eastmoney",
                 sha1=hashlib.sha1(dest.read_bytes()).hexdigest())
    return dest


SCREEN_PROMPT = """下面是一篇券商研报的标题和前几页。判断它能不能用来做「截面选股因子」研究，并且只用 A 股日频行情就能验证。
日频行情只有：开盘价、最高价、最低价、收盘价、成交量、成交额、换手率（每只股票每天一行）。

只输出 JSON：
{{"type": "截面选股因子" | "择时" | "资产配置或行业轮动" | "策略或产品介绍" | "其他",
  "factors": [{{"name": "因子名", "definition": "研报里的定义（一句话，没有就写未给出）",
               "data": "日频行情" | "财务" | "分析师预期" | "日内或逐笔" | "其他"}}],
  "reason": "一句话结论"}}

要求：
- factors 只列研报自己提出或重点检验的因子，最多 8 个；
- data 按研报的定义判断：要用分钟、逐笔、委托、主动买卖方向的，写「日内或逐笔」，不能因为日线能近似就写日频；
- 研报讲风格轮动、指数择时、基金或产品的，type 不要写截面选股因子。

标题：{title}

正文（前几页）：
{text}
"""


def report_text(path: Path) -> str:
    """和 run 命令同样的转换：PDF / PPT 转成带页码的文本，Markdown / 纯文本直接读。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from . import report_pdf
        return report_pdf.pdf_to_text(path)
    if suffix == ".pptx":
        from .report_ppt import pptx_to_text
        return pptx_to_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


# ------------------------------------------------------------------ 收件夹

INBOX = ROOT / "data" / "reports" / "inbox"
INBOX_EXT = {".pdf", ".pptx", ".md", ".txt"}


def scan_inbox(index: ReportIndex, inbox: Path = INBOX, min_age_s: float = 5) -> dict:
    """收件夹里新出现的研报 → 复制进索引目录，状态记为待预筛。
    按内容（sha1）去重：同一份研报改名或重复放入不会再处理；原文件不移动、不删除。
    刚修改过的文件（min_age_s 秒内）先跳过，避免读到还没下载或复制完的文件。"""
    import hashlib
    import shutil

    inbox = Path(inbox)
    inbox.mkdir(parents=True, exist_ok=True)
    known = {x.get("sha1") for x in index.items.values() if x.get("sha1")}
    added, skipped, waiting = [], 0, 0
    for f in sorted(inbox.iterdir()):
        if not f.is_file() or f.name.startswith(".") or f.suffix.lower() not in INBOX_EXT:
            continue
        if time.time() - f.stat().st_mtime < min_age_s:
            waiting += 1
            continue
        h = hashlib.sha1(f.read_bytes()).hexdigest()
        if h in known:
            skipped += 1
            continue
        code = f"LOCAL-{h[:12]}"
        dest = index.root / f"{code}{f.suffix.lower()}"
        shutil.copy2(f, dest)
        index.update(code, title=f.stem, org="本地收件夹", source="inbox", file=f.name, sha1=h,
                     date=time.strftime("%Y-%m-%d", time.localtime(f.stat().st_mtime)), pages=None,
                     path=dest.name, size=dest.stat().st_size, status="downloaded", reason="收件夹新文件，待预筛")
        known.add(h)
        added.append(code)
    return {"added": added, "skipped": skipped, "waiting": waiting, "inbox": str(inbox)}


def run_inbox(llm=None, research_max: int = 0, index: ReportIndex | None = None, inbox: Path = INBOX,
              log=print) -> dict:
    """扫描收件夹 → 预筛新文件 →（可选）研究通过预筛的。"""
    index = index or ReportIndex()
    s = scan_inbox(index, inbox)
    log(f"收件夹 {s['inbox']}：新文件 {len(s['added'])} 个，已处理过 {s['skipped']} 个"
        + (f"，{s['waiting']} 个刚写入、下次再处理" if s["waiting"] else ""))
    if llm is not None:
        for code in s["added"]:
            try:
                r = screen(index, code, llm)
            except Exception as e:  # noqa: BLE001  解析失败（加密 PDF、扫描件等）只影响这一篇
                index.update(code, status="screened_out", stage="parse", reason=f"解析失败：{type(e).__name__}: {e}")
                r = index.items[code]
            log(f"  {'送进研究' if r['status'] == 'queued' else '筛掉'}《{r['title']}》：{r['reason']}")
    ran = []
    for rec in [x for x in index.by_status("queued") if x.get("source") == "inbox"][:research_max]:
        log(f"  研究《{rec['title']}》…")
        r = research(index, rec["code"])
        ran.append(r["run_id"])
        log(f"    {r['reason']}（运行 {r['run_id']}）")
    return {"scan": s, "ran": ran}


SIZE = re.compile(r"市值|规模|size", re.I)


def decide(out: dict) -> tuple[bool, str]:
    """按模型的预筛输出决定是否送进研究（程序复核）。
    市值因子不算：评估时对市值做中性化，市值本身测不出东西（模型常把它归为日频行情）。"""
    factors = [f for f in out.get("factors") or [] if isinstance(f, dict)]
    size = [f for f in factors if SIZE.search(f.get("name", ""))]
    daily = [f for f in factors if f.get("data") in DAILY_DATA and f not in size]
    if out.get("type") != "截面选股因子":
        return False, f"不是截面选股因子研报（{out.get('type')}）：{out.get('reason', '')}"
    if daily:
        return True, f"{len(daily)} 个因子只用日频行情：{'、'.join(f.get('name', '?') for f in daily[:4])}"
    need = sorted({f.get("data", "?") for f in factors if f not in size})
    note = f"市值因子（{'、'.join(f.get('name', '') for f in size)}）在评估时被中性化，不算；" if size else ""
    return False, f"{note}其余因子要用日频行情以外的数据（{'、'.join(need) or '没有其他因子'}）"


def screen(index: ReportIndex, code: str, llm, max_chars: int = 5000) -> dict:
    """程序先查日内数据标记，再用一次模型调用判断；最终是否送进研究流程由程序按模型的输出复核。"""
    from .llm import extract_json

    rec = index.items[code]
    text = report_text(index.root / rec["path"])
    (index.root / f"{code}.text.md").write_text(text, encoding="utf-8")  # 解析结果留档（不和 Markdown 原文重名）
    hits = [m.group(0) for m in INTRADAY.finditer(text)]
    if len(hits) > INTRADAY_HITS_MAX:
        words = sorted(set(hits))[:5]
        index.update(code, status="screened_out", stage="program", intraday_hits=len(hits),
                     reason=f"全文出现 {len(hits)} 处日内数据标记（{'、'.join(words)}），因子多半要用日内或逐笔数据")
        return index.items[code]
    body = re.sub(r"<!--.*?-->", "", text)[:max_chars]
    msgs = [{"role": "system", "content": "你是A股量化研究员，严格按要求输出 JSON，不要输出多余文字。"},
            {"role": "user", "content": SCREEN_PROMPT.format(title=rec["title"], text=body)}]
    try:
        out = extract_json(llm.chat("fetch.screen", msgs, {"report": code}))
    except ValueError as e:
        index.update(code, status="screened_out", stage="model", reason=f"模型输出无法解析：{e}")
        return index.items[code]
    ok, reason = decide(out)
    index.update(code, status="queued" if ok else "screened_out", stage="model", screen=out,
                 intraday_hits=len(hits), reason=reason)
    return index.items[code]


STATUS_CN = {"accepted": "通过", "rejected": "淘汰", "skipped": "跳过", "failed": "实现失败"}


def research(index: ReportIndex, code: str, workers: int = 2, extra: list[str] | None = None) -> dict:
    """子进程跑完整研究流程（和前端发起的一样），结果回写索引。"""
    rec = index.items[code]
    rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    index.update(code, status="researching", run_id=rid)
    report = index.root / rec["path"]
    cmd = [sys.executable, "-m", "qharness", "run", "--report", str(report), "--run-id", rid,
           "--workers", str(workers), *(extra or [])]
    proc = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    results = ROOT / "runs" / rid / "results.json"
    counts: dict[str, int] = {}
    if results.exists():
        for r in json.loads(results.read_text(encoding="utf-8")):
            counts[r.get("status")] = counts.get(r.get("status"), 0) + 1
    index.update(code, status="researched", outcome=counts, returncode=proc.returncode,
                 reason=("研究完成：" + "，".join(f"{STATUS_CN.get(k, k)} {v}" for k, v in counts.items())) if counts
                 else ("研究完成：没有可检验的假设" if proc.returncode == 0 else f"研究失败（退出码 {proc.returncode}）"))
    return index.items[code]


def run_pipeline(days: int = 30, download_max: int = 10, research_max: int = 0, llm=None,
                 index: ReportIndex | None = None, log=print) -> dict:
    """搜集 → 下载 → 预筛 →（可选）研究。每一步都写索引，中断后再跑会接着做。"""
    index = index or ReportIndex()
    c = collect(index, days)
    log(f"列表：{c['begin']} 至 {c['end']} 共 {c['seen']} 篇策略类研报，标题通过的新研报 {c['new']} 篇")
    done = 0
    for rec in index.by_status("listed")[:download_max]:
        try:
            download(index, rec["code"])
            done += 1
            log(f"  下载 {rec['date']} {rec['org']}《{rec['title']}》")
        except Exception as e:  # noqa: BLE001
            index.update(rec["code"], status="screened_out", stage="download", reason=f"下载失败：{e}")
            log(f"  下载失败 {rec['title']}：{e}")
    if llm is not None:
        for rec in index.by_status("downloaded"):
            r = screen(index, rec["code"], llm)
            log(f"  {'送进研究' if r['status'] == 'queued' else '筛掉'}《{r['title']}》：{r['reason']}")
    ran = []
    for rec in index.by_status("queued")[:research_max]:
        log(f"  研究《{rec['title']}》…")
        r = research(index, rec["code"])
        ran.append(r["run_id"])
        log(f"    {r['reason']}（运行 {r['run_id']}）")
    summary = {k: len(index.by_status(k)) for k in ("listed", "downloaded", "screened_out", "queued", "researched")}
    log("索引：" + "，".join(f"{k} {v}" for k, v in summary.items()))
    return {"collect": c, "downloaded": done, "ran": ran, "index": summary}
