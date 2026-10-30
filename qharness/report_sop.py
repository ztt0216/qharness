"""模块一：研报 → 因子假设。固定 SOP（LangGraph StateGraph），每个节点结构化输出、写 trace，出错能定位到节点。

节点：清洗切片 → 主题定位 → 观点提取（带出处）→ 观点归并 → 逻辑树 → 因子假设 → 忠实性核查与去重
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .experience import similarity
from .llm import LLM, extract_json
from .trace import Tracer

BOILERPLATE = re.compile(r"免责声明|分析师声明|重要声明|评级说明|请务必阅读|本报告仅供")
PAGE_MARK = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
META_LINE = re.compile(r"^\s*(日期|作者|分析师|证书编号|联系人)[:：]|^\s*<!--.*-->\s*$")  # 解析统计等注释行不进正文
SUMMARY_SECTIONS = ("核心观点", "摘要", "投资要点")
RISK_SECTIONS = ("风险提示", "风险因素")


@dataclass
class Chunk:
    text: str
    section: str
    page: int
    title: str = ""
    date: str = ""

    def cite(self) -> dict:
        return {"section": self.section, "page": self.page}


@dataclass
class Viewpoint:
    claim: str
    citations: list[dict] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


def grounded(evidence: str, chunk_text: str) -> bool:
    """论据必须出自本块原文：原文子串，或与原文某句高度相似（允许模型轻微改写）。"""
    ev = re.sub(r"\s+", "", evidence or "")
    if len(ev) < 6:
        return False
    if ev in re.sub(r"\s+", "", chunk_text):
        return True
    return any(similarity(ev, sent) >= 0.6 for sent in re.split(r"[。！？\n]", chunk_text) if sent.strip())


def clean_and_chunk(text: str, max_chars: int = 600, overlap: int = 80) -> list[Chunk]:
    title = (re.search(r"^#\s+(.+)$", text, flags=re.M) or [None, ""])[1]
    date = (re.search(r"日期[:：]\s*([\d\-/]+)", text) or [None, ""])[1]
    chunks: list[Chunk] = []
    section, page, buf = "前言", 1, []
    skipping = False

    def flush():
        body = "\n".join(buf).strip()
        if not body:
            return
        # 章节过长：按段落二次切，相邻块少量重叠
        paras, cur = [p for p in body.split("\n\n") if p.strip()], ""
        for p in paras:
            if cur and len(cur) + len(p) > max_chars:
                chunks.append(Chunk(cur.strip(), section, page, title, date))
                cur = cur[-overlap:] + "\n" + p
            else:
                cur = cur + "\n\n" + p if cur else p
        if cur.strip():
            chunks.append(Chunk(cur.strip(), section, page, title, date))

    for line in text.splitlines():
        pm = PAGE_MARK.search(line)
        if pm:
            flush(); buf = []
            page = int(pm.group(1))
            continue
        hm = re.match(r"^#{2,4}\s+(.+)$", line)
        if hm:
            flush(); buf = []
            section = hm.group(1).strip()
            skipping = bool(BOILERPLATE.search(section))
            continue
        if line.startswith("# "):
            continue
        if skipping or BOILERPLATE.search(line) or META_LINE.match(line):
            continue
        buf.append(line)
    flush()
    return chunks


SYS = "你是A股量化研究员，严格按要求输出 JSON，不要输出多余文字。"

# 研报变量依赖比日线更细的数据：出现这些标记时，日线算出来的只能是近似，不是研报的因子。
# 「日内收益」不在其中：用开盘价和收盘价就能算。
INTRADAY = re.compile(r"逐笔|分钟|尾盘|盘中|快照|委托|挂单|订单|大单|小单|主动买|主动卖|Level-?2|"
                      r"(买入|卖出)的?成交(量|额|价)|(高于|低于)收盘价(买入|卖出)|\b(buy|sell)_?i\b|p_?(buy|sell)_?i", re.I)


def needs_intraday(*texts: str) -> str | None:
    """程序兜底：模型判断「能否用日频精确计算」不稳定，多给点上下文就会放宽。"""
    m = INTRADAY.search(" ".join(t for t in texts if t))
    return m.group(0) if m else None


def _ask(llm: LLM, role: str, prompt: str, context: dict) -> dict:
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}]
    reply = llm.chat(role, msgs, context)
    try:
        return extract_json(reply)
    except ValueError:  # 多半是输出过长被截断：要求精简后重来一次，仍失败就抛出，由 trace 定位到节点
        msgs += [{"role": "assistant", "content": reply[-2000:]},
                 {"role": "user", "content": "上面的输出不是完整合法的 JSON（可能过长被截断）。请精简内容，重新输出完整 JSON。"}]
        return extract_json(llm.chat(role + ".retry", msgs, context))


DEFAULT_FIELDS = ["open", "high", "low", "close", "volume", "amount"]


class SOPState(TypedDict, total=False):
    """全部是普通类型（切片、观点存成字典），检查点直接存进 SQLite。"""
    text: str
    chunks: list[dict]
    body: list[dict]          # 正文块（不含风险提示），按与主题的相关度排序
    risks: list[str]
    topic: dict
    raw: list[dict]           # 逐块抽出的观点（带出处）
    merged: list[dict]        # 归并后的观点
    viewpoints: list[dict]
    logic_tree: dict
    hypotheses: list[dict]
    not_testable: list[dict]
    unfaithful: list[dict]


class ReportSOP:
    """固定 SOP 的 LangGraph 实现：每个节点结构化输出、写 trace，出错能定位到节点。

        chunk → topic → viewpoints → merge ─(没有观点)→ finalize → END
                                           └→ logic_tree → hypothesis → faithful → finalize → END
    """

    def __init__(self, llm: LLM, tracer: Tracer, fields: list[str] | None = None,
                 neutralized: bool = False, max_chunks: int = 12, checkpointer=None):
        self.llm, self.tr = llm, tracer
        self.fields, self.neutralized, self.max_chunks = fields or DEFAULT_FIELDS, neutralized, max_chunks
        g = StateGraph(SOPState)
        for name in ("chunk", "topic", "viewpoints", "merge", "logic_tree", "hypothesis", "faithful", "finalize"):
            g.add_node(name, getattr(self, f"_n_{name}"))
        g.add_edge(START, "chunk")
        g.add_edge("chunk", "topic")
        g.add_edge("topic", "viewpoints")
        g.add_edge("viewpoints", "merge")
        g.add_conditional_edges("merge", lambda s: "logic_tree" if s["viewpoints"] else "finalize",
                                ["logic_tree", "finalize"])
        g.add_edge("logic_tree", "hypothesis")
        g.add_edge("hypothesis", "faithful")
        g.add_edge("faithful", "finalize")
        g.add_edge("finalize", END)
        self.graph = g.compile(checkpointer=checkpointer)

    def run(self, text: str, thread_id: str | None = None) -> dict:
        from .checkpoint import run_graph

        init = {"text": text, "hypotheses": [], "not_testable": [], "unfaithful": [], "logic_tree": {}}
        out = run_graph(self.graph, init, thread_id, 50, self.tr, "report_sop")
        return {k: out.get(k) for k in ("topic", "viewpoints", "logic_tree", "hypotheses", "not_testable", "unfaithful")}

    # ---------------------------------------------------------------- 节点

    def _n_chunk(self, s):
        chunks = clean_and_chunk(s["text"])
        self.tr.log("sop.chunk", n_chunks=len(chunks), sections=sorted({c.section for c in chunks}))
        return {"chunks": [asdict(c) for c in chunks],
                "risks": [c.text for c in chunks if any(x in c.section for x in RISK_SECTIONS)],
                "body": [asdict(c) for c in chunks if not any(x in c.section for x in RISK_SECTIONS)]}

    def _n_topic(self, s):
        """主题定位：只读核心观点/摘要。"""
        chunks = [Chunk(**d) for d in s["chunks"]]
        summary = [c for c in chunks if any(x in c.section for x in SUMMARY_SECTIONS)] or chunks[:1]
        summary_text = (f"# {chunks[0].title}\n" if chunks else "") + "\n".join(c.text for c in summary)
        topic = _ask(self.llm, "sop.topic", "阅读研报的核心观点，输出 {\"market\",\"theme\",\"phenomenon\"}：\n" + summary_text,
                     {"summary_text": summary_text})
        self.tr.log("sop.topic", output=topic)
        query = json.dumps(topic, ensure_ascii=False)
        return {"topic": topic, "body": sorted(s["body"], key=lambda c: -similarity(query, c["text"]))}

    def _n_viewpoints(self, s):
        """观点提取：按主题相关度取正文块，逐块抽取，每条带出处；论据必须能在本块原文找到。"""
        query = json.dumps(s["topic"], ensure_ascii=False)
        raw, dropped = [], []
        for c in [Chunk(**d) for d in s["body"] if len(d["text"]) >= 15][:self.max_chunks]:
            ctx = {"chunk": {"text": c.text, **c.cite()}, "topic": s["topic"]}
            out = _ask(self.llm, "sop.viewpoints",
                       "从【原文】中抽取可被数据检验的判断，输出 {\"viewpoints\":[{\"claim\",\"evidence\"}]}。"
                       "【表格】里的回测绩效数字只能作论据，不要单独抽成观点；【公式】是因子定义，不是观点。"
                       "evidence 必须逐字摘自【原文】；【原文】里没有可检验的判断就输出空列表。"
                       "【主题】只用来判断相关性，不能从中抽取观点。\n"
                       f"【主题】{query}\n【原文】（{c.section} 第{c.page}页）\n{c.text}", ctx)
            for v in out.get("viewpoints", []):
                if grounded(v.get("evidence", ""), c.text):
                    raw.append(Viewpoint(v["claim"], [c.cite()], [v.get("evidence", "")]))
                else:
                    dropped.append({"claim": v.get("claim"), **c.cite()})
        self.tr.log("sop.viewpoints", n_raw=len(raw), dropped_ungrounded=dropped)
        return {"raw": [asdict(v) for v in raw]}

    def _n_merge(self, s):
        """观点归并：相似度初筛 + 模型确认，同一判断合并出处和论据；方向相反不合并。"""
        merged: list[Viewpoint] = []
        for v in (Viewpoint(**d) for d in s["raw"]):
            target = None
            for m in merged:
                if similarity(v.claim, m.claim) < 0.3:
                    continue
                if _ask(self.llm, "sop.same_claim",
                        f"两条判断是否是同一个观点（方向相反不算）？输出 {{\"same\": true/false}}\nA：{m.claim}\nB：{v.claim}",
                        {"a": m.claim, "b": v.claim}).get("same"):
                    target = m
                    break
            if target:
                target.citations += [c for c in v.citations if c not in target.citations]
                target.evidence += v.evidence
            else:
                merged.append(v)
        vps = [{"id": i + 1, "claim": m.claim, "citations": m.citations, "n_evidence": len(m.evidence)}
               for i, m in enumerate(merged)]
        self.tr.log("sop.merge", n_merged=len(vps), viewpoints=vps)
        return {"merged": [asdict(m) for m in merged], "viewpoints": vps}

    def _n_logic_tree(self, s):
        """逻辑树：影响变量 → 作用方向 → 适用条件（风险提示 = 失效条件）。"""
        vps, risks = s["viewpoints"], s["risks"]
        tree = _ask(self.llm, "sop.logic_tree",
                    "把观点整理成逻辑树，输出 {\"nodes\":[{\"variable\",\"direction\":\"正向/负向\",\"condition\",\"claim\",\"viewpoint_id\"}]}，"
                    "viewpoint_id 是对应观点的 id。\n"
                    f"观点：{json.dumps(vps, ensure_ascii=False)}\n风险提示：{risks}", {"viewpoints": vps, "risks": risks})
        self.tr.log("sop.logic_tree", output=tree)
        return {"logic_tree": tree}

    def _definitions(self, s, *texts: str) -> list[str]:
        """研报里的定义：【公式】块按名字匹配（定义和观点常不在同一个切片）；
        文字写的定义（如「1 个月动量，即前一个月涨跌幅」）按相似度取最相关的 2 段。"""
        formulas = {m.group(1): m.group(0) for m in re.finditer(r"【公式】([A-Za-z][A-Za-z0-9_]*)：[^\n]*", s["text"])}
        blob = " ".join(texts)
        out = [f for k, f in formulas.items() if re.search(rf"(?<![A-Za-z0-9]){re.escape(k)}(?![A-Za-z0-9])", blob)]
        ranked = sorted(s["body"], key=lambda c: -similarity(blob, c["text"]))[:2]
        return out + [f"（{c['section']} 第{c['page']}页）{c['text'][:400]}" for c in ranked]

    def _n_hypothesis(self, s):
        """逐个观点判断能否写成个股截面因子（一次性从几十个观点里挑，每次挑的都不一样）。"""
        fields, neutralized = self.fields, self.neutralized
        neutral_note = ("评估时因子会先对市值和行业做中性化，所以纯市值或纯行业的判断无法检验；" if neutralized else "")
        nodes = {n.get("viewpoint_id"): n for n in s["logic_tree"].get("nodes", []) if isinstance(n, dict)}
        evid = {i + 1: m["evidence"] for i, m in enumerate(s["merged"])}
        hypotheses, not_testable = [], []
        for vp in s["viewpoints"]:
            ctx = {"viewpoint": vp, "node": nodes.get(vp["id"]), "fields": fields, "neutralized": neutralized}
            out = _ask(self.llm, "sop.hypothesis",
                       "因子是截面选股：每天在股票之间比较，分数高的股票未来收益应更高。"
                       f"可用数据只有个股日频字段 {fields}（turnover 为日换手率），没有财务数据（PE/PB/ROE/盈利增速等）、"
                       "宏观、资金流、指数数据，也没有日内分钟、逐笔成交、委托单数据。" + neutral_note +
                       "研报的变量如果是用日内 / 逐笔 / 订单数据定义的（如尾盘成交占比、大单小单、主动买卖），"
                       "日频字段只能近似，近似出来的不是研报的因子，判为不可检验，reason 写明需要什么数据。"
                       "可检验：描述股票之间差异的判断，例如「高 X 的股票跑赢低 X 的股票」「A 股存在 X 效应」"
                       "（即使研报是在全市场或某类资产上统计的，也是在说股票之间的差异）。"
                       "不可检验：对大盘涨跌、风格何时切换、哪个时期哪种风格占优的判断（择时），以及需要上述缺失数据的判断。"
                       "观点带有「某时期」「某类资产」等条件时，按截面因子检验其核心变量，条件写进 condition。\n"
                       "判断下面这个观点能否直接写成个股截面因子。只量化观点里明确提到的变量，不要联想补充。输出 "
                       "{\"testable\": true/false, \"reason\": \"不超过20字\", \"hypothesis\": null 或 "
                       "{\"name\",\"fields\":[...],\"calculation\",\"window\":int,\"direction\":1或-1,\"rationale\",\"condition\"}}"
                       "，direction=1 表示因子值越大未来收益越高。\n"
                       f"观点：{vp['claim']}\n原文论据：{json.dumps(evid.get(vp['id'], []), ensure_ascii=False)}\n"
                       f"研报中的相关定义和段落：{json.dumps(self._definitions(s, vp['claim'], *evid.get(vp['id'], [])), ensure_ascii=False)}\n"
                       f"逻辑树节点：{json.dumps(nodes.get(vp['id']), ensure_ascii=False)}", ctx)
            h = out.get("hypothesis") if out.get("testable") else None
            if not h:
                not_testable.append({"claim": vp["claim"], "reason": out.get("reason", ""), "viewpoint_id": vp["id"]})
                continue
            h["viewpoint_id"] = vp["id"]
            h["citations"] = list(vp["citations"])
            missing = [f for f in h.get("fields", []) if f not in fields]
            hf = needs_intraday(vp["claim"], *evid.get(vp["id"], []), h.get("name", ""), str(h.get("calculation", "")),
                                h.get("rationale", ""), *[f for f in self._definitions(s, vp["claim"], h.get("name", ""))
                                                          if f.startswith("【公式】")])
            if missing:  # 模型说能算，但用到了没有的字段：程序兜底挪到不可检验
                not_testable.append({"claim": vp["claim"], "reason": f"需要不可得的字段 {missing}", "viewpoint_id": vp["id"]})
            elif hf:
                not_testable.append({"claim": vp["claim"], "reason": f"变量依赖日内/逐笔数据（「{hf}」），日线只能近似",
                                     "viewpoint_id": vp["id"]})
            else:
                hypotheses.append(h)
        return {"hypotheses": hypotheses, "not_testable": not_testable}

    def _n_faithful(self, s):
        """忠实性核查：每个假设对照观点、原文论据和研报定义，不是研报直接描述的、只能近似的都剔除；再按因子身份去重。"""
        from .orchestrator import factor_key

        fields = self.fields
        by_id = {v["id"]: v for v in s["viewpoints"]}
        evid = {i + 1: m["evidence"] for i, m in enumerate(s["merged"])}
        kept, unfaithful, not_testable = [], [], list(s["not_testable"])
        for h in s["hypotheses"]:
            vp = by_id.get(h.get("viewpoint_id"))
            if not vp:
                unfaithful.append({"name": h.get("name"), "reason": "没有对应的观点"})
                continue
            ctx = {"hypothesis": h, "viewpoint": vp["claim"], "evidence": evid.get(vp["id"], [])}
            out = _ask(self.llm, "sop.faithful",
                       "判断两件事：1) 因子假设是不是研报观点或研报其他段落直接描述的变量（研报没提到、由模型联想出来的不算）；"
                       f"2) 只用日频字段 {fields} 能不能精确算出研报定义的这个变量（研报用日内 / 逐笔 / 订单数据或财务数据定义、"
                       "这里只能近似的，算不能；用股价代替市值这类替代也算不能）。"
                       "两条都满足 faithful 才为 true。输出 {\"faithful\": true/false, \"exact\": true/false, \"reason\": \"...\"}\n"
                       f"观点：{vp['claim']}\n原文论据：{json.dumps(evid.get(vp['id'], []), ensure_ascii=False)}\n"
                       f"研报中的相关定义和段落：{json.dumps(self._definitions(s, h.get('name', ''), str(h.get('calculation', '')), vp['claim']), ensure_ascii=False)}\n"
                       f"因子假设：{h.get('name')}｜{h.get('calculation')}｜字段 {h.get('fields')}｜{h.get('rationale')}", ctx)
            if out.get("faithful") and out.get("exact", True):
                kept.append(h)
            elif out.get("exact") is False:  # 是研报的变量，但日频算不出来：归到不可检验，而不是拿近似去测
                not_testable.append({"claim": vp["claim"], "reason": f"日频数据只能近似：{out.get('reason', '')}",
                                     "viewpoint_id": vp["id"]})
            else:
                unfaithful.append({"name": h.get("name"), "reason": out.get("reason", ""), "viewpoint": vp["claim"]})
        # 同一个因子（公式、方向、字段、窗口一致）被多个观点各写一遍：合并成一个，出处合并
        hypotheses, seen = [], {}
        for h in kept:
            k = json.dumps(factor_key(h), ensure_ascii=False)
            if k in seen:
                seen[k]["citations"] += [c for c in h.get("citations", []) if c not in seen[k]["citations"]]
            else:
                seen[k] = h
                hypotheses.append(h)
        self.tr.log("sop.faithful", kept=[h.get("name") for h in hypotheses], n_before_dedup=len(kept), dropped=unfaithful)
        return {"hypotheses": hypotheses, "not_testable": not_testable, "unfaithful": unfaithful}

    def _n_finalize(self, s):
        by_id = {v["id"]: v for v in s.get("viewpoints") or []}
        not_testable = [{**n, "citations": by_id.get(n.get("viewpoint_id"), {}).get("citations", [])}
                        for n in s.get("not_testable") or []]
        self.tr.log("sop.hypothesis", output={"hypotheses": s.get("hypotheses") or [], "not_testable": not_testable})
        return {"not_testable": not_testable}


def run_report_sop(text: str, llm: LLM, tracer: Tracer, fields: list[str] | None = None,
                   neutralized: bool = False, max_chunks: int = 12, checkpointer=None,
                   thread_id: str | None = None) -> dict:
    return ReportSOP(llm, tracer, fields, neutralized, max_chunks, checkpointer).run(text, thread_id)
