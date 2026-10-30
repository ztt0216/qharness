// 一次研究运行的审计轨迹：左边研报原文（按页），右边从研报到判定的每一步；出处可以跳回原文那一页。
import { Fragment, useCallback, useMemo, useRef } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError, useApi, type Citation, type Hypothesis, type Job, type RunDetail as Detail, type TraceEvent } from "../api";
import { ERRORS, RULES, compact, duration, pct, runTime, share, signed, source } from "../format";
import { CountsLine, ErrorState, Loading, Verdict } from "../ui";

const NOTES_LABEL = "演讲者备注（与本页正文同等对待）：";

interface Page { page: number; lines: { kind: "h" | "p" | "notes" | "table"; text: string }[] }

function parseReport(md: string): Page[] {
  const pages: Page[] = [];
  let cur: Page = { page: 1, lines: [] };
  const push = () => { if (cur.lines.length) pages.push(cur); };
  for (const raw of md.split("\n")) {
    const line = raw.trim();
    const m = line.match(/^<!--\s*page:\s*(\d+)\s*-->$/);
    if (m) {
      push();
      const same = pages.find((x) => x.page === +m[1]);  // 页码标记之前的内容（标题、日期）和第 1 页合在一起
      if (same) { pages.splice(pages.indexOf(same), 1); cur = same; } else cur = { page: +m[1], lines: [] };
      continue;
    }
    if (!line || /^<!--.*-->$/.test(line) || /^#\s/.test(line)) continue;
    if (/^#{2,}\s/.test(line)) cur.lines.push({ kind: "h", text: line.replace(/^#+\s*/, "") });
    else if (line.startsWith(NOTES_LABEL)) cur.lines.push({ kind: "notes", text: line.slice(NOTES_LABEL.length) });
    else if (line.startsWith("【表格】") || line.includes(" | ")) cur.lines.push({ kind: "table", text: line });
    else cur.lines.push({ kind: "p", text: line });
  }
  push();
  return pages;
}

interface Round { round: number; events: TraceEvent[] }
interface Cand { key: string; start: TraceEvent; rounds: Round[]; other: TraceEvent[]; end?: TraceEvent; result?: any }

/** 按候选归组：新记录带 cand 标签；老记录（串行）没有，就按 candidate.start / end 之间的顺序归。 */
function groupCandidates(trace: TraceEvent[], results: any[] | null): { cands: Cand[]; skipped: TraceEvent[] } {
  const by = new Map<string, Cand>();
  const order: string[] = [];
  const skipped: TraceEvent[] = [];
  let current: string | null = null, n = 0;
  for (const e of trace) {
    if (e.node === "candidate.start") {
      const key = e.cand ?? `seq-${n++}`;
      if (!by.has(key)) { by.set(key, { key, start: e, rounds: [], other: [] }); order.push(key); }
      current = key;
      continue;
    }
    if (e.node === "research.skip_rejected" && !e.cand) { skipped.push(e); continue; }
    if (!e.node.startsWith("experiment.") && !e.node.startsWith("research.") && e.node !== "candidate.end" && e.node !== "experience.add") continue;
    const key = e.cand ?? current;
    const c = key ? by.get(key) : undefined;
    if (!c) continue;
    if (e.node === "candidate.end") { c.end = e; if (!e.cand) current = null; continue; }
    if (typeof e.round === "number") {
      let r = c.rounds.find((x) => x.round === e.round);
      if (!r) { r = { round: e.round, events: [] }; c.rounds.push(r); }
      r.events.push(e);
    } else c.other.push(e);
  }
  const cands = order.map((k) => by.get(k)!);
  for (const c of cands) c.result = results?.find((r) => r.hypothesis?.name === c.start.hypothesis?.name);
  return { cands, skipped };
}

function Cites({ cites, onJump }: { cites?: Citation[]; onJump: (p: number) => void }) {
  if (!cites?.length) return null;
  return <>{cites.map((c, i) => (
    <button key={i} className="cite" onClick={() => onJump(c.page)} title={`跳到原文第 ${c.page} 页「${c.section}」`}>
      {c.section} p{c.page}
    </button>
  ))}</>;
}

function SegMetrics({ m }: { m: any }) {
  if (!m) return null;
  const segs = (["train", "valid", "test"] as const).filter((k) => m[k]?.valid);
  if (!segs.length) return null;
  const name = { train: "训练段", valid: "验证段", test: "测试段" };
  return (
    <div className="metric-grid num">
      <span className="h"></span><span className="h">RankIC</span><span className="h">t（NW）</span><span className="h">未中性化</span><span className="h">和因子库最大相关</span>
      {segs.map((k) => (
        <Fragment key={k}>
          <span className="ink2">{name[k]}</span>
          <span>{signed(m[k].rank_ic)}</span>
          <span>{m[k].rank_ic_t_nw}</span>
          <span className="ink2">{m[k].rank_ic_raw == null ? "—" : signed(m[k].rank_ic_raw)}</span>
          <span className="ink2">{m[k].max_abs_corr_with_library}</span>
        </Fragment>
      ))}
    </div>
  );
}

function RoundView({ r, metrics }: { r: Round; metrics?: any }) {
  const ev = (n: string) => r.events.filter((e) => e.node === n);
  const retrieve = ev("experiment.retrieve")[0];
  const submits = ev("experiment.submit");
  const execs = ev("experiment.execute");
  const judge = ev("research.judge")[0];
  const dup = ev("experiment.duplicate")[0];
  const flip = ev("research.flip_only")[0];
  const lastExec = execs[execs.length - 1];
  const fixes = execs.filter((e) => !e.ok);
  return (
    <div className="round">
      <div className="round-title">
        <b>第 {r.round + 1} 轮</b>
        {submits[0] && <span>{submits[0].via === "run_qlib_factor" ? "写成 Qlib 表达式" : submits[0].via === "run_factor" ? "写成 Python 代码" : "文本提交"}</span>}
        {retrieve && <span>检索到 {retrieve.n} 条经验{retrieve.kinds?.length ? `（${retrieve.kinds.join("、")}）` : ""}</span>}
        {fixes.length > 0 && <span>自动修复 {fixes.length} 次</span>}
        {flip && <span>只翻转方向（{flip.direction > 0 ? "+1" : "−1"}），沿用上一轮代码</span>}
      </div>
      {fixes.map((e, i) => (
        <div key={i} className="small" style={{ color: "var(--critical)" }}>
          ✕ {ERRORS[e.error_type] ?? e.error_type}：{String(e.message ?? "").slice(0, 160)}
        </div>
      ))}
      {lastExec?.ok && (
        <>
          <pre className="code">{lastExec.code}</pre>
          <div className="checks">
            <span className={lastExec.checks?.static === "pass" ? "ok" : "bad"}>静态检查</span>
            <span className={lastExec.checks?.lookahead === "pass" ? "ok" : "bad"}>未来数据（截断重算差 {lastExec.checks?.lookahead_max_diff ?? "—"}）</span>
            <span className={(lastExec.checks?.order_invariance_max_diff ?? 0) < 1e-9 ? "ok" : "bad"}>按股票分组</span>
            <span>覆盖率 {share(lastExec.checks?.coverage)}</span>
            <span>方向 {lastExec.direction > 0 ? "+1" : "−1"}（由 Harness 施加）</span>
          </div>
        </>
      )}
      {dup && <div className="diagnosis">与本次已评估的「{dup.of}」是同一个因子（截面相关 {signed(dup.corr, 3)}），不再评估。</div>}
      {judge && (
        <>
          <SegMetrics m={metrics} />
          <div style={{ marginTop: 8 }}>
            <Verdict status={judge.verdict === "accept" ? "accepted" : judge.verdict === "reject" ? "rejected" : "running"} />
            <span className="small ink2" style={{ marginLeft: 10 }}>
              {judge.verdict === "refine" ? "研究层要求修改后再试" : ""}
              {judge.t_valid_required ? `验证段 t 门槛 ${judge.t_valid_required}（第 ${judge.n_trials} 次试验）` : ""}
            </span>
          </div>
          {judge.failed?.length > 0 && (
            <div style={{ marginTop: 6 }}>{judge.failed.map((f: string) => <span key={f} className="tag">{RULES[f] ?? f}</span>)}</div>
          )}
          {judge.diagnosis && <p className="diagnosis">{judge.diagnosis}</p>}
          {judge.revision && (
            <p className="small ink2" style={{ marginTop: 6 }}>
              修改：{Object.entries(judge.revision).map(([k, v]) => `${k} → ${Array.isArray(v) ? v.join(", ") : v}`).join("；")}
              {judge.note ? `。${judge.note}` : ""}
            </p>
          )}
        </>
      )}
    </div>
  );
}

function CandidateView({ c, onJump }: { c: Cand; onJump: (p: number) => void }) {
  const h: Hypothesis = c.start.hypothesis;
  const status = c.end?.status ?? c.result?.status ?? "running";
  const rounds = [...c.rounds].sort((a, b) => a.round - b.round);
  const giveUp = c.other.find((e) => e.node === "experiment.give_up");
  return (
    <article className="cand">
      <div className="cand-head">
        <h4>{h.name}</h4>
        <Verdict status={status} />
      </div>
      <div className="cand-meta">
        {h.calculation}；窗口 {h.window}，方向 {h.direction > 0 ? "+1" : "−1"}，字段 {h.fields?.join("、")}
        <Cites cites={h.citations} onJump={onJump} />
        {h.rationale && <div style={{ marginTop: 4 }}>依据：{h.rationale}{h.condition ? `（条件：${h.condition}）` : ""}</div>}
      </div>
      {rounds.map((r, i) => <RoundView key={r.round} r={r} metrics={i === rounds.length - 1 ? c.result?.metrics : (c.result?.history?.[r.round]?.metrics)} />)}
      {c.result?.test_metrics && (
        <div className="round"><div className="round-title"><b>测试段（只开一次）</b></div><SegMetrics m={c.result.test_metrics} /></div>
      )}
      {giveUp && <div className="round small" style={{ color: "var(--critical)" }}>多次修复仍然失败（{ERRORS[giveUp.error_type] ?? giveUp.error_type}），放弃这个候选。</div>}
      {status === "skipped" && c.end?.reason && <div className="round small ink2">{c.end.reason}</div>}
    </article>
  );
}

export default function RunDetail() {
  const { id = "" } = useParams();
  const run = useApi<Detail>(`/api/runs/${id}`, (d) => (!d || d.summary.state === "running" ? 2000 : undefined));
  const jobs = useApi<Job[]>("/api/jobs", (d) => (d?.some((j) => j.run_id === id && j.status === "running") ? 2000 : undefined));
  const job = jobs.data?.find((j) => j.run_id === id);
  const jobLog = useApi<Job>(job ? `/api/jobs/${job.id}` : null, job?.status === "running" ? 2000 : undefined);
  const pane = useRef<HTMLDivElement>(null);

  const jump = useCallback((page: number) => {
    const el = pane.current?.querySelector<HTMLElement>(`[data-page="${page}"]`);
    if (!el || !pane.current) return;
    pane.current.scrollTo({ top: el.offsetTop - 12, behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    el.classList.add("flash");
    window.setTimeout(() => el.classList.remove("flash"), 900);
  }, []);

  const d = run.data;
  const pages = useMemo(() => (d?.report ? parseReport(d.report) : []), [d?.report]);
  const grouped = useMemo(() => (d ? groupCandidates(d.trace, d.results) : { cands: [], skipped: [] }), [d]);

  if (run.error && !d) {
    if (run.error instanceof ApiError && run.error.status === 404 && job?.status === "running") {
      return <div className="page"><h1>正在启动…</h1><p className="ink2" style={{ marginTop: 10 }}>正在加载行情和研报，几秒后开始出现进度。</p></div>;
    }
    if (run.error instanceof ApiError && run.error.status === 404 && job?.status === "failed") {
      return <div className="page"><h1>运行没能启动</h1><pre className="log code" style={{ marginTop: 16 }}>{jobLog.data?.log?.join("\n")}</pre></div>;
    }
    return <div className="page"><ErrorState error={run.error} what="这次运行" /></div>;
  }
  if (!d) return <div className="page"><Loading /></div>;

  const ev = (n: string) => d.trace.find((e) => e.node === n);
  const chunk = ev("sop.chunk"), topic = ev("sop.topic")?.output, vps = ev("sop.viewpoints"), merge = ev("sop.merge");
  const hyp = ev("sop.hypothesis")?.output, faithful = ev("sop.faithful"), start = ev("run.start") ?? ev("run.resume");
  const s = d.summary;
  const llm = Object.values(d.llm_by_role).reduce((a, x) => ({ calls: a.calls + x.calls, tok: a.tok + x.input_tokens + x.output_tokens }), { calls: 0, tok: 0 });

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <p className="small"><Link to="/runs">研究运行</Link> / {runTime(s.id)}</p>
          <h1 style={{ marginTop: 6 }}>{s.theme ?? s.report ?? "研究运行"}</h1>
          <p>
            {source(s)}；模型 {s.model ?? "—"}；{s.n_candidates} 个候选，<CountsLine counts={s.counts} total={s.n_candidates} />；
            模型调用 {llm.calls} 次、{compact(llm.tok)} token，耗时 {duration(s.started, s.ended)}。
          </p>
        </div>
        <Verdict status={s.state} />
      </div>

      {s.state === "running" && jobLog.data?.log && (
        <details open style={{ marginBottom: 24 }}>
          <summary className="small ink2" style={{ cursor: "pointer" }}>运行日志（每 2 秒刷新）</summary>
          <pre className="log code">{jobLog.data.log.slice(-40).join("\n")}</pre>
        </details>
      )}

      <div className="run-split">
        <aside className="report-pane" ref={pane} aria-label="研报原文">
          <h3 style={{ marginBottom: 6 }}>研报原文</h3>
          {pages.length ? (
            <>
              <p className="small muted" style={{ marginBottom: 14 }}>解析后的文本，按页排列。点右边的出处跳到对应页。</p>
              {pages.map((p) => (
                <div key={p.page} className="page-block" data-page={p.page}>
                  <div className="pg">第 {p.page} 页</div>
                  {p.lines.map((l, i) => l.kind === "h" ? <h4 key={i}>{l.text}</h4>
                    : l.kind === "notes" ? <p key={i} className="para notes">演讲者备注：{l.text}</p>
                    : l.kind === "table" ? <p key={i} className="para small code">{l.text}</p>
                    : <p key={i} className="para">{l.text}</p>)}
                </div>
              ))}
            </>
          ) : <p className="ink2">这次运行没有研报原文（直接给的因子假设，或原文件已不在）。</p>}
        </aside>

        <div className="trail">
          <section className="stage">
            <h3>切分研报</h3>
            {chunk ? (
              <p className="ink2">切成 {chunk.n_chunks} 段：{chunk.sections?.join("、")}。</p>
            ) : <p className="ink2">没有研报，跳过。</p>}
            {start?.split && (
              <p className="small muted" style={{ marginTop: 6 }}>
                样本切分：训练段 {start.split.train.join(" 至 ")}，验证段 {start.split.valid.join(" 至 ")}，测试段 {start.split.test.join(" 至 ")}（各段末尾去掉 6 天）。
              </p>
            )}
          </section>

          {topic && (
            <section className="stage">
              <h3>定位主题</h3>
              <p><b>{topic.theme}</b>（{topic.market}）</p>
              {topic.phenomenon && <p className="ink2" style={{ marginTop: 4 }}>{topic.phenomenon}</p>}
            </section>
          )}

          {merge && (
            <section className="stage">
              <h3>抽取观点</h3>
              <p className="small ink2" style={{ marginBottom: 6 }}>
                抽出 {vps?.n_raw ?? "—"} 条，归并为 {merge.n_merged} 条
                {vps?.dropped_ungrounded?.length ? `；${vps.dropped_ungrounded.length} 条在原文里找不到依据，已丢弃` : "；每条都能在原文里找到依据"}。
              </p>
              {merge.viewpoints?.map((v: any) => (
                <div key={v.id} className="viewpoint">
                  {v.claim}<Cites cites={v.citations} onJump={jump} />
                </div>
              ))}
            </section>
          )}

          {hyp && (
            <section className="stage">
              <h3>形成因子假设</h3>
              <p className="ink2">
                {hyp.hypotheses?.length ?? 0} 个可检验的假设
                {hyp.not_testable?.length ? `，${hyp.not_testable.length} 个暂时不可检验` : ""}
                {faithful?.dropped?.length ? `；忠实性核查删掉 ${faithful.dropped.length} 个` : "；都通过了忠实性核查"}。
              </p>
              {hyp.not_testable?.map((x: any, i: number) => (
                <p key={i} className="small muted" style={{ marginTop: 4 }}>不可检验：{typeof x === "string" ? x : `${x.claim ?? x.name ?? ""}${x.reason ? `（${x.reason}）` : ""}`}</p>
              ))}
              {faithful?.dropped?.map((x: any, i: number) => (
                <p key={i} className="small muted" style={{ marginTop: 4 }}>删掉「{x.name}」：{x.reason}</p>
              ))}
            </section>
          )}

          <section className="stage">
            <h3>因子实验和研究层判定</h3>
            {grouped.skipped.map((e, i) => (
              <p key={i} className="small ink2" style={{ marginBottom: 8 }}>↺ 跳过「{e.hypothesis?.name}」：{String(e.reason).slice(0, 140)}</p>
            ))}
            {grouped.cands.length ? grouped.cands.map((c) => <CandidateView key={c.key} c={c} onJump={jump} />)
              : <p className="ink2">{s.state === "running" ? "还在准备候选因子…" : "没有进入实验的候选。"}</p>}
          </section>

          {d.strategy && (
            <section className="stage">
              <h3>策略对照（测试段 {d.strategy.segment}）</h3>
              <table className="data" style={{ marginTop: 4 }}>
                <thead><tr><th></th><th className="r">年化超额（自有回测）</th><th className="r">Qlib 回测</th></tr></thead>
                <tbody>
                  <tr><td>基础因子（{d.strategy.baseline_factors?.length}）</td><td className="r">{pct(d.strategy.baseline?.annual_excess)}</td>
                    <td className="r">{pct(d.strategy.qlib?.baseline?.annual_excess)}</td></tr>
                  {d.strategy.baseline_plus_new && (
                    <tr><td>加入本次通过的因子</td><td className="r">{pct(d.strategy.baseline_plus_new.annual_excess)}</td>
                      <td className="r">{pct(d.strategy.qlib?.baseline_plus_new?.annual_excess)}</td></tr>
                  )}
                </tbody>
              </table>
              {!d.strategy.baseline_plus_new && <p className="small muted" style={{ marginTop: 6 }}>本次没有通过的新因子，只有基线。</p>}
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
