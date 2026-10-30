// 自动搜集的研报：东方财富策略类研报 → 标题筛选 → 下载 → 预筛 → 研究。每篇的状态和原因都在这里。
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { post, useApi, type Job, type RunSummary } from "../api";
import { runTime } from "../format";
import { CountsLine, ErrorState, Loading } from "../ui";

interface Item {
  code: string; title: string; org: string; date: string; pages: number; status: string; reason?: string;
  stage?: string; path?: string; run_id?: string; screen?: { type?: string; factors?: { name: string; definition?: string; data?: string }[] };
  runs: RunSummary[];
}

const STATUS: Record<string, { label: string; icon: string; cls: string }> = {
  listed: { label: "待下载", icon: "·", cls: "skipped" },
  downloaded: { label: "待预筛", icon: "·", cls: "skipped" },
  screened_out: { label: "筛掉", icon: "–", cls: "rejected" },
  queued: { label: "可研究", icon: "✓", cls: "accepted" },
  researching: { label: "研究中", icon: "…", cls: "running" },
  researched: { label: "已研究", icon: "✓", cls: "rejected" },
};

const STAGE: Record<string, string> = { program: "程序", model: "模型", download: "下载" };

export default function Reports() {
  const [jobId, setJobId] = useState<string>();
  const [err, setErr] = useState<string>();
  const [days, setDays] = useState(30);
  const job = useApi<Job>(jobId ? `/api/jobs/${jobId}` : null, (d) => (!d || d.status === "running" ? 2000 : undefined));
  const running = job.data?.status === "running";
  const data = useApi<{ items: Item[]; counts: Record<string, number> }>("/api/reports", running ? 3000 : undefined);
  const nav = useNavigate();

  const fetchNew = async () => {
    setErr(undefined);
    try { setJobId((await post<Job>("/api/jobs/reports", { days, download: 10 })).id); } catch (e) { setErr((e as Error).message); }
  };
  const study = async (it: Item) => {
    setErr(undefined);
    try {
      const j = await post<Job>("/api/jobs/run", { report: `auto/${it.path}`, workers: 2 });
      nav(`/runs/${j.run_id}`);
    } catch (e) { setErr((e as Error).message); }
  };

  const items = data.data?.items ?? [];
  const c = data.data?.counts ?? {};
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>研报来源</h1>
          <p>
            自动从东方财富研报中心的策略类研报里找（金工研报归在这一类）：先按标题筛，下载 PDF 后由程序检查是否依赖日内数据，
            再用一次模型调用判断是不是截面选股因子、能不能只用日线验证。通过的才值得花 2~7 万 token 跑完整研究。
          </p>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexShrink: 0 }}>
          <label className="small ink2" htmlFor="days">往前找</label>
          <select id="days" value={days} onChange={(e) => setDays(+e.target.value)}>
            <option value={7}>7 天</option><option value={30}>30 天</option><option value={90}>90 天</option><option value={365}>一年</option>
          </select>
          <button className="btn btn-primary" onClick={fetchNew} disabled={running}>{running ? "正在搜集…" : "搜集新研报"}</button>
        </div>
      </div>

      {err && <div className="error-box" style={{ marginBottom: 20 }}>{err}</div>}
      {job.data && (
        <details open={job.data.status !== "done"} style={{ marginBottom: 24 }}>
          <summary className="small ink2" style={{ cursor: "pointer" }}>
            {job.data.status === "running" ? "正在拉列表、下载、预筛…" : job.data.status === "done" ? "搜集完成，结果见下表" : `搜集失败（退出码 ${job.data.returncode}）`}
          </summary>
          <pre className="log code">{job.data.log?.slice(-30).join("\n")}</pre>
        </details>
      )}

      {data.error ? <ErrorState error={data.error} what="研报索引" /> : data.loading ? <Loading /> : !items.length ? (
        <div className="empty">
          <p>还没有搜集过研报。</p>
          <p>点「搜集新研报」，或者在命令行运行 <code>python -m qharness reports fetch --days 30</code>。</p>
        </div>
      ) : (
        <>
          <p className="ink2" style={{ marginBottom: 14 }}>
            索引里 {items.length} 篇（都是标题通过的）：可研究 {c.queued ?? 0}，已研究 {c.researched ?? 0}，筛掉 {c.screened_out ?? 0}，
            待处理 {(c.listed ?? 0) + (c.downloaded ?? 0)}。
          </p>
          <div className="table-scroll">
            <table className="data">
              <thead><tr><th>发布</th><th>研报</th><th>状态</th><th>原因</th><th></th></tr></thead>
              <tbody>
                {items.map((it) => {
                  const s = STATUS[it.status] ?? { label: it.status, icon: "·", cls: "skipped" };
                  const run = it.runs[0];
                  return (
                    <tr key={it.code}>
                      <td className="num" style={{ whiteSpace: "nowrap" }}>{it.date}</td>
                      <td style={{ minWidth: 260 }}>
                        {it.title}
                        <div className="small muted">{it.org}，{it.pages} 页</div>
                      </td>
                      <td><span className={`verdict ${s.cls}`}><span className="ic" aria-hidden>{s.icon}</span>{s.label}</span>
                        {it.stage && it.status === "screened_out" && <div className="small muted">{STAGE[it.stage] ?? it.stage}判断</div>}
                      </td>
                      <td className="small" style={{ maxWidth: 420 }}>
                        {it.reason}
                        {it.screen?.factors?.length ? (
                          <div className="muted" style={{ marginTop: 4 }}>
                            {it.screen.factors.slice(0, 6).map((f) => `${f.name}（${f.data ?? "?"}）`).join("、")}
                          </div>
                        ) : null}
                      </td>
                      <td style={{ whiteSpace: "nowrap" }}>
                        {run ? (
                          <Link to={`/runs/${run.id}`}>{runTime(run.id)} 的研究<div className="small muted"><CountsLine counts={run.counts} total={run.n_candidates} /></div></Link>
                        ) : it.run_id ? <Link to={`/runs/${it.run_id}`}>查看研究</Link>
                          : (it.status === "queued" || (it.status === "screened_out" && it.path)) ? (
                            <button className="btn" onClick={() => study(it)}>{it.status === "queued" ? "研究这篇" : "仍然研究"}</button>
                          ) : null}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
