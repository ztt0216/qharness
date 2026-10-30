import { Link, useNavigate } from "react-router-dom";
import { useApi, type RunSummary } from "../api";
import { compact, duration, runTime, source } from "../format";
import { CountsLine, ErrorState, Loading, Verdict } from "../ui";

export default function Runs() {
  const { data, error, loading } = useApi<RunSummary[]>("/api/runs", (d) => (d?.some((r) => r.state === "running") ? 2000 : 8000));
  const nav = useNavigate();
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>研究运行</h1>
          <p>每次运行：研报 → 观点 → 因子假设 → 实验循环 → 研究层判定。点开一次运行，看每个因子为什么通过或被淘汰，出处能跳回研报原文。</p>
        </div>
        <Link to="/new" className="btn">发起研究</Link>
      </div>
      {error ? <ErrorState error={error} what="运行记录" /> : loading ? <Loading /> : !data?.length ? (
        <div className="empty"><p>还没有研究运行。</p><p><Link to="/new">选一份研报发起第一次研究</Link></p></div>
      ) : (
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr><th>时间</th><th>研报和主题</th><th className="r">候选</th><th>结果</th><th className="r">模型调用</th><th className="r">耗时</th><th>状态</th></tr>
            </thead>
            <tbody>
              {data.map((r) => (
                <tr key={r.id} className="link" onClick={() => nav(`/runs/${r.id}`)}>
                  <td className="num" style={{ whiteSpace: "nowrap" }}>{runTime(r.id)}</td>
                  <td>
                    <Link to={`/runs/${r.id}`} onClick={(e) => e.stopPropagation()}>{r.theme ?? "（未定位主题）"}</Link>
                    <div className="small muted">{source(r)}</div>
                  </td>
                  <td className="r">{r.n_candidates}</td>
                  <td><CountsLine counts={r.counts} total={r.n_candidates} /></td>
                  <td className="r">{r.llm.calls} 次<div className="small muted">{compact(r.llm.input_tokens + r.llm.output_tokens)} token</div></td>
                  <td className="r">{duration(r.started, r.ended)}</td>
                  <td><Verdict status={r.state} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
