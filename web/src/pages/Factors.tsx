import { Link } from "react-router-dom";
import { useApi } from "../api";
import { runTime, signed } from "../format";
import { ErrorState, Loading } from "../ui";

export default function Factors() {
  const { data, error, loading } = useApi<any>("/api/factors");
  const mon = useApi<any>("/api/monitor");
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>因子库</h1>
          <p>每日持仓用的因子：5 个基础因子，加上通过研究层判定、并且测试段 RankIC 为正的登记因子，等权合成。</p>
        </div>
      </div>
      {error ? <ErrorState error={error} what="因子库" /> : loading ? <Loading /> : (
        <>
          <section className="section">
            <div className="section-head"><h2>登记因子</h2><span className="note">{data.registered.length} 个</span></div>
            {data.registered.length ? (
              <div className="table-scroll">
                <table className="data">
                  <thead><tr><th>名称</th><th>实现</th><th className="r">验证段 RankIC</th><th className="r">测试段 RankIC</th><th className="r">测试段 t</th><th>来源</th><th>状态</th></tr></thead>
                  <tbody>
                    {data.registered.map((x: any) => (
                      <tr key={x.name}>
                        <td>{x.name}<div className="small muted">{x.hypothesis?.calculation}</div></td>
                        <td><code>{String(x.code).slice(0, 80)}</code></td>
                        <td className="r">{signed(x.metrics?.valid?.rank_ic)}</td>
                        <td className="r">{signed(x.test_metrics?.test?.rank_ic)}</td>
                        <td className="r">{x.test_metrics?.test?.rank_ic_t_nw ?? "—"}</td>
                        <td><Link to={`/runs/${x.run_id}`}>{runTime(x.run_id)}</Link></td>
                        <td>
                          {x.active === false ? <span className="verdict failed"><span className="ic" aria-hidden>✕</span>停用</span>
                            : <span className="verdict accepted"><span className="ic" aria-hidden>✓</span>启用</span>}
                          {x.active === false && x.history?.length > 0 && (
                            <div className="small muted" style={{ maxWidth: 240 }}>{x.history[x.history.length - 1].at.slice(0, 10)}：{x.history[x.history.length - 1].reason}</div>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty">
                <p>还没有登记的因子。</p>
                <p>到目前为止，研报里提出的候选都没能同时通过训练段和验证段的显著性门槛。<Link to="/new">再发起一次研究</Link>，或者到 <Link to="/runs">研究运行</Link> 看每个候选被淘汰的原因。</p>
              </div>
            )}
          </section>
          <section className="section">
            <div className="section-head">
              <h2>衰减监控</h2>
              <span className="note">{mon.data ? `${mon.data.date} 收盘后，窗口 ${mon.data.factors?.[0]?.window?.join(" 至 ") ?? "—"}` : ""}</span>
            </div>
            <p className="ink2" style={{ marginBottom: 12 }}>
              每日任务在选股之前，把每个因子放到最近 120 个已有标签的交易日上重新评估（中性化 RankIC、Newey-West t）。
              登记因子 RankIC ≤ 0 且 t ≤ −1 就自动停用，当天起不再进组合；重新启用要人工确认（<code>qharness factors --activate 名称</code>）。
              最近 60 天转负、或不到登记时测试段一半的只警告。基础因子只报告。
            </p>
            {!mon.data ? <p className="ink2">还没有监控结果。运行 <code>python -m qharness monitor</code>，或等每日任务自动跑。</p> : (
              <div className="table-scroll">
                <table className="data">
                  <thead><tr><th>因子</th><th>类型</th><th className="r">120 天 RankIC</th><th className="r">t</th><th className="r">60 天 RankIC</th><th>状态</th></tr></thead>
                  <tbody>
                    {mon.data.factors.map((r: any) => {
                      const st: Record<string, [string, string, string]> = {
                        ok: ["accepted", "✓", "正常"], warn: ["running", "!", "警告"], deactivate: ["failed", "✕", "停用"], unknown: ["skipped", "?", "数据不足"] };
                      const [cls, ic, label] = st[r.status] ?? ["skipped", "·", r.status];
                      return (
                        <tr key={r.name}>
                          <td><code>{r.name}</code></td>
                          <td className="ink2">{r.kind === "registered" ? "登记" : "基础"}</td>
                          <td className="r">{signed(r.long?.rank_ic)}</td>
                          <td className="r">{r.long?.rank_ic_t_nw ?? "—"}</td>
                          <td className="r">{signed(r.short?.rank_ic)}</td>
                          <td><span className={`verdict ${cls}`}><span className="ic" aria-hidden>{ic}</span>{label}</span>
                            {r.reasons?.length > 0 && <div className="small muted">{r.reasons.join("；")}</div>}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
          <section className="section">
            <div className="section-head"><h2>基础因子</h2><span className="note">新因子要和它们比增量；相关超过 0.7 直接判为重复</span></div>
            <table className="data">
              <tbody>
                {data.baseline.map((b: any) => (
                  <tr key={b.name}><td style={{ width: 200 }}><code>{b.name}</code></td><td>{b.desc}</td></tr>
                ))}
              </tbody>
            </table>
          </section>
        </>
      )}
    </div>
  );
}
