import { Link } from "react-router-dom";
import { useApi } from "../api";
import { runTime, signed } from "../format";
import { ErrorState, Loading } from "../ui";

export default function Factors() {
  const { data, error, loading } = useApi<any>("/api/factors");
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
                        <td>{x.active === false ? "停用" : "启用"}</td>
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
