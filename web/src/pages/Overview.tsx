import { Link, useNavigate } from "react-router-dom";
import { useApi, type RunSummary } from "../api";
import { money, pct, runTime, signed, signedMoney, tone } from "../format";
import { CountsLine, ErrorState, Loading, Verdict } from "../ui";

export default function Overview() {
  const runs = useApi<RunSummary[]>("/api/runs", 5000);
  const paper = useApi<any>("/api/paper");
  const models = useApi<any>("/api/models");
  const factors = useApi<any>("/api/factors");
  const nav = useNavigate();

  const all = runs.data ?? [];
  const cands = all.reduce((a, r) => a + r.n_candidates, 0);
  const accepted = all.reduce((a, r) => a + (r.counts.accepted ?? 0), 0);
  const nReg = factors.data?.registered.length ?? 0;

  const acct = paper.data?.account;
  const navs: { date: string; nav: number }[] = acct?.nav ?? [];
  const last = navs[navs.length - 1], prev = navs[navs.length - 2];
  const wf = models.data?.items.find((m: any) => m.kind === "walk_forward");

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>总览</h1>
          {runs.data && (
            <p>
              {all.length} 次研究运行，共检验 {cands} 个候选因子，{accepted} 个通过研究层判定，因子库登记 {nReg} 个。
              判定标准在看到结果之前就定好，大部分候选被淘汰是预期内的。
            </p>
          )}
        </div>
      </div>

      <div className="grid-2">
        <section>
          <div className="section-head">
            <h2>最近的研究</h2>
            <Link to="/runs" className="small">全部 {all.length} 次</Link>
          </div>
          {runs.error ? <ErrorState error={runs.error} what="运行记录" /> : runs.loading ? <Loading /> : !all.length ? (
            <div className="empty"><p>还没有研究运行。</p><p><Link to="/new">选一份研报发起第一次研究</Link></p></div>
          ) : (
            <table className="data">
              <thead><tr><th>时间</th><th>主题</th><th>结果</th><th>状态</th></tr></thead>
              <tbody>
                {all.slice(0, 8).map((r) => (
                  <tr key={r.id} className="link" onClick={() => nav(`/runs/${r.id}`)}>
                    <td className="num nowrap" style={{ whiteSpace: "nowrap" }}>{runTime(r.id)}</td>
                    <td>
                      <Link to={`/runs/${r.id}`} onClick={(e) => e.stopPropagation()}>{r.theme ?? r.report ?? r.hypothesis_file ?? "（未定位主题）"}</Link>
                      {r.report && r.theme && <div className="small muted">{r.report}</div>}
                    </td>
                    <td><CountsLine counts={r.counts} total={r.n_candidates} /></td>
                    <td><Verdict status={r.state} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        <div>
          <section>
            <div className="section-head"><h2>模拟账户</h2><Link to="/paper" className="small">持仓和成交</Link></div>
            {paper.error ? <ErrorState error={paper.error} what="模拟账户" /> : !acct ? (
              <div className="empty"><p>还没有模拟账户。在「模拟账户」页更新行情并结算后会自动开户（初始 100 万）。</p></div>
            ) : (
              <div className="stats">
                <div className="stat">
                  <div className="label">{last?.date} 净值</div>
                  <div className="value num">{money(last?.nav)}</div>
                  <div className={`sub ${tone(last && prev ? last.nav - prev.nav : null)}`}>
                    当日 {last && prev ? signedMoney(last.nav - prev.nav) : "—"}
                  </div>
                </div>
                <div className="stat">
                  <div className="label">累计收益</div>
                  <div className={`value num ${tone(last ? last.nav / acct.initial - 1 : null)}`}>{pct(last ? last.nav / acct.initial - 1 : null, 2)}</div>
                  <div className="sub">{Object.keys(acct.positions).length} 只持仓，运行 {navs.length} 个交易日</div>
                </div>
              </div>
            )}
          </section>

          <section className="section">
            <div className="section-head"><h2>模型</h2><Link to="/models" className="small">逐折结果</Link></div>
            {wf ? (
              <>
                <p className="ink2">
                  Alpha158 + 梯度提升树按季度滚动训练 {wf.n_folds} 折，样本外 {wf.oos.dates[0]} 至 {wf.oos.dates[1]}。
                </p>
                <table className="data" style={{ marginTop: 10 }}>
                  <thead><tr><th></th><th className="r">样本外 RankIC</th><th className="r">t</th><th className="r">扣费超额（中位数）</th></tr></thead>
                  <tbody>
                    <tr><td>模型</td><td className="r">{signed(wf.oos.model.rank_ic)}</td><td className="r">{wf.oos.model.rank_ic_t_nw}</td>
                      <td className={`r ${tone(wf.backtest_by_phase.model.median)}`}>{pct(wf.backtest_by_phase.model.median)}</td></tr>
                    <tr><td>5 个基础因子等权</td><td className="r">{signed(wf.oos.baseline.rank_ic)}</td><td className="r">{wf.oos.baseline.rank_ic_t_nw}</td>
                      <td className={`r ${tone(wf.backtest_by_phase.baseline.median)}`}>{pct(wf.backtest_by_phase.baseline.median)}</td></tr>
                  </tbody>
                </table>
              </>
            ) : <p className="ink2">还没有滚动训练结果。运行 <code>python -m qharness qlib-train --walk-forward</code>。</p>}
          </section>

          <section className="section">
            <div className="section-head"><h2>因子库</h2><Link to="/factors" className="small">查看</Link></div>
            <p className="ink2">
              {nReg ? `登记 ${nReg} 个因子，` : "还没有登记的因子（登记要求通过研究层判定，并且测试段 RankIC 为正）。"}
              组合里一直在用 5 个基础因子。
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
