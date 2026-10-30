import { useState } from "react";
import { LineChart } from "../charts";
import { post, useApi, type Job } from "../api";
import { money, pct, share, signedMoney, tone } from "../format";
import { ErrorState, Loading, Stat } from "../ui";

const ACTION: Record<string, string> = { buy: "买入", sell: "卖出", hold: "持有" };

export default function Paper() {
  const [jobId, setJobId] = useState<string>();
  const [err, setErr] = useState<string>();
  const job = useApi<Job>(jobId ? `/api/jobs/${jobId}` : null, (d) => (!d || d.status === "running" ? 2000 : undefined));
  const running = job.data?.status === "running";
  const paper = useApi<any>("/api/paper", running ? 4000 : undefined);

  const settle = async () => {
    setErr(undefined);
    try {
      const j = await post<Job>("/api/jobs/daily");
      setJobId(j.id);
    } catch (e) {
      setErr((e as Error).message);
    }
  };

  if (paper.error) return <div className="page"><ErrorState error={paper.error} what="模拟账户" /></div>;
  if (paper.loading || !paper.data) return <div className="page"><Loading /></div>;
  const { account: acct, positions, trades, target, replay } = paper.data;
  const navs: { date: string; nav: number }[] = acct?.nav ?? [];
  const last = navs[navs.length - 1], prev = navs[navs.length - 2];
  const posValue = positions.reduce((a: number, p: any) => a + (p.value ?? 0), 0);
  const pnl = positions.reduce((a: number, p: any) => a + (p.pnl ?? 0), 0);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>模拟账户</h1>
          <p>每个交易日收盘后按因子库选 20 只、等权，5 个交易日调一次仓。当天收盘挂单，下一个交易日开盘成交；整手、佣金万 2.5（最低 5 元）、卖出印花税千 0.5、滑点千 1，停牌和涨跌停不成交。</p>
        </div>
        <button className="btn btn-primary" onClick={settle} disabled={running}>{running ? "正在更新…" : "更新行情并结算"}</button>
      </div>

      {err && <div className="error-box" style={{ marginBottom: 20 }}>{err}</div>}
      {job.data && (
        <details open={job.data.status !== "done"} style={{ marginBottom: 24 }}>
          <summary className="small ink2" style={{ cursor: "pointer" }}>
            {job.data.status === "running" ? "正在拉取最新行情、结算挂单…" : job.data.status === "done" ? "已更新，结果见下方" : `更新失败（退出码 ${job.data.returncode}），日志如下`}
          </summary>
          <pre className="log code">{job.data.log?.slice(-30).join("\n")}</pre>
        </details>
      )}

      {!acct ? (
        <div className="empty"><p>还没有模拟账户。点「更新行情并结算」会以 100 万初始资金开户，当天收盘挂第一批单。</p></div>
      ) : (
        <>
          <div className="stats">
            <Stat label={`${last?.date ?? ""} 净值`} value={<span className="num">{money(last?.nav)}</span>}
              sub={<span className={tone(last && prev ? last.nav - prev.nav : null)}>当日 {last && prev ? signedMoney(last.nav - prev.nav) : "—"}</span>} />
            <Stat label="累计收益" className={tone(last ? last.nav / acct.initial - 1 : null)} value={pct(last ? last.nav / acct.initial - 1 : null, 2)} sub={`初始 ${money(acct.initial)}`} />
            <Stat label="持仓市值" value={money(posValue)} sub={`${positions.length} 只，浮动盈亏 ${signedMoney(pnl)}`} />
            <Stat label="现金" value={money(acct.cash)} sub={acct.pending ? "有待成交的挂单" : `上次调仓 ${acct.last_rebalance ?? "—"}`} />
          </div>

          <section className="section">
            <div className="section-head"><h2>持仓</h2><span className="note">按不复权收盘价计市值</span></div>
            <div className="table-scroll">
              <table className="data">
                <thead><tr><th>股票</th><th className="r">股数</th><th className="r">成本价</th><th className="r">收盘价</th><th className="r">当日涨跌</th><th className="r">市值</th><th className="r">权重</th><th className="r">浮动盈亏</th></tr></thead>
                <tbody>
                  {positions.map((p: any) => (
                    <tr key={p.instrument}>
                      <td>{p.name ?? p.instrument}<div className="small muted">{p.instrument}</div></td>
                      <td className="r">{money(p.shares)}</td>
                      <td className="r">{p.avg_cost?.toFixed(3) ?? "—"}</td>
                      <td className="r">{p.close?.toFixed(2) ?? "—"}</td>
                      <td className={`r ${tone(p.day_change)}`}>{pct(p.day_change, 2)}</td>
                      <td className="r">{money(p.value)}</td>
                      <td className="r">{share(p.weight)}</td>
                      <td className={`r ${tone(p.pnl)}`}>{signedMoney(p.pnl)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}

      {target.rows.length > 0 && (
        <section className="section">
          <div className="section-head"><h2>{target.date} 收盘后的目标持仓</h2><span className="note">下一交易日执行；到调仓日才会挂单</span></div>
          <div className="table-scroll">
            <table className="data">
              <thead><tr><th className="r">排名</th><th>股票</th><th>行业</th><th className="r">合成分数</th><th>操作</th><th>备注</th></tr></thead>
              <tbody>
                {target.rows.map((r: any) => (
                  <tr key={r.instrument}>
                    <td className="r">{r.rank > 0 ? r.rank : "—"}</td>
                    <td>{r.name ?? r.instrument}<span className="small muted" style={{ marginLeft: 8 }}>{r.instrument}</span></td>
                    <td className="ink2">{r.industry}</td>
                    <td className="r">{r.score?.toFixed?.(4) ?? "—"}</td>
                    <td>{ACTION[r.action] ?? r.action}</td>
                    <td className="small ink2">{r.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {replay.nav.length > 0 && (
        <section className="section">
          <div className="section-head">
            <h2>历史回放：同一套规则在测试段上逐日模拟</h2>
            <span className="note">{replay.stats?.start} 至 {replay.stats?.end}</span>
          </div>
          <p className="ink2" style={{ marginBottom: 14 }}>
            总收益 {pct(replay.stats?.total_return)}，相对股票池等权年化超额 {pct(replay.stats?.annual_excess)}，最大回撤 {pct(replay.stats?.max_drawdown)}，
            成交 {replay.stats?.trades_filled} 笔，费用 {money(replay.stats?.fees)}。同期回测的年化超额是 {pct(replay.stats?.backtest_same_period?.annual_excess)}，可以对照模拟撮合和回测的差别。
          </p>
          <div className="panel panel-pad">
            <LineChart
              ariaLabel="模拟账户历史回放净值"
              series={[{ key: "nav", label: "净值", color: "var(--series-model)", points: replay.nav.map((r: any) => ({ x: r.date, y: r.nav })) }]}
              yFormat={(v) => `${(v / 10000).toFixed(0)} 万`}
              xFormat={(x) => x.slice(5)}
              tooltipValue={(_, y) => money(y)}
            />
          </div>
        </section>
      )}

      {trades.length > 0 && (
        <section className="section">
          <div className="section-head"><h2>成交记录</h2><span className="note">{trades.length} 笔</span></div>
          <div className="table-scroll">
            <table className="data">
              <thead><tr><th>日期</th><th>股票</th><th>方向</th><th className="r">股数</th><th className="r">成交价</th><th className="r">金额</th><th className="r">费用</th><th>状态</th></tr></thead>
              <tbody>
                {trades.slice(0, 100).map((t: any, i: number) => (
                  <tr key={i}>
                    <td className="num">{t.date}</td>
                    <td>{t.name ?? t.instrument}<span className="small muted" style={{ marginLeft: 8 }}>{t.instrument}</span></td>
                    <td>{t.side === "buy" ? "买入" : "卖出"}</td>
                    <td className="r">{money(t.shares)}</td>
                    <td className="r">{t.price?.toFixed(2)}</td>
                    <td className="r">{money(t.value)}</td>
                    <td className="r">{t.fee?.toFixed(2)}</td>
                    <td className="small">{t.status === "filled" ? "已成交" : `未成交${t.reason ? `：${t.reason}` : ""}`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
