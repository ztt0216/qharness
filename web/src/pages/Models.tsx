import { GroupedBars } from "../charts";
import { useApi } from "../api";
import { pct, signed, tone } from "../format";
import { ErrorState, Loading } from "../ui";

const shortRange = (r: [string, string]) => `${r[0].slice(2, 7).replace("-", ".")}–${r[1].slice(5, 7)}月`;

export default function Models() {
  const { data, error, loading } = useApi<any>("/api/models");
  if (error) return <div className="page"><ErrorState error={error} what="模型结果" /></div>;
  if (loading || !data) return <div className="page"><Loading /></div>;
  const wf = data.items.find((m: any) => m.kind === "walk_forward");
  const single = data.items.find((m: any) => m.kind === "single");

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>模型</h1>
          <p>Qlib 的 Alpha158（158 个量价特征）+ 梯度提升树，预测和 Harness 同一个标签（T+1 到 T+6 的收益）。评估口径和因子一样：市值行业中性化后的 RankIC，Newey-West t。</p>
        </div>
      </div>

      {wf ? (
        <section className="section">
          <div className="section-head">
            <h2>滚动训练：每个季度重新训练，只预测下一个季度</h2>
            <span className="note">{wf.n_folds} 折，{wf.oos.dates[0]} 至 {wf.oos.dates[1]}</span>
          </div>
          <p className="ink2" style={{ marginBottom: 18 }}>
            样本外整体 RankIC：模型 {signed(wf.oos.model.rank_ic)}（t {wf.oos.model.rank_ic_t_nw}），基础因子等权 {signed(wf.oos.baseline.rank_ic)}（t {wf.oos.baseline.rank_ic_t_nw}）。
            {wf.folds_positive_rank_ic} 折为正，{wf.folds_beat_baseline} 折跑赢基线。扣费后年化超额中位数：模型 {pct(wf.backtest_by_phase.model.median)}，基线 {pct(wf.backtest_by_phase.baseline.median)}。
            {wf.oos.model.rank_ic > 0 && wf.backtest_by_phase.model.median < 0 ? "排序能力有，但不够覆盖 5 天一调仓的成本。" : ""}
          </p>
          <div className="panel panel-pad">
            <GroupedBars
              ariaLabel="每一折预测区间的中性化 RankIC，模型和基础因子对比"
              categories={wf.folds.map((f: any) => f.predict[0])}
              categoryLabel={(c) => { const f = wf.folds.find((x: any) => x.predict[0] === c); return shortRange(f.predict); }}
              series={[
                { key: "m", label: "模型", color: "var(--series-model)", values: wf.folds.map((f: any) => f.model.rank_ic) },
                { key: "b", label: "5 个基础因子等权", color: "var(--series-base)", values: wf.folds.map((f: any) => f.baseline.rank_ic) },
              ]}
              yFormat={(v) => v.toFixed(2)}
              valueFormat={(v) => signed(v)}
              tooltipExtra={(i) => {
                const f = wf.folds[i];
                return <div className="tl" style={{ marginTop: 4 }}>模型 t {f.model.rank_ic_t_nw}，基线 t {f.baseline.rank_ic_t_nw}；训练到 {f.train[1]}</div>;
              }}
            />
          </div>
          <div className="table-scroll" style={{ marginTop: 18 }}>
            <table className="data">
              <thead>
                <tr><th>预测区间</th><th>训练数据截止</th><th className="r">树的数量</th><th className="r">模型 RankIC</th><th className="r">t</th>
                  <th className="r">基线 RankIC</th><th className="r">t</th><th className="r">模型超额</th><th className="r">基线超额</th></tr>
              </thead>
              <tbody>
                {wf.folds.map((f: any) => (
                  <tr key={f.predict[0]}>
                    <td className="num">{f.predict[0]} 至 {f.predict[1]}</td>
                    <td className="num ink2">{f.train[1]}</td>
                    <td className="r">{f.best_iter}</td>
                    <td className="r">{signed(f.model.rank_ic)}</td><td className="r">{f.model.rank_ic_t_nw}</td>
                    <td className="r">{signed(f.baseline.rank_ic)}</td><td className="r">{f.baseline.rank_ic_t_nw}</td>
                    <td className={`r ${tone(f.excess_model)}`}>{pct(f.excess_model)}</td>
                    <td className={`r ${tone(f.excess_baseline)}`}>{pct(f.excess_baseline)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small muted" style={{ marginTop: 10 }}>
            超额是单个季度年化后的数，波动很大。整体超额按 5 个不同的调仓起点各回测一遍：模型 {pct(wf.backtest_by_phase.model.min)} 至 {pct(wf.backtest_by_phase.model.max)}，
            基线 {pct(wf.backtest_by_phase.baseline.min)} 至 {pct(wf.backtest_by_phase.baseline.max)}。
          </p>
          {wf.by_segment && (
            <p className="ink2" style={{ marginTop: 14 }}>
              按原来的切分看：原验证段期间（{wf.by_segment.valid?.dates.join(" 至 ")}）模型 RankIC {signed(wf.by_segment.valid?.model.rank_ic)}{Math.abs(wf.by_segment.valid?.model.rank_ic ?? 1) < 0.01 ? "，基本无效" : ""}；
              原测试段期间 {signed(wf.by_segment.test?.model.rank_ic)}（t {wf.by_segment.test?.model.rank_ic_t_nw}）。
            </p>
          )}
        </section>
      ) : <p className="ink2">还没有滚动训练结果。运行 <code>python -m qharness qlib-train --walk-forward</code>。</p>}

      {single && (
        <section className="section">
          <div className="section-head"><h2>单次切分</h2><span className="note">测试段 {single.segments.test.join(" 至 ")}</span></div>
          <table className="data">
            <thead><tr><th></th><th className="r">测试段 RankIC</th><th className="r">t</th><th className="r">年化超额（自有回测）</th><th className="r">5 个调仓起点</th><th className="r">Qlib 回测</th></tr></thead>
            <tbody>
              <tr><td>模型</td><td className="r">{signed(single.test.rank_ic)}</td><td className="r">{single.test.rank_ic_t_nw}</td>
                <td className={`r ${tone(single.backtest.model.annual_excess)}`}>{pct(single.backtest.model.annual_excess)}</td>
                <td className="r">{pct(single.backtest_by_phase.model.min)} 至 {pct(single.backtest_by_phase.model.max)}</td>
                <td className={`r ${tone(single.backtest_qlib?.model?.annual_excess)}`}>{pct(single.backtest_qlib?.model?.annual_excess)}</td></tr>
              <tr><td>5 个基础因子等权</td><td className="r">{signed(single.baseline_test.rank_ic)}</td><td className="r">{single.baseline_test.rank_ic_t_nw}</td>
                <td className={`r ${tone(single.backtest.baseline.annual_excess)}`}>{pct(single.backtest.baseline.annual_excess)}</td>
                <td className="r">{pct(single.backtest_by_phase.baseline.min)} 至 {pct(single.backtest_by_phase.baseline.max)}</td>
                <td className={`r ${tone(single.backtest_qlib?.baseline?.annual_excess)}`}>{pct(single.backtest_qlib?.baseline?.annual_excess)}</td></tr>
            </tbody>
          </table>
          <p className="small muted" style={{ marginTop: 10 }}>
            单次切分只有一个不到一年的测试段，结论以覆盖更长时间的滚动训练为准。
          </p>
        </section>
      )}

      <section className="section">
        <h2>为什么页面上没有「重新训练」按钮</h2>
        <p className="ink2" style={{ marginTop: 10 }}>
          同一配置的测试段只能打开一次（已记录 {data.test_opened} 次）。在页面上反复点训练，就等于在测试段上挑结果。
          要换配置，在命令行运行 <code>python -m qharness qlib-train --walk-forward --train-days 500</code> 之类的命令，每次都会留下记录。
        </p>
      </section>
    </div>
  );
}
