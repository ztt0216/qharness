import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { post, upload, useApi, type Job } from "../api";
import { ErrorState, Loading } from "../ui";

interface Option { value: string; label: string; size: number }

const kb = (n: number) => (n > 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);

export default function NewRun() {
  const samples = useApi<{ samples: Option[]; uploads: Option[] }>("/api/samples");
  const [choice, setChoice] = useState<string>("");
  const [workers, setWorkers] = useState(2);
  const [noStrategy, setNoStrategy] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string>();
  const nav = useNavigate();

  const onFile = async (f: File | undefined) => {
    if (!f) return;
    setErr(undefined);
    setBusy(true);
    try {
      const o: Option = await upload(f);
      await samples.reload();
      setChoice(o.value);
    } catch (e) {
      setErr(`上传失败：${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const start = async () => {
    setErr(undefined);
    setBusy(true);
    try {
      const job = await post<Job>("/api/jobs/run", { report: choice, workers, no_strategy: noStrategy });
      nav(`/runs/${job.run_id}`);
    } catch (e) {
      setErr(`没能发起：${(e as Error).message}`);
      setBusy(false);
    }
  };

  const opts = [...(samples.data?.uploads ?? []), ...(samples.data?.samples ?? [])];
  return (
    <div className="page"><div style={{ maxWidth: 820 }}>
      <div className="page-head">
        <div>
          <h1>发起研究</h1>
          <p>选一份研报，Harness 会抽取观点、形成因子假设、写代码、在沪深 300 行情上检验，最后给出每个因子的判定。示例研报一次大约 1 分钟、十几次模型调用。</p>
        </div>
      </div>

      {samples.error ? <ErrorState error={samples.error} what="研报列表" /> : samples.loading ? <Loading /> : (
        <>
          <div className="field">
            <div className="label">研报</div>
            <div className="choice-list" role="radiogroup">
              {opts.map((o) => (
                <label key={o.value} className="choice">
                  <input type="radio" name="report" value={o.value} checked={choice === o.value} onChange={() => setChoice(o.value)} />
                  <span style={{ flex: 1 }}>{o.label}</span>
                  <span className="small muted">{o.value.startsWith("uploads/") ? "上传的" : "示例"}，{kb(o.size)}</span>
                </label>
              ))}
            </div>
          </div>
          <div className="field">
            <label htmlFor="file">上传新研报</label>
            <input id="file" type="file" accept=".pdf,.pptx,.md,.txt" disabled={busy} onChange={(e) => onFile(e.target.files?.[0])} />
            <p className="small muted" style={{ marginTop: 6 }}>支持 PDF、PPT（.pptx）、Markdown 和纯文本，30 MB 以内。</p>
          </div>
          <div className="field">
            <label htmlFor="workers">并行候选数</label>
            <select id="workers" value={workers} onChange={(e) => setWorkers(+e.target.value)}>
              <option value={1}>1（串行，后面的候选能用上前面刚沉淀的经验）</option>
              <option value={2}>2</option>
              <option value={4}>4</option>
            </select>
          </div>
          <div className="field">
            <label style={{ fontWeight: 400, display: "flex", gap: 8, alignItems: "center" }}>
              <input type="checkbox" checked={noStrategy} onChange={(e) => setNoStrategy(e.target.checked)} />
              跳过策略对照（不回测，快一些）
            </label>
          </div>
          {err && <div className="error-box" style={{ marginTop: 20 }}>{err}</div>}
          <div style={{ marginTop: 28, display: "flex", gap: 12, alignItems: "center" }}>
            <button className="btn btn-primary" disabled={!choice || busy} onClick={start}>开始研究</button>
            {!choice && <span className="small muted">先选一份研报</span>}
          </div>
        </>
      )}
    </div></div>
  );
}
