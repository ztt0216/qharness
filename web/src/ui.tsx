import type { ReactNode } from "react";
import { STATUS } from "./format";
import type { Counts } from "./api";

export function Verdict({ status }: { status: string }) {
  const s = STATUS[status] ?? { label: status, icon: "·" };
  return <span className={`verdict ${status}`}><span className="ic" aria-hidden>{s.icon}</span>{s.label}</span>;
}

export function CountsLine({ counts, total }: { counts: Counts; total: number }) {
  const order = ["accepted", "rejected", "skipped", "failed"] as const;
  const parts = order.filter((k) => counts[k]).map((k) => `${STATUS[k].label} ${counts[k]}`);
  if (!parts.length) return <span className="muted">{total ? `${total} 个候选` : "没有候选"}</span>;
  return <span>{parts.join("，")}</span>;
}

export function Loading() {
  return <div className="empty">正在读取…</div>;
}

export function ErrorState({ error, what }: { error: Error; what: string }) {
  return (
    <div className="error-box">
      <p>读取{what}失败：{error.message}</p>
      <p className="small ink2" style={{ marginTop: 4 }}>确认后端在运行（<code>python -m qharness serve</code>），然后刷新页面。</p>
    </div>
  );
}

export function Stat({ label, value, sub, className }: { label: string; value: ReactNode; sub?: ReactNode; className?: string }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className={`value ${className ?? ""}`}>{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}
