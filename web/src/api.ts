// 后端接口（qharness/web.py）。字段大多直接来自运行记录的 JSON，类型只写前端用到的部分。
import { useCallback, useEffect, useRef, useState } from "react";

export type Counts = Partial<Record<"accepted" | "rejected" | "skipped" | "failed", number>>;

export interface RunSummary {
  id: string;
  state: "running" | "finished" | "interrupted";
  report: string | null;
  has_report: boolean;
  hypothesis_file: string | null;
  theme: string | null;
  model: string | null;
  started: number | null;
  ended: number | null;
  n_candidates: number;
  counts: Counts;
  llm: { calls: number; input_tokens: number; output_tokens: number; latency_s: number };
}

export interface Citation { section: string; page: number }
export interface Hypothesis {
  name: string; fields: string[]; calculation: string; window: number; direction: number;
  rationale?: string; condition?: string | null; viewpoint_id?: number; citations?: Citation[];
}
export interface TraceEvent { node: string; ts: number; step: number; cand?: string; [k: string]: any }

export interface RunDetail {
  summary: RunSummary;
  args: Record<string, any>;
  report: string | null;
  trace: TraceEvent[];
  results: any[] | null;
  strategy: any | null;
  llm_by_role: Record<string, { calls: number; input_tokens: number; output_tokens: number; latency_s: number }>;
}

export interface Job {
  id: string; kind: string; status: "running" | "done" | "failed"; started: number; ended: number | null;
  returncode: number | null; commands: string[]; run_id?: string; log?: string[];
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) { super(message); this.status = status; }
}

async function parse(r: Response) {
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).detail ?? msg; } catch { /* 不是 JSON */ }
    throw new ApiError(r.status, msg);
  }
  return r.json();
}

export const get = <T,>(url: string): Promise<T> => fetch(url).then(parse);
export const post = <T,>(url: string, body?: unknown): Promise<T> =>
  fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body) }).then(parse);
export const upload = (file: File) => {
  const fd = new FormData();
  fd.append("file", file);
  return fetch("/api/uploads", { method: "POST", body: fd }).then(parse);
};

/** 取数；pollMs 给了就定时刷新（刷新时保留上一次的数据，不闪）。 */
export function useApi<T>(url: string | null, pollMs?: number | ((d: T | undefined) => number | undefined)) {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<ApiError | Error>();
  const [loading, setLoading] = useState(true);
  const dataRef = useRef<T | undefined>(undefined);
  const load = useCallback(async () => {
    if (!url) return;
    try {
      const d = await get<T>(url);
      dataRef.current = d;
      setData(d);
      setError(undefined);
    } catch (e) {
      setError(e as Error);
    } finally {
      setLoading(false);
    }
  }, [url]);
  useEffect(() => {
    setLoading(true);
    load();
  }, [load]);
  useEffect(() => {
    if (!pollMs) return;
    let t: number | undefined;
    const tick = () => {
      const ms = typeof pollMs === "function" ? pollMs(dataRef.current) : pollMs;
      if (!ms) return;
      t = window.setTimeout(async () => { await load(); tick(); }, ms);
    };
    tick();
    return () => window.clearTimeout(t);
  }, [load, pollMs]);
  return { data, error, loading, reload: load };
}
