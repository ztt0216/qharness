// 手写 SVG 图表（按 dataviz 规范：2px 线、≤24px 柱、4px 圆角数据端、2px 间隔、细网格、悬停提示）。
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(720);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

function niceTicks(lo: number, hi: number, n = 4) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const raw = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.floor(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export interface LineSeries { key: string; label: string; color: string; points: { x: string; y: number }[] }

export function LineChart({ series, height = 260, yFormat = (v: number) => `${v}`, xFormat = (x: string) => x,
  tooltipValue, ariaLabel }: {
  series: LineSeries[]; height?: number; yFormat?: (v: number) => string; xFormat?: (x: string) => string;
  tooltipValue?: (s: LineSeries, y: number) => ReactNode; ariaLabel: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const xs = useMemo(() => Array.from(new Set(series.flatMap((s) => s.points.map((p) => p.x)))).sort(), [series]);
  const ys = series.flatMap((s) => s.points.map((p) => p.y));
  const m = { l: 64, r: 16, t: 10, b: 28 };
  const iw = width - m.l - m.r, ih = height - m.t - m.b;
  if (!xs.length) return <div ref={ref} className="empty">没有数据</div>;
  const ticks = niceTicks(Math.min(...ys), Math.max(...ys));
  const lo = ticks[0], hi = ticks[ticks.length - 1];
  const X = (i: number) => m.l + (xs.length === 1 ? iw / 2 : (i / (xs.length - 1)) * iw);
  const Y = (v: number) => m.t + ih - ((v - lo) / (hi - lo)) * ih;
  const idx = new Map(xs.map((x, i) => [x, i]));
  const xTickEvery = Math.max(1, Math.ceil(xs.length / Math.max(2, Math.floor(iw / 110))));

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const box = (e.currentTarget as SVGRectElement).getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * iw;
    setHover(Math.max(0, Math.min(xs.length - 1, Math.round((px / iw) * (xs.length - 1)))));
  };
  const hx = hover != null ? xs[hover] : null;

  return (
    <div className="chart" ref={ref}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={ariaLabel}>
        {ticks.map((t) => (
          <g key={t}>
            <line className="grid" x1={m.l} x2={width - m.r} y1={Y(t)} y2={Y(t)} />
            <text className="tick" x={m.l - 8} y={Y(t) + 4} textAnchor="end">{yFormat(t)}</text>
          </g>
        ))}
        {xs.map((x, i) => (i % xTickEvery === 0 || i === xs.length - 1) && (i === xs.length - 1 || xs.length - 1 - i >= xTickEvery / 2) ? (
          <text key={x} className="tick" x={X(i)} y={height - 8} textAnchor={i === 0 ? "start" : i === xs.length - 1 ? "end" : "middle"}>{xFormat(x)}</text>
        ) : null)}
        {series.map((s) => {
          const d = s.points.map((p, j) => `${j ? "L" : "M"}${X(idx.get(p.x)!).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
          const last = s.points[s.points.length - 1];
          return (
            <g key={s.key}>
              <path d={d} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
              {last && <circle cx={X(idx.get(last.x)!)} cy={Y(last.y)} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} />}
            </g>
          );
        })}
        {hx != null && (
          <g>
            <line x1={X(hover!)} x2={X(hover!)} y1={m.t} y2={m.t + ih} stroke="var(--rule-strong)" strokeWidth={1} />
            {series.map((s) => {
              const p = s.points.find((q) => q.x === hx);
              return p ? <circle key={s.key} cx={X(hover!)} cy={Y(p.y)} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} /> : null;
            })}
          </g>
        )}
        <rect x={m.l} y={m.t} width={iw} height={ih} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
      </svg>
      {hx != null && (
        <div className="tooltip" style={{ left: Math.min(X(hover!) + 12, width - 190), top: 8 }}>
          <div className="tl">{xFormat(hx)}</div>
          {series.map((s) => {
            const p = s.points.find((q) => q.x === hx);
            return p ? (
              <div className="row" key={s.key}>
                <span className="key" style={{ background: s.color }} />
                <span className="tv">{tooltipValue ? tooltipValue(s, p.y) : yFormat(p.y)}</span>
                {series.length > 1 && <span className="tl">{s.label}</span>}
              </div>
            ) : null;
          })}
        </div>
      )}
    </div>
  );
}

export interface BarSeries { key: string; label: string; color: string; values: (number | null)[] }

/** 分组柱状图，围绕 0 轴（正数向上、负数向下）。每根柱子单独悬停。 */
export function GroupedBars({ categories, series, height = 280, yFormat = (v: number) => `${v}`,
  categoryLabel = (c: string) => c, tooltipExtra, ariaLabel, valueFormat }: {
  categories: string[]; series: BarSeries[]; height?: number; yFormat?: (v: number) => string; valueFormat?: (v: number) => string;
  categoryLabel?: (c: string) => string; tooltipExtra?: (ci: number) => ReactNode; ariaLabel: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ c: number; s: number } | null>(null);
  const vals = series.flatMap((s) => s.values.filter((v): v is number => v != null));
  const ticks = niceTicks(Math.min(0, ...vals), Math.max(0, ...vals));
  const lo = ticks[0], hi = ticks[ticks.length - 1];
  const m = { l: 58, r: 8, t: 10, b: 30 };
  const iw = width - m.l - m.r, ih = height - m.t - m.b;
  const band = iw / categories.length;
  const bw = Math.min(24, (band * 0.62 - 2 * (series.length - 1)) / series.length);
  const groupW = bw * series.length + 2 * (series.length - 1);
  const Y = (v: number) => m.t + ih - ((v - lo) / (hi - lo)) * ih;
  const y0 = Y(0);
  const bar = (x: number, v: number) => {
    const y = Y(v), r = Math.min(4, Math.abs(y - y0), bw / 2);
    if (v >= 0) return `M${x},${y0}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${y0}Z`;
    return `M${x},${y0}V${y - r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y - r}V${y0}Z`;
  };
  const labelEvery = Math.max(1, Math.ceil(categories.length / Math.max(2, Math.floor(iw / 90))));

  return (
    <div className="chart" ref={ref}>
      <div className="legend">
        {series.map((s) => <span key={s.key}><span className="sw" style={{ background: s.color }} />{s.label}</span>)}
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={ariaLabel}>
        {ticks.map((t) => (
          <g key={t}>
            <line className={t === 0 ? "axis" : "grid"} x1={m.l} x2={width - m.r} y1={Y(t)} y2={Y(t)} />
            <text className="tick" x={m.l - 8} y={Y(t) + 4} textAnchor="end">{yFormat(t)}</text>
          </g>
        ))}
        {categories.map((c, ci) => {
          const gx = m.l + ci * band + (band - groupW) / 2;
          return (
            <g key={c}>
              {series.map((s, si) => {
                const v = s.values[ci];
                if (v == null) return null;
                const x = gx + si * (bw + 2);
                const on = hover?.c === ci && hover?.s === si;
                return (
                  <g key={s.key}>
                    <path d={bar(x, v)} fill={s.color} opacity={hover && !on ? 0.55 : 1} />
                    <rect x={x - 3} y={m.t} width={bw + 6} height={ih} fill="transparent" tabIndex={0}
                      aria-label={`${categoryLabel(c)} ${s.label} ${(valueFormat ?? yFormat)(v)}`}
                      onPointerEnter={() => setHover({ c: ci, s: si })} onPointerLeave={() => setHover(null)}
                      onFocus={() => setHover({ c: ci, s: si })} onBlur={() => setHover(null)} />
                  </g>
                );
              })}
              {(ci % labelEvery === 0) && (
                <text className="tick" x={m.l + ci * band + band / 2} y={height - 10} textAnchor="middle">{categoryLabel(c)}</text>
              )}
            </g>
          );
        })}
      </svg>
      {hover && (() => {
        const s = series[hover.s], v = s.values[hover.c]!;
        const left = m.l + hover.c * band + band / 2;
        return (
          <div className="tooltip" style={{ left: Math.max(0, Math.min(left - 80, width - 220)), top: 30 }}>
            <div className="tl">{categoryLabel(categories[hover.c])}</div>
            <div className="row"><span className="key" style={{ background: s.color, height: 8, width: 8 }} />
              <span className="tv">{(valueFormat ?? yFormat)(v)}</span><span className="tl">{s.label}</span></div>
            {tooltipExtra?.(hover.c)}
          </div>
        );
      })()}
    </div>
  );
}
