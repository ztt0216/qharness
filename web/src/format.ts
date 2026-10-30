// 数字和文案格式。涨跌一律带正负号，颜色只是辅助。

export const pct = (v: number | null | undefined, d = 1) =>
  v == null || Number.isNaN(v) ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v * 100).toFixed(d)}%`;

export const share = (v: number | null | undefined, d = 1) =>
  v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(d)}%`;

export const signed = (v: number | null | undefined, d = 4) =>
  v == null || Number.isNaN(v) ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}`;

export const money = (v: number | null | undefined, d = 0) =>
  v == null || Number.isNaN(v) ? "—" : v.toLocaleString("zh-CN", { minimumFractionDigits: d, maximumFractionDigits: d });

export const signedMoney = (v: number | null | undefined) =>
  v == null || Number.isNaN(v) ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${money(Math.abs(v))}`;

export const tone = (v: number | null | undefined) => (v == null || v === 0 ? "" : v > 0 ? "up" : "down");

export const compact = (v: number) => (v >= 10000 ? `${(v / 10000).toFixed(1)} 万` : `${v}`);

export function runTime(id: string) {
  // 20261030-220425-8083dc → 10月30日 22:04
  const m = id.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/);
  return m ? `${+m[2]}月${+m[3]}日 ${m[4]}:${m[5]}` : id;
}

export const duration = (a: number | null, b: number | null) => {
  if (a == null || b == null) return "—";
  const s = Math.max(0, Math.round(b - a));
  return s < 60 ? `${s} 秒` : `${Math.floor(s / 60)} 分 ${s % 60} 秒`;
};

// 研究层的预注册规则（agents.Thresholds）
export const RULES: Record<string, string> = {
  weak_train: "训练段 RankIC 太弱",
  weak_valid: "验证段 RankIC 太弱",
  insignificant_train: "训练段不显著",
  insignificant_valid: "验证段不显著",
  invalid_train: "训练段数据不足",
  invalid_valid: "验证段数据不足",
  unstable_years: "分年方向不稳定",
  unstable_train: "训练段不稳定",
  unstable_valid: "验证段不稳定",
  high_corr: "和已有因子高度相关",
  high_turnover: "换手太高",
};

export const ERRORS: Record<string, string> = {
  SYNTAX: "语法错误", FORBIDDEN: "用了禁止的操作", LOOKAHEAD: "用到了未来数据", RUNTIME: "运行出错",
  TIMEOUT: "超时", BAD_OUTPUT: "输出不合格", ORDER_DEPENDENT: "没有按股票分组", SANDBOX: "沙箱没能启动",
};

export const STATUS: Record<string, { label: string; icon: string }> = {
  accepted: { label: "通过", icon: "✓" },
  rejected: { label: "淘汰", icon: "–" },
  skipped: { label: "跳过", icon: "↺" },
  failed: { label: "实现失败", icon: "✕" },
  running: { label: "进行中", icon: "…" },
  finished: { label: "已完成", icon: "✓" },
  interrupted: { label: "已中断", icon: "‖" },
};

export const source = (r: { report: string | null; has_report: boolean; hypothesis_file: string | null }) =>
  r.report ? `研报 ${r.report}` : r.has_report ? "研报（早期运行没有记录文件名）"
    : r.hypothesis_file ? `假设文件 ${r.hypothesis_file}` : "直接给的假设";
