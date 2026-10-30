"""从 data/hs300 缓存切一份测试用的真实行情：按代码排序取前 N 只、最近 D 个交易日。

  python tests/fixtures/make_fixture.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qharness.market import load_hs300  # noqa: E402

N_STOCKS, N_DAYS = 80, 420

df, risk = load_hs300()
insts = sorted(df.index.get_level_values("instrument").unique())[:N_STOCKS]
dates = sorted(df.index.get_level_values("datetime").unique())[-N_DAYS:]
here = Path(__file__).parent
part = df.loc[(dates, insts), :].reset_index()
part.to_csv(here / "hs300_sample.csv.gz", index=False, float_format="%.6g")
print("panel", part.shape, list(part.columns))
if risk is not None:
    r = risk.loc[(dates, insts), :].reset_index()
    r.to_csv(here / "hs300_risk.csv.gz", index=False, float_format="%.6g")
    print("risk", r.shape)
