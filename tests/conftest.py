import os
from pathlib import Path

import pytest

from qharness.data import load_csv_panel
from qharness.evaluate import Evaluator

FIXTURE = Path(__file__).parent / "fixtures" / "hs300_sample.csv.gz"
RISK_FIXTURE = Path(__file__).parent / "fixtures" / "hs300_risk.csv.gz"


def pytest_configure(config):
    config.addinivalue_line("markers", "llm: 调用真实大模型（需要 QH_ENV_FILE 或 ANTHROPIC_API_KEY）")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("QH_ENV_FILE") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("QH_LLM_API_KEY"):
        return
    skip = pytest.mark.skip(reason="未配置模型密钥：设置 QH_ENV_FILE=<.env> 后运行")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def df():
    """沪深 300 真实日线切片（tests/fixtures/make_fixture.py 从 data/hs300 生成）。"""
    return load_csv_panel(str(FIXTURE))


@pytest.fixture(scope="session")
def llm():
    from qharness.llm import make_llm
    return make_llm(os.environ.get("QH_LLM_KIND", "anthropic"))


@pytest.fixture(scope="session")
def risk(df):
    """同一切片的市值、申万一级行业、停牌、涨跌停。"""
    import pandas as pd
    r = pd.read_csv(RISK_FIXTURE, parse_dates=["datetime"]).set_index(["datetime", "instrument"])
    for c in ("halted", "limit_up", "limit_down"):
        r[c] = r[c].astype(bool)
    return r.reindex(df.index)


@pytest.fixture()
def ev(df):
    return Evaluator(df)


@pytest.fixture()
def ev_risk(df, risk):
    return Evaluator(df, risk=risk)
