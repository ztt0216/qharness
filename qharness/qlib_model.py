"""Qlib 模型训练（对应 qrun 的 workflow：数据处理器 → 数据集 → 模型 → 信号记录 → 信号分析）。

- 特征：Qlib 的 Alpha158（158 个量价特征）；可选再加上因子库里写成 Qlib 表达式的已登记因子
- 标签：和 Harness 同口径 Ref($close,-6)/Ref($close,-1)-1；训练时按截面做 Z-score（Alpha158 默认）
- 切分：沿用 Evaluator 的训练 / 验证 / 测试段，每段末尾去掉 EMBARGO 天
- 模型：LightGBM（Qlib 的 LGBModel）；LightGBM 加载不了（macOS 缺 OpenMP 运行库）时改用
  sklearn 的 HistGradientBoostingRegressor（同为直方图梯度提升树），用验证段 L2 早停，与 LGBModel 的做法一致
- 记录：qlib.workflow.R（MLflow），SignalRecord 存预测、SigAnaRecord 算 IC
- 评估：预测值当作一个因子交给 Evaluator（中性化、Newey-West t），测试段同样只开一次；
  再和 5 个基础因子等权合成的基线比回测（自有回测 + Qlib 回测）
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import qlib_bridge as qb
from .data import EMBARGO, LABEL_HORIZON, segment_dates
from .evaluate import Evaluator

LABEL = f"Ref($close, -{LABEL_HORIZON + 1})/Ref($close, -1) - 1"


def lightgbm_ok() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except OSError:  # 动态库缺失（libomp）
        return False


class HGBModel:
    """Qlib Model 接口（fit(dataset) / predict(dataset, segment)）的 sklearn 实现。"""

    def __init__(self, learning_rate: float = 0.05, max_leaf_nodes: int = 63, min_samples_leaf: int = 200,
                 l2_regularization: float = 1.0, step: int = 20, max_iter: int = 1000, patience: int = 3,
                 seed: int = 0):
        self.kw = dict(learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                       l2_regularization=l2_regularization, random_state=seed)
        self.step, self.max_iter, self.patience = step, max_iter, patience
        self.model, self.history = None, []

    def fit(self, dataset, reweighter=None):
        from qlib.data.dataset.handler import DataHandlerLP
        from sklearn.ensemble import HistGradientBoostingRegressor

        tr, va = dataset.prepare(["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        xt, yt = tr["feature"].to_numpy(np.float32), tr["label"].to_numpy().ravel()
        xv, yv = va["feature"].to_numpy(np.float32), va["label"].to_numpy().ravel()
        m = HistGradientBoostingRegressor(warm_start=True, early_stopping=False, max_iter=0, **self.kw)
        best, best_it, bad = np.inf, 0, 0
        for it in range(self.step, self.max_iter + 1, self.step):  # 每加 step 棵树看一次验证段 L2
            m.set_params(max_iter=it)
            m.fit(xt, yt)
            loss = float(np.mean((m.predict(xv) - yv) ** 2))
            self.history.append((it, round(loss, 6)))
            if loss < best - 1e-6:
                best, best_it, bad = loss, it, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        m.set_params(warm_start=False, max_iter=best_it)
        self.model = m.fit(xt, yt)  # 用最优轮数重训
        self.best_iter = best_it
        return self

    def predict(self, dataset, segment="test"):
        from qlib.data.dataset.handler import DataHandlerLP

        x = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(x.to_numpy(np.float32)), index=x.index)


def _bounds(ev: Evaluator) -> dict[str, tuple[str, str]]:
    out = {}
    for name in ("train", "valid", "test"):
        d = segment_dates(ev.df, getattr(ev.split, name))
        out[name] = (str(d[0].date()), str(d[-1].date()))
    return out


def registry_exprs(items: list[dict]) -> dict[str, str]:
    """已登记因子里能直接交给 Qlib 的（Qlib 表达式写的）。登记的代码已经包含 Harness 施加的方向，不再处理。"""
    return {x["name"]: qb.normalize(qb.expr_of(x["code"])) for x in items if qb.is_expr(x.get("code", ""))}


def _handler_cls():
    from qlib.contrib.data.handler import Alpha158

    class Alpha158Plus(Alpha158):
        """Alpha158 + 额外的 Qlib 表达式特征。"""

        def __init__(self, extra: dict[str, str] | None = None, **kw):
            self._extra = dict(sorted((extra or {}).items()))  # 父类 __init__ 里会调 get_feature_config
            super().__init__(**kw)

        def get_feature_config(self):
            fields, names = super().get_feature_config()
            return list(fields) + list(self._extra.values()), list(names) + list(self._extra)

    return Alpha158Plus


def _make_model(model: str, seed: int, model_kw: dict | None):
    use = "lgb" if model == "lgb" or (model == "auto" and lightgbm_ok()) else "hgb"
    if use == "lgb":
        from qlib.contrib.model.gbdt import LGBModel

        return use, LGBModel(loss="mse", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, lambda_l2=1.0,
                             early_stopping_rounds=50, num_boost_round=1000, seed=seed, verbose=-1)
    return use, HGBModel(seed=seed, **(model_kw or {}))


def train(ev: Evaluator, out_dir: str | Path, model: str = "auto", extra: dict[str, str] | None = None,
          topk: int = 20, seed: int = 0, model_kw: dict | None = None) -> dict:
    """训练 → 预测测试段 → 用 Harness 的口径评估。返回摘要并写入 out_dir/summary.json。"""
    from qlib.data.dataset import DatasetH
    from qlib.workflow import R
    from qlib.workflow.record_temp import SigAnaRecord, SignalRecord

    from .strategy import backtest, combine

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # MLflow 3 默认拒绝文件存储；记录和预测文件都放在本次输出目录里
    qb.init()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = _bounds(ev)
    insts = sorted(ev.df.index.get_level_values("instrument").unique())
    dates = ev.df.index.get_level_values("datetime")
    handler = _handler_cls()(extra=extra, instruments=insts, start_time=str(dates.min().date()),
                             end_time=str(dates.max().date()), fit_start_time=seg["train"][0],
                             fit_end_time=seg["train"][1], label=([LABEL], ["LABEL0"]))
    dataset = DatasetH(handler, segments=seg)
    use, m = _make_model(model, seed, model_kw)

    warnings.filterwarnings("ignore", message="Mean of empty slice")  # Qlib 的 IC 分析在全空截面上逐日刷屏
    with R.start(experiment_name="qharness_alpha158", uri=(out_dir / "mlruns").resolve().as_uri()):
        m.fit(dataset)
        rec = R.get_recorder()
        R.log_params(model=use, features=len(handler.get_cols()) - 1, label=LABEL, **{f"seg_{k}": "~".join(v)
                                                                                      for k, v in seg.items()})
        SignalRecord(model=m, dataset=dataset, recorder=rec).generate()
        SigAnaRecord(recorder=rec, ana_long_short=False).generate()
        pred = rec.load_object("pred.pkl")["score"]
        qlib_ic = {k: float(rec.load_object(f"sig_analysis/{k}.pkl").mean()) for k in ("ic", "ric")}
        rec_id = rec.id

    pred.index = pred.index.set_names(["datetime", "instrument"])
    pred = pred.reindex(ev.df.index)
    # 测试段锁的键包含超参：同一配置只能看一次测试段；换配置会留下新的记录（次数可查，见 test_opened.txt）
    key = "qlib_model:" + hashlib.sha1(json.dumps([use, seed, sorted(extra or {}), seg, model_kw or {}],
                                                  ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    test = ev.evaluate_test(pred, key)["test"]

    test_dates = segment_dates(ev.df, ev.split.test)
    base_score = combine(ev, list(ev.raw_library.values()))
    model_score = pred.unstack("instrument").reindex(index=base_score.index, columns=base_score.columns)
    bt = {"baseline": backtest(ev.df, base_score, test_dates, topk, risk=ev.risk),
          "model": backtest(ev.df, model_score, test_dates, topk, risk=ev.risk)}
    # 调仓日相位：5 天调一次，从测试段第 0~4 天开始各回测一遍。测试段只有一年不到，起点不同结果能差十几个点
    phase = {k: [backtest(ev.df, sc, test_dates[i:], topk, risk=ev.risk)["annual_excess"] for i in range(5)]
             for k, sc in (("baseline", base_score), ("model", model_score))}
    try:
        bt_qlib = {k: qb.qlib_backtest(s, test_dates[0], test_dates[-1], topk)
                   for k, s in (("baseline", base_score), ("model", model_score))}
    except Exception as e:  # noqa: BLE001
        bt_qlib = {"error": f"{type(e).__name__}: {e}"}
    base_test = ev._segment_metrics(ev._prep(base_score.stack().rename("s")), test_dates)

    summary = {
        "model": use, "n_features": len(handler.get_cols()) - 1, "extra_features": sorted(extra or {}),
        "segments": seg, "label": LABEL, "mlflow_recorder": rec_id,
        "best_iter": getattr(m, "best_iter", None),
        "qlib_sig_analysis_test": {k: round(v, 4) for k, v in qlib_ic.items()},  # 未中性化
        "test": {k: test.get(k) for k in ("rank_ic", "rank_icir", "rank_ic_t_nw", "rank_ic_raw", "long_short",
                                          "turnover", "max_abs_corr_with_library", "n_days")},
        "baseline_test": {k: base_test.get(k) for k in ("rank_ic", "rank_icir", "rank_ic_t_nw", "long_short")},
        "valid_l2_history": getattr(m, "history", None),
        "backtest": bt, "backtest_qlib": bt_qlib,
        "backtest_by_phase": {k: {"min": min(v), "median": float(np.median(v)), "max": max(v), "all": v}
                              for k, v in phase.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                          encoding="utf-8")
    pred.dropna().to_frame("score").to_pickle(out_dir / "pred.pkl")
    return summary


# ------------------------------------------------------------------ 滚动训练

def folds(cal: pd.DatetimeIndex, step: int = 63, valid_days: int = 120, min_train: int = 250,
          train_days: int | None = None) -> list[dict[str, tuple[str, str]]]:
    """滚动切分：每折预测接下来 step 天；验证段是预测段前的 valid_days 天，训练段是再往前的数据
    （train_days=None 时从头扩张，否则只取最近 train_days 天）。训练 / 验证 / 预测之间各隔 EMBARGO 天，
    标签（看 t+1~t+6 的价格）不会跨进下一段。"""
    e, out = EMBARGO, []
    p0 = min_train + e + valid_days + e
    while p0 < len(cal):
        p1 = min(p0 + step, len(cal))
        v1 = p0 - e
        v0 = v1 - valid_days
        t1 = v0 - e
        t0 = 0 if train_days is None else max(0, t1 - train_days)
        d = lambda i: str(cal[i].date())  # noqa: E731
        out.append({"train": (d(t0), d(t1 - 1)), "valid": (d(v0), d(v1 - 1)), "test": (d(p0), d(p1 - 1))})
        p0 = p1
    return out


def walk_forward(ev: Evaluator, out_dir: str | Path, model: str = "auto", extra: dict[str, str] | None = None,
                 topk: int = 20, seed: int = 0, model_kw: dict | None = None, step: int = 63,
                 valid_days: int = 120, min_train: int = 250, train_days: int | None = None) -> dict:
    """按季度向前滚：每折重新训练（验证段早停），只预测下一个季度；各折的样本外预测拼成一条，
    逐折、逐段（验证段 / 测试段）和整体评估。拼起来的预测覆盖了测试段，所以照样经过测试段锁。"""
    from qlib.data.dataset import DatasetH

    from .strategy import backtest, combine

    qb.init()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal = ev.df.index.get_level_values("datetime").unique().sort_values()
    plan = folds(cal, step, valid_days, min_train, train_days)
    if not plan:
        raise ValueError(f"数据只有 {len(cal)} 天，不够切出一折（至少要 {min_train + 2 * EMBARGO + valid_days + 1} 天）")
    insts = sorted(ev.df.index.get_level_values("instrument").unique())
    handler = _handler_cls()(extra=extra, instruments=insts, start_time=str(cal[0].date()),
                             end_time=str(cal[-1].date()), fit_start_time=plan[0]["train"][0],
                             fit_end_time=plan[0]["train"][1], label=([LABEL], ["LABEL0"]))  # 特征只算一次，各折复用
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    preds, meta = [], []
    for f in plan:
        use, m = _make_model(model, seed, model_kw)
        m.fit(DatasetH(handler, segments=f))
        p = m.predict(DatasetH(handler, segments=f), segment="test")
        preds.append(p)
        meta.append({**f, "best_iter": getattr(m, "best_iter", None)})
    pred = pd.concat(preds)
    pred.index = pred.index.set_names(["datetime", "instrument"])
    pred = pred[~pred.index.duplicated()].reindex(ev.df.index)

    key = "qlib_walk_forward:" + hashlib.sha1(json.dumps(
        [use, seed, sorted(extra or {}), plan, model_kw or {}], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    test = ev.evaluate_test(pred, key)["test"]  # 登记测试段被打开（拼接预测覆盖了测试段）

    base_score = combine(ev, list(ev.raw_library.values()))
    model_score = pred.unstack("instrument").reindex(index=base_score.index, columns=base_score.columns)
    fm, fb = ev._prep(pred), ev._prep(base_score.stack().rename("s"))
    keys = ("rank_ic", "rank_ic_t_nw", "n_days")

    def ric(fw, d):
        r = ev._segment_metrics(fw, d) if len(d) >= 10 else {}
        return {k: r.get(k) for k in keys}

    rows = []
    for f in meta:
        d = cal[(cal >= f["test"][0]) & (cal <= f["test"][1])]
        d = d[:-EMBARGO] if f is not meta[-1] else d  # 末尾 6 天的标签落在下一折，和下一折重叠
        d = d[ev.label_w.loc[d].notna().sum(axis=1).to_numpy() > 0]  # 最后几天还没有标签
        bm = backtest(ev.df, model_score, d, topk, risk=ev.risk) if len(d) > 10 else {}
        bb = backtest(ev.df, base_score, d, topk, risk=ev.risk) if len(d) > 10 else {}
        rows.append({"predict": f["test"], "train": f["train"], "best_iter": f["best_iter"],
                     "model": ric(fm, d), "baseline": ric(fb, d),
                     "excess_model": bm.get("annual_excess"), "excess_baseline": bb.get("annual_excess")})

    oos = cal[(cal >= plan[0]["test"][0])]
    oos = oos[ev.label_w.loc[oos].notna().sum(axis=1).to_numpy() > 0]
    by_seg = {}
    for name in ("train", "valid", "test"):
        d = segment_dates(ev.df, getattr(ev.split, name)).intersection(oos)
        if len(d) >= 10:
            by_seg[name] = {"dates": (str(d[0].date()), str(d[-1].date())), "model": ric(fm, d), "baseline": ric(fb, d)}
    phase = {k: [backtest(ev.df, sc, oos[i:], topk, risk=ev.risk)["annual_excess"] for i in range(5)]
             for k, sc in (("baseline", base_score), ("model", model_score))}
    pos = [r for r in rows if r["model"]["rank_ic"] is not None]
    summary = {
        "model": use, "n_features": len(handler.get_cols()) - 1, "extra_features": sorted(extra or {}),
        "label": LABEL, "n_folds": len(plan), "step": step, "valid_days": valid_days, "min_train": min_train,
        "train_window": train_days or "expanding",
        "oos": {"dates": (str(oos[0].date()), str(oos[-1].date())), "model": ric(fm, oos), "baseline": ric(fb, oos)},
        "by_segment": by_seg,  # 落在原切分的哪一段（训练段的部分对滚动模型来说也是样本外）
        "test_segment": {k: test.get(k) for k in keys},
        "folds_positive_rank_ic": f"{sum(r['model']['rank_ic'] > 0 for r in pos)}/{len(pos)}",
        "folds_beat_baseline": f"{sum(r['model']['rank_ic'] > (r['baseline']['rank_ic'] or 0) for r in pos)}/{len(pos)}",
        "folds": rows,
        "backtest_by_phase": {k: {"min": min(v), "median": float(np.median(v)), "max": max(v), "all": v}
                              for k, v in phase.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                          encoding="utf-8")
    pred.dropna().to_frame("score").to_pickle(out_dir / "pred.pkl")
    return summary
