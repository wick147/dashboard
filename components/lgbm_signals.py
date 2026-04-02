"""
LightGBM 多周期选股信号生成器 — Qlib Alpha158 模式。

每个预测周期（5/10/20日）独立训练一个 LightGBM 模型，
目标变量为持仓期 n 日总收益的截面排名（per-day rank normalization）。

数据来源: qlib Alpha158 特征 (chenditc/investment_data)
不依赖任何实时行情 API，可在 GitHub Actions 上稳定运行。
"""
from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytz

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LGBM_PARAMS, LGBM_EARLY_STOP, LGBM_VERBOSE_EVAL,
    SIGNAL_TOP_N, HORIZONS,
    SIGNALS_QLIB_CACHE, QLIB_DATA_PATH, TZ,
)

logger = logging.getLogger(__name__)


# ── qlib 可用性检查 ────────────────────────────────────────────────────────────

def _qlib_available() -> bool:
    try:
        import qlib  # noqa: F401
    except ImportError:
        return False
    data_path = Path(QLIB_DATA_PATH).expanduser()
    return data_path.exists() and any(data_path.iterdir())


# ── qlib Alpha158 模式 ─────────────────────────────────────────────────────────

def _generate_signals_qlib(progress_cb: Optional[Callable] = None) -> dict:
    """
    qlib Alpha158 特征 + 3个独立 LightGBM 模型（5/10/20日）。

    Alpha158 handler 只初始化一次，label 列表传入3个周期的总收益表达式。
    总收益与 GMR 截面排名等价（单调变换），所以用总收益训练排名模型即可。
    特征维度：158个 Alpha 因子（MACD/RSI/量价等）。
    股票池: CSI300（沪深300成分股）。
    """
    import qlib
    from qlib.config import REG_CN
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    tz = pytz.timezone(TZ)
    today       = datetime.now(tz)
    train_start = "2020-01-01"
    train_end   = (today - timedelta(days=60)).strftime("%Y-%m-%d")
    pred_start  = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    pred_end    = today.strftime("%Y-%m-%d")

    qlib.init(provider_uri=QLIB_DATA_PATH, region=REG_CN)
    if progress_cb: progress_cb(0.05)

    # ── 1. 一次性加载 Alpha158 特征 + 3列标签 ────────────────────────────────
    handler = Alpha158(
        start_time=train_start,
        end_time=pred_end,
        fit_start_time=train_start,
        fit_end_time=train_end,
        instruments="csi300",
        label=[
            "Ref($close, -5)/$close - 1",    # LABEL0: 5日总收益
            "Ref($close, -10)/$close - 1",   # LABEL1: 10日总收益
            "Ref($close, -20)/$close - 1",   # LABEL2: 20日总收益
        ],
    )
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (train_start, train_end),
            "test":  (pred_start,  pred_end),
        },
    )
    if progress_cb: progress_cb(0.2)

    train_df = dataset.prepare(
        "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    )
    X_train = train_df["feature"]
    y_all   = train_df["label"]        # columns: LABEL0, LABEL1, LABEL2

    test_df = dataset.prepare(
        "test", col_set=["feature"], data_key=DataHandlerLP.DK_I
    )
    X_test = test_df["feature"]
    if progress_cb: progress_cb(0.35)

    # ── 2. 为每个周期独立训练一个 LightGBM ───────────────────────────────────
    out: dict = {"feature_importance": {}}
    label_cols = list(y_all.columns)   # ["LABEL0", "LABEL1", "LABEL2"]
    latest_dt  = None

    for i, h in enumerate(HORIZONS):
        label_col = label_cols[i]

        # 去掉标签缺失行（前向标签在时间轴末端会有 NaN）
        y_h = y_all[label_col].dropna()
        X_h = X_train.loc[y_h.index].fillna(0)

        # 截面 rank 归一化：每个交易日内对标签排名，消除绝对收益水平的影响
        y_rank = (
            y_h.groupby(level="datetime").rank(pct=True) - 0.5
        ).values

        params = {**LGBM_PARAMS}
        n_est  = params.pop("n_estimators", 150)

        model_h = lgb.train(
            params,
            lgb.Dataset(X_h.values, label=y_rank, feature_name=list(X_h.columns)),
            num_boost_round=n_est,
            callbacks=[lgb.log_evaluation(period=LGBM_VERBOSE_EVAL)],
        )
        if progress_cb: progress_cb(0.35 + (i + 1) / len(HORIZONS) * 0.50)

        # 预测最新一个交易日的分数
        scores    = model_h.predict(X_test.fillna(0).values)
        score_idx = X_test.index.get_level_values("datetime")
        latest_dt = score_idx.max()
        mask      = score_idx == latest_dt

        latest_scores = pd.Series(
            scores[mask],
            index=X_test.index[mask].get_level_values("instrument"),
            name="score",
        ).sort_values(ascending=False)

        top = latest_scores.head(SIGNAL_TOP_N).reset_index()
        top.columns = ["code", "score"]
        out[f"h{h}"] = [
            {
                "code":       str(r["code"]),
                "name":       str(r["code"]),   # qlib 模式用代码作为名称
                "rank_score": round(float(r["score"]), 4),
            }
            for _, r in top.iterrows()
        ]

        # 特征重要性（gain）
        fi_names  = model_h.feature_name()
        fi_values = model_h.feature_importance(importance_type="gain")
        fi_sorted = sorted(zip(fi_names, fi_values), key=lambda x: -x[1])[:10]
        out["feature_importance"][f"h{h}"] = [
            {"feature": str(f), "importance": round(float(v), 4)}
            for f, v in fi_sorted
        ]

    if progress_cb: progress_cb(1.0)

    out.update({
        "train_start": train_start,
        "train_end":   train_end,
        "pred_start":  pred_start,
        "pred_end":    pred_end,
        "pred_date":   str(latest_dt.date()) if hasattr(latest_dt, "date") else str(latest_dt)[:10],
    })
    return out


# ── 主入口 ────────────────────────────────────────────────────────────────────

def generate_signals(
    progress_cb: Optional[Callable] = None,
    use_cache:   bool = True,
    cache_path:  Optional[Path] = None,
) -> dict:
    """生成 qlib Alpha158 多周期选股信号，结果写入 JSON 缓存。"""
    tz        = pytz.timezone(TZ)
    read_path = cache_path or SIGNALS_QLIB_CACHE

    if use_cache and read_path.exists():
        try:
            data  = json.loads(read_path.read_text())
            age_h = (datetime.now(tz) - datetime.fromisoformat(data["updated_at"])).total_seconds() / 3600
            if age_h < 12 and not data.get("error"):
                return data
        except Exception:
            pass

    result: dict = {
        "mode":               "qlib",
        "updated_at":         datetime.now(tz).isoformat(),
        "h5": [], "h10": [], "h20": [],
        "feature_importance": {},
        "error":              None,
    }

    try:
        if not _qlib_available():
            raise RuntimeError(f"qlib 数据不可用，请检查路径: {QLIB_DATA_PATH}")
        horizon_data = _generate_signals_qlib(progress_cb)
        result.update(horizon_data)
        result["error"] = None
    except Exception:
        result["error"] = traceback.format_exc()

    write_path = cache_path or SIGNALS_QLIB_CACHE
    write_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
