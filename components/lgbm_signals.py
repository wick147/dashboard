"""
LightGBM 多周期选股信号生成器。

每个预测周期（5/10/20日）独立训练一个 LightGBM 模型，
目标变量为持仓期几何平均日收益率：
    gmr_n = (close[t+n] / close[t]) ^ (1/n) - 1

模式（auto-detect）：
  qlib   — Alpha158 特征 + qlib LGBModel
  akshare — OHLCV 技术因子 + 标准 LightGBM
"""
from __future__ import annotations

import json
import logging
import pickle
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import akshare as ak
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytz

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LGBM_PARAMS, LGBM_EARLY_STOP, LGBM_VERBOSE_EVAL,
    UNIVERSE_SIZE, SIGNAL_TOP_N, HORIZONS,
    FEATURE_LOOKBACK, FETCH_WORKERS,
    SIGNALS_CACHE, CACHE_DIR,
    QLIB_DATA_PATH, TZ,
)

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "ret1", "ret5", "ret10", "ret20",
    "vol_rv5", "vol_rv10", "vol_rv20",
    "vol_ratio5", "vol_ratio20",
    "rsi14", "bb_pos", "macd_sig", "hi20_pos",
]


# ── universe ──────────────────────────────────────────────────────────────────

def get_csi300_universe() -> list[str]:
    try:
        df = ak.index_stock_cons_weight_csindex(symbol="000300")
        return df["成分券代码"].astype(str).str.zfill(6).tolist()
    except Exception:
        pass
    try:
        spot = ak.stock_zh_a_spot_em()
        spot["总市值"] = pd.to_numeric(spot.get("总市值", spot.get("市值", 0)), errors="coerce")
        return spot.dropna(subset=["总市值"]).nlargest(UNIVERSE_SIZE, "总市值")["代码"].astype(str).str.zfill(6).tolist()
    except Exception:
        return []


# ── AKShare 数据获取 ───────────────────────────────────────────────────────────

def _fetch_one(code: str, start: str, end: str) -> tuple[str, pd.DataFrame | None]:
    try:
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start, end_date=end, adjust="hfq",
        )
        if df is None or df.empty:
            return code, None
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        return code, df.set_index("date").sort_index()
    except Exception:
        return code, None


def fetch_stock_data(codes: list[str], progress_cb: Optional[Callable] = None) -> dict[str, pd.DataFrame]:
    tz = pytz.timezone(TZ)
    end   = datetime.now(tz).strftime("%Y%m%d")
    start = (datetime.now(tz) - timedelta(days=FEATURE_LOOKBACK + 40)).strftime("%Y%m%d")
    results: dict[str, pd.DataFrame] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, c, start, end): c for c in codes}
        for fut in as_completed(futures):
            code, df = fut.result()
            if df is not None and len(df) >= 40:
                results[code] = df
            done += 1
            if progress_cb:
                progress_cb(done / len(codes))
    return results


# ── 特征工程 ──────────────────────────────────────────────────────────────────

def _compute_features(df: pd.DataFrame, code: str) -> pd.DataFrame:
    c = df["close"].astype(float)
    v = df["volume"].astype(float)

    feat = pd.DataFrame(index=df.index)
    for d in [1, 5, 10, 20]:
        feat[f"ret{d}"] = c.pct_change(d)
    for w in [5, 10, 20]:
        feat[f"vol_rv{w}"] = c.pct_change().rolling(w).std()
    feat["vol_ratio5"]  = v / v.rolling(5).mean()
    feat["vol_ratio20"] = v / v.rolling(20).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    feat["rsi14"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    feat["bb_pos"] = (c - ma20) / (2 * std20.replace(0, 1e-9))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    feat["macd_sig"] = macd - macd.ewm(span=9, adjust=False).mean()
    feat["hi20_pos"] = c / c.rolling(20).max()

    # 各周期几何平均日收益率（前向标签）
    for n in HORIZONS:
        feat[f"label_{n}"] = (c.shift(-n) / c) ** (1.0 / n) - 1

    feat["code"] = code
    # 只丢弃特征为 NaN 的行，标签 NaN 留给各模型自己处理
    return feat.dropna(subset=FEATURE_COLS)


def build_panel(stock_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for code, df in stock_data.items():
        try:
            frames.append(_compute_features(df, code))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


# ── 训练 ──────────────────────────────────────────────────────────────────────

def train_lgbm_horizon(panel: pd.DataFrame, horizon: int) -> lgb.Booster:
    """训练预测 horizon 日 GMR 的 LightGBM 模型。"""
    label_col = f"label_{horizon}"
    dates = panel.index.unique().sort_values()

    # 最后 horizon+5 个交易日没有完整的前向标签，排除
    if len(dates) <= horizon + 5:
        raise ValueError(f"数据不足以训练 {horizon}日 模型（共 {len(dates)} 个交易日）")
    cutoff = dates[-(horizon + 5)]
    train = panel[panel.index <= cutoff].dropna(subset=FEATURE_COLS + [label_col])
    if len(train) < 100:
        raise ValueError(f"{horizon}日 模型训练样本不足：{len(train)} 条")

    X = train[FEATURE_COLS].values
    # 截面 rank 归一化（对齐 qlib 做法）
    rank_y = (
        train.assign(_y=train[label_col])
        .groupby(level=0)["_y"]
        .rank(pct=True)
        .values - 0.5
    )
    params = {**LGBM_PARAMS}
    n_est = params.pop("n_estimators", 400)
    model = lgb.train(
        params,
        lgb.Dataset(X, label=rank_y),
        num_boost_round=n_est,
        callbacks=[lgb.log_evaluation(period=LGBM_VERBOSE_EVAL)],
    )
    logger.info("训练完成：%d日 模型，样本数=%d", horizon, len(train))
    return model


# ── 预测 ──────────────────────────────────────────────────────────────────────

def predict_top_n(
    model: lgb.Booster,
    panel: pd.DataFrame,
    horizon: int,
    spot_df: Optional[pd.DataFrame] = None,
    n: int = SIGNAL_TOP_N,
) -> list[dict]:
    """用最新交易日特征预测，返回 top-n 列表。"""
    last_date = panel.index.max()
    latest = panel[panel.index == last_date].copy()
    if latest.empty:
        return []

    X = latest[FEATURE_COLS].fillna(0).values
    latest = latest.copy()
    latest["score"] = model.predict(X)
    latest = latest.sort_values("score", ascending=False).head(n)

    name_map: dict = {}
    if spot_df is not None and "代码" in spot_df.columns and "名称" in spot_df.columns:
        name_map = spot_df.set_index("代码")["名称"].to_dict()

    result = []
    for _, row in latest.iterrows():
        code = row["code"]
        # 实际预测的 GMR（非 rank 分数）
        label_col = f"label_{horizon}"
        gmr_val = float(row[label_col]) if label_col in row and pd.notna(row[label_col]) else None
        result.append({
            "code":         code,
            "name":         name_map.get(code, ""),
            "rank_score":   round(float(row["score"]), 4),
            "gmr_daily":    round(gmr_val * 100, 4) if gmr_val is not None else None,
            "total_ret":    round(((1 + gmr_val) ** horizon - 1) * 100, 2) if gmr_val is not None else None,
        })
    return result


# ── qlib 模式 ────────────────────────────────────────────────────────────────

def _qlib_available() -> bool:
    try:
        import qlib  # noqa: F401
    except ImportError:
        return False
    data_path = Path(QLIB_DATA_PATH).expanduser()
    return data_path.exists() and any(data_path.iterdir())


def _generate_signals_qlib(progress_cb: Optional[Callable] = None) -> dict:
    """qlib Alpha158 + 多周期 LGBModel。"""
    import qlib
    from qlib.config import REG_CN
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH

    tz = pytz.timezone(TZ)
    today      = datetime.now(tz)
    train_end  = (today - timedelta(days=365)).strftime("%Y-%m-%d")
    pred_start = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    pred_end   = today.strftime("%Y-%m-%d")

    qlib.init(provider_uri=QLIB_DATA_PATH, region=REG_CN)
    if progress_cb: progress_cb(0.1)

    out: dict = {}
    for i, h in enumerate(HORIZONS):
        handler = Alpha158(
            start_time="2020-01-01", end_time=pred_end,
            fit_start_time="2020-01-01", fit_end_time=train_end,
            instruments="csi300",
            label=[f"Ref($close, -{h}) / $close - 1"],   # h日总收益
        )
        dataset = DatasetH(
            handler=handler,
            segments={"train": ("2020-01-01", train_end), "test": (pred_start, pred_end)},
        )
        model = LGBModel(
            loss="mse", colsample_bytree=0.8879, learning_rate=0.1,
            subsample=0.8789, num_leaves=63, max_depth=6,
            num_boost_round=300, early_stopping_rounds=LGBM_EARLY_STOP,
        )
        model.fit(dataset)
        pred = model.predict(dataset, segment="test").reset_index()
        pred.columns = ["date", "code", "score"]
        latest = pred[pred["date"] == pred["date"].max()].sort_values("score", ascending=False)
        out[f"h{h}"] = latest.head(SIGNAL_TOP_N)[["code", "score"]].assign(
            name="", gmr_daily=None, total_ret=None
        ).to_dict("records")
        if progress_cb: progress_cb(0.1 + (i + 1) / len(HORIZONS) * 0.85)

    return out


# ── 主入口 ────────────────────────────────────────────────────────────────────

def generate_signals(
    mode: str = "auto",
    progress_cb: Optional[Callable] = None,
    use_cache: bool = True,
) -> dict:
    """
    生成多周期选股信号。

    返回 dict：
      mode, updated_at, error
      h5  / h10 / h20  : list[{code, name, rank_score, gmr_daily, total_ret}]
      feature_importance: {h5: [...], h10: [...], h20: [...]}
    """
    tz = pytz.timezone(TZ)

    if use_cache and SIGNALS_CACHE.exists():
        try:
            data = json.loads(SIGNALS_CACHE.read_text())
            age_h = (datetime.now(tz) - datetime.fromisoformat(data["updated_at"])).total_seconds() / 3600
            if age_h < 12:
                return data
        except Exception:
            pass

    effective_mode = mode
    if mode == "auto":
        effective_mode = "qlib" if _qlib_available() else "akshare"

    result: dict = {
        "mode": effective_mode,
        "updated_at": datetime.now(tz).isoformat(),
        "h5": [], "h10": [], "h20": [],
        "feature_importance": {},
        "error": None,
    }

    try:
        if effective_mode == "qlib":
            horizon_data = _generate_signals_qlib(progress_cb)
            result.update(horizon_data)
        else:
            # ── AKShare 模式 ──────────────────────────────────────────────────
            if progress_cb: progress_cb(0.03)
            codes = get_csi300_universe()[:UNIVERSE_SIZE]
            if not codes:
                raise ValueError("无法获取股票池")

            # 使用缓存的 OHLCV 数据
            stock_data: dict[str, pd.DataFrame] = {}
            STOCK_DATA_CACHE = CACHE_DIR / "stock_data.pkl"
            if use_cache and STOCK_DATA_CACHE.exists():
                try:
                    cached = pickle.loads(STOCK_DATA_CACHE.read_bytes())
                    age_h  = (datetime.now(tz).timestamp() - cached.get("timestamp", 0)) / 3600
                    if age_h < 12:
                        stock_data = cached["data"]
                except Exception:
                    pass

            if not stock_data:
                if progress_cb: progress_cb(0.05)
                stock_data = fetch_stock_data(
                    codes,
                    progress_cb=lambda p: progress_cb(0.05 + p * 0.35) if progress_cb else None,
                )
                STOCK_DATA_CACHE.write_bytes(pickle.dumps({
                    "timestamp": datetime.now(tz).timestamp(), "data": stock_data,
                }))

            if progress_cb: progress_cb(0.42)
            panel = build_panel(stock_data)
            if panel.empty:
                raise ValueError("面板数据为空")

            # 获取股票名称
            spot_df = None
            try:
                spot_df = ak.stock_zh_a_spot_em()[["代码", "名称"]]
            except Exception:
                pass

            # 为每个周期训练模型并预测
            fi_dict: dict = {}
            for i, h in enumerate(HORIZONS):
                base_prog = 0.44 + i * 0.17
                if progress_cb: progress_cb(base_prog)

                model_path = CACHE_DIR / f"lgbm_h{h}.pkl"
                model_ts   = CACHE_DIR / f"lgbm_h{h}.ts"
                model: lgb.Booster

                if use_cache and model_path.exists() and model_ts.exists():
                    try:
                        age_h = (datetime.now(tz).timestamp() - float(model_ts.read_text())) / 3600
                        if age_h < 24:
                            model = pickle.loads(model_path.read_bytes())
                        else:
                            raise ValueError("stale")
                    except Exception:
                        model = train_lgbm_horizon(panel, h)
                        model_path.write_bytes(pickle.dumps(model))
                        model_ts.write_text(str(datetime.now(tz).timestamp()))
                else:
                    model = train_lgbm_horizon(panel, h)
                    model_path.write_bytes(pickle.dumps(model))
                    model_ts.write_text(str(datetime.now(tz).timestamp()))

                result[f"h{h}"] = predict_top_n(model, panel, h, spot_df)

                # 特征重要性
                imp = dict(zip(model.feature_name(), model.feature_importance(importance_type="gain")))
                fi_dict[f"h{h}"] = [
                    {"feature": k, "importance": float(v)}
                    for k, v in sorted(imp.items(), key=lambda x: -x[1])[:8]
                ]
                if progress_cb: progress_cb(base_prog + 0.15)

            result["feature_importance"] = fi_dict

    except Exception as exc:
        result["error"] = traceback.format_exc()
        logger.exception("信号生成失败: %s", exc)

    if progress_cb: progress_cb(1.0)
    SIGNALS_CACHE.write_text(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return result
