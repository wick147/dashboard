"""
LightGBM stock-selection signal generator.

Two modes (auto-detected):
  1. qlib mode  — uses Alpha158 features from the local qlib data store.
  2. akshare mode — fetches OHLCV from AKShare and computes technical features.
"""
from __future__ import annotations

import json
import logging
import os
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
from sklearn.preprocessing import RobustScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LGBM_PARAMS, LGBM_EARLY_STOP, LGBM_VERBOSE_EVAL,
    UNIVERSE_SIZE, SIGNAL_TOP_N, FEATURE_LOOKBACK,
    FETCH_WORKERS, SIGNALS_CACHE, MODEL_CACHE, STOCK_DATA_CACHE,
    QLIB_DATA_PATH, TZ,
)

logger = logging.getLogger(__name__)


# ── universe ──────────────────────────────────────────────────────────────────

def get_csi300_universe() -> list[str]:
    """Return 6-digit stock codes in CSI300."""
    try:
        df = ak.index_stock_cons_weight_csindex(symbol="000300")
        codes = df["成分券代码"].astype(str).str.zfill(6).tolist()
        return codes
    except Exception:
        pass
    # fallback: use spot data market-cap top stocks
    try:
        spot = ak.stock_zh_a_spot_em()
        spot["总市值"] = pd.to_numeric(spot.get("总市值", spot.get("市值", 0)), errors="coerce")
        spot = spot.dropna(subset=["总市值"]).nlargest(UNIVERSE_SIZE, "总市值")
        return spot["代码"].astype(str).str.zfill(6).tolist()
    except Exception:
        return []


# ── AKShare feature engineering ───────────────────────────────────────────────

def _fetch_one_stock(code: str, start: str, end: str) -> tuple[str, pd.DataFrame | None]:
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
            "成交额": "amount", "涨跌幅": "pct",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return code, df
    except Exception:
        return code, None


def fetch_stock_data(codes: list[str], lookback: int = FEATURE_LOOKBACK,
                     progress_cb: Optional[Callable] = None) -> dict[str, pd.DataFrame]:
    """Parallel-fetch OHLCV for each code; returns {code: df}."""
    tz = pytz.timezone(TZ)
    end = datetime.now(tz).strftime("%Y%m%d")
    start = (datetime.now(tz) - timedelta(days=lookback + 30)).strftime("%Y%m%d")

    results: dict[str, pd.DataFrame] = {}
    total = len(codes)
    done = 0
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {pool.submit(_fetch_one_stock, c, start, end): c for c in codes}
        for fut in as_completed(futures):
            code, df = fut.result()
            if df is not None and len(df) >= 30:
                results[code] = df
            done += 1
            if progress_cb:
                progress_cb(done / total)
    return results


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features for a single stock DataFrame."""
    c = df["close"].astype(float)
    v = df["volume"].astype(float)

    feat = pd.DataFrame(index=df.index)
    for d in [1, 5, 10, 20]:
        feat[f"ret{d}"] = c.pct_change(d)
    for w in [5, 10, 20]:
        feat[f"vol_rv{w}"] = c.pct_change().rolling(w).std()
    feat["vol_ratio5"]  = v / v.rolling(5).mean()
    feat["vol_ratio20"] = v / v.rolling(20).mean()

    # RSI-14
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    feat["rsi14"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

    # Bollinger position
    ma20  = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    feat["bb_pos"] = (c - ma20) / (2 * std20.replace(0, 1e-9))

    # MACD signal
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    feat["macd_sig"] = macd - macd.ewm(span=9, adjust=False).mean()

    # Price vs. 20-day high (momentum)
    feat["hi20_pos"] = c / c.rolling(20).max()

    # Next-day return as label
    feat["label"] = c.pct_change(1).shift(-1)

    feat["code"] = df.name if hasattr(df, "name") else "unknown"
    return feat.dropna()


def build_panel(stock_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-stock feature frames into a cross-sectional panel."""
    frames = []
    for code, df in stock_data.items():
        df = df.copy()
        df.name = code
        try:
            feat = _compute_features(df)
            feat["code"] = code
            frames.append(feat)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames).sort_index()
    return panel


FEATURE_COLS = [
    "ret1", "ret5", "ret10", "ret20",
    "vol_rv5", "vol_rv10", "vol_rv20",
    "vol_ratio5", "vol_ratio20",
    "rsi14", "bb_pos", "macd_sig", "hi20_pos",
]


def train_lgbm(panel: pd.DataFrame) -> lgb.Booster:
    """Train LightGBM on historical cross-sectional panel."""
    # Use all but the last trading day for training
    dates = panel.index.unique().sort_values()
    if len(dates) < 10:
        raise ValueError("Not enough history to train model.")
    train_dates = dates[:-1]
    train = panel[panel.index.isin(train_dates)].dropna(subset=FEATURE_COLS + ["label"])

    X = train[FEATURE_COLS].values
    y = train["label"].values

    # Cross-sectional rank normalise labels per day
    rank_labels = (
        train.assign(_y=y)
        .groupby(level=0)["_y"]
        .rank(pct=True)
        .values - 0.5
    )

    dtrain = lgb.Dataset(X, label=rank_labels)
    params = {**LGBM_PARAMS}
    n_est = params.pop("n_estimators", 400)
    model = lgb.train(
        params, dtrain,
        num_boost_round=n_est,
        callbacks=[lgb.log_evaluation(period=LGBM_VERBOSE_EVAL)],
    )
    return model


def predict_signals(
    model: lgb.Booster,
    panel: pd.DataFrame,
    spot_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Predict scores for the latest trading day."""
    last_date = panel.index.max()
    latest = panel[panel.index == last_date].copy()
    if latest.empty:
        return pd.DataFrame()
    X = latest[FEATURE_COLS].fillna(0).values
    latest["score"] = model.predict(X)
    latest = latest.sort_values("score", ascending=False)

    # Attach names from spot data if available
    if spot_df is not None and "代码" in spot_df.columns:
        name_map = spot_df.set_index("代码")["名称"].to_dict() if "名称" in spot_df.columns else {}
        latest["name"] = latest["code"].map(name_map).fillna("")
    else:
        latest["name"] = ""
    return latest[["code", "name", "score"] + FEATURE_COLS].reset_index()


# ── qlib mode ─────────────────────────────────────────────────────────────────

def _qlib_available() -> bool:
    # qlib 必须能 import，且本地二进制数据目录非空
    try:
        import qlib  # noqa: F401
    except ImportError:
        return False
    data_path = Path(QLIB_DATA_PATH).expanduser()
    return data_path.exists() and any(data_path.iterdir())


def generate_signals_qlib(progress_cb: Optional[Callable] = None) -> pd.DataFrame:
    """Use qlib Alpha158 + LGBModel for signal generation."""
    import qlib
    from qlib.config import REG_CN
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    tz = pytz.timezone(TZ)
    today = datetime.now(tz)
    train_end = (today - timedelta(days=365)).strftime("%Y-%m-%d")
    pred_start = (today - timedelta(days=10)).strftime("%Y-%m-%d")
    pred_end   = today.strftime("%Y-%m-%d")

    qlib.init(provider_uri=QLIB_DATA_PATH, region=REG_CN)
    if progress_cb:
        progress_cb(0.2)

    handler = Alpha158(
        start_time="2020-01-01",
        end_time=pred_end,
        fit_start_time="2020-01-01",
        fit_end_time=train_end,
        instruments="csi300",
    )
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": ("2020-01-01", train_end),
            "test":  (pred_start, pred_end),
        },
    )
    if progress_cb:
        progress_cb(0.5)

    model = LGBModel(
        loss="mse",
        colsample_bytree=LGBM_PARAMS["colsample_bytree"],
        learning_rate=0.1,
        subsample=LGBM_PARAMS["subsample"],
        num_leaves=LGBM_PARAMS["num_leaves"],
        max_depth=LGBM_PARAMS["max_depth"],
        num_boost_round=300,
        early_stopping_rounds=LGBM_EARLY_STOP,
    )
    model.fit(dataset)
    if progress_cb:
        progress_cb(0.85)

    pred = model.predict(dataset, segment="test")
    pred_df = pred.reset_index()
    pred_df.columns = ["date", "code", "score"]
    latest_date = pred_df["date"].max()
    result = pred_df[pred_df["date"] == latest_date].sort_values("score", ascending=False)
    if progress_cb:
        progress_cb(1.0)
    return result.reset_index(drop=True)


# ── main entry ─────────────────────────────────────────────────────────────────

def generate_signals(
    mode: str = "auto",
    progress_cb: Optional[Callable] = None,
    use_cache: bool = True,
) -> dict:
    """
    Generate LightGBM stock signals.

    Returns dict with keys:
      mode, updated_at, top_buy (list), top_sell (list), feature_importance (list)
    """
    # ── check cache freshness ─────────────────────────────────────────────────
    if use_cache and SIGNALS_CACHE.exists():
        try:
            data = json.loads(SIGNALS_CACHE.read_text())
            updated = datetime.fromisoformat(data["updated_at"])
            age_h = (datetime.now(pytz.timezone(TZ)) - updated).total_seconds() / 3600
            if age_h < 12:
                return data
        except Exception:
            pass

    tz = pytz.timezone(TZ)
    updated_at = datetime.now(tz).isoformat()

    # ── choose mode ───────────────────────────────────────────────────────────
    effective_mode = mode
    if mode == "auto":
        effective_mode = "qlib" if _qlib_available() else "akshare"

    result: dict = {"mode": effective_mode, "updated_at": updated_at,
                    "top_buy": [], "top_sell": [], "feature_importance": [],
                    "error": None}
    try:
        if effective_mode == "qlib":
            signals_df = generate_signals_qlib(progress_cb)
            top_buy  = signals_df.head(SIGNAL_TOP_N)[["code", "score"]].to_dict("records")
            top_sell = signals_df.tail(SIGNAL_TOP_N)[["code", "score"]].to_dict("records")
            result["top_buy"]  = top_buy
            result["top_sell"] = top_sell
        else:
            # AKShare mode
            if progress_cb:
                progress_cb(0.05)
            codes = get_csi300_universe()[:UNIVERSE_SIZE]
            if not codes:
                raise ValueError("Failed to get stock universe.")

            # Try to reuse cached stock data
            stock_data: dict[str, pd.DataFrame] = {}
            if use_cache and STOCK_DATA_CACHE.exists():
                try:
                    cached = pickle.loads(STOCK_DATA_CACHE.read_bytes())
                    stamp  = cached.get("timestamp", 0)
                    age_h  = (datetime.now(tz).timestamp() - stamp) / 3600
                    if age_h < 12:
                        stock_data = cached["data"]
                except Exception:
                    pass

            if not stock_data:
                if progress_cb:
                    progress_cb(0.1)
                fetched_pct = [0.0]

                def _pcb(p: float) -> None:
                    fetched_pct[0] = p
                    if progress_cb:
                        progress_cb(0.1 + p * 0.4)

                stock_data = fetch_stock_data(codes, progress_cb=_pcb)
                STOCK_DATA_CACHE.write_bytes(pickle.dumps({
                    "timestamp": datetime.now(tz).timestamp(),
                    "data": stock_data,
                }))

            if progress_cb:
                progress_cb(0.55)

            panel = build_panel(stock_data)
            if panel.empty:
                raise ValueError("No panel data built.")

            if progress_cb:
                progress_cb(0.65)

            # Train or load model
            model: lgb.Booster
            if use_cache and MODEL_CACHE.exists():
                try:
                    ts_path = MODEL_CACHE.with_suffix(".ts")
                    if ts_path.exists():
                        age_h = (datetime.now(tz).timestamp() - float(ts_path.read_text())) / 3600
                        if age_h < 24:
                            model = pickle.loads(MODEL_CACHE.read_bytes())
                        else:
                            raise ValueError("stale")
                    else:
                        raise ValueError("no ts")
                except Exception:
                    model = train_lgbm(panel)
                    MODEL_CACHE.write_bytes(pickle.dumps(model))
                    MODEL_CACHE.with_suffix(".ts").write_text(str(datetime.now(tz).timestamp()))
            else:
                model = train_lgbm(panel)
                MODEL_CACHE.write_bytes(pickle.dumps(model))
                MODEL_CACHE.with_suffix(".ts").write_text(str(datetime.now(tz).timestamp()))

            if progress_cb:
                progress_cb(0.9)

            # Get spot names
            spot_df = None
            try:
                spot_df = ak.stock_zh_a_spot_em()[["代码", "名称"]]
            except Exception:
                pass

            signals_df = predict_signals(model, panel, spot_df)
            if signals_df.empty:
                raise ValueError("Empty prediction result.")

            top_buy  = signals_df.head(SIGNAL_TOP_N)[["code", "name", "score", "ret1", "ret5"]].to_dict("records")
            top_sell = signals_df.tail(SIGNAL_TOP_N)[::-1][["code", "name", "score", "ret1", "ret5"]].to_dict("records")

            # Feature importance
            imp = pd.Series(
                dict(zip(model.feature_name(), model.feature_importance(importance_type="gain")))
            ).nlargest(10)
            result["feature_importance"] = [{"feature": k, "importance": float(v)}
                                             for k, v in imp.items()]
            result["top_buy"]  = [
                {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
                for r in top_buy
            ]
            result["top_sell"] = [
                {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
                for r in top_sell
            ]
    except Exception as exc:
        result["error"] = traceback.format_exc()
        logger.exception("Signal generation failed: %s", exc)

    if progress_cb:
        progress_cb(1.0)

    SIGNALS_CACHE.write_text(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return result
