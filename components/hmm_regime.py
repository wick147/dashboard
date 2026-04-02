"""
HMM market-regime detector.

Fits a Gaussian HMM with 3 hidden states on CSI300 daily features:
  • daily log-return
  • 5-day realised volatility
  • 20-day trend strength (price / MA20 – 1)

States are post-hoc labelled as:
  bull  — highest mean return state
  bear  — lowest mean return state
  side  — middle (sideways / transition)
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from hmmlearn.hmm import GaussianHMM

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    HMM_N_STATES, HMM_N_ITER, HMM_COVARIANCE,
    HMM_LOOKBACK_DAYS,
    REGIME_CACHE, CACHE_DIR, TZ,
)

_HMM_MODEL_CACHE = CACHE_DIR / "hmm_model.pkl"

REGIME_LABELS = {0: "熊市 Bear", 1: "震荡 Sideways", 2: "牛市 Bull"}
REGIME_COLORS = {"熊市 Bear": "#EF5350", "震荡 Sideways": "#FFA726", "牛市 Bull": "#66BB6A"}


# ── feature engineering ───────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> np.ndarray:
    c = df["close"].astype(float)
    log_ret  = np.log(c / c.shift(1))
    rv5      = log_ret.rolling(5).std()
    ma20     = c.rolling(20).mean()
    trend    = (c / ma20) - 1

    feat = pd.DataFrame({"ret": log_ret, "rv5": rv5, "trend": trend})
    feat = feat.dropna()
    return feat


# ── fit / label states ────────────────────────────────────────────────────────

def _label_states(model: GaussianHMM) -> dict[int, str]:
    """
    Assign bull / side / bear based on mean return of each hidden state.
    """
    mean_rets = model.means_[:, 0]   # first feature is log-return
    order = np.argsort(mean_rets)    # ascending
    mapping = {}
    labels = ["熊市 Bear", "震荡 Sideways", "牛市 Bull"]
    for rank, state_idx in enumerate(order):
        mapping[int(state_idx)] = labels[rank]
    return mapping


# ── main ──────────────────────────────────────────────────────────────────────

def fit_hmm(hist_df: pd.DataFrame) -> tuple[GaussianHMM, dict[int, str], pd.DataFrame]:
    """Fit HMM on historical CSI300 data; return (model, label_map, feature_df)."""
    feat_df = _build_features(hist_df)
    X = feat_df.values

    model = GaussianHMM(
        n_components=HMM_N_STATES,
        covariance_type=HMM_COVARIANCE,
        n_iter=HMM_N_ITER,
        random_state=42,
    )
    model.fit(X)
    label_map = _label_states(model)
    return model, label_map, feat_df


def detect_regime(hist_df: pd.DataFrame, use_cache: bool = True) -> dict:
    """
    Run HMM on CSI300 history and return regime info dict.

    Keys: updated_at, current_regime, current_state, history (list of {date, state, regime})
          state_stats (list of {state, regime, mean_ret, mean_rv})
          transition_matrix (list of lists)
    """
    tz = pytz.timezone(TZ)

    if use_cache and REGIME_CACHE.exists():
        try:
            data = json.loads(REGIME_CACHE.read_text())
            updated = datetime.fromisoformat(data["updated_at"])
            age_h = (datetime.now(tz) - updated).total_seconds() / 3600
            if age_h < 12:
                return data
        except Exception:
            pass

    model, label_map, feat_df = fit_hmm(hist_df)

    hidden_states = model.predict(feat_df.values)
    current_state = int(hidden_states[-1])
    current_regime = label_map[current_state]

    history = []
    for date, state in zip(feat_df.index, hidden_states):
        history.append({
            "date":   str(date)[:10],
            "state":  int(state),
            "regime": label_map[int(state)],
        })

    # Per-state statistics
    state_stats = []
    for s in range(HMM_N_STATES):
        mask = hidden_states == s
        if mask.sum() == 0:
            continue
        state_stats.append({
            "state":    s,
            "regime":   label_map[s],
            "mean_ret": float(feat_df["ret"].values[mask].mean()) * 252,  # annualised
            "mean_rv":  float(feat_df["rv5"].values[mask].mean()) * np.sqrt(252),
            "days":     int(mask.sum()),
            "pct":      round(float(mask.mean()) * 100, 1),
        })

    result = {
        "updated_at":       datetime.now(tz).isoformat(),
        "current_regime":   current_regime,
        "current_state":    current_state,
        "history":          history[-500:],   # last 500 days for charts
        "state_stats":      state_stats,
        "transition_matrix": model.transmat_.tolist(),
        "error":            None,
    }

    # Cache model
    _HMM_MODEL_CACHE.write_bytes(pickle.dumps((model, label_map)))
    REGIME_CACHE.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
