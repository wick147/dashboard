from pathlib import Path
import os

DASHBOARD_DIR = Path(__file__).parent
CACHE_DIR     = DASHBOARD_DIR / "cache"
RESULTS_DIR   = DASHBOARD_DIR / "results"

for _d in [CACHE_DIR, RESULTS_DIR]:
    _d.mkdir(exist_ok=True)

# ── Qlib ──────────────────────────────────────────────────────────────────────
QLIB_DATA_PATH = os.environ.get(
    "QLIB_DATA_PATH",
    str(Path.home() / ".qlib/qlib_data/cn_data"),
)

# ── LightGBM (mirrors qlib Alpha158 benchmark params) ─────────────────────────
LGBM_PARAMS = {
    "objective":        "regression",
    "boosting_type":    "gbdt",
    "num_leaves":       31,
    "max_depth":        5,
    "learning_rate":    0.05,
    "n_estimators":     150,
    "colsample_bytree": 0.8879,
    "subsample":        0.8789,
    "reg_alpha":        2.0,
    "reg_lambda":       5.0,
    "verbose":          -1,
    "n_jobs":           2,
}
LGBM_EARLY_STOP   = 40
LGBM_VERBOSE_EVAL = 50

# ── HMM ───────────────────────────────────────────────────────────────────────
HMM_N_STATES    = 3        # bull / sideways / bear
HMM_N_ITER      = 200
HMM_COVARIANCE  = "full"
HMM_LOOKBACK_DAYS = 500

# ── Signals ───────────────────────────────────────────────────────────────────
SIGNAL_TOP_N = 10
HORIZONS     = [5, 10, 20]

# ── Cache paths ───────────────────────────────────────────────────────────────
SIGNALS_QLIB_CACHE    = RESULTS_DIR / "signals_qlib.json"
REGIME_CACHE          = RESULTS_DIR / "regime.json"
MARKET_HISTORY_CACHE  = RESULTS_DIR / "market_history.json"

# ── Regime display colors (A股惯例: 红涨绿跌) ──────────────────────────────────
REGIME_COLORS = {
    "牛市 Bull":      "#EF5350",   # 红
    "震荡 Sideways":  "#FFA726",   # 橙
    "熊市 Bear":      "#26A69A",   # 绿
}

# ── Misc ──────────────────────────────────────────────────────────────────────
TZ = "Asia/Shanghai"

# ── GitHub Actions trigger ────────────────────────────────────────────────────
GITHUB_REPO     = os.environ.get("GITHUB_REPO",     "wick147/dashboard")
GITHUB_WORKFLOW = "qlib-signals.yml"
