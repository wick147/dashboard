from pathlib import Path
import os

DASHBOARD_DIR = Path(__file__).parent
CACHE_DIR = DASHBOARD_DIR / "cache"
RESULTS_DIR = DASHBOARD_DIR / "results"

for _d in [CACHE_DIR, RESULTS_DIR]:
    _d.mkdir(exist_ok=True)

# ── Qlib ──────────────────────────────────────────────────────────────────────
QLIB_DATA_PATH = os.environ.get(
    "QLIB_DATA_PATH",
    str(Path.home() / ".qlib/qlib_data/cn_data"),
)

# ── LightGBM (mirrors qlib Alpha158 benchmark params) ─────────────────────────
LGBM_PARAMS = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 150,
    "colsample_bytree": 0.8879,
    "subsample": 0.8789,
    "reg_alpha": 2.0,
    "reg_lambda": 5.0,
    "verbose": -1,
    "n_jobs": 2,
}
LGBM_EARLY_STOP = 40
LGBM_VERBOSE_EVAL = 50

# ── HMM ───────────────────────────────────────────────────────────────────────
HMM_N_STATES = 3          # bull / sideways / bear
HMM_N_ITER = 200
HMM_COVARIANCE = "full"
HMM_LOOKBACK_DAYS = 500   # days of history for fitting

# ── Universe & signals ────────────────────────────────────────────────────────
UNIVERSE_SIZE = 9999       # 全市场不限制（GitHub Actions 跑），实际约 4800 支（除北交所/ST）
SIGNAL_TOP_N = 10          # top N per horizon
HORIZONS = [5, 10, 20]     # forward return horizons (trading days)
FEATURE_LOOKBACK = 160     # calendar days of OHLCV history (needs room for 20d labels)
FETCH_WORKERS = 16         # parallel AKShare fetch threads（Actions 有 2 核，可加速）

# ── Index symbols (AKShare convention) ───────────────────────────────────────
INDICES = {
    "上证指数": "sh000001",
    "深证成指": "sz399001",
    "沪深300": "sh000300",
    "中证500": "sh000905",
    "创业板指": "sz399006",
    "科创50":  "sh000688",
}
CSI300_SYMBOL = "sh000300"

# ── Cache filenames ───────────────────────────────────────────────────────────
SIGNALS_CACHE         = RESULTS_DIR / "signals.json"       # backward compat
SIGNALS_AKSHARE_CACHE = RESULTS_DIR / "signals_akshare.json"
SIGNALS_QLIB_CACHE    = RESULTS_DIR / "signals_qlib.json"
REGIME_CACHE   = RESULTS_DIR / "regime.json"
MARKET_CACHE   = RESULTS_DIR / "market.json"
NEWS_CACHE     = RESULTS_DIR / "news.json"
MODEL_CACHE    = CACHE_DIR   / "lgbm_model.pkl"
STOCK_DATA_CACHE = CACHE_DIR / "stock_data.pkl"

# ── Regime colors ────────────────────────────────────────────────────────────
REGIME_COLORS = {
    "熊市 Bear":     "#EF5350",
    "震荡 Sideways": "#FFA726",
    "牛市 Bull":     "#66BB6A",
}

# ── Misc ──────────────────────────────────────────────────────────────────────
TZ = "Asia/Shanghai"
NEWS_COUNT = 20
CACHE_TTL_HOURS = 12
