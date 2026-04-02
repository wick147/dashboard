#!/usr/bin/env python3
"""
Qlib Alpha158 选股信号 + HMM市场状态生成器（GitHub Actions 专用）。

流程：
  1. 用 ~/.qlib/qlib_data/cn_data 的离线 qlib 数据
  2. 训练 5/10/20 日 LightGBM 横截面排名模型 → results/signals_qlib.json
  3. 从 qlib 数据提取 CSI300 价格历史     → results/market_history.json
  4. 用价格历史拟合 HMM 市场状态           → results/regime.json
  5. 不依赖 AKShare / 任何中国实时行情 API

用法:
    python run_qlib_signals.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

DASH_DIR = Path(__file__).parent
sys.path.insert(0, str(DASH_DIR))

from config import (
    SIGNALS_QLIB_CACHE, REGIME_CACHE, MARKET_HISTORY_CACHE,
    QLIB_DATA_PATH, TZ,
)
from components.lgbm_signals import generate_signals
from components.hmm_regime import detect_regime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("run_qlib_signals")


def _get_csi300_history(start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    从已初始化的 qlib 数据中提取 CSI300 价格历史。

    优先尝试直接读取 sh000300 指数；
    若不存在，则用 CSI300 成分股归一化均值（等权合成指数）作为代理。
    返回包含 'close' 列、DatetimeIndex 的 DataFrame；失败则返回 None。
    """
    from qlib.data import D

    # ── 优先: 直接读取 sh000300 指数 ──────────────────────────────────────────
    try:
        df = D.features(
            ["sh000300"],
            fields=["$close"],
            start_time=start_date,
            end_time=end_date,
            freq="day",
        )
        if not df.empty:
            close = df["$close"].unstack(level=1).iloc[:, 0]
            close.index = pd.to_datetime(close.index)
            log.info("  sh000300 指数数据: %d 条", len(close))
            return pd.DataFrame({"close": close})
    except Exception as e:
        log.warning("  sh000300 直接读取失败，尝试成分股均值: %s", e)

    # ── 备选: CSI300 成分股归一化均值 ─────────────────────────────────────────
    try:
        df = D.features(
            "csi300",
            fields=["$close"],
            start_time=start_date,
            end_time=end_date,
            freq="day",
        )
        pivot = df["$close"].unstack(level=1)           # date x instrument
        normed = pivot.div(pivot.iloc[0])               # 归一化到第一天=1
        avg = normed.mean(axis=1) * 3000                # 缩放到 CSI300 量级
        avg.index = pd.to_datetime(avg.index)
        log.info("  CSI300 成分股均值(代理): %d 条", len(avg))
        return pd.DataFrame({"close": avg})
    except Exception as e:
        log.error("  CSI300 历史数据获取失败: %s", e)
        return None


def main() -> None:
    start  = time.time()
    tz     = pytz.timezone(TZ)
    today  = datetime.now(tz)
    today_str   = today.strftime("%Y-%m-%d")
    hist_start  = (today - timedelta(days=700)).strftime("%Y-%m-%d")

    log.info("=" * 60)
    log.info("  Qlib 量化信号生成器")
    log.info("  运行日期: %s", today_str)
    log.info("=" * 60)

    # ── Step 1: LightGBM 选股信号 ─────────────────────────────────────────────
    log.info("[1/3] LightGBM Alpha158 选股信号 (5/10/20日)...")

    step_pct = [0]
    def _progress(p: float) -> None:
        pct = int(p * 100)
        if pct != step_pct[0] and pct % 10 == 0:
            log.info("  进度 %d%%", pct)
        step_pct[0] = pct

    signals = generate_signals(
        progress_cb=_progress,
        use_cache=False,
        cache_path=SIGNALS_QLIB_CACHE,
    )

    if signals.get("error"):
        log.error("LightGBM 信号生成失败:\n%s", signals["error"][:800])
        sys.exit(1)

    log.info(
        "  完成  H5: %d只  H10: %d只  H20: %d只  预测日: %s",
        len(signals.get("h5", [])),
        len(signals.get("h10", [])),
        len(signals.get("h20", [])),
        signals.get("pred_date", "?"),
    )

    # ── Step 2: CSI300 历史数据 ───────────────────────────────────────────────
    log.info("[2/3] 提取 CSI300 历史数据...")

    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri=QLIB_DATA_PATH, region=REG_CN)

    hist_df = _get_csi300_history(hist_start, today_str)

    if hist_df is not None and not hist_df.empty:
        market_history = {
            "updated_at": today.isoformat(),
            "data": [
                {"date": str(idx.date()), "close": round(float(row["close"]), 2)}
                for idx, row in hist_df.iterrows()
            ],
        }
        MARKET_HISTORY_CACHE.write_text(
            json.dumps(market_history, ensure_ascii=False, indent=2)
        )
        log.info("  已写入 market_history.json (%d 条)", len(market_history["data"]))
    else:
        log.warning("  无法获取 CSI300 数据，跳过写入")

    # ── Step 3: HMM 市场状态 ──────────────────────────────────────────────────
    log.info("[3/3] HMM 市场状态识别...")

    if hist_df is not None and not hist_df.empty:
        try:
            regime = detect_regime(hist_df, use_cache=False)
            log.info("  当前状态: %s", regime.get("current_regime", "?"))
        except Exception as e:
            log.error("  HMM 计算失败: %s", e)
    else:
        log.warning("  无 CSI300 数据，跳过 HMM")

    elapsed = time.time() - start
    log.info("=" * 60)
    log.info("全部完成！耗时 %.1fs", elapsed)
    log.info("  signals_qlib.json     → %s", SIGNALS_QLIB_CACHE)
    log.info("  market_history.json   → %s", MARKET_HISTORY_CACHE)
    log.info("  regime.json           → %s", REGIME_CACHE)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
