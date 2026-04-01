#!/usr/bin/env python3
"""
Daily pre-market runner — executes at 08:00 CST and pre-computes:
  • Market data  (indices, sectors, breadth)
  • LightGBM signals
  • HMM regime
  • News briefing

Results are saved to dashboard/results/ as JSON files so the
Streamlit app can display them instantly without re-running models.

Usage:
    # Manual
    python dashboard/run_daily.py

    # Via cron (08:00 Beijing time = 00:00 UTC)
    # 0 0 * * 1-5  /path/to/venv/bin/python /path/to/dashboard/run_daily.py >> /tmp/qlib_dashboard.log 2>&1

    # Via Claude Code schedule
    # /schedule  (see instructions at bottom of this file)
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pytz

DASH_DIR = Path(__file__).parent
sys.path.insert(0, str(DASH_DIR))

from config import TZ, RESULTS_DIR
from components.market_data import fetch_market_data, get_index_history
from components.lgbm_signals import generate_signals
from components.hmm_regime import detect_regime
from components.news import fetch_news
from components.notify import send_wechat
from config import SIGNALS_AKSHARE_CACHE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("run_daily")


def banner(msg: str) -> None:
    bar = "─" * 60
    log.info("%s\n  %s\n%s", bar, msg, bar)


def run(signal_mode: str = "auto") -> None:
    tz = pytz.timezone(TZ)
    start = time.time()
    today = datetime.now(tz).strftime("%Y-%m-%d %H:%M CST")
    banner(f"A股量化监控日报  {today}")

    # ── 1. Market data ────────────────────────────────────────────────────────
    banner("Step 1/4 — 行情数据 (AKShare)")
    try:
        market = fetch_market_data()
        n_idx = len(market.get("indices", []))
        n_sec = len(market.get("sector", []))
        log.info("OK  indices=%d  sectors=%d", n_idx, n_sec)
    except Exception as exc:
        log.error("行情数据失败: %s", exc)

    # ── 2. LightGBM signals ───────────────────────────────────────────────────
    banner(f"Step 2/4 — LightGBM 选股信号 (mode={signal_mode})")
    try:
        step_pct = [0]

        def _progress(p: float) -> None:
            pct = int(p * 100)
            if pct != step_pct[0] and pct % 10 == 0:
                log.info("  信号进度 %d%%", pct)
            step_pct[0] = pct

        signals = generate_signals(mode=signal_mode, progress_cb=_progress,
                                   use_cache=False, cache_path=SIGNALS_AKSHARE_CACHE)
        err = signals.get("error")
        if err:
            log.error("信号生成出错:\n%s", err[:500])
        else:
            log.info(
                "OK  mode=%s  top_buy=%d  top_sell=%d",
                signals.get("mode"), len(signals.get("top_buy", [])),
                len(signals.get("top_sell", [])),
            )
    except Exception as exc:
        log.error("LightGBM 信号失败: %s", exc)

    # ── 3. HMM regime ─────────────────────────────────────────────────────────
    banner("Step 3/4 — HMM 市场状态")
    try:
        hist_df = get_index_history()
        if hist_df is not None and not hist_df.empty:
            regime = detect_regime(hist_df, use_cache=False)
            log.info(
                "OK  当前状态: %s  (state=%d)",
                regime.get("current_regime", "?"),
                regime.get("current_state", -1),
            )
        else:
            log.warning("无法获取指数历史数据，跳过 HMM")
    except Exception as exc:
        log.error("HMM 失败: %s", exc)

    # ── 4. News ────────────────────────────────────────────────────────────────
    banner("Step 4/4 — 财经要闻")
    try:
        news = fetch_news(use_cache=False)
        log.info("OK  新闻条数=%d", len(news.get("items", [])))
    except Exception as exc:
        log.error("新闻获取失败: %s", exc)

    # ── 5. WeChat notification ────────────────────────────────────────────────
    banner("Step 5/5 — 微信推送 (Server酱)")
    try:
        sig_data    = json.loads(RESULTS_DIR.joinpath("signals.json").read_text()) if RESULTS_DIR.joinpath("signals.json").exists() else {}
        regime_data = json.loads(RESULTS_DIR.joinpath("regime.json").read_text())  if RESULTS_DIR.joinpath("regime.json").exists()  else {}
        mkt_data    = json.loads(RESULTS_DIR.joinpath("market.json").read_text())  if RESULTS_DIR.joinpath("market.json").exists()  else {}
        ok = send_wechat(sig_data, regime_data, mkt_data)
        if not ok:
            log.info("（未配置 SERVERCHAN_KEY，跳过）")
    except Exception as exc:
        log.error("微信推送失败: %s", exc)

    elapsed = time.time() - start
    banner(f"完成！耗时 {elapsed:.1f}s  结果存于 {RESULTS_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A股量化监控日报生成器")
    parser.add_argument(
        "--mode", choices=["auto", "qlib", "akshare"], default="auto",
        help="LightGBM 信号模式 (default: auto)",
    )
    args = parser.parse_args()
    run(signal_mode=args.mode)
