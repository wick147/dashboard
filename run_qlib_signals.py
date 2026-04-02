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
import requests

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

    用 D.list_instruments(D.instruments("csi300")) 获取成分股列表，
    再调 D.features() 拿收盘价，归一化后合成等权指数（缩放至 ~3000 量级）。
    返回包含 'close' 列、DatetimeIndex 的 DataFrame；失败则返回 None。
    """
    from qlib.data import D

    try:
        # 正确获取 CSI300 成分股列表（不能直接把字符串 "csi300" 传给 D.features）
        inst = D.list_instruments(
            D.instruments("csi300"),
            start_time=start_date,
            end_time=end_date,
            as_list=True,
        )
        if not inst:
            log.error("  CSI300 成分股列表为空")
            return None
        log.info("  CSI300 成分股: %d 只", len(inst))

        df = D.features(inst, ["$close"], start_time=start_date, end_time=end_date)
        pivot  = df["$close"].unstack(level=1)     # date x instrument
        normed = pivot.div(pivot.iloc[0])           # 归一化到第一天=1
        avg    = normed.mean(axis=1) * 3000         # 缩放到 CSI300 量级
        avg.index = pd.to_datetime(avg.index)
        log.info("  CSI300 等权合成指数: %d 条", len(avg))
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

    # ── 补充中文股票名称（Eastmoney 原始 API，不依赖 AKShare）────────────────
    log.info("  获取股票中文名称...")
    name_map: dict[str, str] = {}
    try:
        for market in ["m:0+t:6,m:0+t:13,m:1+t:2,m:1+t:23"]:   # 沪深A股
            resp = requests.get(
                "http://push2.eastmoney.com/api/qt/clist/get",
                params={
                    "pn": 1, "pz": 5000, "po": 1, "np": 1,
                    "fltt": 2, "invt": 2, "fid": "f3",
                    "fs": market,
                    "fields": "f12,f14",   # f12=代码, f14=名称
                    "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                },
                timeout=15,
            )
            items = resp.json().get("data", {}).get("diff", [])
            for item in items:
                code = str(item.get("f12", ""))
                name = str(item.get("f14", ""))
                if code and name:
                    name_map[code] = name
        log.info("  获取名称 %d 只", len(name_map))
    except Exception as e:
        log.warning("  名称获取失败（展示代码）: %s", e)

    def _apply_names(stock_list: list) -> list:
        for s in stock_list:
            raw = s.get("code", "")           # e.g. "SH600588"
            short = raw[2:] if raw[:2] in ("SH", "SZ", "BJ") else raw
            s["name"] = name_map.get(short, short)
        return stock_list

    for key in ["h5", "h10", "h20"]:
        signals[key] = _apply_names(signals.get(key, []))

    SIGNALS_QLIB_CACHE.write_text(json.dumps(signals, ensure_ascii=False, indent=2))
    log.info("  名称已写入 signals_qlib.json")

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
