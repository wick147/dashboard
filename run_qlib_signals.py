#!/usr/bin/env python3
"""
Qlib Alpha158 选股信号生成器（GitHub Actions 专用）。

流程：
  1. 用 ~/.qlib/qlib_data/cn_data 的离线数据
  2. 训练 5/10/20 日 LightGBM 横截面排名模型
  3. 结果保存到 results/signals_qlib.json
  4. 不依赖 AKShare / 任何中国 API

用法：
    python run_qlib_signals.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

DASH_DIR = Path(__file__).parent
sys.path.insert(0, str(DASH_DIR))

from config import SIGNALS_QLIB_CACHE
from components.lgbm_signals import generate_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("run_qlib_signals")


def main() -> None:
    start = time.time()
    log.info("=" * 60)
    log.info("  Qlib Alpha158 选股信号生成（5/10/20日横截面排名）")
    log.info("=" * 60)

    step_pct = [0]

    def _progress(p: float) -> None:
        pct = int(p * 100)
        if pct != step_pct[0] and pct % 10 == 0:
            log.info("  进度 %d%%", pct)
        step_pct[0] = pct

    signals = generate_signals(
        mode="qlib",
        progress_cb=_progress,
        use_cache=False,
        cache_path=SIGNALS_QLIB_CACHE,
    )

    elapsed = time.time() - start
    err = signals.get("error")

    if err:
        log.error("信号生成出错:\n%s", err[:1000])
        sys.exit(1)

    log.info("-" * 60)
    log.info("完成！耗时 %.1fs", elapsed)
    log.info("预测日期: %s", signals.get("pred_date", "?"))
    log.info("训练区间: %s → %s", signals.get("train_start", "?"), signals.get("train_end", "?"))
    log.info("H5  Top持仓: %d 只", len(signals.get("h5", [])))
    log.info("H10 Top持仓: %d 只", len(signals.get("h10", [])))
    log.info("H20 Top持仓: %d 只", len(signals.get("h20", [])))
    log.info("结果已写入: %s", SIGNALS_QLIB_CACHE)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
