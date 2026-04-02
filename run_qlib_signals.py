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
import re
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

NAME_API_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://finance.sina.com.cn/",
}


def _normalize_stock_code(raw: str) -> str | None:
    """Normalize a stock code into Qlib style, e.g. SH600588 / SZ000001."""
    code = str(raw or "").strip().upper()
    if len(code) >= 8 and code[:2] in {"SH", "SZ", "BJ"} and code[2:].isdigit():
        return code[:8]

    digits = re.sub(r"\D", "", code)
    if len(digits) < 6:
        return None
    digits = digits[-6:]

    prefix = "SH" if digits[0] in {"5", "6", "9"} else "SZ"
    return f"{prefix}{digits}"


def _vendor_symbol(code: str) -> str | None:
    norm = _normalize_stock_code(code)
    if not norm:
        return None
    market = norm[:2].lower()
    return f"{market}{norm[2:]}"


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _extract_signal_codes(signals: dict) -> list[str]:
    codes: set[str] = set()
    for key in ["h5", "h10", "h20"]:
        for stock in signals.get(key, []):
            norm = _normalize_stock_code(stock.get("code", ""))
            if norm:
                codes.add(norm)
    return sorted(codes)


def _fetch_names_sina(codes: list[str]) -> dict[str, str]:
    """
    Fetch stock names from Sina HQ in small batches.

    Example response line:
      var hq_str_sh600588="用友网络,16.34,...";
    """
    if not codes:
        return {}

    session = requests.Session()
    session.headers.update(NAME_API_HEADERS)

    name_map: dict[str, str] = {}
    symbols = []
    symbol_to_code: dict[str, str] = {}

    for code in codes:
        symbol = _vendor_symbol(code)
        if symbol:
            symbols.append(symbol)
            symbol_to_code[symbol] = _normalize_stock_code(code) or code

    for chunk in _chunked(symbols, 120):
        resp = session.get(
            "https://hq.sinajs.cn/list=" + ",".join(chunk),
            timeout=12,
        )
        resp.raise_for_status()

        text = resp.content.decode("gb18030", errors="ignore")
        for line in text.splitlines():
            m = re.match(r'var hq_str_([a-z0-9]+)="([^"]*)";?', line.strip())
            if not m:
                continue
            symbol, payload = m.groups()
            name = payload.split(",", 1)[0].strip()
            code = symbol_to_code.get(symbol)
            if code and name:
                name_map[code] = name

    return name_map


def _fetch_names_tencent(codes: list[str]) -> dict[str, str]:
    """
    Fallback name source from Tencent quote API.

    Example response line:
      v_sh600588="1~用友网络~600588~16.34~...";
    """
    if not codes:
        return {}

    session = requests.Session()
    session.headers.update(NAME_API_HEADERS)

    name_map: dict[str, str] = {}
    symbols = []
    symbol_to_code: dict[str, str] = {}

    for code in codes:
        symbol = _vendor_symbol(code)
        if symbol:
            symbols.append(symbol)
            symbol_to_code[symbol] = _normalize_stock_code(code) or code

    for chunk in _chunked(symbols, 80):
        resp = session.get(
            "https://qt.gtimg.cn/q=" + ",".join(chunk),
            timeout=12,
        )
        resp.raise_for_status()

        text = resp.content.decode("gb18030", errors="ignore")
        for line in text.splitlines():
            m = re.match(r'v_([a-z0-9]+)="([^"]*)";?', line.strip())
            if not m:
                continue
            symbol, payload = m.groups()
            parts = payload.split("~")
            if len(parts) < 2:
                continue
            name = parts[1].strip()
            code = symbol_to_code.get(symbol)
            if code and name:
                name_map[code] = name

    return name_map


def _enrich_signal_names(signals: dict) -> dict[str, str]:
    codes = _extract_signal_codes(signals)
    if not codes:
        return {}

    resolved: dict[str, str] = {}

    for source_name, fetcher in [
        ("Sina", _fetch_names_sina),
        ("Tencent", _fetch_names_tencent),
    ]:
        missing = [code for code in codes if code not in resolved]
        if not missing:
            break
        try:
            fetched = fetcher(missing)
            resolved.update(fetched)
            log.info(
                "  %s 名称补全: %d/%d",
                source_name,
                len(fetched),
                len(missing),
            )
        except Exception as e:
            log.warning("  %s 名称获取失败: %s", source_name, e)

    unresolved = [code for code in codes if code not in resolved]
    if unresolved:
        log.warning("  仍有 %d 只股票未补全名称", len(unresolved))
    else:
        log.info("  股票名称已全部补全")

    return resolved


def _normalize_date_index(index: pd.Index) -> pd.DatetimeIndex:
    """
    Normalize qlib/pandas date-like indexes into a clean DatetimeIndex.

    qlib data may surface dates as strings like "2026-04-01", integers like
    20260401, or already-parsed timestamps. If integer dates are passed
    directly to pd.to_datetime, pandas interprets them as Unix epoch numbers,
    which yields 1970-era timestamps.
    """
    if isinstance(index, pd.DatetimeIndex):
        return pd.DatetimeIndex(index)

    parsed = []
    for value in index:
        if isinstance(value, pd.Timestamp):
            parsed.append(value)
            continue

        if isinstance(value, (int, np.integer)):
            text = str(int(value))
            if len(text) == 8 and text.isdigit():
                parsed.append(pd.to_datetime(text, format="%Y%m%d", errors="coerce"))
                continue

        text = str(value).strip()
        if len(text) == 8 and text.isdigit():
            parsed.append(pd.to_datetime(text, format="%Y%m%d", errors="coerce"))
            continue

        parsed.append(pd.to_datetime(value, errors="coerce"))

    parsed_index = pd.DatetimeIndex(parsed)
    if parsed_index.isna().any():
        bad_count = int(parsed_index.isna().sum())
        raise ValueError(f"存在无法解析的交易日索引: {bad_count} 条")
    return parsed_index


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

        close_series = df["$close"]
        index_names = list(close_series.index.names)
        if "instrument" in index_names:
            pivot = close_series.unstack(level="instrument")
        else:
            # Fallback for unnamed or legacy index orders: put instruments on columns.
            pivot = close_series.unstack(level=0 if index_names and index_names[0] != "datetime" else 1)

        normed = pivot.div(pivot.iloc[0])           # 归一化到第一天=1
        avg    = normed.mean(axis=1) * 3000         # 缩放到 CSI300 量级
        avg.index = _normalize_date_index(avg.index)
        avg = avg.sort_index()
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

    # ── 补充中文股票名称（只拉 top 信号股票，避免整市场请求不稳定）──────────────
    log.info("  获取股票中文名称...")
    name_map = _enrich_signal_names(signals)

    def _apply_names(stock_list: list) -> list:
        for s in stock_list:
            norm = _normalize_stock_code(s.get("code", ""))
            if norm and norm in name_map:
                s["name"] = name_map[norm]
            elif norm:
                s["name"] = norm[2:]
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
