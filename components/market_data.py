"""
Market data component — fetches index performance, sector rotation,
and market movers from AKShare.
"""
from __future__ import annotations

import json
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
import pytz

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import INDICES, CSI300_SYMBOL, TZ, MARKET_CACHE


# ── helpers ───────────────────────────────────────────────────────────────────

def _today_str() -> str:
    tz = pytz.timezone(TZ)
    return datetime.now(tz).strftime("%Y%m%d")


def _safe(fn, *args, default=None, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


# ── index performance ─────────────────────────────────────────────────────────

def get_index_performance() -> pd.DataFrame:
    """Return latest close + pct-change for each major index."""
    rows = []
    for name, symbol in INDICES.items():
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
            df = df.tail(2)
            prev, curr = df.iloc[-2]["close"], df.iloc[-1]["close"]
            rows.append({
                "名称": name,
                "代码": symbol,
                "最新价": round(curr, 2),
                "涨跌幅": round((curr - prev) / prev * 100, 2),
                "昨收": round(prev, 2),
                "日期": str(df.index[-1])[:10],
            })
        except Exception as exc:
            rows.append({"名称": name, "代码": symbol, "最新价": None,
                         "涨跌幅": None, "昨收": None, "日期": None})
    return pd.DataFrame(rows)


def get_index_history(symbol: str = CSI300_SYMBOL, days: int = 500) -> pd.DataFrame | None:
    """Return OHLCV history for an index."""
    try:
        df = ak.stock_zh_index_daily(symbol=symbol)
        df.index = pd.to_datetime(df.index)
        return df.tail(days)
    except Exception:
        return None


# ── sector performance ────────────────────────────────────────────────────────

def get_sector_performance() -> pd.DataFrame | None:
    """Return industry board performance from East Money."""
    try:
        df = ak.stock_board_industry_spot_em()
        df = df[["板块名称", "涨跌幅", "涨跌额", "总市值", "换手率", "上涨家数", "下跌家数"]].copy()
        df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
        df = df.dropna(subset=["涨跌幅"]).sort_values("涨跌幅", ascending=False)
        return df.reset_index(drop=True)
    except Exception:
        return None


# ── market breadth ────────────────────────────────────────────────────────────

def get_market_breadth() -> dict:
    """Count A-share advancing / declining / flat stocks."""
    try:
        spot = ak.stock_zh_a_spot_em()
        chg = pd.to_numeric(spot["涨跌幅"], errors="coerce").dropna()
        return {
            "up": int((chg > 0).sum()),
            "down": int((chg < 0).sum()),
            "flat": int((chg == 0).sum()),
            "limit_up": int((chg >= 9.9).sum()),
            "limit_down": int((chg <= -9.9).sum()),
        }
    except Exception:
        return {"up": None, "down": None, "flat": None,
                "limit_up": None, "limit_down": None}


# ── main entry ────────────────────────────────────────────────────────────────

def fetch_market_data() -> dict:
    """Fetch all market data and return as a serialisable dict."""
    index_df = get_index_performance()
    sector_df = get_sector_performance()
    breadth = get_market_breadth()

    out = {
        "updated_at": datetime.now(pytz.timezone(TZ)).isoformat(),
        "indices": index_df.to_dict(orient="records"),
        "sector": sector_df.to_dict(orient="records") if sector_df is not None else [],
        "breadth": breadth,
    }
    MARKET_CACHE.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def load_market_data() -> dict:
    """Load from cache if fresh, otherwise re-fetch."""
    if MARKET_CACHE.exists():
        try:
            data = json.loads(MARKET_CACHE.read_text())
            updated = datetime.fromisoformat(data["updated_at"])
            age_h = (datetime.now(pytz.timezone(TZ)) - updated).total_seconds() / 3600
            if age_h < 12:
                return data
        except Exception:
            pass
    return fetch_market_data()
