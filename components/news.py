"""
Financial morning briefing — aggregates headlines from multiple AKShare sources.
"""
from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path

import akshare as ak
import pytz

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NEWS_CACHE, NEWS_COUNT, TZ


# ── individual sources ────────────────────────────────────────────────────────

def _fetch_cctv() -> list[dict]:
    """CCTV financial news."""
    try:
        tz = pytz.timezone(TZ)
        today = datetime.now(tz).strftime("%Y%m%d")
        df = ak.news_cctv(date=today)
        if df is None or df.empty:
            yesterday = (datetime.now(tz).replace(hour=0) - __import__("datetime").timedelta(days=1)).strftime("%Y%m%d")
            df = ak.news_cctv(date=yesterday)
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "source": "CCTV财经",
                "time":   str(r.get("date", r.get("datetime", ""))),
                "title":  str(r.get("title", r.get("content", ""))),
                "content": str(r.get("content", "")),
            })
        return rows[:NEWS_COUNT]
    except Exception:
        return []


def _fetch_eastmoney_news() -> list[dict]:
    """East Money (东方财富) real-time news headlines."""
    try:
        df = ak.stock_news_em(symbol="")
        if df is None or df.empty:
            return []
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "source":  "东方财富",
                "time":    str(r.get("发布时间", r.get("time", ""))),
                "title":   str(r.get("新闻标题", r.get("title", ""))),
                "content": str(r.get("新闻内容", r.get("content", ""))),
            })
        return rows[:NEWS_COUNT]
    except Exception:
        return []


def _fetch_cls_alerts() -> list[dict]:
    """财联社 (CLS) real-time market alerts."""
    try:
        df = ak.stock_zh_a_alerts_cls()
        if df is None or df.empty:
            return []
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "source":  "财联社",
                "time":    str(r.get("时间", "")),
                "title":   str(r.get("内容", r.get("标题", ""))),
                "content": "",
            })
        return rows[:NEWS_COUNT]
    except Exception:
        return []


def _fetch_hot_concepts() -> list[dict]:
    """Hot concept boards from East Money."""
    try:
        df = ak.stock_board_concept_spot_em()
        if df is None or df.empty:
            return []
        top = df.nlargest(5, "涨跌幅") if "涨跌幅" in df.columns else df.head(5)
        rows = []
        for _, r in top.iterrows():
            chg = r.get("涨跌幅", "")
            rows.append({
                "source":  "热门板块",
                "time":    "",
                "title":   f"【{r.get('板块名称', '')}】涨幅 {chg}%",
                "content": f"领涨股: {r.get('领涨股票', '')}",
            })
        return rows
    except Exception:
        return []


# ── main entry ────────────────────────────────────────────────────────────────

def fetch_news(use_cache: bool = True) -> dict:
    tz = pytz.timezone(TZ)

    if use_cache and NEWS_CACHE.exists():
        try:
            data = json.loads(NEWS_CACHE.read_text())
            updated = datetime.fromisoformat(data["updated_at"])
            age_h = (datetime.now(tz) - updated).total_seconds() / 3600
            if age_h < 2:   # news refreshes every 2 hours
                return data
        except Exception:
            pass

    all_news: list[dict] = []

    # Hot concepts first (quick)
    all_news.extend(_fetch_hot_concepts())

    # CLS alerts (real-time)
    cls = _fetch_cls_alerts()
    if cls:
        all_news.extend(cls)
    else:
        # fallback to CCTV
        all_news.extend(_fetch_cctv())

    # East Money headlines
    em = _fetch_eastmoney_news()
    all_news.extend(em)

    # De-duplicate by title
    seen: set[str] = set()
    unique: list[dict] = []
    for item in all_news:
        key = item["title"][:40]
        if key not in seen and len(key) > 5:
            seen.add(key)
            unique.append(item)

    result = {
        "updated_at": datetime.now(tz).isoformat(),
        "items": unique[:NEWS_COUNT],
        "error": None,
    }
    NEWS_CACHE.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
