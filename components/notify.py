"""
Server酱推送 — 每日信号摘要发到微信。
需在环境变量或 GitHub Secrets 中设置 SERVERCHAN_KEY。
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pytz
import requests

logger = logging.getLogger(__name__)

SERVERCHAN_KEY = os.environ.get("SERVERCHAN_KEY", "")


def _build_content(signals: dict, regime: dict, market: dict) -> tuple[str, str]:
    """生成微信通知的标题和正文（Markdown）。"""
    tz = pytz.timezone("Asia/Shanghai")
    today = datetime.now(tz).strftime("%Y-%m-%d")

    # ── Regime ────────────────────────────────────────────────────────────────
    cur_regime = regime.get("current_regime", "未知")
    regime_emoji = (
        "🟢" if "Bull" in cur_regime
        else "🔴" if "Bear" in cur_regime
        else "🟡"
    )

    # ── Market breadth ────────────────────────────────────────────────────────
    breadth = market.get("breadth", {})
    up   = breadth.get("up",   "-")
    down = breadth.get("down", "-")
    lu   = breadth.get("limit_up",   "-")
    ld   = breadth.get("limit_down", "-")

    # ── Index row ─────────────────────────────────────────────────────────────
    idx_rows = ""
    for idx in market.get("indices", []):
        name = idx.get("名称", "")
        chg  = idx.get("涨跌幅")
        if chg is not None:
            arrow = "▲" if chg >= 0 else "▼"
            idx_rows += f"**{name}** {arrow}{chg:+.2f}%  "

    def _horizon_table(items: list[dict], horizon: int, top: int = 5) -> str:
        rows = f"| 代码 | 名称 | 截面排名分 |\n|------|------|------|\n"
        for item in items[:top]:
            code  = item.get("code", "")
            name  = item.get("name", "") or ""
            score = item.get("rank_score", 0)
            rows += f"| {code} | {name} | {float(score):.4f} |\n"
        return rows

    h5_tbl  = _horizon_table(signals.get("h5",  []), 5)
    h10_tbl = _horizon_table(signals.get("h10", []), 10)
    h20_tbl = _horizon_table(signals.get("h20", []), 20)
    sig_mode = signals.get("mode", "akshare")

    title = f"📊 A股量化日报 {today} {regime_emoji} {cur_regime}"
    body = f"""## {regime_emoji} 市场状态
**{cur_regime}**

## 📈 主要指数
{idx_rows}

## 市场宽度
上涨 **{up}** | 下跌 **{down}** | 涨停 **{lu}** | 跌停 **{ld}**

## 📅 5日持仓 TOP 5
{h5_tbl}
## 📅 10日持仓 TOP 5
{h10_tbl}
## 📅 20日持仓 TOP 5
{h20_tbl}
---
*信号来源: {sig_mode} | 仅供研究参考，不构成投资建议*
"""
    return title, body


def send_wechat(signals: dict, regime: dict, market: dict) -> bool:
    """通过 Server酱 发送微信通知。返回是否成功。"""
    if not SERVERCHAN_KEY:
        logger.warning("SERVERCHAN_KEY 未设置，跳过微信推送")
        return False

    title, body = _build_content(signals, regime, market)
    url = f"https://sctapi.ftqq.com/{SERVERCHAN_KEY}.send"
    try:
        resp = requests.post(
            url,
            data={"title": title, "desp": body},
            timeout=15,
        )
        result = resp.json()
        if result.get("code") == 0:
            logger.info("微信推送成功 ✓")
            return True
        else:
            logger.warning("微信推送失败: %s", result)
            return False
    except Exception as exc:
        logger.error("微信推送异常: %s", exc)
        return False
