"""
A股量化信号监控 Dashboard
─────────────────────────────────────────────────────────────
• AKShare 最新行情  （指数 / 板块 / 市场宽度）
• LightGBM 选股信号 （qlib Alpha158 or AKShare 技术因子）
• HMM 市场 Regime   （牛市 / 震荡 / 熊市 三态）
• 财经早餐 / 要闻
─────────────────────────────────────────────────────────────
Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
DASH_DIR = Path(__file__).parent
sys.path.insert(0, str(DASH_DIR))

from config import TZ, SIGNAL_TOP_N, REGIME_COLORS
from components.market_data import load_market_data, get_index_history
from components.lgbm_signals import generate_signals
from components.hmm_regime import detect_regime
from components.news import fetch_news


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A股量化信号监控",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e; border-radius: 10px; padding: 16px 20px;
        border-left: 4px solid #7c3aed;
    }
    .regime-bull  { background: #1a3a2a; border-left: 4px solid #66BB6A; }
    .regime-bear  { background: #3a1a1a; border-left: 4px solid #EF5350; }
    .regime-side  { background: #2a2a1a; border-left: 4px solid #FFA726; }
    .news-item    { padding: 8px 0; border-bottom: 1px solid #333; }
    .section-hdr  { font-size:1.2rem; font-weight:700; margin: 1rem 0 0.5rem; }
    div[data-testid="stMetric"] label { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _now_bj() -> str:
    tz = pytz.timezone(TZ)
    return datetime.now(tz).strftime("%Y-%m-%d  %H:%M  CST")


def _delta_color(val: float | None) -> str:
    if val is None:
        return "off"
    return "normal" if val >= 0 else "inverse"


def _pct_fmt(v) -> str:
    if v is None:
        return "N/A"
    return f"{v:+.2f}%"


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/stocks.png", width=48)
    st.title("A股量化监控")
    st.caption(_now_bj())

    signal_mode = st.radio(
        "信号模式",
        options=["auto", "qlib", "akshare"],
        index=0,
        help="auto: 自动检测 qlib 数据目录；qlib: 使用 Alpha158；akshare: 技术因子",
    )

    use_cache = st.checkbox("使用缓存（< 12 h）", value=True)
    refresh_btn = st.button("🔄 立即刷新", use_container_width=True)
    st.divider()
    st.caption("每日 08:00 (北京时间) 自动运行\n\n"
               "`python dashboard/run_daily.py`\n\nor via cron / Claude Code schedule")


# ── header ────────────────────────────────────────────────────────────────────

st.markdown("## 📊 A股量化信号监控 Dashboard")
st.caption(f"更新时间: {_now_bj()}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Market overview
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-hdr">📈 指数行情</p>', unsafe_allow_html=True)

with st.spinner("加载行情数据..."):
    market = load_market_data() if not refresh_btn else __import__(
        "components.market_data", fromlist=["fetch_market_data"]
    ).fetch_market_data()

# Index metrics
idx_df = pd.DataFrame(market.get("indices", []))
if not idx_df.empty:
    cols = st.columns(len(idx_df))
    for i, row in idx_df.iterrows():
        chg = row.get("涨跌幅")
        price = row.get("最新价")
        delta_color = _delta_color(chg)
        with cols[i]:
            st.metric(
                label=row["名称"],
                value=f"{price:,.2f}" if price else "N/A",
                delta=_pct_fmt(chg),
                delta_color=delta_color,
            )

# Market breadth
breadth = market.get("breadth", {})
if breadth.get("up") is not None:
    st.markdown("")
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("上涨", breadth.get("up", "-"), delta_color="off")
    b2.metric("下跌", breadth.get("down", "-"), delta_color="off")
    b3.metric("平盘", breadth.get("flat", "-"), delta_color="off")
    b4.metric("涨停", breadth.get("limit_up", "-"), delta_color="off")
    b5.metric("跌停", breadth.get("limit_down", "-"), delta_color="off")

# ─────────────────────────────────────────────────────────────────────────────
# 2. HMM Regime (left) + Sector heatmap (right)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-hdr">🧠 HMM 市场状态 & 板块表现</p>', unsafe_allow_html=True)
col_regime, col_sector = st.columns([1, 2])

with col_regime:
    with st.spinner("计算 HMM Regime..."):
        hist_df = get_index_history()
        regime_data: dict = {}
        if hist_df is not None and not hist_df.empty:
            try:
                regime_data = detect_regime(hist_df, use_cache=(use_cache and not refresh_btn))
            except Exception as exc:
                st.error(f"HMM 计算失败: {exc}")

    if regime_data:
        cur_regime = regime_data.get("current_regime", "未知")
        color_map = REGIME_COLORS
        cur_color = color_map.get(cur_regime, "#888")

        regime_css = (
            "regime-bull" if "Bull" in cur_regime
            else "regime-bear" if "Bear" in cur_regime
            else "regime-side"
        )
        st.markdown(f"""
        <div class="metric-card {regime_css}">
            <div style="font-size:0.85rem; color:#aaa">当前 Regime</div>
            <div style="font-size:2rem; font-weight:bold; color:{cur_color}">{cur_regime}</div>
        </div>""", unsafe_allow_html=True)

        # State stats table
        ss = regime_data.get("state_stats", [])
        if ss:
            ss_df = pd.DataFrame(ss)
            ss_df = ss_df.rename(columns={
                "regime": "状态", "mean_ret": "年化收益",
                "mean_rv": "年化波动", "days": "天数", "pct": "占比%",
            })
            ss_df["年化收益"] = ss_df["年化收益"].map("{:.1%}".format)
            ss_df["年化波动"] = ss_df["年化波动"].map("{:.1%}".format)
            st.dataframe(
                ss_df[["状态", "年化收益", "年化波动", "占比%"]],
                use_container_width=True, hide_index=True,
            )

        # Regime history chart
        hist = regime_data.get("history", [])
        if hist:
            hdf = pd.DataFrame(hist)
            hdf["date"] = pd.to_datetime(hdf["date"])
            hdf["color"] = hdf["regime"].map(color_map).fillna("#888")
            hdf["y"] = 1

            fig_r = px.bar(
                hdf.tail(252), x="date", y="y", color="regime",
                color_discrete_map=color_map,
                height=160,
                labels={"date": "", "y": "", "regime": ""},
            )
            fig_r.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=-0.4),
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                showlegend=True,
                bargap=0, barmode="stack",
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.info("等待 HMM 数据...")

with col_sector:
    sector_data = market.get("sector", [])
    if sector_data:
        sdf = pd.DataFrame(sector_data)
        sdf["涨跌幅"] = pd.to_numeric(sdf.get("涨跌幅", 0), errors="coerce").fillna(0)
        sdf = sdf.sort_values("涨跌幅", ascending=False)

        fig_s = go.Figure(go.Bar(
            x=sdf["涨跌幅"].values[:20],
            y=sdf["板块名称"].values[:20],
            orientation="h",
            marker_color=[
                "#EF5350" if v < 0 else "#66BB6A"
                for v in sdf["涨跌幅"].values[:20]
            ],
            text=[f"{v:+.2f}%" for v in sdf["涨跌幅"].values[:20]],
            textposition="outside",
        ))
        fig_s.update_layout(
            height=420,
            margin=dict(l=100, r=60, t=20, b=20),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="#555",
                       ticksuffix="%", color="#aaa"),
            yaxis=dict(showgrid=False, color="#ddd", tickfont=dict(size=11)),
        )
        st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.info("板块数据不可用")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LightGBM Signals
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-hdr">🤖 LightGBM 选股信号</p>', unsafe_allow_html=True)

signals_placeholder = st.empty()

with signals_placeholder.container():
    progress = st.progress(0, text="生成选股信号中...")

    def _update_progress(p: float) -> None:
        progress.progress(min(int(p * 100), 100), text=f"生成选股信号... {min(int(p*100),100)}%")

    try:
        signals = generate_signals(
            mode=signal_mode,
            progress_cb=_update_progress,
            use_cache=(use_cache and not refresh_btn),
        )
        progress.empty()
    except Exception as exc:
        progress.empty()
        st.error(f"信号生成失败: {exc}")
        signals = {"h5": [], "h10": [], "h20": [], "mode": "error", "error": str(exc)}

# ── 信号展示 ──────────────────────────────────────────────────────────────────
sig_mode_tag = signals.get("mode", "unknown")
sig_updated  = signals.get("updated_at", "")[:19]
sig_error    = signals.get("error")

if sig_error:
    st.warning(f"信号生成出错（显示缓存结果）: {str(sig_error)[:300]}")

st.caption(f"信号来源: `{sig_mode_tag}` | 更新: {sig_updated}")


def _render_horizon_table(items: list[dict], horizon: int) -> None:
    if not items:
        st.info("暂无数据")
        return
    df = pd.DataFrame(items)
    df.insert(0, "排名", range(1, len(df) + 1))
    rename = {"code": "代码", "name": "名称", "rank_score": "模型评分",
              "gmr_daily": f"日均GMR(%)", "total_ret": f"{horizon}日预期收益(%)"}
    df = df.rename(columns=rename)
    # 颜色高亮：预期收益列
    ret_col = f"{horizon}日预期收益(%)"
    if ret_col in df.columns:
        df[ret_col] = df[ret_col].map(
            lambda x: f"+{x:.2f}%" if x is not None and x >= 0
            else (f"{x:.2f}%" if x is not None else "-")
        )
    if f"日均GMR(%)" in df.columns:
        df["日均GMR(%)"] = df["日均GMR(%)"].map(
            lambda x: f"{x:.4f}%" if x is not None else "-"
        )
    show_cols = [c for c in ["排名", "代码", "名称", "模型评分", "日均GMR(%)", ret_col] if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)


tab5, tab10, tab20 = st.tabs(["📅 5日持仓", "📅 10日持仓", "📅 20日持仓"])

with tab5:
    st.caption("预测未来 **5个交易日** 几何平均日收益率最高的10支票")
    _render_horizon_table(signals.get("h5", []), 5)

with tab10:
    st.caption("预测未来 **10个交易日** 几何平均日收益率最高的10支票")
    _render_horizon_table(signals.get("h10", []), 10)

with tab20:
    st.caption("预测未来 **20个交易日** 几何平均日收益率最高的10支票")
    _render_horizon_table(signals.get("h20", []), 20)

# 特征重要性
feat_imp_dict = signals.get("feature_importance", {})
if feat_imp_dict:
    with st.expander("📊 各周期特征重要性"):
        fi_tabs = st.tabs(["5日模型", "10日模型", "20日模型"])
        for fi_tab, hkey in zip(fi_tabs, ["h5", "h10", "h20"]):
            with fi_tab:
                fi_list = feat_imp_dict.get(hkey, [])
                if fi_list:
                    fi_df = pd.DataFrame(fi_list).sort_values("importance", ascending=True)
                    fig_fi = go.Figure(go.Bar(
                        x=fi_df["importance"], y=fi_df["feature"],
                        orientation="h", marker_color="#7c3aed",
                    ))
                    fig_fi.update_layout(
                        height=260, margin=dict(l=80, r=20, t=10, b=20),
                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                        xaxis=dict(showgrid=False, color="#aaa"),
                        yaxis=dict(showgrid=False, color="#ddd"),
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CSI300 price chart with regime overlay
# ─────────────────────────────────────────────────────────────────────────────
if hist_df is not None and not hist_df.empty and regime_data:
    st.markdown('<p class="section-hdr">📉 沪深300 走势 + Regime 背景</p>', unsafe_allow_html=True)
    hist_recent = hist_df.tail(252).copy()

    fig_idx = go.Figure()
    fig_idx.add_trace(go.Scatter(
        x=hist_recent.index,
        y=hist_recent["close"],
        line=dict(color="#60a5fa", width=2),
        name="沪深300",
    ))

    # Overlay regime bands
    hist_map = {r["date"]: r["regime"] for r in regime_data.get("history", [])}
    if hist_map:
        idx_dates = hist_recent.index.strftime("%Y-%m-%d").tolist()
        for i, d in enumerate(idx_dates):
            regime = hist_map.get(d)
            if regime:
                col = REGIME_COLORS.get(regime, "rgba(128,128,128,0.1)")
                fig_idx.add_vrect(
                    x0=hist_recent.index[i],
                    x1=hist_recent.index[min(i + 1, len(idx_dates) - 1)],
                    fillcolor=col, opacity=0.12,
                    layer="below", line_width=0,
                )

    fig_idx.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        xaxis=dict(showgrid=False, color="#aaa"),
        yaxis=dict(showgrid=False, color="#aaa", tickformat=","),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
    )
    st.plotly_chart(fig_idx, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Morning briefing / news
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-hdr">📰 财经早餐 / 要闻</p>', unsafe_allow_html=True)

with st.spinner("加载要闻..."):
    news_data = fetch_news(use_cache=(use_cache and not refresh_btn))

news_updated = news_data.get("updated_at", "")[:19]
st.caption(f"新闻更新时间: {news_updated}")

news_items = news_data.get("items", [])
if news_items:
    for item in news_items:
        source  = item.get("source", "")
        t       = item.get("time", "")
        title   = item.get("title", "")
        content = item.get("content", "")
        if not title:
            continue
        time_str = t[:16] if len(t) >= 16 else t
        badge = f"`{source}`" if source else ""
        st.markdown(
            f'<div class="news-item">'
            f'<span style="color:#888;font-size:0.8rem">{time_str}&nbsp;</span>'
            f'<span style="color:#ccc;font-size:0.78rem">{badge}&nbsp;</span>'
            f'<span style="font-weight:600">{title}</span>'
            + (f'<br><span style="color:#999;font-size:0.8rem">{content[:120]}</span>' if content else "")
            + "</div>",
            unsafe_allow_html=True,
        )
else:
    st.info("暂时没有新闻数据")

# ─────────────────────────────────────────────────────────────────────────────
# footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "数据来源: AKShare · 模型: qlib LightGBM Alpha158 / 技术因子 · HMM: hmmlearn · "
    "仅供研究参考，不构成投资建议"
)
