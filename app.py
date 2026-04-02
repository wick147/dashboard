"""
A股量化信号监控 Dashboard
专注于展示 GitHub Actions 自动运行的 Qlib Alpha158 LightGBM 多周期选股信号。

数据来源: chenditc/investment_data (qlib 格式离线数据)
不依赖任何实时行情 API。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytz
import requests
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
DASH_DIR    = Path(__file__).parent
RESULTS_DIR = DASH_DIR / "results"

# ── Constants ─────────────────────────────────────────────────────────────────
TZ = "Asia/Shanghai"

REGIME_COLORS = {
    "牛市 Bull":     "#EF5350",
    "震荡 Sideways": "#FFA726",
    "熊市 Bear":     "#26A69A",
}

HORIZONS = [
    ("h5",  "📅 5日持仓"),
    ("h10", "📅 10日持仓"),
    ("h20", "📅 20日持仓"),
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A股量化信号",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.regime-card {
    padding: 24px 20px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 16px;
}
.regime-card .rname {
    font-size: 2rem;
    font-weight: 700;
    color: white;
    letter-spacing: 1px;
}
.regime-card .rlabel {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.8);
    margin-top: 6px;
}
.info-banner {
    background: rgba(100,181,246,0.08);
    border-left: 4px solid #64B5F6;
    padding: 14px 18px;
    border-radius: 0 8px 8px 0;
    margin: 12px 0 20px 0;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _fmt_dt(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso).astimezone(pytz.timezone(TZ))
        return dt.strftime("%Y-%m-%d %H:%M CST")
    except Exception:
        return iso


def _trigger_workflow() -> tuple[bool, str]:
    """通过 GitHub API 触发 qlib-signals workflow。"""
    token = st.secrets.get("GITHUB_TOKEN", "")
    repo  = st.secrets.get("GITHUB_REPO", "wick147/dashboard")
    if not token:
        return False, "未配置 GITHUB_TOKEN（在 Streamlit Secrets 中设置后生效）"
    url  = f"https://api.github.com/repos/{repo}/actions/workflows/qlib-signals.yml/dispatches"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept":        "application/vnd.github+json",
        },
        json={"ref": "main"},
        timeout=10,
    )
    if resp.status_code == 204:
        return True, "✅ 已触发！约 40-60 分钟后结果自动更新到仓库。"
    return False, f"触发失败 ({resp.status_code}): {resp.text[:200]}"


# ── Load cached results ───────────────────────────────────────────────────────
signals = _load(RESULTS_DIR / "signals_qlib.json")
regime  = _load(RESULTS_DIR / "regime.json")
history = _load(RESULTS_DIR / "market_history.json")

has_signals = signals and not signals.get("error") and any(
    signals.get(k) for k in ["h5", "h10", "h20"]
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 A股量化信号监控")
    st.caption("Qlib Alpha158 · LightGBM · GitHub Actions")
    st.divider()

    if has_signals:
        st.markdown("**⏱ 最近更新**")
        st.markdown(f"`{_fmt_dt(signals.get('updated_at'))}`")
        st.markdown(f"**📅 预测日期** `{signals.get('pred_date', '—')}`")
        t0 = signals.get("train_start", "")
        t1 = signals.get("train_end", "")
        if t0 and t1:
            st.markdown("**📆 训练区间**")
            st.markdown(f"`{t0}` ~ `{t1}`")
    elif signals and signals.get("error"):
        st.error("信号生成出错（见主页面详情）")
    else:
        st.info("暂无信号数据")

    st.divider()

    if st.button("▶️ 触发 GitHub Actions 重跑", use_container_width=True, type="primary"):
        with st.spinner("正在触发..."):
            ok, msg = _trigger_workflow()
        if ok:
            st.success(msg)
        else:
            st.warning(msg)

    st.divider()
    st.caption(
        "⏰ 每日 **09:30 CST** 自动运行\n\n"
        "📦 数据: [chenditc/investment_data]"
        "(https://github.com/chenditc/investment_data)"
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🤖 A股 Qlib Alpha158 选股信号")

# ── No-data / Error state ─────────────────────────────────────────────────────
if not has_signals:
    if signals and signals.get("error"):
        st.error("上次运行出错，详情如下：")
        with st.expander("错误日志"):
            st.code(signals["error"][:2000])
    else:
        st.markdown(
            '<div class="info-banner">'
            '暂无信号数据。GitHub Actions 将于下一个工作日 09:30 CST 自动生成，'
            '或点击左侧栏「触发 GitHub Actions 重跑」手动启动。'
            '</div>',
            unsafe_allow_html=True,
        )
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — 多周期持仓推荐
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📋 多周期持仓推荐")
st.caption(
    f"预测日期: **{signals.get('pred_date', '—')}** · "
    "股票池: CSI300 · "
    "158个 Alpha 因子 · "
    "截面排名分越高 = 预测超额收益越大"
)

tabs = st.tabs([label for _, label in HORIZONS])

for tab, (key, label) in zip(tabs, HORIZONS):
    with tab:
        stocks = signals.get(key, [])
        if not stocks:
            st.info(f"{label} 暂无数据")
            continue

        df = pd.DataFrame(stocks)
        df.index = range(1, len(df) + 1)
        df.index.name = "排名"

        rename = {"code": "代码", "name": "名称", "rank_score": "截面排名分"}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        if "截面排名分" in df.columns:
            df["截面排名分"] = df["截面排名分"].map(lambda x: f"{x:.4f}")

        show = [c for c in ["代码", "名称", "截面排名分"] if c in df.columns]
        st.dataframe(df[show], use_container_width=True, height=420)

# ── Feature importance ────────────────────────────────────────────────────────
feat_imp = signals.get("feature_importance", {})
if any(feat_imp.get(k) for k in ["h5", "h10", "h20"]):
    with st.expander("📊 特征重要性"):
        fi_tabs = st.tabs(["5日模型", "10日模型", "20日模型"])
        for fi_tab, (key, _) in zip(fi_tabs, HORIZONS):
            with fi_tab:
                items = feat_imp.get(key, [])
                if not items:
                    st.info("暂无数据")
                    continue
                fi_df = pd.DataFrame(items).sort_values("importance").tail(12)
                fig = go.Figure(go.Bar(
                    x=fi_df["importance"],
                    y=fi_df["feature"],
                    orientation="h",
                    marker_color="#5C6BC0",
                ))
                fig.update_layout(
                    height=340,
                    margin=dict(l=0, r=20, t=10, b=0),
                    xaxis_title="重要性 (gain)",
                    yaxis_title=None,
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — HMM 市场状态
# ═══════════════════════════════════════════════════════════════════════════════
if regime and not regime.get("error"):
    st.subheader("🧠 市场状态 (HMM · 3状态高斯隐马尔可夫)")

    col_left, col_right = st.columns([2, 3])

    with col_left:
        cur   = regime.get("current_regime", "震荡 Sideways")
        color = REGIME_COLORS.get(cur, "#90A4AE")
        st.markdown(
            f'<div class="regime-card" style="background:{color};">'
            f'<div class="rname">{cur}</div>'
            f'<div class="rlabel">当前市场状态</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        stats = regime.get("state_stats", [])
        if stats:
            rows = [{
                "状态":    s.get("regime", "?"),
                "年化收益": f"{s.get('mean_ret', 0)*100:.1f}%",
                "年化波动": f"{s.get('mean_rv', 0)*100:.1f}%",
                "占比":    f"{s.get('pct', 0):.1f}%",
            } for s in stats]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with col_right:
        hist = regime.get("history", [])
        if hist:
            h_df   = pd.DataFrame(hist).tail(252)
            colors = [REGIME_COLORS.get(r, "#90A4AE") for r in h_df.get("regime", [])]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=h_df["date"],
                y=[1] * len(h_df),
                marker_color=colors,
                showlegend=False,
            ))
            for name, c in REGIME_COLORS.items():
                fig.add_trace(go.Bar(
                    x=[None], y=[None],
                    marker_color=c,
                    name=name,
                    showlegend=True,
                ))
            fig.update_layout(
                title="近252个交易日市场状态",
                height=220,
                barmode="stack",
                bargap=0,
                margin=dict(l=0, r=0, t=36, b=0),
                yaxis=dict(showticklabels=False, showgrid=False),
                xaxis=dict(showgrid=False),
                legend=dict(orientation="h", y=-0.25),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — CSI300 价格走势
# ═══════════════════════════════════════════════════════════════════════════════
if history and history.get("data"):
    st.subheader("📈 沪深300走势")

    data   = history["data"]
    dates  = [d["date"] for d in data]
    closes = [d["close"] for d in data]

    fig = go.Figure()

    # Regime 背景色带
    if regime and regime.get("history"):
        reg_map = {d["date"]: d.get("regime", "") for d in regime["history"]}
        cur_reg, seg_start = None, None
        for date in dates:
            r = reg_map.get(date, cur_reg)
            if r != cur_reg:
                if cur_reg and seg_start:
                    fig.add_vrect(
                        x0=seg_start, x1=date,
                        fillcolor=REGIME_COLORS.get(cur_reg, "#90A4AE"),
                        opacity=0.12, layer="below", line_width=0,
                    )
                cur_reg, seg_start = r, date
        if cur_reg and seg_start:
            fig.add_vrect(
                x0=seg_start, x1=dates[-1],
                fillcolor=REGIME_COLORS.get(cur_reg, "#90A4AE"),
                opacity=0.12, layer="below", line_width=0,
            )

    fig.add_trace(go.Scatter(
        x=dates, y=closes,
        mode="lines",
        name="沪深300",
        line=dict(color="#1E88E5", width=2),
        hovertemplate="%{x}<br>收盘: %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(
            title="收盘价",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.15)",
        ),
        hovermode="x unified",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "📦 数据来源: [chenditc/investment_data](https://github.com/chenditc/investment_data) · "
    "🤖 模型: qlib LightGBM Alpha158 (CSI300) · "
    "📊 市场状态: HMM hmmlearn · "
    "⚠️ 仅供研究参考，不构成投资建议"
)
