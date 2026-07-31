"""
AEGIS — Institutional Quantitative Trading Platform
=====================================================
B2B SaaS dashboard — Midnight SaaS aesthetic.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ta
import json

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Aegis · Quant Platform",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# DESIGN SYSTEM — Midnight SaaS
# ============================================================================
# Aesthetic: Linear.app × Bloomberg Terminal × Vercel Dashboard
# Fonts: Bricolage Grotesque (display) · Outfit (body) · JetBrains Mono (data)
# Palette: Near-black base, indigo accent (#6366F1), emerald/rose signals
# ============================================================================

st.markdown("""
<style>
    /* ─── Typography ─── */
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,200;12..96,300;12..96,400;12..96,500;12..96,600;12..96,700;12..96,800&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

    /* ─── Design Tokens ─── */
    :root {
        /* Surfaces */
        --bg-root:        #09090B;
        --bg-primary:     #0F0F12;
        --bg-secondary:   #16161A;
        --bg-tertiary:    #1C1C21;
        --bg-elevated:    #222228;
        --bg-hover:       #27272D;
        --bg-active:      #2E2E35;

        /* Accent — Indigo */
        --accent:         #6366F1;
        --accent-hover:   #818CF8;
        --accent-muted:   rgba(99, 102, 241, 0.12);
        --accent-border:  rgba(99, 102, 241, 0.25);
        --accent-glow:    0 0 20px rgba(99, 102, 241, 0.15);

        /* Signal Colors */
        --success:        #10B981;
        --success-muted:  rgba(16, 185, 129, 0.12);
        --danger:         #F43F5E;
        --danger-muted:   rgba(244, 63, 94, 0.12);
        --warning:        #F59E0B;
        --warning-muted:  rgba(245, 158, 11, 0.12);
        --info:           #38BDF8;
        --info-muted:     rgba(56, 189, 248, 0.12);

        /* Text */
        --text-primary:   #FAFAFA;
        --text-secondary: #A1A1AA;
        --text-tertiary:  #71717A;
        --text-muted:     #52525B;

        /* Borders */
        --border:         #27272A;
        --border-subtle:  #1E1E22;
        --border-hover:   #3F3F46;

        /* Typography */
        --font-display:   'Bricolage Grotesque', system-ui, sans-serif;
        --font-body:      'Outfit', system-ui, sans-serif;
        --font-mono:      'JetBrains Mono', 'Menlo', monospace;

        /* Effects */
        --radius-xs:      4px;
        --radius-sm:      6px;
        --radius:         8px;
        --radius-lg:      12px;
        --shadow-sm:      0 1px 2px rgba(0,0,0,0.3);
        --shadow-md:      0 2px 8px rgba(0,0,0,0.4);
        --shadow-lg:      0 8px 32px rgba(0,0,0,0.5);
    }

    /* ─── GLOBAL ─── */
    .stApp {
        background: var(--bg-root);
        font-family: var(--font-body);
        color: var(--text-primary);
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    div[data-testid="stDecoration"] { display: none; }

    /* ─── SIDEBAR ─── */
    section[data-testid="stSidebar"] {
        background: var(--bg-primary);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] .stRadio > div {
        gap: 2px;
    }

    section[data-testid="stSidebar"] .stRadio > div > label {
        background: transparent;
        border: none;
        border-radius: var(--radius-sm);
        padding: 0.55rem 0.75rem;
        font-family: var(--font-body);
        font-size: 0.88rem;
        font-weight: 450;
        color: var(--text-secondary);
        transition: all 0.15s ease;
        cursor: pointer;
    }

    section[data-testid="stSidebar"] .stRadio > div > label:hover {
        background: var(--bg-hover);
        color: var(--text-primary);
    }

    section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
    section[data-testid="stSidebar"] .stRadio > div [data-checked="true"] ~ label {
        background: var(--accent-muted);
        color: var(--accent-hover);
        font-weight: 550;
    }

    /* ─── PAGE HEADER ─── */
    .page-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 0 1.25rem 0;
        margin-bottom: 1.25rem;
        border-bottom: 1px solid var(--border);
    }
    .page-header-left {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .page-title {
        font-family: var(--font-display);
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .page-subtitle {
        font-family: var(--font-body);
        font-size: 0.82rem;
        font-weight: 400;
        color: var(--text-tertiary);
    }
    .page-header-right {
        display: flex;
        gap: 0.75rem;
        align-items: center;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.3rem 0.75rem;
        border-radius: 99px;
        font-family: var(--font-mono);
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.03em;
    }
    .badge-live {
        background: var(--success-muted);
        color: var(--success);
        border: 1px solid rgba(16,185,129,0.2);
    }
    .badge-ai {
        background: var(--accent-muted);
        color: var(--accent-hover);
        border: 1px solid var(--accent-border);
    }
    .badge-warn {
        background: var(--warning-muted);
        color: var(--warning);
        border: 1px solid rgba(245,158,11,0.2);
    }
    .live-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: currentColor;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* ─── METRIC CARDS ─── */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 1px;
        background: var(--border);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        overflow: hidden;
        margin-bottom: 1.25rem;
    }
    .metric-cell {
        background: var(--bg-primary);
        padding: 1rem 1.15rem;
        transition: background 0.15s ease;
    }
    .metric-cell:hover {
        background: var(--bg-secondary);
    }
    .metric-label {
        font-family: var(--font-body);
        font-size: 0.72rem;
        font-weight: 500;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-family: var(--font-mono);
        font-size: 1.35rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .metric-delta {
        font-family: var(--font-mono);
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 0.2rem;
    }
    .delta-up { color: var(--success); }
    .delta-down { color: var(--danger); }

    /* ─── CARDS / PANELS ─── */
    .card {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        overflow: hidden;
        margin-bottom: 1rem;
    }
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.85rem 1.15rem;
        border-bottom: 1px solid var(--border);
    }
    .card-title {
        font-family: var(--font-body);
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .card-body {
        padding: 1rem 1.15rem;
    }

    /* ─── DATA TABLE ROWS ─── */
    .data-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.55rem 0;
        border-bottom: 1px solid var(--border-subtle);
    }
    .data-row:last-child { border-bottom: none; }
    .data-row:hover {
        background: var(--bg-secondary);
        margin: 0 -1.15rem;
        padding-left: 1.15rem;
        padding-right: 1.15rem;
    }
    .data-sym {
        font-family: var(--font-mono);
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--text-primary);
    }
    .data-val {
        font-family: var(--font-mono);
        font-size: 0.82rem;
        font-weight: 500;
    }
    .val-pos { color: var(--success); }
    .val-neg { color: var(--danger); }
    .val-neutral { color: var(--text-secondary); }

    /* ─── AGENT STATUS CARDS ─── */
    .agent-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.6rem;
        margin-bottom: 1rem;
    }
    .agent-chip {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.75rem 0.6rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    .agent-chip:hover {
        border-color: var(--border-hover);
        background: var(--bg-tertiary);
    }
    .agent-chip.ok { border-left: 3px solid var(--success); }
    .agent-chip.warn { border-left: 3px solid var(--warning); }
    .agent-chip.err { border-left: 3px solid var(--danger); }
    .agent-chip-icon { font-size: 1.1rem; margin-bottom: 0.25rem; }
    .agent-chip-name {
        font-family: var(--font-body);
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.3;
    }
    .agent-chip-status {
        font-family: var(--font-mono);
        font-size: 0.6rem;
        font-weight: 500;
        margin-top: 0.15rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* ─── BUTTONS ─── */
    .stButton > button {
        background: var(--accent);
        color: #fff;
        border: none;
        border-radius: var(--radius-sm);
        padding: 0.55rem 1.3rem;
        font-family: var(--font-body);
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.01em;
        transition: all 0.15s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        background: var(--accent-hover);
        box-shadow: var(--accent-glow);
    }

    /* ─── INPUTS ─── */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.85rem !important;
    }
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus-within {
        border-color: var(--accent-border) !important;
        box-shadow: 0 0 0 2px var(--accent-muted) !important;
    }

    /* ─── TABS ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-secondary);
        border-radius: var(--radius-sm);
        padding: 2px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: var(--radius-xs);
        padding: 0.45rem 1rem;
        color: var(--text-tertiary);
        font-family: var(--font-body);
        font-weight: 500;
        font-size: 0.82rem;
    }
    .stTabs [aria-selected="true"] {
        background: var(--bg-elevated);
        color: var(--text-primary) !important;
        font-weight: 600;
        box-shadow: var(--shadow-sm);
    }

    /* ─── SCROLLBAR ─── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-root); }
    ::-webkit-scrollbar-thumb { background: var(--bg-hover); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--bg-active); }

    /* ─── STREAMLIT OVERRIDES ─── */
    .stMetric label {
        font-family: var(--font-body) !important;
        font-size: 0.72rem !important;
        font-weight: 500 !important;
        color: var(--text-tertiary) !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-family: var(--font-mono) !important;
        font-weight: 600 !important;
        font-size: 1.3rem !important;
        color: var(--text-primary) !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        font-family: var(--font-mono) !important;
    }

    div[data-testid="stExpander"] {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
    }
    div[data-testid="stExpander"]:hover {
        border-color: var(--border-hover);
    }

    /* Plotly container */
    .js-plotly-plot .plotly .main-svg { border-radius: var(--radius); }

    /* Signal intensity */
    .sig-strong-buy  { color: var(--success); font-weight: 600; }
    .sig-buy         { color: #6EE7B7; }
    .sig-neutral     { color: var(--text-secondary); }
    .sig-sell        { color: #FDA4AF; }
    .sig-strong-sell { color: var(--danger); font-weight: 600; }

    /* Section separator */
    .section-sep {
        height: 1px;
        background: var(--border);
        border: none;
        margin: 1.5rem 0;
    }
    .section-title {
        font-family: var(--font-body);
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
    }
    .section-subtitle {
        font-family: var(--font-body);
        font-size: 0.78rem;
        color: var(--text-tertiary);
        margin-bottom: 0.75rem;
    }

    /* Streamlit dataframe tweaks */
    .stDataFrame { border-radius: var(--radius) !important; }
    [data-testid="stDataFrame"] > div { border-radius: var(--radius) !important; }

    /* Kill default padding bloat */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }

    /* JSON viewer */
    .stJson { background: var(--bg-secondary) !important; border-radius: var(--radius) !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA FUNCTIONS
# ============================================================================

DATA_CACHE = Path(__file__).parent.parent / "data_cache"


@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        cache_path = DATA_CACHE / f"{ticker.replace('^', 'IDX_').replace('=', '_')}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [c.lower() for c in data.columns]
        return data
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_realtime_quote(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "price": info.get("regularMarketPrice", info.get("currentPrice", 0)),
            "change": info.get("regularMarketChange", 0),
            "change_pct": info.get("regularMarketChangePercent", 0),
            "volume": info.get("regularMarketVolume", 0),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "high": info.get("dayHigh", 0),
            "low": info.get("dayLow", 0),
            "open": info.get("open", 0),
            "prev_close": info.get("previousClose", 0),
        }
    except Exception:
        return {"price": 0, "change": 0, "change_pct": 0, "volume": 0,
                "market_cap": 0, "pe_ratio": 0, "name": ticker, "sector": "N/A",
                "high": 0, "low": 0, "open": 0, "prev_close": 0}


def load_alpha_signals() -> pd.DataFrame:
    path = DATA_CACHE / "alpha_signals.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def load_fundamentals() -> pd.DataFrame:
    path = DATA_CACHE / "fundamentals.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=3600)
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 20:
        return df
    df = df.copy()
    close = df["close"]
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["ema_12"] = close.ewm(span=12).mean()
    df["ema_26"] = close.ewm(span=26).mean()
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], close, window=14
    ).average_true_range()
    return df


def fmt(num: float, prefix: str = "$") -> str:
    if num is None or num == 0:
        return f"{prefix}0"
    if abs(num) >= 1e12:
        return f"{prefix}{num/1e12:.2f}T"
    if abs(num) >= 1e9:
        return f"{prefix}{num/1e9:.2f}B"
    if abs(num) >= 1e6:
        return f"{prefix}{num/1e6:.2f}M"
    if abs(num) >= 1e3:
        return f"{prefix}{num/1e3:.1f}K"
    return f"{prefix}{num:.2f}"


def signal_class(val: float) -> str:
    if val > 0.5:   return "sig-strong-buy"
    if val > 0.2:   return "sig-buy"
    if val > -0.2:  return "sig-neutral"
    if val > -0.5:  return "sig-sell"
    return "sig-strong-sell"


# ============================================================================
# CHART SYSTEM — Midnight SaaS Plotly Theme
# ============================================================================

C_GREEN  = "#10B981"
C_RED    = "#F43F5E"
C_INDIGO = "#6366F1"
C_SKY    = "#38BDF8"
C_AMBER  = "#F59E0B"
C_BG     = "rgba(9,9,11,0)"
C_GRID   = "rgba(255,255,255,0.04)"
C_FONT   = dict(family="JetBrains Mono, monospace", color="#71717A", size=10)


def _base_layout(height: int = 400, **overrides) -> dict:
    base = dict(
        template="plotly_dark",
        paper_bgcolor=C_BG,
        plot_bgcolor="rgba(15,15,18,0.9)",
        height=height,
        margin=dict(l=48, r=16, t=28, b=28),
        font=C_FONT,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10)),
    )
    base.update(overrides)
    return base


def create_advanced_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="OHLC",
        increasing_line_color=C_GREEN, decreasing_line_color=C_RED,
        increasing_fillcolor=C_GREEN, decreasing_fillcolor=C_RED,
    ), row=1, col=1)

    if "sma_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_20"], name="SMA 20",
                                 line=dict(color=C_INDIGO, width=1.2)), row=1, col=1)
    if "sma_50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="SMA 50",
                                 line=dict(color=C_SKY, width=1.2)), row=1, col=1)
    if "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper",
                                 line=dict(color="rgba(99,102,241,0.3)", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower",
                                 line=dict(color="rgba(99,102,241,0.3)", dash="dash"),
                                 fill="tonexty", fillcolor="rgba(99,102,241,0.04)"), row=1, col=1)

    vol_colors = [C_GREEN if c >= o else C_RED for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume",
                         marker_color=vol_colors, opacity=0.6), row=2, col=1)

    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
                                 line=dict(color=C_INDIGO, width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=C_RED, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=C_GREEN, row=3, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.015)", row=3, col=1)

    if "macd" in df.columns:
        macd_colors = [C_GREEN if v >= 0 else C_RED for v in df["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="MACD Hist",
                             marker_color=macd_colors), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD",
                                 line=dict(color=C_SKY, width=1)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
                                 line=dict(color=C_INDIGO, width=1)), row=4, col=1)

    fig.update_layout(**_base_layout(height=680), xaxis_rangeslider_visible=False)
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=C_GRID, row=i, col=1)
        fig.update_yaxes(gridcolor=C_GRID, row=i, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    return fig


# ============================================================================
# HELPERS — Page Header & Metric Grid
# ============================================================================

def page_header(title: str, subtitle: str = "", badges: list = None):
    badges_html = ""
    if badges:
        for b in badges:
            btype = b.get("type", "live")
            label = b.get("label", "")
            dot = '<span class="live-dot"></span>' if b.get("dot") else ""
            badges_html += f'<span class="status-badge badge-{btype}">{dot}{label}</span>'

    st.markdown(f"""
    <div class="page-header">
        <div class="page-header-left">
            <div class="page-title">{title}</div>
            {"<div class='page-subtitle'>" + subtitle + "</div>" if subtitle else ""}
        </div>
        <div class="page-header-right">{badges_html}</div>
    </div>
    """, unsafe_allow_html=True)


def metric_grid(items: list):
    """Render a connected metric strip. items = [{"label": ..., "value": ..., "delta": ...}, ...]"""
    cells = ""
    for item in items:
        delta_html = ""
        if item.get("delta"):
            d = item["delta"]
            cls = "delta-up" if not str(d).startswith("-") else "delta-down"
            delta_html = f'<div class="metric-delta {cls}">{d}</div>'
        cells += f"""
        <div class="metric-cell">
            <div class="metric-label">{item["label"]}</div>
            <div class="metric-value">{item["value"]}</div>
            {delta_html}
        </div>"""
    st.markdown(f'<div class="metric-grid">{cells}</div>', unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    # Brand
    st.markdown("""
    <div style="padding:0.8rem 0 0.5rem 0;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:32px; height:32px; background:linear-gradient(135deg, #6366F1, #818CF8);
                        border-radius:8px; display:flex; align-items:center; justify-content:center;
                        font-size:1rem; color:#fff; font-weight:700;">A</div>
            <div>
                <div style="font-family:var(--font-display); font-size:1.1rem; font-weight:700;
                            color:var(--text-primary); letter-spacing:-0.02em; line-height:1.2;">Aegis</div>
                <div style="font-family:var(--font-mono); font-size:0.58rem; color:var(--text-muted);
                            letter-spacing:0.1em; text-transform:uppercase;">Quant Platform</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview", "Analysis", "Signals", "Portfolio", "India", "Research", "Agents", "Settings"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # Quick Quote
    st.markdown('<div class="section-title" style="font-size:0.78rem;">Quick Quote</div>', unsafe_allow_html=True)
    ticker_input = st.text_input("Symbol", value="AAPL", key="sidebar_ticker",
                                 label_visibility="collapsed").upper()

    if ticker_input:
        q = fetch_realtime_quote(ticker_input)
        if q["price"] > 0:
            d_cls = "delta-up" if q["change"] >= 0 else "delta-down"
            d_sgn = "+" if q["change"] >= 0 else ""
            st.markdown(f"""
            <div style="background:var(--bg-secondary); border:1px solid var(--border);
                        border-radius:var(--radius); padding:0.85rem; margin-top:0.5rem;">
                <div style="font-family:var(--font-body); font-size:0.72rem; color:var(--text-tertiary);
                            text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.3rem;">
                    {q['name']}</div>
                <div style="font-family:var(--font-mono); font-size:1.25rem; font-weight:600;
                            color:var(--text-primary);">${q['price']:.2f}</div>
                <div class="{d_cls}" style="font-family:var(--font-mono); font-size:0.78rem; margin-top:0.15rem;">
                    {d_sgn}{q['change']:.2f} ({d_sgn}{q['change_pct']:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # Status
    now = datetime.now()
    mkt_open = 9 <= now.hour < 16 and now.weekday() < 5
    st_color = "var(--success)" if mkt_open else "var(--text-muted)"
    st_text = "Market Open" if mkt_open else "Market Closed"

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; padding:0.5rem 0;">
        <div style="width:6px; height:6px; border-radius:50%; background:{st_color};"></div>
        <span style="font-family:var(--font-mono); font-size:0.72rem; color:var(--text-tertiary);">
            {st_text} · {now.strftime("%H:%M")}</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PAGE: OVERVIEW (Dashboard)
# ============================================================================

if page == "Overview":
    page_header("Overview", "Market data & portfolio snapshot",
                [{"type": "live", "label": "Live", "dot": True},
                 {"type": "ai", "label": "Real-Time Data"}])

    # Market indices strip
    indices = {
        "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DOW 30": "^DJI",
        "Russell 2K": "^RUT", "VIX": "^VIX", "10Y Yield": "^TNX",
    }

    items = []
    for name, tkr in indices.items():
        q = fetch_realtime_quote(tkr)
        d_sgn = "+" if q["change_pct"] >= 0 else ""
        items.append({
            "label": name,
            "value": f"{q['price']:,.2f}",
            "delta": f"{d_sgn}{q['change_pct']:.2f}%",
        })
    metric_grid(items)

    # Main content grid
    col_chart, col_side = st.columns([2.5, 1])

    with col_chart:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <span class="card-title">S&P 500</span>
                <span class="status-badge badge-live" style="font-size:0.62rem;">6M</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sp500 = fetch_stock_data("^GSPC", "6mo")
        if not sp500.empty:
            pct = (sp500["close"].iloc[-1] / sp500["close"].iloc[0] - 1) * 100
            c = C_GREEN if pct >= 0 else C_RED
            fill_c = "rgba(16,185,129,0.06)" if pct >= 0 else "rgba(244,63,94,0.06)"

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sp500.index, y=sp500["close"], fill="tozeroy",
                fillcolor=fill_c, line=dict(color=c, width=2), name="S&P 500",
            ))
            fig.update_layout(**_base_layout(height=340, showlegend=False,
                              xaxis=dict(showgrid=False),
                              yaxis=dict(showgrid=True, gridcolor=C_GRID)))
            st.plotly_chart(fig, width="stretch")

    with col_side:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <span class="card-title">Top Signals</span>
                <span class="status-badge badge-ai" style="font-size:0.62rem;">AI</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        alpha_df = load_alpha_signals()
        if not alpha_df.empty:
            top = alpha_df.nlargest(8, "composite_alpha")
            rows_html = ""
            for _, row in top.iterrows():
                sc = signal_class(row["composite_alpha"])
                rows_html += f"""
                <div class="data-row">
                    <span class="data-sym">{row['symbol']}</span>
                    <span class="data-val {sc}">{row['composite_alpha']:.3f}</span>
                </div>"""
            st.markdown(f'<div style="padding:0 0.2rem;">{rows_html}</div>', unsafe_allow_html=True)
        else:
            st.info("Run `python download_data.py` to generate alpha signals")

    # Bottom grid
    col_sector, col_breadth = st.columns(2)

    with col_sector:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <span class="card-title">Sector Performance</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sectors = {
            "XLK": "Tech", "XLF": "Finance", "XLV": "Health",
            "XLE": "Energy", "XLI": "Industrial", "XLP": "Staples",
            "XLY": "Discret.", "XLU": "Utilities",
        }
        sec_data = []
        for tkr, nm in sectors.items():
            q = fetch_realtime_quote(tkr)
            if q["price"] > 0:
                sec_data.append({"Sector": nm, "Change": q["change_pct"]})

        if sec_data:
            sec_df = pd.DataFrame(sec_data).sort_values("Change", ascending=True)
            bar_colors = [C_GREEN if x >= 0 else C_RED for x in sec_df["Change"]]
            fig = go.Figure(go.Bar(
                x=sec_df["Change"], y=sec_df["Sector"], orientation="h",
                marker_color=bar_colors,
                text=[f"{x:+.2f}%" for x in sec_df["Change"]],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=10),
            ))
            fig.update_layout(**_base_layout(height=280,
                              showlegend=False,
                              xaxis=dict(showgrid=True, gridcolor=C_GRID,
                                         zeroline=True, zerolinecolor="rgba(255,255,255,0.08)"),
                              yaxis=dict(showgrid=False),
                              margin=dict(l=0, r=50, t=6, b=6)))
            st.plotly_chart(fig, width="stretch")

    with col_breadth:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <span class="card-title">Market Breadth</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            sp500_data = fetch_stock_data("^GSPC", "1mo")
            nasdaq_data = fetch_stock_data("^IXIC", "1mo")
            
            def calc_breadth(data, name):
                if data.empty or len(data) < 2:
                    return {"Exchange": name, "Advancing": "N/A", "Declining": "N/A", "Unchanged": "N/A"}
                changes = data["close"].diff()
                return {
                    "Exchange": name,
                    "Advancing": int((changes > 0).sum()),
                    "Declining": int((changes < 0).sum()),
                    "Unchanged": int((changes == 0).sum()),
                }
            
            breadth_data = [
                calc_breadth(sp500_data, "S&P 500"),
                calc_breadth(nasdaq_data, "NASDAQ"),
            ]
            st.dataframe(pd.DataFrame(breadth_data), width="stretch", hide_index=True)
        except Exception:
            st.caption("Market breadth data unavailable")


# ============================================================================
# PAGE: ANALYSIS
# ============================================================================

elif page == "Analysis":
    page_header("Technical Analysis", "Multi-timeframe charting with indicators")

    ac1, ac2, ac3 = st.columns([2, 1, 1])
    with ac1:
        ticker = st.text_input("Symbol", value="AAPL", key="analysis_ticker").upper()
    with ac2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    with ac3:
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    if ticker:
        df = fetch_stock_data(ticker, period)
        if not df.empty:
            df = calculate_indicators(df)
            q = fetch_realtime_quote(ticker)

            d_sgn = "+" if q["change_pct"] >= 0 else ""
            metric_grid([
                {"label": "Price", "value": f"${q['price']:.2f}", "delta": f"{d_sgn}{q['change_pct']:.2f}%"},
                {"label": "Open", "value": f"${q['open']:.2f}"},
                {"label": "Day High", "value": f"${q['high']:.2f}"},
                {"label": "Day Low", "value": f"${q['low']:.2f}"},
                {"label": "Volume", "value": fmt(q["volume"], "")},
                {"label": "Mkt Cap", "value": fmt(q["market_cap"])},
            ])

            st.plotly_chart(create_advanced_chart(df, ticker), width="stretch")

            # Technical Summary
            st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Technical Summary</div>', unsafe_allow_html=True)

            tc1, tc2, tc3, tc4 = st.columns(4)
            current = df["close"].iloc[-1]

            with tc1:
                sma20 = df["sma_20"].iloc[-1] if "sma_20" in df.columns else 0
                sma50 = df["sma_50"].iloc[-1] if "sma_50" in df.columns else 0
                st.markdown("**Moving Averages**")
                if sma20 > 0:
                    sig_c = "var(--success)" if current > sma20 else "var(--danger)"
                    sig_t = "Above" if current > sma20 else "Below"
                    st.markdown(f'<span style="color:{sig_c}; font-family:var(--font-mono); font-size:0.82rem;">SMA 20: {sig_t}</span>', unsafe_allow_html=True)
                if sma50 > 0:
                    sig_c = "var(--success)" if current > sma50 else "var(--danger)"
                    sig_t = "Above" if current > sma50 else "Below"
                    st.markdown(f'<span style="color:{sig_c}; font-family:var(--font-mono); font-size:0.82rem;">SMA 50: {sig_t}</span>', unsafe_allow_html=True)
            with tc2:
                if "rsi" in df.columns:
                    rsi = df["rsi"].iloc[-1]
                    st.markdown("**RSI (14)**")
                    if rsi > 70:
                        st.markdown(f'<span style="color:var(--danger); font-family:var(--font-mono); font-size:0.82rem;">Overbought ({rsi:.1f})</span>', unsafe_allow_html=True)
                    elif rsi < 30:
                        st.markdown(f'<span style="color:var(--success); font-family:var(--font-mono); font-size:0.82rem;">Oversold ({rsi:.1f})</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span style="color:var(--text-secondary); font-family:var(--font-mono); font-size:0.82rem;">Neutral ({rsi:.1f})</span>', unsafe_allow_html=True)
            with tc3:
                if "macd" in df.columns:
                    st.markdown("**MACD**")
                    is_bull = df["macd"].iloc[-1] > df["macd_signal"].iloc[-1]
                    sig_c = "var(--success)" if is_bull else "var(--danger)"
                    sig_t = "Bullish" if is_bull else "Bearish"
                    st.markdown(f'<span style="color:{sig_c}; font-family:var(--font-mono); font-size:0.82rem;">{sig_t}</span>', unsafe_allow_html=True)
            with tc4:
                if "bb_upper" in df.columns:
                    bb_range = df["bb_upper"].iloc[-1] - df["bb_lower"].iloc[-1]
                    if bb_range > 0:
                        bb_pos = (current - df["bb_lower"].iloc[-1]) / bb_range * 100
                        st.markdown("**Bollinger %B**")
                        st.markdown(f'<span style="font-family:var(--font-mono); font-size:0.82rem; color:var(--text-secondary);">{bb_pos:.1f}%</span>', unsafe_allow_html=True)


# ============================================================================
# PAGE: SIGNALS
# ============================================================================

elif page == "Signals":
    page_header("Alpha Signals", "AI-powered factor decomposition",
                [{"type": "ai", "label": "ML Engine", "dot": True}])

    alpha_df = load_alpha_signals()

    if not alpha_df.empty:
        metric_grid([
            {"label": "Total Signals", "value": str(len(alpha_df))},
            {"label": "Strong Buy", "value": str(len(alpha_df[alpha_df["composite_alpha"] > 0.5]))},
            {"label": "Strong Sell", "value": str(len(alpha_df[alpha_df["composite_alpha"] < -0.5]))},
            {"label": "Avg Confidence", "value": f"{alpha_df['confidence'].mean() * 100:.1f}%"},
        ])

        tab1, tab2, tab3 = st.tabs(["Long Ideas", "Short Ideas", "Full Matrix"])

        with tab1:
            longs = alpha_df.nlargest(15, "composite_alpha")
            st.dataframe(
                longs[["symbol", "composite_alpha", "confidence", "alpha_rank"]].rename(
                    columns={"symbol": "Symbol", "composite_alpha": "Alpha",
                             "confidence": "Confidence", "alpha_rank": "Rank"}),
                width="stretch", hide_index=True)
        with tab2:
            shorts = alpha_df.nsmallest(15, "composite_alpha")
            st.dataframe(
                shorts[["symbol", "composite_alpha", "confidence", "alpha_rank"]].rename(
                    columns={"symbol": "Symbol", "composite_alpha": "Alpha",
                             "confidence": "Confidence", "alpha_rank": "Rank"}),
                width="stretch", hide_index=True)
        with tab3:
            st.dataframe(alpha_df.sort_values("alpha_rank"),
                         width="stretch", hide_index=True)
    else:
        st.info("No alpha signals available. Run `python download_data.py` to generate.")


# ============================================================================
# PAGE: PORTFOLIO
# ============================================================================

elif page == "Portfolio":
    page_header("Portfolio", "Position tracking & risk attribution")

    PORTFOLIO_FILE = DATA_CACHE / "portfolio.json"
    
    def load_portfolio():
        if PORTFOLIO_FILE.exists():
            with open(PORTFOLIO_FILE) as f:
                return json.load(f)
        return {
            "AAPL": {"shares": 100, "cost": 175.00},
            "MSFT": {"shares": 50, "cost": 380.00},
            "NVDA": {"shares": 30, "cost": 450.00},
            "GOOGL": {"shares": 25, "cost": 140.00},
        }
    
    def save_portfolio(portfolio):
        PORTFOLIO_FILE.parent.mkdir(exist_ok=True)
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolio, f, indent=2)
    
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = load_portfolio()

    total_value = total_cost = 0
    holdings = []

    for sym, data in st.session_state.portfolio.items():
        q = fetch_realtime_quote(sym)
        price = q["price"] if q["price"] > 0 else data["cost"]
        value = price * data["shares"]
        cost = data["cost"] * data["shares"]
        pnl = value - cost
        pnl_pct = (pnl / cost) * 100 if cost > 0 else 0
        total_value += value
        total_cost += cost
        holdings.append({
            "Symbol": sym, "Shares": data["shares"],
            "Cost Basis": f"${data['cost']:.2f}", "Price": f"${price:.2f}",
            "Value": f"${value:,.2f}", "P&L": f"${pnl:,.2f}",
            "Return": f"{pnl_pct:+.2f}%",
        })

    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
    d_sgn = "+" if total_pnl >= 0 else ""

    metric_grid([
        {"label": "Portfolio Value", "value": f"${total_value:,.0f}"},
        {"label": "Total Cost", "value": f"${total_cost:,.0f}"},
        {"label": "Total P&L", "value": f"${total_pnl:,.0f}", "delta": f"{d_sgn}{total_pnl_pct:.2f}%"},
        {"label": "Positions", "value": str(len(st.session_state.portfolio))},
    ])

    ph1, ph2 = st.columns([2.2, 1])

    with ph1:
        st.markdown('<div class="section-title">Holdings</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(holdings), width="stretch", hide_index=True)
        if st.button("Save Portfolio", type="secondary"):
            save_portfolio(st.session_state.portfolio)
            st.success("Portfolio saved to disk")

    with ph2:
        st.markdown('<div class="section-title">Allocation</div>', unsafe_allow_html=True)

        alloc = []
        for sym, data in st.session_state.portfolio.items():
            q = fetch_realtime_quote(sym)
            price = q["price"] if q["price"] > 0 else data["cost"]
            alloc.append({"Symbol": sym, "Value": price * data["shares"]})
        alloc_df = pd.DataFrame(alloc)

        fig = go.Figure(data=[go.Pie(
            labels=alloc_df["Symbol"], values=alloc_df["Value"], hole=0.6,
            marker=dict(colors=[C_INDIGO, C_SKY, C_GREEN, C_AMBER, C_RED]),
            textfont=dict(family="JetBrains Mono", size=11),
        )])
        fig.update_layout(**_base_layout(height=280,
                          margin=dict(l=16, r=16, t=16, b=16),
                          legend=dict(font=dict(size=10))))
        st.plotly_chart(fig, width="stretch")


# ============================================================================
# PAGE: RESEARCH
# ============================================================================

elif page == "Research":
    page_header("Research Lab", "Fundamental screening & factor analysis")

    fundamentals = load_fundamentals()

    if not fundamentals.empty:
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            sector_filter = st.selectbox("Sector",
                ["All"] + fundamentals["sector"].dropna().unique().tolist())
        with rc2:
            pe_max = st.slider("Max P/E Ratio", 0, 100, 50)
        with rc3:
            sort_by = st.selectbox("Sort By",
                ["market_cap", "pe_ratio", "profit_margin", "roe"])

        filtered = fundamentals.copy()
        if sector_filter != "All":
            filtered = filtered[filtered["sector"] == sector_filter]
        filtered = filtered[filtered["pe_ratio"] <= pe_max]
        filtered = filtered.sort_values(sort_by, ascending=False)

        st.dataframe(
            filtered[["symbol", "name", "sector", "market_cap",
                       "pe_ratio", "profit_margin", "roe"]].head(20),
            width="stretch", hide_index=True)
    else:
        st.info("Run `python download_data.py` to load fundamental data.")


# ============================================================================
# PAGE: AGENTS
# ============================================================================

elif page == "Agents":
    page_header("Agent Pipeline", "Cooperative multi-agent execution engine",
                [{"type": "ai", "label": "Reasoning Engine", "dot": True}])

    # Import agent system
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from agents.agent_registry import create_default_registry
        from agents.orchestrator import AgentOrchestrator
        from core.event_bus import EventBus
        AGENTS_OK = True
    except Exception as _e:
        AGENTS_OK = False
        st.error(f"Agent system unavailable: {_e}")

    if AGENTS_OK:
        @st.cache_resource
        def get_orchestrator():
            o = AgentOrchestrator()
            o.start()
            return o

        orch = get_orchestrator()
        registry = orch.registry
        bus = EventBus()

        # ── Pipeline Status ──
        st.markdown('<div class="section-title">Pipeline Status</div>', unsafe_allow_html=True)

        pipeline_stages = [
            ("Data Ingest", "DataIngestionAgent", "01"),
            ("Quality", "DataQualityAgent", "02"),
            ("Features", "FeatureEngineeringAgent", "03"),
            ("Regime", "RegimeDetectionAgent", "04"),
            ("Model", "ModelingAgent", "05"),
            ("Decision", "DecisionAgent", "06"),
            ("Risk", "RiskAgent", "07"),
            ("Scenario", "ScenarioAgent", "08"),
            ("Monitor", "MonitoringAgent", "09"),
            ("Lifecycle", "LifecycleAgent", "10"),
        ]

        # Two rows of 5
        chips_html = '<div class="agent-grid">'
        for label, agent_name, num in pipeline_stages:
            agent = registry.get(agent_name)
            if agent:
                h = agent.health_check()
                status = h.get("status", "unknown")
            else:
                status = "missing"

            chip_cls = "ok" if status == "healthy" else ("warn" if status == "degraded" else "err")
            color_map = {"healthy": "var(--success)", "degraded": "var(--warning)",
                         "error": "var(--danger)", "missing": "var(--danger)"}
            col = color_map.get(status, "var(--danger)")

            chips_html += f"""
            <div class="agent-chip {chip_cls}">
                <div class="agent-chip-icon" style="font-family:var(--font-mono); font-size:0.65rem; color:var(--text-muted);">{num}</div>
                <div class="agent-chip-name">{label}</div>
                <div class="agent-chip-status" style="color:{col};">{status}</div>
            </div>"""
        chips_html += '</div>'
        st.markdown(chips_html, unsafe_allow_html=True)

        # System Metrics
        health_data = registry.health_check_all()
        sys_status = health_data.get("system_status", "unknown")

        metric_grid([
            {"label": "System", "value": sys_status.upper()},
            {"label": "Agents", "value": str(health_data.get("agent_count", 0))},
            {"label": "Total Events", "value": str(bus.stats.get("total_events", 0))},
            {"label": "Unique Processed", "value": str(bus.stats.get("unique_processed", 0))},
        ])

        # ── Run Pipeline ──
        st.markdown('<div class="section-title">Run Pipeline</div>', unsafe_allow_html=True)

        rp1, rp2, rp3 = st.columns(3)
        with rp1:
            agent_symbols = st.text_input("Symbols (comma-separated)",
                                          value="AAPL, MSFT, GOOGL", key="agent_sym")
        with rp2:
            agent_source = st.selectbox("Data Source", ["yahoo", "openbb"], key="agent_src")
        with rp3:
            agent_period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1, key="agent_per")

        if st.button("Run Pipeline", type="primary"):
            symbols = [s.strip() for s in agent_symbols.split(",") if s.strip()]
            with st.spinner(f"Executing pipeline for {symbols}..."):
                results = orch.run_pipeline(symbols, source=agent_source,
                                            period=agent_period)
                for sym, result in results.items():
                    d = result.to_dict()
                    icon = "+" if d["success"] else "!"
                    with st.expander(f"[{icon}] {sym} — {d['duration_ms']:.0f}ms", expanded=True):
                        completed = d["stages_completed"]
                        for sl, _, sn in pipeline_stages:
                            sk = sl.replace(" ", "")
                            done = any(sk.lower() in s.lower().replace(" ", "")
                                       for s in completed)
                            mark = "done" if done else "pending"
                            c = "var(--success)" if done else "var(--text-muted)"
                            st.markdown(f'<span style="color:{c}; font-family:var(--font-mono); font-size:0.8rem;">[{mark}] {sn} {sl}</span>', unsafe_allow_html=True)

                        data = d.get("data", {})
                        if data.get("signal"):
                            dr1, dr2, dr3, dr4 = st.columns(4)
                            dr1.metric("Signal", data.get("signal", "—"))
                            _conf = data.get('confidence', 0)
                            dr2.metric("Confidence", f"{float(_conf):.1%}" if _conf not in (None, '') else "—")
                            dr3.metric("Regime", data.get("regime", "—"))
                            _conv = data.get('conviction', 0)
                            dr4.metric("Conviction", f"{float(_conv):.1%}" if _conv not in (None, '') else "—")
                        if data.get("resilience_score") is not None:
                            _res = data['resilience_score']
                            st.metric("Resilience", f"{float(_res):.1%}" if _res not in (None, '') else "—")

        st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

        # ── Agent Inspector ──
        st.markdown('<div class="section-title">Agent Inspector</div>', unsafe_allow_html=True)

        selected_agent = st.selectbox("Select Agent",
            [n for n in registry.agent_names], key="agent_detail")

        if selected_agent:
            agent = registry.get(selected_agent)
            if agent:
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.markdown('<div class="section-subtitle">Health</div>', unsafe_allow_html=True)
                    st.json(agent.health)
                with dc2:
                    st.markdown('<div class="section-subtitle">Metrics</div>', unsafe_allow_html=True)
                    st.json(agent.metrics)

                st.markdown('<div class="section-subtitle">Recent Logs</div>', unsafe_allow_html=True)
                logs = agent.logs[-20:]
                if logs:
                    for entry in reversed(logs):
                        if "[ERROR]" in entry:
                            st.markdown(f'<span style="color:var(--danger); font-family:var(--font-mono); font-size:0.78rem;">{entry}</span>', unsafe_allow_html=True)
                        elif "[WARNING]" in entry:
                            st.markdown(f'<span style="color:var(--warning); font-family:var(--font-mono); font-size:0.78rem;">{entry}</span>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<span style="color:var(--text-secondary); font-family:var(--font-mono); font-size:0.78rem;">{entry}</span>', unsafe_allow_html=True)
                else:
                    st.caption("No logs yet — run the pipeline to generate activity.")

        st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Event Bus</div>', unsafe_allow_html=True)
        st.json(bus.stats)


# ============================================================================
# PAGE: INDIA
# ============================================================================

elif page == "India":
    page_header("Indian Markets", "NSE / BSE real-time data & analysis",
                [{"type": "live", "label": "NSE Live", "dot": True}])
    
    # Indian market indices
    in_indices = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "NIFTY Bank": "^NSEBANK",
        "NIFTY IT": "^CNXIT",
    }
    
    items = []
    for name, tkr in in_indices.items():
        q = fetch_realtime_quote(tkr)
        if q["price"] and q["price"] > 0:
            d_sgn = "+" if q["change_pct"] >= 0 else ""
            items.append({
                "label": name,
                "value": f"{q['price']:,.2f}",
                "delta": f"{d_sgn}{q['change_pct']:.2f}%",
            })
    if items:
        metric_grid(items)
    
    # Indian stock search
    ic1, ic2, ic3 = st.columns([2, 1, 1])
    with ic1:
        in_ticker = st.text_input("Indian Stock (e.g. RELIANCE.NS)", value="RELIANCE.NS", key="in_ticker").upper()
    with ic2:
        in_period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3, key="in_period")
    with ic3:
        in_exchange = st.selectbox("Exchange", ["NSE (.NS)", "BSE (.BO)"], key="in_exchange")
    
    # Auto-append suffix if not present
    if in_ticker and not in_ticker.endswith((".NS", ".BO")):
        suffix = ".NS" if "NSE" in in_exchange else ".BO"
        in_ticker = in_ticker + suffix
    
    if in_ticker:
        in_df = fetch_stock_data(in_ticker, in_period)
        if not in_df.empty:
            in_df = calculate_indicators(in_df)
            in_q = fetch_realtime_quote(in_ticker)
            
            d_sgn = "+" if in_q.get("change_pct", 0) >= 0 else ""
            metric_grid([
                {"label": "Price", "value": f"\u20b9{in_q.get('price', 0):,.2f}", "delta": f"{d_sgn}{in_q.get('change_pct', 0):.2f}%"},
                {"label": "Open", "value": f"\u20b9{in_q.get('open', 0):,.2f}"},
                {"label": "Day High", "value": f"\u20b9{in_q.get('high', 0):,.2f}"},
                {"label": "Day Low", "value": f"\u20b9{in_q.get('low', 0):,.2f}"},
                {"label": "Volume", "value": fmt(in_q.get("volume", 0), "")},
                {"label": "Mkt Cap", "value": fmt(in_q.get("market_cap", 0))},
            ])
            
            st.plotly_chart(create_advanced_chart(in_df, in_ticker), width="stretch")
        else:
            st.warning(f"No data found for {in_ticker}. Make sure the symbol is correct and ends with .NS (NSE) or .BO (BSE).")
    
    # Popular Indian stocks quick access
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Popular Indian Stocks</div>', unsafe_allow_html=True)
    
    popular = {
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "Tata Consultancy",
        "HDFCBANK.NS": "HDFC Bank",
        "INFY.NS": "Infosys",
        "ICICIBANK.NS": "ICICI Bank",
        "SBIN.NS": "State Bank of India",
        "BHARTIARTL.NS": "Bharti Airtel",
        "ITC.NS": "ITC Limited",
        "KOTAKBANK.NS": "Kotak Mahindra",
        "LT.NS": "Larsen & Toubro",
    }
    
    pop_html = '<div style="display:grid; grid-template-columns:repeat(5, 1fr); gap:0.5rem;">'
    for sym, name in popular.items():
        q = fetch_realtime_quote(sym)
        price = q.get("price", 0)
        chg = q.get("change_pct", 0)
        c = "var(--success)" if chg >= 0 else "var(--danger)"
        d_sgn = "+" if chg >= 0 else ""
        pop_html += f"""
        <div style="background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius); padding:0.7rem;">
            <div style="font-family:var(--font-mono); font-weight:600; font-size:0.82rem; color:var(--text-primary);">{sym.replace('.NS','')}</div>
            <div style="font-family:var(--font-body); font-size:0.68rem; color:var(--text-tertiary);">{name}</div>
            <div style="font-family:var(--font-mono); font-size:0.78rem; color:var(--text-primary); margin-top:0.3rem;">\u20b9{price:,.2f}</div>
            <div style="font-family:var(--font-mono); font-size:0.72rem; color:{c};">{d_sgn}{chg:.2f}%</div>
        </div>"""
    pop_html += '</div>'
    st.markdown(pop_html, unsafe_allow_html=True)


# ============================================================================
# PAGE: SETTINGS
# ============================================================================

elif page == "Settings":
    page_header("Settings", "Configuration & system info")

    st.markdown('<div class="section-title">Data Management</div>', unsafe_allow_html=True)

    if st.button("Clear Cache & Refresh"):
        st.cache_data.clear()
        st.success("Cache cleared — data will refresh on next load.")

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">System Information</div>', unsafe_allow_html=True)

    data_files = list(DATA_CACHE.glob("*.parquet")) if DATA_CACHE.exists() else []

    info_items = [
        ("Cache Location", str(DATA_CACHE)),
        ("Cached Files", str(len(data_files))),
        ("Framework", "Streamlit"),
        ("Design System", "Midnight SaaS"),
        ("Chart Engine", "Plotly"),
        ("Agent Engine", "Aegis Orchestrator"),
    ]

    rows_html = ""
    for label, value in info_items:
        rows_html += f"""
        <div style="display:flex; justify-content:space-between; padding:0.5rem 0;
                    border-bottom:1px solid var(--border-subtle);">
            <span style="font-family:var(--font-body); font-size:0.82rem; color:var(--text-tertiary);">{label}</span>
            <span style="font-family:var(--font-mono); font-size:0.82rem; color:var(--text-primary);">{value}</span>
        </div>"""

    st.markdown(f"""
    <div class="card">
        <div class="card-body">{rows_html}</div>
    </div>
    """, unsafe_allow_html=True)
