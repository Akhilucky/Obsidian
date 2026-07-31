"""
AEGIS — Institutional Quantitative Trading Platform
=====================================================
Art-Deco-inspired editorial finance dashboard with
gold-on-obsidian luxury aesthetic.
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
    page_title="AEGIS Terminal",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# DESIGN SYSTEM — Obsidian & Gold Editorial Finance
# ============================================================================

st.markdown("""
<style>
    /* ─── Typography ─── */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400;1,500&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700&family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');

    /* ─── Design Tokens ─── */
    :root {
        /* Backgrounds */
        --obsidian:       #0B0C10;
        --onyx:           #111318;
        --charcoal:       #181B22;
        --slate:          #1F222C;
        --graphite:       #282C38;

        /* Gold spectrum */
        --gold:           #C9A84C;
        --gold-bright:    #E8C866;
        --gold-dim:       #8B7332;
        --gold-glow:      rgba(201, 168, 76, 0.12);
        --gold-border:    rgba(201, 168, 76, 0.22);

        /* Signal palette */
        --emerald:        #3ECF8E;
        --emerald-dim:    rgba(62, 207, 142, 0.15);
        --crimson:        #EF4444;
        --crimson-dim:    rgba(239, 68, 68, 0.15);
        --amber:          #F59E0B;
        --sky:            #38BDF8;

        /* Text */
        --text-primary:   #E8E6E1;
        --text-secondary: #8B8D94;
        --text-muted:     #54565E;

        /* Typography */
        --font-display:   'Playfair Display', Georgia, serif;
        --font-body:      'DM Sans', -apple-system, sans-serif;
        --font-mono:      'IBM Plex Mono', 'Menlo', monospace;

        /* Borders & Effects */
        --border:         rgba(255, 255, 255, 0.06);
        --border-hover:   rgba(201, 168, 76, 0.35);
        --shadow-sm:      0 1px 3px rgba(0,0,0,0.4);
        --shadow-md:      0 4px 14px rgba(0,0,0,0.5);
        --shadow-lg:      0 10px 40px rgba(0,0,0,0.6);
        --shadow-gold:    0 4px 20px rgba(201, 168, 76, 0.15);
        --radius:         10px;
        --radius-sm:      6px;
    }

    /* ─── GLOBAL ─── */
    .stApp {
        background: var(--obsidian);
        font-family: var(--font-body);
        color: var(--text-primary);
    }

    /* Grain texture overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.025'/%3E%3C/svg%3E");
        pointer-events: none;
        z-index: 0;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    div[data-testid="stDecoration"] {display: none;}

    /* ─── ANIMATIONS ─── */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse-gold {
        0%, 100% { box-shadow: 0 0 0 0 rgba(201,168,76,0.4); }
        50%      { box-shadow: 0 0 0 6px rgba(201,168,76,0); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    @keyframes breathe {
        0%, 100% { opacity: 0.6; }
        50%      { opacity: 1; }
    }

    .reveal {
        animation: fadeUp 0.5s ease-out both;
    }
    .reveal-1 { animation-delay: 0.05s; }
    .reveal-2 { animation-delay: 0.10s; }
    .reveal-3 { animation-delay: 0.15s; }
    .reveal-4 { animation-delay: 0.20s; }
    .reveal-5 { animation-delay: 0.25s; }
    .reveal-6 { animation-delay: 0.30s; }

    /* ─── MASTHEAD ─── */
    .masthead {
        background: linear-gradient(135deg, var(--onyx) 0%, var(--charcoal) 50%, var(--onyx) 100%);
        border: 1px solid var(--border);
        border-bottom: 2px solid var(--gold-border);
        border-radius: var(--radius);
        padding: 1.6rem 2.2rem;
        margin-bottom: 1.8rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        animation: fadeUp 0.4s ease-out both;
        position: relative;
        overflow: hidden;
    }

    /* Diagonal art-deco accent */
    .masthead::before {
        content: '';
        position: absolute;
        top: -50%; right: -5%;
        width: 200px; height: 200%;
        background: linear-gradient(135deg, transparent 30%, var(--gold-glow) 50%, transparent 70%);
        transform: rotate(25deg);
        pointer-events: none;
    }

    .masthead-title {
        font-family: var(--font-display);
        font-size: 1.7rem;
        font-weight: 700;
        color: var(--gold);
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .masthead-subtitle {
        font-family: var(--font-mono);
        font-size: 0.7rem;
        color: var(--text-muted);
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 2px;
    }

    .masthead-status {
        display: flex;
        gap: 1.8rem;
        align-items: center;
        font-family: var(--font-mono);
        font-size: 0.78rem;
        color: var(--text-secondary);
        z-index: 1;
    }

    .live-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: var(--emerald);
        display: inline-block;
        margin-right: 6px;
        animation: pulse-gold 2s infinite;
    }

    /* ─── METRIC CARDS ─── */
    .kpi-card {
        background: linear-gradient(160deg, var(--onyx) 0%, var(--charcoal) 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.2rem 1.4rem;
        transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1);
        position: relative;
        overflow: hidden;
    }

    .kpi-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--gold-dim), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .kpi-card:hover {
        border-color: var(--border-hover);
        transform: translateY(-3px);
        box-shadow: var(--shadow-gold);
    }
    .kpi-card:hover::after {
        opacity: 1;
    }

    .kpi-label {
        font-family: var(--font-mono);
        font-size: 0.68rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.55rem;
    }

    .kpi-value {
        font-family: var(--font-mono);
        font-size: 1.45rem;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.2;
    }

    .kpi-delta {
        font-family: var(--font-mono);
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }
    .kpi-delta.up   { color: var(--emerald); }
    .kpi-delta.down { color: var(--crimson); }

    /* ─── PANELS ─── */
    .panel {
        background: var(--onyx);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.5rem 1.6rem;
        margin-bottom: 1.4rem;
    }

    .panel-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.85rem;
        border-bottom: 1px solid var(--border);
    }

    .panel-title {
        font-family: var(--font-display);
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: 0.5px;
    }

    .panel-badge {
        font-family: var(--font-mono);
        font-size: 0.6rem;
        padding: 0.25rem 0.65rem;
        border-radius: 3px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .badge-gold { background: var(--gold-glow); color: var(--gold); border: 1px solid var(--gold-border); }
    .badge-live { background: var(--emerald-dim); color: var(--emerald); border: 1px solid rgba(62,207,142,0.25); }

    /* ─── DATA TABLE ─── */
    .data-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid var(--border);
        transition: background 0.2s ease;
    }
    .data-row:hover {
        background: var(--gold-glow);
    }
    .data-sym {
        font-family: var(--font-mono);
        font-weight: 600;
        font-size: 0.88rem;
        color: var(--text-primary);
    }
    .data-val {
        font-family: var(--font-mono);
        font-size: 0.85rem;
    }
    .val-pos { color: var(--emerald); }
    .val-neg { color: var(--crimson); }
    .val-neutral { color: var(--text-secondary); }

    /* ─── AGENT PIPELINE CARDS ─── */
    .agent-card {
        background: linear-gradient(160deg, var(--onyx) 0%, var(--charcoal) 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.9rem;
        text-align: center;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }
    .agent-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    .agent-card.healthy  { border-color: rgba(62,207,142,0.35); }
    .agent-card.degraded { border-color: rgba(245,158,11,0.35); }
    .agent-card.error    { border-color: rgba(239,68,68,0.35); }

    .agent-icon { font-size: 1.4rem; margin-bottom: 0.3rem; }
    .agent-name {
        font-family: var(--font-body);
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .agent-status {
        font-family: var(--font-mono);
        font-size: 0.6rem;
        margin-top: 0.2rem;
        letter-spacing: 0.5px;
    }

    /* ─── SIDEBAR ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--onyx) 0%, var(--obsidian) 100%);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] .stRadio > label {
        background: transparent;
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 0.75rem 1rem;
        margin-bottom: 0.35rem;
        font-family: var(--font-body);
        font-size: 0.88rem;
        transition: all 0.2s ease;
    }

    section[data-testid="stSidebar"] .stRadio > label:hover {
        border-color: var(--gold-border);
        background: var(--gold-glow);
    }

    /* ─── BUTTONS ─── */
    .stButton > button {
        background: linear-gradient(135deg, var(--gold-dim) 0%, var(--gold) 100%);
        color: var(--obsidian);
        border: none;
        border-radius: var(--radius-sm);
        padding: 0.7rem 1.6rem;
        font-family: var(--font-body);
        font-weight: 600;
        font-size: 0.88rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-gold);
        background: linear-gradient(135deg, var(--gold) 0%, var(--gold-bright) 100%);
    }

    /* ─── INPUTS ─── */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {
        background: var(--charcoal) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.88rem !important;
    }
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus-within {
        border-color: var(--gold-border) !important;
        box-shadow: 0 0 0 1px var(--gold-border) !important;
    }

    /* ─── TABS ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--onyx);
        border-radius: var(--radius-sm);
        padding: 3px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 4px;
        padding: 0.5rem 1.1rem;
        color: var(--text-secondary);
        font-family: var(--font-body);
        font-weight: 500;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: var(--gold);
        color: var(--obsidian) !important;
        font-weight: 600;
    }

    /* ─── SCROLLBAR ─── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--obsidian); }
    ::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 3px; }

    /* ─── STREAMLIT OVERRIDES ─── */
    .stMetric label { font-family: var(--font-mono) !important; font-size: 0.7rem !important;
        color: var(--text-muted) !important; text-transform: uppercase; letter-spacing: 1px; }
    .stMetric [data-testid="stMetricValue"] {
        font-family: var(--font-mono) !important; font-weight: 600 !important; color: var(--text-primary) !important; }
    .stMetric [data-testid="stMetricDelta"] { font-family: var(--font-mono) !important; }

    div[data-testid="stExpander"] {
        background: var(--onyx);
        border: 1px solid var(--border);
        border-radius: var(--radius);
    }
    div[data-testid="stExpander"]:hover {
        border-color: var(--gold-border);
    }

    /* Plotly background override */
    .js-plotly-plot .plotly .main-svg { border-radius: var(--radius); }

    /* Signal colors */
    .sig-strong-buy  { color: var(--emerald); font-weight: 600; }
    .sig-buy         { color: #6EE7A8; }
    .sig-neutral     { color: var(--text-secondary); }
    .sig-sell        { color: #FCA5A5; }
    .sig-strong-sell { color: var(--crimson); font-weight: 600; }

    /* Divider line with gold center */
    .gold-rule {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--gold-dim), transparent);
        border: none;
        margin: 1.6rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA FUNCTIONS  (identical logic, kept compact)
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
    if num >= 1e12:
        return f"{prefix}{num/1e12:.2f}T"
    if num >= 1e9:
        return f"{prefix}{num/1e9:.2f}B"
    if num >= 1e6:
        return f"{prefix}{num/1e6:.2f}M"
    if num >= 1e3:
        return f"{prefix}{num/1e3:.2f}K"
    return f"{prefix}{num:.2f}"


def signal_class(val: float) -> str:
    if val > 0.5:   return "sig-strong-buy"
    if val > 0.2:   return "sig-buy"
    if val > -0.2:  return "sig-neutral"
    if val > -0.5:  return "sig-sell"
    return "sig-strong-sell"


# ============================================================================
# CHART THEME — Obsidian & Gold Plotly defaults
# ============================================================================

CHART_BG     = "rgba(11,12,16,0)"
CHART_GRID   = "rgba(255,255,255,0.04)"
CHART_FONT   = dict(family="IBM Plex Mono, monospace", color="#8B8D94", size=11)
CHART_GREEN  = "#3ECF8E"
CHART_RED    = "#EF4444"
CHART_GOLD   = "#C9A84C"
CHART_SKY    = "#38BDF8"
CHART_AMBER  = "#F59E0B"


def _base_layout(height: int = 400, **overrides) -> dict:
    """Base Plotly layout with Obsidian & Gold theme. Overrides win."""
    base = dict(
        template="plotly_dark",
        paper_bgcolor=CHART_BG,
        plot_bgcolor="rgba(17,19,24,0.85)",
        height=height,
        margin=dict(l=50, r=20, t=30, b=30),
        font=CHART_FONT,
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

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="OHLC",
        increasing_line_color=CHART_GREEN, decreasing_line_color=CHART_RED,
        increasing_fillcolor=CHART_GREEN, decreasing_fillcolor=CHART_RED,
    ), row=1, col=1)

    if "sma_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_20"], name="SMA 20",
                                 line=dict(color=CHART_GOLD, width=1)), row=1, col=1)
    if "sma_50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="SMA 50",
                                 line=dict(color=CHART_SKY, width=1)), row=1, col=1)
    if "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper",
                                 line=dict(color="rgba(201,168,76,0.3)", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower",
                                 line=dict(color="rgba(201,168,76,0.3)", dash="dash"),
                                 fill="tonexty", fillcolor="rgba(201,168,76,0.04)"), row=1, col=1)

    # Volume
    vol_colors = [CHART_GREEN if c >= o else CHART_RED for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume",
                         marker_color=vol_colors, opacity=0.65), row=2, col=1)

    # RSI
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
                                 line=dict(color=CHART_GOLD, width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=CHART_RED, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=CHART_GREEN, row=3, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.015)", row=3, col=1)

    # MACD
    if "macd" in df.columns:
        macd_colors = [CHART_GREEN if v >= 0 else CHART_RED for v in df["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="MACD Hist",
                             marker_color=macd_colors), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD",
                                 line=dict(color=CHART_SKY, width=1)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
                                 line=dict(color=CHART_GOLD, width=1)), row=4, col=1)

    fig.update_layout(**_base_layout(height=700), xaxis_rangeslider_visible=False)
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=CHART_GRID, row=i, col=1)
        fig.update_yaxes(gridcolor=CHART_GRID, row=i, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    return fig


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    # Brand
    st.markdown("""
    <div style="text-align:center; padding:1.2rem 0 0.6rem 0;">
        <div style="font-family:var(--font-display); font-size:1.6rem; font-weight:700;
                    color:var(--gold); letter-spacing:3px;">
            ◆ AEGIS
        </div>
        <div style="font-family:var(--font-mono); font-size:0.6rem; color:var(--text-muted);
                    letter-spacing:3px; margin-top:4px;">
            QUANTITATIVE TERMINAL
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["◈ Dashboard", "◈ Analysis", "◈ Alpha Signals",
         "◈ Portfolio", "◈ Research", "◈ Agents", "◈ Settings"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    # Quick Quote
    st.markdown(f"""
    <div style="font-family:var(--font-display); font-size:0.9rem; font-weight:600;
                color:var(--text-primary); margin-bottom:0.6rem;">Quick Quote</div>
    """, unsafe_allow_html=True)
    ticker_input = st.text_input("Symbol", value="AAPL", key="sidebar_ticker",
                                 label_visibility="collapsed").upper()

    if ticker_input:
        q = fetch_realtime_quote(ticker_input)
        if q["price"] > 0:
            delta_cls = "up" if q["change"] >= 0 else "down"
            delta_sign = "+" if q["change"] >= 0 else ""
            st.markdown(f"""
            <div class="kpi-card reveal" style="margin-top:0.6rem;">
                <div class="kpi-label">{q['name']}</div>
                <div class="kpi-value">${q['price']:.2f}</div>
                <div class="kpi-delta {delta_cls}">
                    {delta_sign}{q['change']:.2f} ({delta_sign}{q['change_pct']:.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    # Market Clock
    now = datetime.now()
    mkt_open = 9 <= now.hour < 16 and now.weekday() < 5
    status_txt = "MARKET OPEN" if mkt_open else "MARKET CLOSED"
    status_col = "var(--emerald)" if mkt_open else "var(--crimson)"

    st.markdown(f"""
    <div style="text-align:center; padding:0.5rem; background:var(--charcoal);
                border:1px solid var(--border); border-radius:var(--radius-sm);">
        <div style="font-family:var(--font-mono); font-size:0.65rem; color:var(--text-muted);
                    letter-spacing:2px; text-transform:uppercase;">Status</div>
        <div style="font-family:var(--font-mono); font-size:0.82rem; color:{status_col}; margin-top:3px;">
            ● {status_txt}
        </div>
        <div style="font-family:var(--font-mono); font-size:0.72rem; color:var(--text-muted); margin-top:3px;">
            {now.strftime("%H:%M:%S EST")}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

if page == "◈ Dashboard":
    # Masthead
    st.markdown("""
    <div class="masthead">
        <div>
            <div class="masthead-title">Aegis Terminal</div>
            <div class="masthead-subtitle">Institutional-Grade Quantitative Platform</div>
        </div>
        <div class="masthead-status">
            <span><span class="live-dot"></span>LIVE</span>
            <span>DATA : REAL-TIME</span>
            <span>LATENCY : 12ms</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Market Overview ──
    indices = {
        "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DOW": "^DJI",
        "RUSSELL": "^RUT", "VIX": "^VIX", "10Y YIELD": "^TNX",
    }

    cols = st.columns(6)
    for i, (name, tkr) in enumerate(indices.items()):
        with cols[i]:
            q = fetch_realtime_quote(tkr)
            d_cls = "up" if q["change"] >= 0 else "down"
            d_sgn = "+" if q["change_pct"] >= 0 else ""
            st.markdown(f"""
            <div class="kpi-card reveal reveal-{i+1}">
                <div class="kpi-label">{name}</div>
                <div class="kpi-value">{q['price']:,.2f}</div>
                <div class="kpi-delta {d_cls}">{d_sgn}{q['change_pct']:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main Grid ──
    col_chart, col_alpha = st.columns([2.2, 1])

    with col_chart:
        st.markdown("""
        <div class="panel reveal reveal-2">
            <div class="panel-head">
                <span class="panel-title">S&P 500 — 6 Month</span>
                <span class="panel-badge badge-live">LIVE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sp500 = fetch_stock_data("^GSPC", "6mo")
        if not sp500.empty:
            pct = (sp500["close"].iloc[-1] / sp500["close"].iloc[0] - 1) * 100
            c = CHART_GREEN if pct >= 0 else CHART_RED
            fill_c = "rgba(62,207,142,0.08)" if pct >= 0 else "rgba(239,68,68,0.08)"

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sp500.index, y=sp500["close"], fill="tozeroy",
                fillcolor=fill_c, line=dict(color=c, width=2), name="S&P 500",
            ))
            fig.update_layout(**_base_layout(height=360, showlegend=False,
                              xaxis=dict(showgrid=False),
                              yaxis=dict(showgrid=True, gridcolor=CHART_GRID)))
            st.plotly_chart(fig, width="stretch")

    with col_alpha:
        st.markdown("""
        <div class="panel reveal reveal-3">
            <div class="panel-head">
                <span class="panel-title">Top Alpha Signals</span>
                <span class="panel-badge badge-gold">AI</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        alpha_df = load_alpha_signals()
        if not alpha_df.empty:
            top = alpha_df.nlargest(8, "composite_alpha")
            for _, row in top.iterrows():
                sc = signal_class(row["composite_alpha"])
                st.markdown(f"""
                <div class="data-row">
                    <span class="data-sym">{row['symbol']}</span>
                    <span class="data-val {sc}">{row['composite_alpha']:.3f}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Run `python download_data.py` to generate alpha signals")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bottom Grid ──
    col_sector, col_breadth = st.columns(2)

    with col_sector:
        st.markdown("""
        <div class="panel reveal reveal-4">
            <div class="panel-head">
                <span class="panel-title">Sector Performance</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sectors = {
            "XLK": "Technology", "XLF": "Financials", "XLV": "Healthcare",
            "XLE": "Energy", "XLI": "Industrials", "XLP": "Staples",
            "XLY": "Discretionary", "XLU": "Utilities",
        }
        sec_data = []
        for tkr, nm in sectors.items():
            q = fetch_realtime_quote(tkr)
            if q["price"] > 0:
                sec_data.append({"Sector": nm, "Change": q["change_pct"]})

        if sec_data:
            sec_df = pd.DataFrame(sec_data).sort_values("Change", ascending=True)
            bar_colors = [CHART_GREEN if x >= 0 else CHART_RED for x in sec_df["Change"]]
            fig = go.Figure(go.Bar(
                x=sec_df["Change"], y=sec_df["Sector"], orientation="h",
                marker_color=bar_colors,
                text=[f"{x:+.2f}%" for x in sec_df["Change"]],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=11),
            ))
            fig.update_layout(**_base_layout(height=300,
                              showlegend=False,
                              xaxis=dict(showgrid=True, gridcolor=CHART_GRID,
                                         zeroline=True, zerolinecolor="rgba(255,255,255,0.12)"),
                              yaxis=dict(showgrid=False),
                              margin=dict(l=0, r=50, t=0, b=0)))
            st.plotly_chart(fig, width="stretch")

    with col_breadth:
        st.markdown("""
        <div class="panel reveal reveal-5">
            <div class="panel-head">
                <span class="panel-title">Market Breadth</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        breadth = pd.DataFrame({
            "Metric": ["Advancing", "Declining", "Unchanged", "New Highs", "New Lows"],
            "NYSE": [1842, 1156, 89, 124, 45],
            "NASDAQ": [2156, 1678, 134, 89, 67],
        })
        st.dataframe(breadth, width="stretch", hide_index=True)


# ============================================================================
# PAGE: ANALYSIS
# ============================================================================

elif page == "◈ Analysis":
    st.markdown("""
    <div class="masthead">
        <div>
            <div class="masthead-title">Technical Analysis</div>
            <div class="masthead-subtitle">Multi-Timeframe Chart Suite</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

            # Metric strip
            mcols = st.columns(6)
            mdata = [
                ("Price", f"${q['price']:.2f}", f"{q['change_pct']:+.2f}%"),
                ("Open", f"${q['open']:.2f}", None),
                ("High", f"${q['high']:.2f}", None),
                ("Low", f"${q['low']:.2f}", None),
                ("Volume", fmt(q["volume"], ""), None),
                ("Mkt Cap", fmt(q["market_cap"]), None),
            ]
            for col, (lbl, val, delta) in zip(mcols, mdata):
                with col:
                    st.metric(lbl, val, delta) if delta else st.metric(lbl, val)

            st.plotly_chart(create_advanced_chart(df, ticker), width="stretch")

            # Technical Summary
            st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)
            st.markdown("""<div style="font-family:var(--font-display); font-size:1.1rem;
                        font-weight:600; color:var(--gold); margin-bottom:1rem;">
                        Technical Summary</div>""", unsafe_allow_html=True)

            tc1, tc2, tc3, tc4 = st.columns(4)
            current = df["close"].iloc[-1]

            with tc1:
                sma20 = df["sma_20"].iloc[-1] if "sma_20" in df.columns else 0
                sma50 = df["sma_50"].iloc[-1] if "sma_50" in df.columns else 0
                st.markdown("**Moving Averages**")
                if sma20 > 0:
                    sig = "🟢 ABOVE" if current > sma20 else "🔴 BELOW"
                    st.markdown(f"SMA 20: {sig}")
                if sma50 > 0:
                    sig = "🟢 ABOVE" if current > sma50 else "🔴 BELOW"
                    st.markdown(f"SMA 50: {sig}")
            with tc2:
                if "rsi" in df.columns:
                    rsi = df["rsi"].iloc[-1]
                    st.markdown("**RSI (14)**")
                    if rsi > 70:
                        st.markdown(f"🔴 OVERBOUGHT ({rsi:.1f})")
                    elif rsi < 30:
                        st.markdown(f"🟢 OVERSOLD ({rsi:.1f})")
                    else:
                        st.markdown(f"⚪ NEUTRAL ({rsi:.1f})")
            with tc3:
                if "macd" in df.columns:
                    st.markdown("**MACD**")
                    if df["macd"].iloc[-1] > df["macd_signal"].iloc[-1]:
                        st.markdown("🟢 BULLISH")
                    else:
                        st.markdown("🔴 BEARISH")
            with tc4:
                if "bb_upper" in df.columns:
                    bb_pos = (current - df["bb_lower"].iloc[-1]) / \
                             (df["bb_upper"].iloc[-1] - df["bb_lower"].iloc[-1]) * 100
                    st.markdown("**Bollinger %B**")
                    st.markdown(f"{bb_pos:.1f}%")


# ============================================================================
# PAGE: ALPHA SIGNALS
# ============================================================================

elif page == "◈ Alpha Signals":
    st.markdown("""
    <div class="masthead">
        <div>
            <div class="masthead-title">Alpha Signal Matrix</div>
            <div class="masthead-subtitle">AI-Powered Factor Decomposition</div>
        </div>
        <div class="masthead-status">
            <span><span class="live-dot"></span>AI-POWERED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    alpha_df = load_alpha_signals()

    if not alpha_df.empty:
        ac1, ac2, ac3, ac4 = st.columns(4)
        with ac1:
            st.metric("Total Signals", len(alpha_df))
        with ac2:
            st.metric("Strong Buy", len(alpha_df[alpha_df["composite_alpha"] > 0.5]))
        with ac3:
            st.metric("Strong Sell", len(alpha_df[alpha_df["composite_alpha"] < -0.5]))
        with ac4:
            st.metric("Avg Confidence", f"{alpha_df['confidence'].mean() * 100:.1f}%")

        st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

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
        st.warning("No alpha signals found. Run `python download_data.py` to generate.")


# ============================================================================
# PAGE: PORTFOLIO
# ============================================================================

elif page == "◈ Portfolio":
    st.markdown("""
    <div class="masthead">
        <div>
            <div class="masthead-title">Portfolio Management</div>
            <div class="masthead-subtitle">Position Tracking & Risk Attribution</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "portfolio" not in st.session_state:
        st.session_state.portfolio = {
            "AAPL": {"shares": 100, "cost": 175.00},
            "MSFT": {"shares": 50, "cost": 380.00},
            "NVDA": {"shares": 30, "cost": 450.00},
            "GOOGL": {"shares": 25, "cost": 140.00},
        }

    total_value = total_cost = 0
    holdings = []

    for sym, data in st.session_state.portfolio.items():
        q = fetch_realtime_quote(sym)
        price = q["price"] if q["price"] > 0 else data["cost"]
        value = price * data["shares"]
        cost = data["cost"] * data["shares"]
        pnl = value - cost
        pnl_pct = (pnl / cost) * 100
        total_value += value
        total_cost += cost
        holdings.append({
            "Symbol": sym, "Shares": data["shares"],
            "Cost": f"${data['cost']:.2f}", "Price": f"${price:.2f}",
            "Value": f"${value:,.2f}", "P&L": f"${pnl:,.2f}",
            "P&L %": f"{pnl_pct:+.2f}%",
        })

    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) * 100

    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        st.metric("Total Value", f"${total_value:,.2f}")
    with pc2:
        st.metric("Total Cost", f"${total_cost:,.2f}")
    with pc3:
        st.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl_pct:+.2f}%")
    with pc4:
        st.metric("Positions", len(st.session_state.portfolio))

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    ph1, ph2 = st.columns([2, 1])

    with ph1:
        st.markdown("""<div style="font-family:var(--font-display); font-size:1.05rem;
                    font-weight:600; color:var(--text-primary); margin-bottom:0.8rem;">
                    Holdings</div>""", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(holdings), width="stretch", hide_index=True)

    with ph2:
        st.markdown("""<div style="font-family:var(--font-display); font-size:1.05rem;
                    font-weight:600; color:var(--text-primary); margin-bottom:0.8rem;">
                    Allocation</div>""", unsafe_allow_html=True)

        alloc = []
        for sym, data in st.session_state.portfolio.items():
            q = fetch_realtime_quote(sym)
            price = q["price"] if q["price"] > 0 else data["cost"]
            alloc.append({"Symbol": sym, "Value": price * data["shares"]})
        alloc_df = pd.DataFrame(alloc)

        fig = go.Figure(data=[go.Pie(
            labels=alloc_df["Symbol"], values=alloc_df["Value"], hole=0.55,
            marker=dict(colors=[CHART_GOLD, CHART_SKY, CHART_GREEN, CHART_AMBER, CHART_RED]),
            textfont=dict(family="IBM Plex Mono"),
        )])
        fig.update_layout(**_base_layout(height=300,
                          margin=dict(l=20, r=20, t=20, b=20),
                          legend=dict(font=dict(size=10))))
        st.plotly_chart(fig, width="stretch")


# ============================================================================
# PAGE: RESEARCH
# ============================================================================

elif page == "◈ Research":
    st.markdown("""
    <div class="masthead">
        <div>
            <div class="masthead-title">Research Lab</div>
            <div class="masthead-subtitle">Fundamental Screening & Factor Analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fundamentals = load_fundamentals()

    if not fundamentals.empty:
        st.markdown("""<div style="font-family:var(--font-display); font-size:1.05rem;
                    font-weight:600; color:var(--gold); margin-bottom:1rem;">
                    Fundamental Screener</div>""", unsafe_allow_html=True)

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            sector_filter = st.selectbox("Sector",
                ["All"] + fundamentals["sector"].dropna().unique().tolist())
        with rc2:
            pe_max = st.slider("Max P/E", 0, 100, 50)
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
        st.info("Run `python download_data.py` to download fundamental data")


# ============================================================================
# PAGE: AGENTS
# ============================================================================

elif page == "◈ Agents":
    st.markdown("""
    <div class="masthead">
        <div>
            <div class="masthead-title">Agent Command Center</div>
            <div class="masthead-subtitle">Cooperative Multi-Agent Pipeline</div>
        </div>
        <div class="masthead-status">
            <span><span class="live-dot"></span>REASONING ENGINE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        st.error(f"Agent system not available: {_e}")

    if AGENTS_OK:
        @st.cache_resource
        def get_orchestrator():
            o = AgentOrchestrator()
            o.start()
            return o

        orch = get_orchestrator()
        registry = orch.registry
        bus = EventBus()

        # ── Pipeline Visualization ──
        st.markdown("""<div style="font-family:var(--font-display); font-size:1.05rem;
                    font-weight:600; color:var(--gold); margin-bottom:1rem;">
                    Pipeline Status</div>""", unsafe_allow_html=True)

        pipeline_stages = [
            ("Data Ingestion", "DataIngestionAgent", "📡"),
            ("Data Quality", "DataQualityAgent", "🧪"),
            ("Feature Eng.", "FeatureEngineeringAgent", "📐"),
            ("Regime Detect", "RegimeDetectionAgent", "🌦️"),
            ("Modeling", "ModelingAgent", "🤖"),
            ("Decision", "DecisionAgent", "🧠"),
            ("Risk Eval", "RiskAgent", "⚖️"),
            ("Scenario Sim", "ScenarioAgent", "🔬"),
            ("Monitoring", "MonitoringAgent", "📊"),
            ("Lifecycle", "LifecycleAgent", "🗂️"),
        ]

        pcols = st.columns(5)
        for i, (label, agent_name, icon) in enumerate(pipeline_stages):
            agent = registry.get(agent_name)
            if agent:
                h = agent.health_check()
                status = h.get("status", "unknown")
            else:
                status = "missing"

            status_cls = "healthy" if status == "healthy" else \
                         ("degraded" if status == "degraded" else "error")
            color_map = {"healthy": "var(--emerald)", "degraded": "var(--amber)",
                         "error": "var(--crimson)", "missing": "var(--crimson)"}
            col = color_map.get(status, "var(--crimson)")

            with pcols[i % 5]:
                st.markdown(f"""
                <div class="agent-card {status_cls} reveal reveal-{(i%5)+1}">
                    <div class="agent-icon">{icon}</div>
                    <div class="agent-name">{label}</div>
                    <div class="agent-status" style="color:{col};">● {status.upper()}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

        # System metrics
        health_data = registry.health_check_all()
        sys_status = health_data.get("system_status", "unknown")

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("System Status", sys_status.upper())
        sc2.metric("Agents", health_data.get("agent_count", 0))
        sc3.metric("Events Published", bus.stats.get("total_published", 0))
        sc4.metric("Events Consumed", bus.stats.get("total_consumed", 0))

        st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

        # ── Run Pipeline ──
        st.markdown("""<div style="font-family:var(--font-display); font-size:1.05rem;
                    font-weight:600; color:var(--gold); margin-bottom:1rem;">
                    Run Pipeline</div>""", unsafe_allow_html=True)

        rp1, rp2, rp3 = st.columns(3)
        with rp1:
            agent_symbols = st.text_input("Symbols (comma-separated)",
                                          value="AAPL, MSFT, GOOGL", key="agent_sym")
        with rp2:
            agent_source = st.selectbox("Data Source", ["yahoo", "openbb"], key="agent_src")
        with rp3:
            agent_period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1, key="agent_per")

        if st.button("▶  Run Full Pipeline", type="primary"):
            symbols = [s.strip() for s in agent_symbols.split(",") if s.strip()]
            with st.spinner(f"Running pipeline for {symbols}..."):
                results = orch.run_pipeline(symbols, source=agent_source,
                                            period=agent_period)
                for sym, result in results.items():
                    d = result.to_dict()
                    icon = "✅" if d["success"] else "⚠️"
                    with st.expander(f"{icon} {sym} — {d['duration_ms']:.0f}ms", expanded=True):
                        completed = d["stages_completed"]
                        for sl, _, si in pipeline_stages:
                            sk = sl.replace(" ", "").replace(".", "")
                            done = any(sk.lower() in s.lower().replace(" ", "")
                                       for s in completed)
                            st.text(f"  {'✅' if done else '⬜'} {si} {sl}")

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

        st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

        # ── Agent Details ──
        st.markdown("""<div style="font-family:var(--font-display); font-size:1.05rem;
                    font-weight:600; color:var(--gold); margin-bottom:1rem;">
                    Agent Inspector</div>""", unsafe_allow_html=True)

        selected_agent = st.selectbox("Select Agent",
            [n for n in registry.agent_names], key="agent_detail")

        if selected_agent:
            agent = registry.get(selected_agent)
            if agent:
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.markdown("**Health**")
                    st.json(agent.health)
                with dc2:
                    st.markdown("**Metrics**")
                    st.json(agent.metrics)

                st.markdown("**Recent Logs**")
                logs = agent.logs[-20:]
                if logs:
                    for entry in reversed(logs):
                        if "[ERROR]" in entry:
                            st.markdown(f"🔴 `{entry}`")
                        elif "[WARNING]" in entry:
                            st.markdown(f"🟡 `{entry}`")
                        else:
                            st.markdown(f"🟢 `{entry}`")
                else:
                    st.info("No logs yet — run the pipeline to generate activity.")

        st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)
        st.markdown("**Event Bus**")
        st.json(bus.stats)


# ============================================================================
# PAGE: SETTINGS
# ============================================================================

elif page == "◈ Settings":
    st.markdown("""
    <div class="masthead">
        <div>
            <div class="masthead-title">Settings</div>
            <div class="masthead-subtitle">Configuration & Data Management</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div style="font-family:var(--font-display); font-size:1.05rem;
                font-weight:600; color:var(--gold); margin-bottom:1rem;">
                Data Management</div>""", unsafe_allow_html=True)

    if st.button("Refresh Market Data"):
        st.cache_data.clear()
        st.success("Cache cleared — data will refresh on next load.")

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    st.markdown("""<div style="font-family:var(--font-display); font-size:1.05rem;
                font-weight:600; color:var(--gold); margin-bottom:1rem;">
                System Info</div>""", unsafe_allow_html=True)

    data_files = list(DATA_CACHE.glob("*.parquet")) if DATA_CACHE.exists() else []

    st.markdown(f"""
    <div class="panel">
        <table style="width:100%; font-family:var(--font-mono); font-size:0.85rem;">
            <tr><td style="color:var(--text-muted); padding:0.4rem 0;">Cache Location</td>
                <td style="color:var(--text-primary); padding:0.4rem 0;">{DATA_CACHE}</td></tr>
            <tr><td style="color:var(--text-muted); padding:0.4rem 0;">Cached Files</td>
                <td style="color:var(--text-primary); padding:0.4rem 0;">{len(data_files)}</td></tr>
            <tr><td style="color:var(--text-muted); padding:0.4rem 0;">Framework</td>
                <td style="color:var(--text-primary); padding:0.4rem 0;">Streamlit</td></tr>
            <tr><td style="color:var(--text-muted); padding:0.4rem 0;">Design System</td>
                <td style="color:var(--gold); padding:0.4rem 0;">Obsidian & Gold — Editorial Finance</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
