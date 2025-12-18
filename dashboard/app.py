"""
Quantum Trading Terminal - Institutional-Grade Trading Platform
================================================================
Professional trading terminal with advanced analytics and risk management.
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
# PAGE CONFIG & THEME
# ============================================================================

st.set_page_config(
    page_title="Quantum Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional dark theme CSS
st.markdown("""
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-tertiary: #1a1a25;
        --accent-blue: #00d4ff;
        --accent-green: #00ff88;
        --accent-red: #ff3366;
        --accent-yellow: #ffcc00;
        --accent-purple: #a855f7;
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
        --border-color: rgba(255,255,255,0.08);
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, #0f0f18 50%, var(--bg-primary) 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main header */
    .terminal-header {
        background: linear-gradient(90deg, rgba(0,212,255,0.1), rgba(168,85,247,0.1));
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .terminal-logo {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    
    .terminal-status {
        display: flex;
        gap: 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    
    .status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--text-secondary);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent-green);
        box-shadow: 0 0 10px var(--accent-green);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Metric cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: var(--accent-blue);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,212,255,0.15);
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .metric-change {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }
    
    .metric-change.positive { color: var(--accent-green); }
    .metric-change.negative { color: var(--accent-red); }
    
    /* Section panels */
    .panel {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .panel-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .panel-badge {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        color: white;
        font-size: 0.65rem;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    /* Data tables */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    
    .data-table th {
        text-align: left;
        padding: 0.75rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 1px;
        border-bottom: 1px solid var(--border-color);
    }
    
    .data-table td {
        padding: 0.75rem;
        color: var(--text-primary);
        border-bottom: 1px solid var(--border-color);
    }
    
    .data-table tr:hover td {
        background: rgba(0,212,255,0.05);
    }
    
    /* Ticker tape */
    .ticker-tape {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 1.5rem;
        overflow: hidden;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    
    .ticker-item {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-right: 2rem;
    }
    
    .ticker-symbol {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    .ticker-price {
        color: var(--text-secondary);
    }
    
    /* Sidebar styles */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] .stRadio > label {
        background: transparent;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }
    
    section[data-testid="stSidebar"] .stRadio > label:hover {
        border-color: var(--accent-blue);
        background: rgba(0,212,255,0.05);
    }
    
    /* Button styles */
    .stButton > button {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,212,255,0.3);
    }
    
    /* Input styles */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-secondary);
        border-radius: 8px;
        padding: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        color: white;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent-blue);
        border-radius: 3px;
    }
    
    /* Alpha signal colors */
    .alpha-strong-buy { color: #00ff88; font-weight: 600; }
    .alpha-buy { color: #66ffaa; }
    .alpha-neutral { color: #a0a0a0; }
    .alpha-sell { color: #ff9999; }
    .alpha-strong-sell { color: #ff3366; font-weight: 600; }
    
    /* Risk indicator */
    .risk-low { color: var(--accent-green); }
    .risk-medium { color: var(--accent-yellow); }
    .risk-high { color: var(--accent-red); }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA FUNCTIONS
# ============================================================================

DATA_CACHE = Path(__file__).parent.parent / "data_cache"

@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch stock data with caching."""
    try:
        # Try cache first
        cache_path = DATA_CACHE / f"{ticker.replace('^', 'IDX_').replace('=', '_')}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            return df
        
        # Fallback to yfinance
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
    """Fetch real-time quote."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'price': info.get('regularMarketPrice', info.get('currentPrice', 0)),
            'change': info.get('regularMarketChange', 0),
            'change_pct': info.get('regularMarketChangePercent', 0),
            'volume': info.get('regularMarketVolume', 0),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'name': info.get('shortName', ticker),
            'sector': info.get('sector', 'N/A'),
            'high': info.get('dayHigh', 0),
            'low': info.get('dayLow', 0),
            'open': info.get('open', 0),
            'prev_close': info.get('previousClose', 0),
        }
    except:
        return {'price': 0, 'change': 0, 'change_pct': 0, 'volume': 0, 'market_cap': 0, 
                'pe_ratio': 0, 'name': ticker, 'sector': 'N/A', 'high': 0, 'low': 0, 
                'open': 0, 'prev_close': 0}

def load_alpha_signals() -> pd.DataFrame:
    """Load pre-computed alpha signals."""
    path = DATA_CACHE / "alpha_signals.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()

def load_fundamentals() -> pd.DataFrame:
    """Load fundamentals data."""
    path = DATA_CACHE / "fundamentals.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
    if df.empty or len(df) < 20:
        return df
    
    df = df.copy()
    close = df['close']
    
    # Moving Averages
    df['sma_20'] = close.rolling(20).mean()
    df['sma_50'] = close.rolling(50).mean()
    df['ema_12'] = close.ewm(span=12).mean()
    df['ema_26'] = close.ewm(span=26).mean()
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(close)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], close, window=14).average_true_range()
    
    return df

def format_number(num: float, prefix: str = "$") -> str:
    """Format large numbers."""
    if num >= 1e12:
        return f"{prefix}{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"{prefix}{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{prefix}{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{prefix}{num/1e3:.2f}K"
    else:
        return f"{prefix}{num:.2f}"

def get_signal_class(value: float) -> str:
    """Get CSS class for alpha signal."""
    if value > 0.5:
        return "alpha-strong-buy"
    elif value > 0.2:
        return "alpha-buy"
    elif value > -0.2:
        return "alpha-neutral"
    elif value > -0.5:
        return "alpha-sell"
    else:
        return "alpha-strong-sell"

# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def create_advanced_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create professional candlestick chart."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(None, None, None, None)
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff3366',
        increasing_fillcolor='#00ff88',
        decreasing_fillcolor='#ff3366'
    ), row=1, col=1)
    
    # Moving Averages
    if 'sma_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', 
                                 line=dict(color='#ffcc00', width=1)), row=1, col=1)
    if 'sma_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50', 
                                 line=dict(color='#00d4ff', width=1)), row=1, col=1)
    
    # Bollinger Bands
    if 'bb_upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper',
                                 line=dict(color='rgba(168,85,247,0.3)', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower',
                                 line=dict(color='rgba(168,85,247,0.3)', dash='dash'),
                                 fill='tonexty', fillcolor='rgba(168,85,247,0.05)'), row=1, col=1)
    
    # Volume
    colors = ['#00ff88' if c >= o else '#ff3366' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', 
                        marker_color=colors, opacity=0.7), row=2, col=1)
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI',
                                 line=dict(color='#a855f7', width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff3366", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00ff88", row=3, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.02)", row=3, col=1)
    
    # MACD
    if 'macd' in df.columns:
        macd_colors = ['#00ff88' if v >= 0 else '#ff3366' for v in df['macd_hist'].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='MACD Hist',
                            marker_color=macd_colors), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD',
                                 line=dict(color='#00d4ff', width=1)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='Signal',
                                 line=dict(color='#ffcc00', width=1)), row=4, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,10,15,0.8)',
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=10)
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=30, b=30),
        font=dict(family="JetBrains Mono, monospace", color="#a0a0a0")
    )
    
    # Update axes
    for i in range(1, 5):
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)', row=i, col=1)
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', row=i, col=1)
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    
    return fig

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700;
                    background: linear-gradient(90deg, #00d4ff, #a855f7);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            ⚡ QUANTUM
        </div>
        <div style="font-size: 0.75rem; color: #666; margin-top: 0.25rem;">
            TRADING TERMINAL v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "📈 Analysis", "🎯 Alpha Signals", "💼 Portfolio", "🔬 Research", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick quote
    st.markdown("### Quick Quote")
    ticker_input = st.text_input("Symbol", value="AAPL", key="sidebar_ticker").upper()
    
    if ticker_input:
        quote = fetch_realtime_quote(ticker_input)
        if quote['price'] > 0:
            change_class = "positive" if quote['change'] >= 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{quote['name']}</div>
                <div class="metric-value">${quote['price']:.2f}</div>
                <div class="metric-change {change_class}">
                    {'+' if quote['change'] >= 0 else ''}{quote['change']:.2f} 
                    ({'+' if quote['change_pct'] >= 0 else ''}{quote['change_pct']:.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Market status
    now = datetime.now()
    market_open = now.hour >= 9 and now.hour < 16 and now.weekday() < 5
    status = "MARKET OPEN" if market_open else "MARKET CLOSED"
    status_color = "#00ff88" if market_open else "#ff3366"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 0.5rem; background: rgba(255,255,255,0.03); border-radius: 8px;">
        <div style="font-size: 0.7rem; color: #666; text-transform: uppercase;">Status</div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: {status_color};">
            ● {status}
        </div>
        <div style="font-size: 0.75rem; color: #666; margin-top: 0.25rem;">
            {now.strftime("%H:%M:%S EST")}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

if page == "📊 Dashboard":
    # Header
    st.markdown("""
    <div class="terminal-header">
        <div class="terminal-logo">QUANTUM TERMINAL</div>
        <div class="terminal-status">
            <div class="status-item">
                <div class="status-dot"></div>
                <span>LIVE</span>
            </div>
            <div class="status-item">
                <span>DATA: REALTIME</span>
            </div>
            <div class="status-item">
                <span>LATENCY: 12ms</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Market Overview
    indices = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "DOW": "^DJI",
        "RUSSELL": "^RUT",
        "VIX": "^VIX",
        "10Y": "^TNX"
    }
    
    cols = st.columns(6)
    for i, (name, ticker) in enumerate(indices.items()):
        with cols[i]:
            quote = fetch_realtime_quote(ticker)
            change_class = "positive" if quote['change'] >= 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{name}</div>
                <div class="metric-value">{quote['price']:,.2f}</div>
                <div class="metric-change {change_class}">
                    {'+' if quote['change_pct'] >= 0 else ''}{quote['change_pct']:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main content grid
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">📈 S&P 500 INTRADAY</div>
                <div class="panel-badge">LIVE</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        sp500_data = fetch_stock_data("^GSPC", "6mo")
        if not sp500_data.empty:
            fig = go.Figure()
            
            # Calculate color based on performance
            pct_change = (sp500_data['close'].iloc[-1] / sp500_data['close'].iloc[0] - 1) * 100
            color = '#00ff88' if pct_change >= 0 else '#ff3366'
            
            fig.add_trace(go.Scatter(
                x=sp500_data.index,
                y=sp500_data['close'],
                fill='tozeroy',
                fillcolor=f'rgba({0 if pct_change >= 0 else 255},{255 if pct_change >= 0 else 51},{136 if pct_change >= 0 else 102},0.1)',
                line=dict(color=color, width=2),
                name='S&P 500'
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">🎯 TOP ALPHA SIGNALS</div>
                <div class="panel-badge">AI</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        alpha_df = load_alpha_signals()
        if not alpha_df.empty:
            top_alpha = alpha_df.nlargest(8, 'composite_alpha')
            for _, row in top_alpha.iterrows():
                signal_class = get_signal_class(row['composite_alpha'])
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; 
                            border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <span style="font-family: 'JetBrains Mono', monospace; font-weight: 600;">
                        {row['symbol']}
                    </span>
                    <span class="{signal_class}" style="font-family: 'JetBrains Mono', monospace;">
                        {row['composite_alpha']:.3f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Run download_data.py to generate alpha signals")
    
    # Bottom section
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">🔥 SECTOR PERFORMANCE</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        sectors = {
            'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare',
            'XLE': 'Energy', 'XLI': 'Industrials', 'XLP': 'Staples',
            'XLY': 'Discretionary', 'XLU': 'Utilities'
        }
        
        sector_data = []
        for ticker, name in sectors.items():
            quote = fetch_realtime_quote(ticker)
            if quote['price'] > 0:
                sector_data.append({
                    'Sector': name,
                    'Change': quote['change_pct']
                })
        
        if sector_data:
            sector_df = pd.DataFrame(sector_data).sort_values('Change', ascending=True)
            
            colors = ['#00ff88' if x >= 0 else '#ff3366' for x in sector_df['Change']]
            
            fig = go.Figure(go.Bar(
                x=sector_df['Change'],
                y=sector_df['Sector'],
                orientation='h',
                marker_color=colors,
                text=[f"{x:+.2f}%" for x in sector_df['Change']],
                textposition='outside',
                textfont=dict(family="JetBrains Mono", size=11)
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                margin=dict(l=0, r=50, t=0, b=0),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=True,
                          zerolinecolor='rgba(255,255,255,0.2)'),
                yaxis=dict(showgrid=False),
                font=dict(family="JetBrains Mono", color="#a0a0a0")
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">📊 MARKET BREADTH</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulated market breadth
        breadth_data = {
            'Metric': ['Advancing', 'Declining', 'Unchanged', 'New Highs', 'New Lows'],
            'NYSE': [1842, 1156, 89, 124, 45],
            'NASDAQ': [2156, 1678, 134, 89, 67]
        }
        
        st.dataframe(
            pd.DataFrame(breadth_data),
            use_container_width=True,
            hide_index=True
        )

elif page == "📈 Analysis":
    st.markdown("""
    <div class="terminal-header">
        <div class="terminal-logo">TECHNICAL ANALYSIS</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.text_input("Symbol", value="AAPL", key="analysis_ticker").upper()
    with col2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    with col3:
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    
    if ticker:
        df = fetch_stock_data(ticker, period)
        
        if not df.empty:
            df = calculate_indicators(df)
            quote = fetch_realtime_quote(ticker)
            
            # Quote metrics
            cols = st.columns(6)
            metrics = [
                ("Price", f"${quote['price']:.2f}", f"{quote['change_pct']:+.2f}%"),
                ("Open", f"${quote['open']:.2f}", None),
                ("High", f"${quote['high']:.2f}", None),
                ("Low", f"${quote['low']:.2f}", None),
                ("Volume", format_number(quote['volume'], ""), None),
                ("Mkt Cap", format_number(quote['market_cap']), None),
            ]
            
            for col, (label, value, delta) in zip(cols, metrics):
                with col:
                    if delta:
                        st.metric(label, value, delta)
                    else:
                        st.metric(label, value)
            
            # Chart
            st.plotly_chart(create_advanced_chart(df, ticker), use_container_width=True)
            
            # Technical Summary
            st.markdown("### 📊 Technical Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current = df['close'].iloc[-1]
                sma20 = df['sma_20'].iloc[-1] if 'sma_20' in df.columns else 0
                sma50 = df['sma_50'].iloc[-1] if 'sma_50' in df.columns else 0
                
                st.markdown("**Moving Averages**")
                if sma20 > 0:
                    signal = "🟢 ABOVE" if current > sma20 else "🔴 BELOW"
                    st.markdown(f"SMA 20: {signal}")
                if sma50 > 0:
                    signal = "🟢 ABOVE" if current > sma50 else "🔴 BELOW"
                    st.markdown(f"SMA 50: {signal}")
            
            with col2:
                if 'rsi' in df.columns:
                    rsi = df['rsi'].iloc[-1]
                    st.markdown("**RSI (14)**")
                    if rsi > 70:
                        st.markdown(f"🔴 OVERBOUGHT ({rsi:.1f})")
                    elif rsi < 30:
                        st.markdown(f"🟢 OVERSOLD ({rsi:.1f})")
                    else:
                        st.markdown(f"⚪ NEUTRAL ({rsi:.1f})")
            
            with col3:
                if 'macd' in df.columns:
                    macd = df['macd'].iloc[-1]
                    signal = df['macd_signal'].iloc[-1]
                    st.markdown("**MACD**")
                    if macd > signal:
                        st.markdown("🟢 BULLISH")
                    else:
                        st.markdown("🔴 BEARISH")
            
            with col4:
                if 'bb_upper' in df.columns:
                    bb_pos = (current - df['bb_lower'].iloc[-1]) / (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) * 100
                    st.markdown("**Bollinger %B**")
                    st.markdown(f"{bb_pos:.1f}%")

elif page == "🎯 Alpha Signals":
    st.markdown("""
    <div class="terminal-header">
        <div class="terminal-logo">ALPHA SIGNAL MATRIX</div>
        <div class="terminal-status">
            <div class="status-item">
                <div class="status-dot"></div>
                <span>AI-POWERED</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    alpha_df = load_alpha_signals()
    
    if not alpha_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Signals", len(alpha_df))
        with col2:
            strong_buy = len(alpha_df[alpha_df['composite_alpha'] > 0.5])
            st.metric("Strong Buy", strong_buy)
        with col3:
            strong_sell = len(alpha_df[alpha_df['composite_alpha'] < -0.5])
            st.metric("Strong Sell", strong_sell)
        with col4:
            avg_conf = alpha_df['confidence'].mean() * 100
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        
        st.markdown("---")
        
        # Signal table
        tab1, tab2, tab3 = st.tabs(["📈 Long Ideas", "📉 Short Ideas", "📊 Full Matrix"])
        
        with tab1:
            longs = alpha_df.nlargest(15, 'composite_alpha')
            st.dataframe(
                longs[['symbol', 'composite_alpha', 'confidence', 'alpha_rank']].rename(columns={
                    'symbol': 'Symbol',
                    'composite_alpha': 'Alpha',
                    'confidence': 'Confidence',
                    'alpha_rank': 'Rank'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        with tab2:
            shorts = alpha_df.nsmallest(15, 'composite_alpha')
            st.dataframe(
                shorts[['symbol', 'composite_alpha', 'confidence', 'alpha_rank']].rename(columns={
                    'symbol': 'Symbol',
                    'composite_alpha': 'Alpha',
                    'confidence': 'Confidence',
                    'alpha_rank': 'Rank'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        with tab3:
            st.dataframe(
                alpha_df.sort_values('alpha_rank'),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.warning("No alpha signals found. Run `python download_data.py` to generate signals.")

elif page == "💼 Portfolio":
    st.markdown("""
    <div class="terminal-header">
        <div class="terminal-logo">PORTFOLIO MANAGEMENT</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize portfolio
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {
            'AAPL': {'shares': 100, 'cost': 175.00},
            'MSFT': {'shares': 50, 'cost': 380.00},
            'NVDA': {'shares': 30, 'cost': 450.00},
            'GOOGL': {'shares': 25, 'cost': 140.00}
        }
    
    # Portfolio metrics
    total_value = 0
    total_cost = 0
    holdings = []
    
    for symbol, data in st.session_state.portfolio.items():
        quote = fetch_realtime_quote(symbol)
        price = quote['price'] if quote['price'] > 0 else data['cost']
        value = price * data['shares']
        cost = data['cost'] * data['shares']
        pnl = value - cost
        pnl_pct = (pnl / cost) * 100
        
        total_value += value
        total_cost += cost
        
        holdings.append({
            'Symbol': symbol,
            'Shares': data['shares'],
            'Cost': f"${data['cost']:.2f}",
            'Price': f"${price:.2f}",
            'Value': f"${value:,.2f}",
            'P&L': f"${pnl:,.2f}",
            'P&L %': f"{pnl_pct:+.2f}%"
        })
    
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) * 100
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Value", f"${total_value:,.2f}")
    with col2:
        st.metric("Total Cost", f"${total_cost:,.2f}")
    with col3:
        st.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl_pct:+.2f}%")
    with col4:
        st.metric("Positions", len(st.session_state.portfolio))
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Holdings")
        st.dataframe(pd.DataFrame(holdings), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### Allocation")
        
        alloc_data = []
        for symbol, data in st.session_state.portfolio.items():
            quote = fetch_realtime_quote(symbol)
            price = quote['price'] if quote['price'] > 0 else data['cost']
            alloc_data.append({'Symbol': symbol, 'Value': price * data['shares']})
        
        alloc_df = pd.DataFrame(alloc_data)
        
        fig = go.Figure(data=[go.Pie(
            labels=alloc_df['Symbol'],
            values=alloc_df['Value'],
            hole=0.5,
            marker=dict(colors=['#00d4ff', '#a855f7', '#00ff88', '#ffcc00', '#ff3366'])
        )])
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True,
            legend=dict(font=dict(size=10))
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔬 Research":
    st.markdown("""
    <div class="terminal-header">
        <div class="terminal-logo">RESEARCH LAB</div>
    </div>
    """, unsafe_allow_html=True)
    
    fundamentals = load_fundamentals()
    
    if not fundamentals.empty:
        st.markdown("### 📊 Fundamental Screener")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sector_filter = st.selectbox("Sector", ["All"] + fundamentals['sector'].dropna().unique().tolist())
        with col2:
            pe_max = st.slider("Max P/E", 0, 100, 50)
        with col3:
            sort_by = st.selectbox("Sort By", ["market_cap", "pe_ratio", "profit_margin", "roe"])
        
        # Filter
        filtered = fundamentals.copy()
        if sector_filter != "All":
            filtered = filtered[filtered['sector'] == sector_filter]
        filtered = filtered[filtered['pe_ratio'] <= pe_max]
        filtered = filtered.sort_values(sort_by, ascending=False)
        
        st.dataframe(
            filtered[['symbol', 'name', 'sector', 'market_cap', 'pe_ratio', 'profit_margin', 'roe']].head(20),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Run `python download_data.py` to download fundamental data")

elif page == "⚙️ Settings":
    st.markdown("""
    <div class="terminal-header">
        <div class="terminal-logo">SETTINGS</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Data Management")
    
    if st.button("🔄 Refresh Market Data"):
        st.cache_data.clear()
        st.success("Cache cleared! Data will be refreshed on next load.")
    
    st.markdown("### System Info")
    
    data_files = list(DATA_CACHE.glob("*.parquet")) if DATA_CACHE.exists() else []
    
    st.markdown(f"""
    - **Data Cache Location**: `{DATA_CACHE}`
    - **Cached Files**: {len(data_files)}
    - **Python Version**: 3.10+
    - **Framework**: Streamlit
    """)
