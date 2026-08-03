"""
Obsidian Terminal — Flask API server.

Serves JSON endpoints for the React frontend. Wraps the same data layer
used by the legacy Streamlit dashboard.

Run:  python api/server.py
"""
import sys
import os
import json
import re
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import yfinance as yf
import ta
from flask import Flask, jsonify, request
from flask_cors import CORS

from agents.agent_registry import create_default_registry

DATA_CACHE = ROOT / "data_cache"
DATA_CACHE.mkdir(exist_ok=True)
WATCHLIST_FILE = DATA_CACHE / "watchlist.json"

_cache: dict = {}


def _cached(key: str, ttl: int, fn):
    now = datetime.now().timestamp()
    if key in _cache and now - _cache[key][0] < ttl:
        return _cache[key][1]
    value = fn()
    _cache[key] = (now, value)
    return value


# ============================================================================
# NEWS SENTIMENT — lightweight finance lexicon (no external deps)
# ============================================================================

_POS_WORDS = {
    "surge", "surges", "surged", "rally", "rallies", "rallied", "beat", "beats",
    "record", "records", "jump", "jumps", "jumped", "gain", "gains", "gained",
    "rise", "rises", "rose", "growth", "growing", "soar", "soars", "soared",
    "upgrade", "upgrades", "upgraded", "outperform", "outperformed", "profit",
    "profits", "profitable", "strong", "stronger", "bullish", "buy", "boost",
    "boosted", "boost", "positive", "higher", "high", "growth", "milestone",
    "win", "wins", "winning", "success", "successful", "expanding", "expansion",
    "launch", "launches", "launched", "partnership", "deal", "deals", "acquire",
    "acquisition", "innovation", "breakthrough", "dividend", "buyback",
    "revenue", "exceed", "exceeds", "exceeded", "top", "tops", "tops", "climb",
    "climbs", "climbed", "advance", "advances", "advanced", "rebound",
}

_NEG_WORDS = {
    "plunge", "plunges", "plunged", "crash", "crashes", "crashed", "miss",
    "misses", "missed", "downgrade", "downgrades", "downgraded", "underperform",
    "underperformed", "loss", "losses", "losing", "decline", "declines",
    "declined", "drop", "drops", "dropped", "fall", "falls", "fell", "slump",
    "slumps", "slumped", "bearish", "sell", "selloff", "negative", "lower",
    "weak", "weaker", "weakness", "lawsuit", "sued", "sues", "fine", "fines",
    "fined", "investigation", "probe", "scrutiny", "warning", "warns",
    "warned", "risk", "risks", "uncertainty", "layoff", "layoffs", "cut",
    "cuts", "cutting", "bankruptcy", "bankrupt", "fraud", "scandal", "recall",
    "recalls", "recalled", "short", "shortfall", "resign", "resigns",
    "resigned", "exit", "exits", "delay", "delays", "delayed", "disappoint",
    "disappoints", "disappointed", "sink", "sinks", "sank", "tumble", "tumbles",
    "tumbled", "banned", "ban", "penalty", "penalties", "defective",
}

_NEG_PREFIXES = ("not ", "no ", "misses on ", "fails to ", "below ", "slower than ")


def _sentiment_score(text: str) -> float:
    """Score text in [-1, 1] using a finance word lexicon."""
    if not text:
        return 0.0
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0
    pos = neg = 0
    for i, w in enumerate(words):
        prev = words[i - 1] + " " if i > 0 else ""
        negated = any(prev.startswith(p) or prev == p.strip() for p in _NEG_PREFIXES)
        if w in _POS_WORDS and not negated:
            pos += 1
        elif w in _NEG_WORDS:
            neg += 1
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def _sentiment_label(score: float) -> str:
    if score > 0.25:
        return "bullish"
    if score < -0.25:
        return "bearish"
    return "neutral"


def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    def _load():
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
    try:
        return _cached(f"data:{ticker}:{period}", 300, _load)
    except Exception:
        return pd.DataFrame()


def fetch_realtime_quote(ticker: str) -> dict:
    def _load():
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
    try:
        return _cached(f"quote:{ticker}", 60, _load)
    except Exception:
        return {"price": 0, "change": 0, "change_pct": 0, "volume": 0,
                "market_cap": 0, "pe_ratio": 0, "name": ticker, "sector": "N/A",
                "high": 0, "low": 0, "open": 0, "prev_close": 0}


def batch_quotes(symbols: list, period: str = "5d") -> dict:
    """Fetch light-weight quotes for many symbols in a single HTTP request."""
    def _load():
        result = {}
        df = yf.download(symbols, period=period, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return result
        try:
            close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
            close.columns = [str(c) if not isinstance(c, tuple) else str(c[1]) for c in close.columns]
        except Exception:
            return result
        try:
            open_ = df["Open"] if isinstance(df.columns, pd.MultiIndex) else df[["Open"]]
            high = df["High"] if isinstance(df.columns, pd.MultiIndex) else df[["High"]]
            low = df["Low"] if isinstance(df.columns, pd.MultiIndex) else df[["Low"]]
            volume = df["Volume"] if isinstance(df.columns, pd.MultiIndex) else df[["Volume"]]
        except Exception:
            open_ = high = low = volume = None
        for sym in symbols:
            try:
                if sym not in close.columns:
                    continue
                series = close[sym].dropna()
                if series.empty:
                    continue
                price = float(series.iloc[-1])
                prev = float(series.iloc[-2]) if len(series) > 1 else price
                result[sym] = {
                    "price": price,
                    "change": round(price - prev, 4),
                    "change_pct": round((price - prev) / prev * 100, 4) if prev else 0,
                    "volume": float(volume[sym].dropna().iloc[-1]) if volume is not None and sym in volume.columns and not volume[sym].dropna().empty else 0,
                    "high": float(high[sym].dropna().iloc[-1]) if high is not None and sym in high.columns and not high[sym].dropna().empty else price,
                    "low": float(low[sym].dropna().iloc[-1]) if low is not None and sym in low.columns and not low[sym].dropna().empty else price,
                    "open": float(open_[sym].dropna().iloc[-1]) if open_ is not None and sym in open_.columns and not open_[sym].dropna().empty else price,
                    "prev_close": prev,
                }
            except Exception:
                continue
        return result
    try:
        return _cached(f"batch:{','.join(sorted(symbols))}", 120, _load)
    except Exception:
        return {}


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


def df_to_records(df: pd.DataFrame, limit: int = 1000) -> list:
    df = df.dropna().tail(limit)
    out = []
    for idx, row in df.iterrows():
        rec = {"date": str(idx.date() if hasattr(idx, "date") else idx)}
        for col in df.columns:
            v = row[col]
            if isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, (np.integer,)):
                v = int(v)
            if v is None or (isinstance(v, float) and (v != v)):  # NaN
                continue
            rec[str(col)] = v
        out.append(rec)
    return out


app = Flask(__name__)
CORS(app)

INDICES = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW 30": "^DJI",
    "Russell 2K": "^RUT",
    "VIX": "^VIX",
    "10Y Yield": "^TNX",
}

INDIA_INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTY Bank": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
}

INDIA_UNIVERSE = {
    "IT & Software": [
        ("TCS.NS", "Tata Consultancy"), ("INFY.NS", "Infosys"),
        ("WIPRO.NS", "Wipro"), ("HCLTECH.NS", "HCL Technologies"),
        ("TECHM.NS", "Tech Mahindra"), ("LTIM.NS", "LTIMindtree"),
        ("PERSISTENT.NS", "Persistent Systems"), ("COFORGE.NS", "Coforge"),
        ("MPHASIS.NS", "Mphasis"), ("OFSS.NS", "Oracle Financial"),
    ],
    "Banking & Finance": [
        ("HDFCBANK.NS", "HDFC Bank"), ("ICICIBANK.NS", "ICICI Bank"),
        ("SBIN.NS", "State Bank of India"), ("KOTAKBANK.NS", "Kotak Mahindra"),
        ("AXISBANK.NS", "Axis Bank"), ("INDUSINDBK.NS", "IndusInd Bank"),
        ("BAJFINANCE.NS", "Bajaj Finance"), ("BAJAJFINSV.NS", "Bajaj Finserv"),
        ("HDFCLIFE.NS", "HDFC Life"), ("SBILIFE.NS", "SBI Life"),
    ],
    "Energy & Oil": [
        ("RELIANCE.NS", "Reliance Industries"), ("ONGC.NS", "ONGC"),
        ("NTPC.NS", "NTPC"), ("POWERGRID.NS", "Power Grid"),
        ("TATAPOWER.NS", "Tata Power"), ("ADANIGREEN.NS", "Adani Green"),
        ("IOC.NS", "Indian Oil"), ("BPCL.NS", "BPCL"),
        ("GAIL.NS", "GAIL India"), ("OIL.NS", "Oil India"),
    ],
    "Auto & Auto Parts": [
        ("TATAMOTORS.NS", "Tata Motors"), ("M&M.NS", "Mahindra & Mahindra"),
        ("MARUTI.NS", "Maruti Suzuki"), ("HEROMOTOCO.NS", "Hero MotoCorp"),
        ("BAJAJ-AUTO.NS", "Bajaj Auto"), ("EICHERMOT.NS", "Eicher Motors"),
        ("ASHOKLEY.NS", "Ashok Leyland"), ("TVSMOTOR.NS", "TVS Motor"),
        ("BOSCHLTD.NS", "Bosch"), ("MRF.NS", "MRF"),
    ],
    "Pharma & Healthcare": [
        ("SUNPHARMA.NS", "Sun Pharma"), ("DRREDDY.NS", "Dr. Reddy's"),
        ("CIPLA.NS", "Cipla"), ("DIVISLAB.NS", "Divi's Labs"),
        ("APOLLOHOSP.NS", "Apollo Hospitals"), ("AUROPHARMA.NS", "Aurobindo Pharma"),
        ("LUPIN.NS", "Lupin"), ("BIOCON.NS", "Biocon"),
        ("GLENMARK.NS", "Glenmark"), ("ALKYLAMINE.NS", "Alkyl Amines"),
    ],
    "FMCG & Consumer": [
        ("ITC.NS", "ITC Limited"), ("HINDUNILVR.NS", "Hindustan Unilever"),
        ("NESTLEIND.NS", "Nestlé India"), ("BRITANNIA.NS", "Britannia"),
        ("DABUR.NS", "Dabur"), ("MARICO.NS", "Marico"),
        ("TATACONSUM.NS", "Tata Consumer"), ("GODREJCP.NS", "Godrej Consumer"),
        ("COLPAL.NS", "Colgate-Palmolive"), ("PGHH.NS", "P&G Hygiene"),
    ],
    "Metals & Mining": [
        ("TATASTEEL.NS", "Tata Steel"), ("JSWSTEEL.NS", "JSW Steel"),
        ("HINDALCO.NS", "Hindalco"), ("VEDL.NS", "Vedanta"),
        ("COALINDIA.NS", "Coal India"), ("SAIL.NS", "SAIL"),
        ("JINDALSTEL.NS", "Jindal Steel"), ("NMDC.NS", "NMDC"),
        ("NATIONALUM.NS", "National Aluminium"), ("HINDZINC.NS", "Hindustan Zinc"),
    ],
    "Infra & Cement": [
        ("LT.NS", "Larsen & Toubro"), ("ULTRACEMCO.NS", "UltraTech Cement"),
        ("ADANIENT.NS", "Adani Enterprises"), ("ADANIPORTS.NS", "Adani Ports"),
        ("AMBUJACEM.NS", "Ambuja Cement"), ("ACC.NS", "ACC"),
        ("DLF.NS", "DLF"), ("GODREJPROP.NS", "Godrej Properties"),
        ("OBEROIRLTY.NS", "Oberoi Realty"), ("IRCTC.NS", "IRCTC"),
    ],
    "Telecom & Media": [
        ("BHARTIARTL.NS", "Bharti Airtel"), ("JIOFIN.NS", "Jio Financial"),
        ("IDEA.NS", "Vodafone Idea"), ("TATACOMM.NS", "Tata Communications"),
        ("ZOMATO.NS", "Zomato"), ("NAUKRI.NS", "Info Edge"),
        ("INDIGO.NS", "InterGlobe Aviation"), ("PAYTM.NS", "One97 Communications"),
        ("DMART.NS", "Avenue Supermarts"), ("TITAN.NS", "Titan Company"),
    ],
}

INDIA_POPULAR = {sym: name for sector in INDIA_UNIVERSE.values() for sym, name in sector}

US_UNIVERSE = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "ORCL",
        "CRM", "ADBE", "AMD", "INTC", "QCOM", "TXN", "CSCO", "IBM",
        "NOW", "INTU", "UBER", "SHOP",
    ],
    "Consumer Discretionary": [
        "TSLA", "NFLX", "HD", "MCD", "BKNG", "SBUX", "NKE", "LULU",
        "CMG", "ABNB", "TGT", "COST",
    ],
    "Financials": [
        "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "BLK",
        "SCHW", "C", "PYPL", "COIN",
    ],
    "Healthcare": [
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR",
        "BMY", "AMGN", "GILD", "ISRG", "VRTX",
    ],
    "Industrials": [
        "CAT", "GE", "BA", "HON", "UNP", "UPS", "DE", "RTX", "LMT", "ETN",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "VLO", "KMI",
    ],
    "Communication Services": [
        "T", "VZ", "CMCSA", "DIS", "TMUS", "SPOT", "EA",
    ],
    "Consumer Staples": [
        "WMT", "PG", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "KMB",
    ],
    "Real Estate & Utilities": [
        "PLD", "AMT", "EQIX", "NEE", "SO", "DUK", "AEP",
    ],
}
US_POPULAR = {sym: sym for sector, syms in US_UNIVERSE.items() for sym in syms}

FOCUS_FILE = DATA_CACHE / "focus.json"
PORTFOLIO_FILE = DATA_CACHE / "portfolio.json"


@app.route("/api/health")
def health():
    now = datetime.now()
    mkt_open = 9 <= now.hour < 16 and now.weekday() < 5
    return jsonify({
        "status": "ok",
        "engine": "Obsidian Core",
        "backend": "flask",
        "time": now.strftime("%H:%M:%S"),
        "market_open": mkt_open,
        "cached_entries": len(_cache),
    })


@app.route("/api/indices")
def indices():
    symbols = request.args.get("market", "us")
    source = INDIA_INDICES if symbols == "india" else INDICES
    result = {}
    for name, sym in source.items():
        q = fetch_realtime_quote(sym)
        result[name] = {
            "symbol": sym,
            "price": q.get("price", 0),
            "change": q.get("change", 0),
            "change_pct": q.get("change_pct", 0),
            "name": q.get("name", name),
        }
    return jsonify(result)


@app.route("/api/chart")
def chart():
    ticker = request.args.get("ticker", "AAPL")
    period = request.args.get("period", "6mo")
    df = fetch_stock_data(ticker, period)
    if df.empty:
        return jsonify({"ticker": ticker, "points": []})
    df = calculate_indicators(df)
    return jsonify({"ticker": ticker, "points": df_to_records(df, 1200)})


@app.route("/api/quote")
def quote():
    ticker = request.args.get("ticker", "AAPL")
    return jsonify(fetch_realtime_quote(ticker))


@app.route("/api/india")
def india():
    ticker = request.args.get("ticker", "RELIANCE.NS")
    period = request.args.get("period", "1y")
    df = fetch_stock_data(ticker, period)
    q = fetch_realtime_quote(ticker)
    quotes = batch_quotes(list(INDIA_POPULAR.keys()), period="5d")
    popular = []
    for sym, name in INDIA_POPULAR.items():
        qq = quotes.get(sym, {})
        popular.append({
            "symbol": sym,
            "name": name,
            "price": qq.get("price", 0),
            "change_pct": qq.get("change_pct", 0),
        })
    sectors = [
        {"name": sector, "stocks": [{"symbol": s, "name": n} for s, n in stocks]}
        for sector, stocks in INDIA_UNIVERSE.items()
    ]
    return jsonify({
        "ticker": ticker,
        "quote": q,
        "points": df_to_records(calculate_indicators(df), 1200) if not df.empty else [],
        "popular": popular,
        "sectors": sectors,
    })


# ============================================================================
# FOCUS TICKER — the stock featured on the main dashboard (persisted)
# ============================================================================

def _load_focus() -> str:
    if FOCUS_FILE.exists():
        try:
            return json.loads(FOCUS_FILE.read_text()).get("ticker", "AAPL")
        except Exception:
            return "AAPL"
    return "AAPL"


def _save_focus(ticker: str):
    FOCUS_FILE.write_text(json.dumps({"ticker": ticker.upper()}, indent=2))


@app.route("/api/focus", methods=["GET", "POST"])
def focus():
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        ticker = str(body.get("ticker", "AAPL")).upper()
        _save_focus(ticker)
        return jsonify({"ticker": ticker})
    ticker = _load_focus()
    q = fetch_realtime_quote(ticker)
    return jsonify({"ticker": ticker, "quote": q})


# ============================================================================
# WORLD MARKETS — WEI-style global indices, FX & commodities (one batch call)
# ============================================================================

WORLD_MARKETS = {
    "United States": {
        "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DOW 30": "^DJI",
        "Russell 2K": "^RUT", "VIX": "^VIX", "10Y Yield": "^TNX",
    },
    "Asia Pacific": {
        "Nikkei 225": "^N225", "Hang Seng": "^HSI", "Shanghai": "000001.SS",
        "S&P ASX 200": "^AXJO", "KOSPI": "^KS11", "NIFTY 50": "^NSEI",
        "Sensex": "^BSESN",
    },
    "Europe": {
        "FTSE 100": "^FTSE", "DAX": "^GDAXI", "CAC 40": "^FCHI",
        "Euro Stoxx 50": "^STOXX50E", "IBEX 35": "^IBEX",
    },
    "Commodities & FX": {
        "Gold": "GC=F", "Silver": "SI=F", "Crude WTI": "CL=F",
        "Brent": "BZ=F", "Natural Gas": "NG=F", "Copper": "HG=F",
        "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X", "USD/INR": "INR=X",
        "USD/JPY": "JPY=X",
    },
}


@app.route("/api/markets")
def markets():
    flat = {name: sym for region in WORLD_MARKETS.values() for name, sym in region.items()}
    quotes = batch_quotes(list(flat.values()), period="5d")
    out = {}
    for name, sym in flat.items():
        q = quotes.get(sym, {})
        out[name] = {
            "symbol": sym,
            "price": q.get("price", 0),
            "change_pct": q.get("change_pct", 0),
            "name": name,
        }
    return jsonify({
        "groups": {region: list(names.keys()) for region, names in WORLD_MARKETS.items()},
        "quotes": out,
    })


# ============================================================================
# MOVERS — MOV/MOST-style top gainers, losers & most active
# ============================================================================

@app.route("/api/movers")
def movers():
    market = request.args.get("market", "us")
    source = INDIA_POPULAR if market == "india" else US_POPULAR
    quotes = batch_quotes(list(source.keys()), period="5d")
    rows = []
    for sym, q in quotes.items():
        rows.append({
            "symbol": sym,
            "name": source.get(sym, sym),
            "price": q.get("price", 0),
            "change_pct": q.get("change_pct", 0),
            "volume": q.get("volume", 0),
        })
    rows = [r for r in rows if r["price"] > 0]
    gainers = sorted(rows, key=lambda r: -r["change_pct"])[:10]
    losers = sorted(rows, key=lambda r: r["change_pct"])[:10]
    active = sorted(rows, key=lambda r: -r["volume"])[:10]
    return jsonify({
        "gainers": gainers,
        "losers": losers,
        "most_active": active,
    })


# ============================================================================
# SCREENER — EQS-style equity screening
# ============================================================================

import concurrent.futures as _cf

SCREENER_UNIVERSE = US_POPULAR


@app.route("/api/screener")
def screener():
    market = request.args.get("market", "us")
    sector = request.args.get("sector", "")
    min_cap = float(request.args.get("min_cap", 0))
    max_pe = float(request.args.get("max_pe", 0))
    source = INDIA_POPULAR if market == "india" else SCREENER_UNIVERSE
    sectors_map = US_UNIVERSE if market == "us" else INDIA_UNIVERSE

    quotes = batch_quotes(list(source.keys()), period="3mo")
    rows = []
    for sym, q in quotes.items():
        if sector and sector not in sectors_map:
            continue
        price = q.get("price", 0)
        change = q.get("change_pct", 0)
        if price <= 0:
            continue
        rows.append({
            "symbol": sym,
            "name": source.get(sym, sym),
            "price": price,
            "change_pct": change,
        })
    rows = sorted(rows, key=lambda r: -abs(r["change_pct"]))

    # Enrich top candidates with fundamentals (P/E, market cap, dividend yield)
    def _enrich(row):
        try:
            info = yf.Ticker(row["symbol"]).info
            row["market_cap"] = info.get("marketCap", 0)
            row["pe_ratio"] = info.get("trailingPE", 0) or 0
            row["dividend_yield"] = (info.get("dividendYield", 0) or 0) * 100
            row["sector"] = info.get("sector", "")
        except Exception:
            row["market_cap"] = 0
            row["pe_ratio"] = 0
            row["dividend_yield"] = 0
            row["sector"] = ""
        return row

    def _screener():
        selected = rows[:24]
        with _cf.ThreadPoolExecutor(max_workers=8) as ex:
            enriched = list(ex.map(_enrich, selected))
        filtered = [
            r for r in enriched
            if (min_cap <= 0 or r["market_cap"] >= min_cap)
            and (max_pe <= 0 or (0 < r["pe_ratio"] <= max_pe))
            and (not sector or r["sector"] == sector or sector == "Any")
        ]
        return {"rows": filtered, "total": len(filtered)}

    return jsonify(_cached(f"screener:{market}:{sector}:{min_cap}:{max_pe}", 300, _screener))


# ============================================================================
# FUNDAMENTALS — FA-style income statement, balance sheet, cash flow
# ============================================================================

@app.route("/api/fundamentals")
def fundamentals():
    ticker = request.args.get("ticker", "AAPL").upper()

    def _load():
        t = yf.Ticker(ticker)
        out = {"ticker": ticker, "income": [], "balance": [], "cashflow": []}

        def _rows(df, keys):
            if df is None or df.empty:
                return []
            rows = []
            for key in keys:
                if key not in df.index:
                    continue
                series = df.loc[key]
                vals = []
                for ts, v in series.items():
                    if pd.isna(v):
                        continue
                    vals.append({
                        "date": str(ts.date()) if hasattr(ts, "date") else str(ts),
                        "value": float(v),
                    })
                rows.append({"metric": key, "values": vals[:4]})
            return rows

        out["income"] = _rows(t.income_stmt, [
            "Total Revenue", "Gross Profit", "Operating Income", "Net Income",
            "Diluted EPS", "EBITDA",
        ])
        out["balance"] = _rows(t.balance_sheet, [
            "Total Assets", "Total Liabilities Net Minority Interest",
            "Total Debt", "Cash And Cash Equivalents", "Stockholders Equity",
        ])
        out["cashflow"] = _rows(t.cashflow, [
            "Operating Cash Flow", "Capital Expenditure", "Free Cash Flow",
            "Cash Dividends Paid", "Repurchase Of Capital Stock",
        ])
        info = t.info
        out["snapshot"] = {
            "beta": info.get("beta", 0),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh", 0),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow", 0),
            "shares_outstanding": info.get("sharesOutstanding", 0),
            "target_mean_price": info.get("targetMeanPrice", 0),
            "dividend_yield": (info.get("trailingAnnualDividendYield", info.get("dividendYield", 0)) or 0) * 100,
            "forward_pe": info.get("forwardPE", 0),
            "profit_margin": (info.get("profitMargins", 0) or 0) * 100,
            "return_on_equity": (info.get("returnOnEquity", 0) or 0) * 100,
        }
        return out

    return jsonify(_cached(f"fundamentals:{ticker}", 300, _load))


# ============================================================================
# PEERS — RV-style relative valuation vs sector comparables
# ============================================================================

@app.route("/api/peers")
def peers():
    ticker = request.args.get("ticker", "AAPL").upper()

    def _load():
        info = yf.Ticker(ticker).info
        sector = info.get("sector", "")
        peers = []
        for syms in US_UNIVERSE.values():
            if ticker in syms:
                peers = [s for s in syms if s != ticker]
                break
        if not peers:
            peers = US_UNIVERSE.get("Technology", [])[:8]
        quotes = batch_quotes(peers, period="3mo")
        out = []
        for sym in peers:
            q = quotes.get(sym, {})
            if not q.get("price"):
                continue
            try:
                pi = yf.Ticker(sym).info
                pe = pi.get("trailingPE", 0) or 0
                mcap = pi.get("marketCap", 0)
            except Exception:
                pe, mcap = 0, 0
            out.append({
                "symbol": sym,
                "price": q.get("price", 0),
                "change_pct": q.get("change_pct", 0),
                "pe_ratio": pe,
                "market_cap": mcap,
            })
        return {"ticker": ticker, "sector": sector, "peers": sorted(out, key=lambda r: -r["market_cap"])}

    return jsonify(_cached(f"peers:{ticker}", 300, _load))


# ============================================================================
# DIVIDENDS — DVD-style history + yield
# ============================================================================

@app.route("/api/dividends")
def dividends():
    ticker = request.args.get("ticker", "AAPL").upper()

    def _load():
        t = yf.Ticker(ticker)
        divs = t.dividends
        out = {"ticker": ticker, "history": []}
        if divs is not None and not divs.empty:
            tail = divs.tail(12)
            for ts, v in tail.items():
                out["history"].append({
                    "date": str(ts.date()),
                    "amount": float(v),
                })
        try:
            out["yield"] = (t.info.get("trailingAnnualDividendYield", t.info.get("dividendYield", 0)) or 0) * 100
            out["payout_ratio"] = (t.info.get("payoutRatio", 0) or 0) * 100
        except Exception:
            out["yield"] = 0
            out["payout_ratio"] = 0
        return out

    return jsonify(_cached(f"dividends:{ticker}", 300, _load))


# ============================================================================
# COMPARE — COMP-style normalized total-return overlay
# ============================================================================

@app.route("/api/compare")
def compare():
    tickers = [t.strip().upper() for t in request.args.get("tickers", "AAPL,MSFT").split(",") if t.strip()]
    period = request.args.get("period", "6mo")

    def _load():
        series = {}
        for tk in tickers:
            df = fetch_stock_data(tk, period)
            if df.empty or len(df) < 2:
                continue
            base = df["close"].dropna()
            if base.empty:
                continue
            base = base / base.iloc[0] * 100
            series[tk] = [{"date": str(i.date()), "value": round(float(v), 2)}
                          for i, v in base.items()]
        return {"tickers": list(series.keys()), "series": series}

    return jsonify(_cached(f"compare:{','.join(tickers)}:{period}", 300, _load))


# ============================================================================
# OPTIONS — OMON-style nearest-expiry chain snapshot
# ============================================================================

@app.route("/api/options")
def options():
    ticker = request.args.get("ticker", "AAPL").upper()

    def _load():
        t = yf.Ticker(ticker)
        out = {"ticker": ticker, "expirations": [], "chain": {}}
        try:
            exps = t.options[:4]
            out["expirations"] = list(exps)
            if exps:
                exp = exps[0]
                ch = t.option_chain(exp)
                out["expiry"] = exp
                out["underlying"] = ch.underlying.get("lastPrice", 0) if hasattr(ch, "underlying") else 0
                for kind, df in (("calls", ch.calls), ("puts", ch.puts)):
                    rows = []
                    for _, r in df.head(10).iterrows():
                        rows.append({
                            "strike": float(r.get("strike", 0)),
                            "last_price": float(r.get("lastPrice", 0)),
                            "bid": float(r.get("bid", 0)),
                            "ask": float(r.get("ask", 0)),
                            "implied_vol": float(r.get("impliedVolatility", 0) or 0),
                            "open_interest": float(r.get("openInterest", 0) or 0),
                            "volume": float(r.get("volume", 0) or 0),
                        })
                    out["chain"][kind] = rows
        except Exception:
            pass
        return out

    return jsonify(_cached(f"options:{ticker}", 300, _load))


# ============================================================================
# CALENDAR — ERN/ECO-style upcoming earnings across the watchlist
# ============================================================================

@app.route("/api/calendar")
def calendar():
    def _load():
        symbols = _load_watchlist()
        events = []
        for sym in symbols:
            try:
                ed = yf.Ticker(sym).get_earnings_dates(limit=4)
                if ed is None or ed.empty:
                    continue
                upcoming = None
                for idx, row in ed.iterrows():
                    est = row.get("EPS Estimate")
                    actual = row.get("Reported EPS")
                    if pd.isna(actual):
                        upcoming = {
                            "symbol": sym,
                            "type": "Earnings",
                            "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                            "eps_estimate": float(est) if pd.notna(est) else None,
                        }
                        break
                if upcoming:
                    events.append(upcoming)
            except Exception:
                continue
        return sorted(events, key=lambda e: e["date"])

    return jsonify({"events": _cached("calendar", 600, _load)})


# ============================================================================
# PORTFOLIO — Aladdin-style holdings, P&L, risk & stress analytics
# ============================================================================

def _load_portfolio() -> dict:
    default = {
        "cash": 125400,
        "holdings": [
            {"symbol": "AAPL", "qty": 120, "avg_cost": 178.4},
            {"symbol": "MSFT", "qty": 60, "avg_cost": 338.2},
            {"symbol": "NVDA", "qty": 95, "avg_cost": 89.4},
            {"symbol": "GOOGL", "qty": 80, "avg_cost": 142.1},
            {"symbol": "AMZN", "qty": 55, "avg_cost": 158.9},
            {"symbol": "META", "qty": 42, "avg_cost": 312.6},
            {"symbol": "TSLA", "qty": 60, "avg_cost": 244.3},
            {"symbol": "JPM", "qty": 90, "avg_cost": 158.7},
            {"symbol": "V", "qty": 70, "avg_cost": 241.5},
            {"symbol": "AMD", "qty": 110, "avg_cost": 128.9},
        ],
    }
    if PORTFOLIO_FILE.exists():
        try:
            return json.loads(PORTFOLIO_FILE.read_text())
        except Exception:
            return default
    return default


def _save_portfolio(data: dict):
    PORTFOLIO_FILE.write_text(json.dumps(data, indent=2))


def _portfolio_analytics() -> dict:
    data = _load_portfolio()
    holdings = data.get("holdings", [])
    cash = data.get("cash", 0)
    symbols = [h["symbol"] for h in holdings]
    quotes = batch_quotes(symbols, period="5d")

    enriched = []
    for h in holdings:
        q = quotes.get(h["symbol"], {})
        last = q.get("price", 0)
        enriched.append({
            "symbol": h["symbol"],
            "qty": h["qty"],
            "avg_cost": h["avg_cost"],
            "last": last,
            "change_pct": q.get("change_pct", 0),
            "value": last * h["qty"],
            "cost": h["avg_cost"] * h["qty"],
            "pnl": (last - h["avg_cost"]) * h["qty"],
            "pnl_pct": ((last - h["avg_cost"]) / h["avg_cost"] * 100) if h["avg_cost"] else 0,
        })

    invested = sum(r["cost"] for r in enriched)
    market_value = sum(r["value"] for r in enriched)
    total = market_value + cash
    day_pnl = sum(r["qty"] * r["last"] * (r["change_pct"] / 100) for r in enriched if r["change_pct"])

    # Risk: historical + parametric VaR (1-day, 95%) using 1y daily returns
    var95_h = var95_p = 0.0
    if holdings:
        hist = yf.download(symbols, period="1y", progress=False)
        if isinstance(hist.columns, pd.MultiIndex) and not hist.empty:
            closes = hist["Close"].copy()
            closes.columns = [c[1] if isinstance(c, tuple) else c for c in closes.columns]
            rets = closes.pct_change().dropna()
            if not rets.empty:
                weights = [h["qty"] for h in holdings]
                wsum = sum(weights) or 1
                weights = [w / wsum for w in weights]
                port_ret = rets.mul(weights, axis=1).sum(axis=1)
                var95_h = abs(float(np.percentile(port_ret, 5)))
                var95_p = abs(float(port_ret.mean() - 1.645 * port_ret.std()))
    var95_h = var95_h * market_value
    var95_p = var95_p * market_value

    # Stress scenarios (haircuts on equity value)
    tech = {"AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AMD", "INTC", "AVGO", "ORCL", "CRM", "ADBE", "UBER", "NOW", "INTU", "SHOP"}
    fin = {"JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "SCHW", "C", "PYPL", "COIN"}
    ene = {"XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "VLO", "KMI"}

    def _sector_value(names: set) -> float:
        return sum(r["value"] for r in enriched if r["symbol"] in names)

    tech_v, fin_v, ene_v, other_v = _sector_value(tech), _sector_value(fin), _sector_value(ene), market_value - _sector_value(tech | fin | ene)
    scenarios = [
        {"name": "Market Crash (-20%)", "impact": -0.20 * market_value, "pct": -20.0},
        {"name": "Tech Rout (-30%)", "impact": -0.30 * tech_v, "pct": (-0.30 * tech_v / market_value * 100) if market_value else 0},
        {"name": "Financial Stress (-25%)", "impact": -0.25 * fin_v, "pct": (-0.25 * fin_v / market_value * 100) if market_value else 0},
        {"name": "Oil Shock (-20%)", "impact": -0.20 * ene_v, "pct": (-0.20 * ene_v / market_value * 100) if market_value else 0},
        {"name": "Broad Recession (-35%)", "impact": -0.35 * market_value, "pct": -35.0},
    ]

    # Compliance: concentration limits
    limits = [
        {"name": "Single position > 10%", "limit": 10.0},
        {"name": "Single position > 25%", "limit": 25.0},
    ]
    violations = []
    for r in enriched:
        w = (r["value"] / total * 100) if total else 0
        for lim in limits:
            if w > lim["limit"]:
                violations.append({
                    "rule": f"{r['symbol']} at {w:.1f}%",
                    "limit": f">{lim['limit']}%",
                    "severity": "high" if lim["limit"] == 25 else "medium",
                })

    sectors = {}
    for r in enriched:
        try:
            sec = yf.Ticker(r["symbol"]).info.get("sector", "Other")
        except Exception:
            sec = "Other"
        sectors.setdefault(sec, 0)
        sectors[sec] += r["value"]

    return {
        "cash": cash,
        "holdings": enriched,
        "invested": invested,
        "market_value": market_value,
        "total": total,
        "day_pnl": day_pnl,
        "day_pnl_pct": (day_pnl / total * 100) if total else 0,
        "total_pnl": market_value - invested,
        "total_pnl_pct": ((market_value - invested) / invested * 100) if invested else 0,
        "var_95_historical": var95_h,
        "var_95_parametric": var95_p,
        "var_pct_historical": (var95_h / market_value * 100) if market_value else 0,
        "scenarios": scenarios,
        "violations": violations,
        "sector_exposure": [{"name": k, "pct": (v / total * 100) if total else 0} for k, v in sorted(sectors.items(), key=lambda x: -x[1])],
        "position_count": len(enriched),
    }


@app.route("/api/portfolio", methods=["GET"])
def portfolio_get():
    return jsonify(_cached("portfolio:analytics", 120, _portfolio_analytics))


@app.route("/api/portfolio/holdings", methods=["POST"])
def portfolio_add():
    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "")).upper()
    qty = float(body.get("qty", 0))
    avg_cost = float(body.get("avg_cost", 0))
    if not symbol or qty <= 0:
        return jsonify({"error": "symbol and qty required"}), 400
    data = _load_portfolio()
    for h in data["holdings"]:
        if h["symbol"] == symbol:
            h["qty"] += qty
            h["avg_cost"] = (h["avg_cost"] * (h["qty"] - qty) + avg_cost * qty) / h["qty"] if qty < h["qty"] else avg_cost
            _save_portfolio(data)
            return jsonify({"ok": True, "holdings": data["holdings"]})
    data["holdings"].append({"symbol": symbol, "qty": qty, "avg_cost": avg_cost})
    _save_portfolio(data)
    return jsonify({"ok": True, "holdings": data["holdings"]})


@app.route("/api/portfolio/holdings/<symbol>", methods=["DELETE"])
def portfolio_remove(symbol: str):
    data = _load_portfolio()
    data["holdings"] = [h for h in data["holdings"] if h["symbol"].upper() != symbol.upper()]
    _save_portfolio(data)
    return jsonify({"ok": True, "holdings": data["holdings"]})


@app.route("/api/portfolio/holdings/<symbol>", methods=["PUT"])
def portfolio_update(symbol: str):
    body = request.get_json(silent=True) or {}
    data = _load_portfolio()
    for h in data["holdings"]:
        if h["symbol"].upper() == symbol.upper():
            if "qty" in body:
                h["qty"] = float(body["qty"])
            if "avg_cost" in body:
                h["avg_cost"] = float(body["avg_cost"])
            _save_portfolio(data)
            return jsonify({"ok": True, "holdings": data["holdings"]})
    return jsonify({"error": "not found"}), 404

def _load_watchlist() -> list:
    if WATCHLIST_FILE.exists():
        try:
            return json.loads(WATCHLIST_FILE.read_text()).get("symbols", [])
        except Exception:
            return []
    return ["AAPL", "MSFT", "NVDA", "RELIANCE.NS", "TCS.NS"]


def _save_watchlist(symbols: list):
    WATCHLIST_FILE.write_text(json.dumps({"symbols": symbols}, indent=2))


@app.route("/api/watchlist")
def get_watchlist():
    symbols = _load_watchlist()
    quotes = {}
    for sym in symbols:
        q = fetch_realtime_quote(sym)
        quotes[sym] = {
            "price": q.get("price", 0),
            "change_pct": q.get("change_pct", 0),
            "name": q.get("name", sym),
        }
    return jsonify({"symbols": symbols, "quotes": quotes})


@app.route("/api/watchlist/<symbol>", methods=["POST"])
def watchlist_add(symbol: str):
    symbols = _load_watchlist()
    if symbol not in symbols:
        symbols.append(symbol.upper())
        _save_watchlist(symbols)
    return jsonify({"symbols": symbols})


@app.route("/api/watchlist/<symbol>", methods=["DELETE"])
def watchlist_remove(symbol: str):
    symbols = _load_watchlist()
    if symbol in symbols:
        symbols.remove(symbol)
        _save_watchlist(symbols)
    return jsonify({"symbols": symbols})


# ============================================================================
# COMPANY INTELLIGENCE — analysts, earnings, news
# ============================================================================

@app.route("/api/analysts")
def analysts():
    ticker = request.args.get("ticker", "AAPL").upper()
    return jsonify(_cached(f"analysts:{ticker}", 300, lambda: _analysts(ticker)))


def _analysts(ticker: str):
    try:
        t = yf.Ticker(ticker)
        rec = t.get_recommendations_summary()
        result = {"ticker": ticker, "periods": []}
        if rec is not None and not rec.empty:
            for _, row in rec.iterrows():
                entry = {}
                for col in rec.columns:
                    v = row[col]
                    if pd.api.types.is_numeric_dtype(rec[col]):
                        entry[str(col)] = float(v) if pd.notna(v) else 0
                    else:
                        entry[str(col)] = str(v)
                result["periods"].append(entry)
            latest = result["periods"][0]
            total = sum(latest.get(c, 0) for c in
                        ["strongBuy", "buy", "hold", "sell", "strongSell"])
            score = (latest.get("strongBuy", 0) * 2 + latest.get("buy", 0)
                     - latest.get("sell", 0) - latest.get("strongSell", 0) * 2)
            result["total"] = total
            result["consensus"] = score / total if total else 0
            result["consensus_label"] = (
                "Strong Buy" if score / total >= 1.0 else
                "Buy" if score / total > 0.3 else
                "Hold" if score / total >= -0.3 else
                "Sell" if score / total > -1.0 else "Strong Sell"
            ) if total else "N/A"
        return result
    except Exception as e:
        return {"ticker": ticker, "periods": [], "error": str(e)}


@app.route("/api/earnings")
def earnings():
    ticker = request.args.get("ticker", "AAPL").upper()
    return jsonify(_cached(f"earnings:{ticker}", 300, lambda: _earnings(ticker)))


def _earnings(ticker: str):
    try:
        t = yf.Ticker(ticker)
        ed = t.get_earnings_dates(limit=12)
        result = {"ticker": ticker, "quarters": []}
        if ed is not None and not ed.empty:
            for idx, row in ed.iterrows():
                est = row.get("EPS Estimate")
                actual = row.get("Reported EPS")
                surprise = row.get("Surprise(%)")
                result["quarters"].append({
                    "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                    "eps_estimate": float(est) if pd.notna(est) else None,
                    "eps_actual": float(actual) if pd.notna(actual) else None,
                    "surprise_pct": float(surprise) if pd.notna(surprise) else None,
                })
            past = [q for q in result["quarters"] if q["eps_actual"] is not None]
            if past:
                beats = sum(1 for q in past if (q["surprise_pct"] or 0) > 0)
                result["beat_rate"] = beats / len(past)
            upcoming = [q for q in result["quarters"] if q["eps_actual"] is None]
            if upcoming:
                result["next_earnings"] = upcoming[0]["date"]
        return result
    except Exception as e:
        return {"ticker": ticker, "quarters": [], "error": str(e)}


@app.route("/api/news")
def news():
    ticker = request.args.get("ticker", "AAPL").upper()
    limit = min(int(request.args.get("limit", 12)), 30)
    return jsonify(_cached(f"news:{ticker}:{limit}", 300, lambda: _news(ticker, limit)))


def _news(ticker: str, limit: int):
    try:
        t = yf.Ticker(ticker)
        items = t.news if hasattr(t, "news") else []
        result = {"ticker": ticker, "items": [], "net_sentiment": 0.0}
        for n in items[:limit]:
            content = n.get("content") if isinstance(n, dict) and "content" in n else n
            if isinstance(content, dict):
                title = content.get("title", "")
                publisher = content.get("provider", {}).get("displayName", "Unknown") \
                    if isinstance(content.get("provider"), dict) else content.get("publisher", "Unknown")
                link = content.get("canonicalUrl", {}).get("url", "") \
                    if isinstance(content.get("canonicalUrl"), dict) else content.get("link", "")
                ts = content.get("pubDate", 0) or content.get("providerPublishTime", 0)
            else:
                title = n.get("title", "") if isinstance(n, dict) else ""
                publisher = n.get("publisher", "Unknown") if isinstance(n, dict) else "Unknown"
                link = n.get("link", "") if isinstance(n, dict) else ""
                ts = n.get("providerPublishTime", 0) if isinstance(n, dict) else 0
            if isinstance(ts, str):
                try:
                    ts = time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))
                except ValueError:
                    ts = 0
            score = _sentiment_score(title)
            result["items"].append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "time": int(ts),
                "sentiment": round(score, 3),
                "label": _sentiment_label(score),
            })
        if result["items"]:
            result["net_sentiment"] = round(
                sum(i["sentiment"] for i in result["items"]) / len(result["items"]), 3)
        result["net_label"] = _sentiment_label(result["net_sentiment"])
        return result
    except Exception as e:
        return {"ticker": ticker, "items": [], "net_sentiment": 0.0,
                "error": str(e)}


@app.route("/api/agents")
def agents():
    try:
        registry = create_default_registry()
        health = registry.health_check_all()
        agents_list = []
        for name, status in health.get("agents", {}).items():
            if isinstance(status, str):
                status = {"status": status}
            agents_list.append({
                "name": name,
                "status": status.get("status", "unknown"),
                "latency_ms": status.get("latency_ms", 0),
                "last_run": status.get("last_run", None),
            })
        return jsonify({"agents": agents_list})
    except Exception as e:
        return jsonify({"agents": [], "error": str(e)})


@app.route("/api/settings")
def settings_info():
    data_files = list(DATA_CACHE.glob("*.parquet")) if DATA_CACHE.exists() else []
    return jsonify({
        "cache_location": str(DATA_CACHE),
        "cached_files": len(data_files),
        "framework": "React Terminal",
        "design_system": "Obsidian Institutional",
        "chart_engine": "Recharts",
        "agent_engine": "Obsidian Orchestrator",
    })


@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    _cache.clear()
    for f in DATA_CACHE.glob("*.parquet"):
        f.unlink()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    print("Obsidian Terminal API — http://localhost:8000")
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
