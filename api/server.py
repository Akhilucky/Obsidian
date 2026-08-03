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
    popular = []
    for sym, name in INDIA_POPULAR.items():
        qq = fetch_realtime_quote(sym)
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
# WATCHLIST — pinned stocks (persisted to disk)
# ============================================================================

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
