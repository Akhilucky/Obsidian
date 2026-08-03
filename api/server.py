"""
Obsidian Terminal — Flask API server.

Serves JSON endpoints for the React frontend. Wraps the same data layer
used by the legacy Streamlit dashboard.

Run:  python api/server.py
"""
import sys
import os
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

_cache: dict = {}


def _cached(key: str, ttl: int, fn):
    now = datetime.now().timestamp()
    if key in _cache and now - _cache[key][0] < ttl:
        return _cache[key][1]
    value = fn()
    _cache[key] = (now, value)
    return value


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

INDIA_POPULAR = {
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
    return jsonify({
        "ticker": ticker,
        "quote": q,
        "points": df_to_records(calculate_indicators(df), 1200) if not df.empty else [],
        "popular": popular,
    })


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
