"""
Aegis — Quant Platform API Server
===================================
FastAPI backend serving market data, alpha signals, and agent orchestration.
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import ta
import json
import sys
import time
import traceback

# ── Path Setup ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_CACHE = ROOT / "data_cache"

# ── App ──
app = FastAPI(title="Aegis Quant Platform", version="2.0", docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# SIMPLE CACHE
# ============================================================================
_cache: dict = {}
_cache_ts: dict = {}


def get_cached(key: str, ttl: int = 120):
    if key in _cache and time.time() - _cache_ts.get(key, 0) < ttl:
        return _cache[key]
    return None


def set_cached(key: str, value):
    _cache[key] = value
    _cache_ts[key] = time.time()


# ============================================================================
# DATA FUNCTIONS
# ============================================================================

def fetch_stock(ticker: str, period: str = "1y") -> pd.DataFrame:
    ck = f"stock:{ticker}:{period}"
    cached = get_cached(ck, ttl=300)
    if cached is not None:
        return cached
    try:
        fp = DATA_CACHE / f"{ticker.replace('^', 'IDX_').replace('=', '_')}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
            set_cached(ck, df)
            return df
    except Exception:
        pass
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [c.lower() for c in data.columns]
        set_cached(ck, data)
        return data
    except Exception:
        return pd.DataFrame()


def fetch_quote(ticker: str) -> dict:
    ck = f"quote:{ticker}"
    cached = get_cached(ck, ttl=60)
    if cached is not None:
        return cached
    try:
        info = yf.Ticker(ticker).info
        result = {
            "price": info.get("regularMarketPrice", info.get("currentPrice", 0)) or 0,
            "change": info.get("regularMarketChange", 0) or 0,
            "change_pct": info.get("regularMarketChangePercent", 0) or 0,
            "volume": info.get("regularMarketVolume", 0) or 0,
            "market_cap": info.get("marketCap", 0) or 0,
            "pe_ratio": info.get("trailingPE", 0) or 0,
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "high": info.get("dayHigh", 0) or 0,
            "low": info.get("dayLow", 0) or 0,
            "open": info.get("open", 0) or 0,
            "prev_close": info.get("previousClose", 0) or 0,
        }
        set_cached(ck, result)
        return result
    except Exception:
        return {
            "price": 0, "change": 0, "change_pct": 0, "volume": 0,
            "market_cap": 0, "pe_ratio": 0, "name": ticker, "sector": "N/A",
            "high": 0, "low": 0, "open": 0, "prev_close": 0,
        }


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 20:
        return df
    df = df.copy()
    c = df["close"]
    df["sma_20"] = c.rolling(20).mean()
    df["sma_50"] = c.rolling(50).mean()
    df["ema_12"] = c.ewm(span=12).mean()
    df["ema_26"] = c.ewm(span=26).mean()
    df["rsi"] = ta.momentum.RSIIndicator(c, window=14).rsi()
    macd_obj = ta.trend.MACD(c)
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"] = macd_obj.macd_diff()
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], c, window=14
    ).average_true_range()
    return df


def df_to_columns(df: pd.DataFrame) -> dict:
    """Convert DataFrame to column-oriented dict for Plotly.js consumption."""
    result = {}
    df_r = df.reset_index()
    for col in df_r.columns:
        s = df_r[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            result[col.lower() if col != "Date" else "date"] = (
                s.dt.strftime("%Y-%m-%d").tolist()
            )
        elif pd.api.types.is_numeric_dtype(s):
            result[col] = [
                None if pd.isna(v) else round(float(v), 6) for v in s
            ]
        else:
            result[col] = s.fillna("").astype(str).tolist()
    if "Date" in result:
        result["date"] = result.pop("Date")
    return result


# ============================================================================
# AGENT SINGLETON
# ============================================================================
_orchestrator = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        try:
            from agents.orchestrator import AgentOrchestrator
            _orchestrator = AgentOrchestrator()
            _orchestrator.start()
        except Exception:
            pass
    return _orchestrator


# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/api/quote/{ticker}")
async def api_quote(ticker: str):
    return fetch_quote(ticker.upper())


@app.get("/api/stock/{ticker}")
async def api_stock(ticker: str, period: str = "1y"):
    df = fetch_stock(ticker.upper(), period)
    if df.empty:
        return JSONResponse({"error": f"No data for {ticker}"}, status_code=404)
    return df_to_columns(df)


@app.get("/api/indicators/{ticker}")
async def api_indicators(ticker: str, period: str = "1y"):
    df = fetch_stock(ticker.upper(), period)
    if df.empty:
        return JSONResponse({"error": f"No data for {ticker}"}, status_code=404)
    df = calc_indicators(df)
    return df_to_columns(df)


@app.get("/api/indices")
async def api_indices():
    tickers = {
        "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DOW 30": "^DJI",
        "Russell 2K": "^RUT", "VIX": "^VIX", "10Y Yield": "^TNX",
    }
    out = []
    for name, tkr in tickers.items():
        q = fetch_quote(tkr)
        out.append({"name": name, "ticker": tkr, **q})
    return out


@app.get("/api/sectors")
async def api_sectors():
    sectors = {
        "XLK": "Technology", "XLF": "Financials", "XLV": "Healthcare",
        "XLE": "Energy", "XLI": "Industrials", "XLP": "Staples",
        "XLY": "Consumer Disc.", "XLU": "Utilities",
    }
    out = []
    for tkr, nm in sectors.items():
        q = fetch_quote(tkr)
        q.pop("sector", None)  # Remove ETF sector to keep our label
        out.append({"sector": nm, "ticker": tkr, **q})
    return out


@app.get("/api/signals")
async def api_signals():
    path = DATA_CACHE / "alpha_signals.parquet"
    if not path.exists():
        return {"data": [], "count": 0}
    df = pd.read_parquet(path)
    return {"data": json.loads(df.to_json(orient="records")), "count": len(df)}


@app.get("/api/fundamentals")
async def api_fundamentals():
    path = DATA_CACHE / "fundamentals.parquet"
    if not path.exists():
        return {"data": [], "count": 0}
    df = pd.read_parquet(path)
    return {"data": json.loads(df.to_json(orient="records")), "count": len(df)}


@app.get("/api/agents/status")
async def api_agents_status():
    orch = get_orchestrator()
    if orch is None:
        return {"error": "Agent system not available", "agents": [], "system_status": "offline"}
    try:
        registry = orch.registry
        from core.event_bus import EventBus
        bus = EventBus()
        agents = []
        for name in registry.agent_names:
            agent = registry.get(name)
            if agent:
                h = agent.health_check()
                agents.append({
                    "name": name,
                    "status": h.get("status", "unknown"),
                    "health": h,
                    "metrics": agent.metrics,
                })
        health = registry.health_check_all()
        return {
            "system_status": health.get("system_status", "unknown"),
            "agent_count": health.get("agent_count", 0),
            "agents": agents,
            "bus_stats": bus.stats,
        }
    except Exception as e:
        return {"error": str(e), "agents": [], "system_status": "error"}


@app.post("/api/agents/run")
async def api_agents_run(request: Request):
    orch = get_orchestrator()
    if orch is None:
        return JSONResponse({"error": "Agent system not available"}, status_code=503)
    try:
        body = await request.json()
        symbols = body.get("symbols", [])
        source = body.get("source", "yahoo")
        period = body.get("period", "1y")
        results = orch.run_pipeline(symbols, source=source, period=period)
        return {sym: result.to_dict() for sym, result in results.items()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================================
# STATIC FILES + ROOT
# ============================================================================
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    print("\n  ⬡  Aegis Quant Platform")
    print("  ─────────────────────────")
    print("  http://localhost:8501\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8501, reload=True)
