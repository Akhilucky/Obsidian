"""
Multi-provider market data layer.

Quotes, history and news can be served by any configured provider:

  1. Alpha Vantage (ALPHA_VANTAGE_KEY)       — 25 req/day free, EOD + News & Sentiment
  2. Financial Modeling Prep (FMP_API_KEY)    — 250 req/day free, EOD + fundamentals
  3. Polygon.io / Massive (POLYGON_API_KEY)   — 5 req/min free, EOD
  4. yfinance (no key required)               — always available fallback

Providers are key-gated: if the env var is missing the provider is skipped.
A per-day quota ledger (data_cache/provider_quota.json) tracks free-tier limits so
the chain degrades gracefully instead of burning a paid quota.

Ordering can be overridden with DATA_PROVIDER_ORDER (comma-separated names) or with
data_cache/provider_trials.json written by `research/tool_trials.py --apply`,
which is the auto-integration step of the tool-trial harness.
"""

import json
import os
import time
from datetime import date

import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

DATA_CACHE = os.environ.get("OBSIDIAN_DATA_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache"))
os.makedirs(DATA_CACHE, exist_ok=True)

QUOTA_FILE = os.path.join(DATA_CACHE, "provider_quota.json")
TRIALS_FILE = os.path.join(DATA_CACHE, "provider_trials.json")
HTTP_TIMEOUT = 12

# Free-tier daily limits (0 = unlimited / no tracking)
DAILY_LIMITS = {"alphavantage": 25, "fmp": 250, "polygon": 10000}

_PROVIDER_NAMES = ["alphavantage", "fmp", "polygon", "yfinance"]
_KEY_ENV = {
    "alphavantage": "ALPHA_VANTAGE_KEY",
    "fmp": "FMP_API_KEY",
    "polygon": "POLYGON_API_KEY",
}


class QuotaLedger:
    """Persistent per-day request counter for free-tier providers."""

    def __init__(self, path: str = QUOTA_FILE):
        self.path = path
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path) as fh:
                self._data = json.load(fh)
        except Exception:
            self._data = {}

    def _save(self) -> None:
        try:
            with open(self.path, "w") as fh:
                json.dump(self._data, fh, indent=2)
        except Exception:
            pass

    def used_today(self, provider: str) -> int:
        day = str(date.today())
        return int(self._data.get(day, {}).get(provider, 0))

    def remaining(self, provider: str) -> int:
        limit = DAILY_LIMITS.get(provider, 0)
        return 0 if limit and self.used_today(provider) >= limit else (limit - self.used_today(provider) if limit else 99999)

    def record(self, provider: str) -> None:
        day = str(date.today())
        self._data.setdefault(day, {})
        self._data[day][provider] = int(self._data[day].get(provider, 0)) + 1
        self._save()

    def can_use(self, provider: str) -> bool:
        limit = DAILY_LIMITS.get(provider, 0)
        return not limit or self.used_today(provider) < limit


quota = QuotaLedger()


def _http_json(url: str) -> dict:
    resp = requests.get(url, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    if isinstance(payload, dict) and payload.get("Note", "").startswith("Thank you for using Alpha Vantage"):
        raise RuntimeError("Alpha Vantage rate limit hit")
    return payload


class BaseProvider:
    name = "base"
    requires_key = False

    def __init__(self) -> None:
        self.hits = 0
        self.errors = 0
        self.last_ok = None  # unix ts of last successful call
        self.last_error = None

    @property
    def configured(self) -> bool:
        env = _KEY_ENV.get(self.name)
        return (not env) or bool(os.environ.get(env))

    def _guard(self) -> None:
        if not self.configured:
            raise RuntimeError(f"{self.name}: missing key {_KEY_ENV.get(self.name)}")
        if not quota.can_use(self.name):
            raise RuntimeError(f"{self.name}: daily quota exhausted ({DAILY_LIMITS[self.name]} used)")

    def _ok(self) -> None:
        self.hits += 1
        self.last_ok = time.time()

    def _fail(self, err: Exception) -> None:
        self.errors += 1
        self.last_error = f"{type(err).__name__}: {err}"

    def get_quote(self, ticker: str):  # -> dict | None
        raise NotImplementedError

    def get_history(self, ticker: str, period: str = "1y"):  # -> pd.DataFrame | None
        raise NotImplementedError

    def get_news(self, ticker: str, limit: int = 12):  # -> list | None
        raise NotImplementedError


class AlphaVantageProvider(BaseProvider):
    """Free tier: 25 requests/day, end-of-day data, news + sentiment."""

    name = "alphavantage"
    requires_key = True
    BASE = "https://www.alphavantage.co/query"

    def _call(self, params: dict) -> dict:
        params["apikey"] = os.environ["ALPHA_VANTAGE_KEY"]
        return _http_json(f"{self.BASE}?{requests.compat.urlencode(params)}")

    def get_quote(self, ticker: str):
        self._guard()
        try:
            data = self._call({"function": "GLOBAL_QUOTE", "symbol": ticker})
            q = (data or {}).get("Global Quote") or {}
            if not q:
                raise RuntimeError("empty GLOBAL_QUOTE")
            self._ok()
            quota.record(self.name)
            price = float(q.get("05. price") or 0)
            prev = float(q.get("08. previous close") or 0)
            return {
                "price": price,
                "change": float(q.get("09. change") or (price - prev)),
                "change_pct": float(q.get("10. change percent", "0").rstrip("%") or 0),
                "volume": int(float(q.get("06. volume") or 0)),
                "market_cap": 0,
                "pe_ratio": 0,
                "name": ticker,
                "sector": "N/A",
                "high": float(q.get("03. high") or 0),
                "low": float(q.get("04. low") or 0),
                "open": float(q.get("02. open") or 0),
                "prev_close": prev,
            }
        except Exception as exc:
            self._fail(exc)
            return None

    def get_news(self, ticker: str, limit: int = 12):
        self._guard()
        try:
            data = self._call({"function": "NEWS_SENTIMENT", "tickers": ticker, "limit": min(limit, 25), "sort": "LATEST"})
            raw = (data or {}).get("feed") or []
            self._ok()
            quota.record(self.name)
            items = []
            for n in raw[:limit]:
                ts = n.get("time_published", "")
                try:
                    t = time.mktime(time.strptime(ts, "%Y%m%dT%H%M%S"))
                except Exception:
                    t = 0
                items.append({
                    "title": n.get("title", ""),
                    "publisher": n.get("source", "Unknown"),
                    "link": n.get("url", ""),
                    "time": int(t),
                    "provider_label": n.get("overall_sentiment_label", "neutral"),
                })
            return items
        except Exception as exc:
            self._fail(exc)
            return None

    def get_history(self, ticker: str, period: str = "1y"):
        self._guard()
        try:
            data = self._call({"function": "TIME_SERIES_DAILY", "symbol": ticker, "outputsize": "compact"})
            series = (data or {}).get("Time Series (Daily)") or {}
            if not series:
                raise RuntimeError("empty TIME_SERIES_DAILY")
            rows = []
            for day, r in sorted(series.items()):
                rows.append({
                    "date": day,
                    "open": float(r["1. open"]),
                    "high": float(r["2. high"]),
                    "low": float(r["3. low"]),
                    "close": float(r["4. close"]),
                    "volume": int(float(r["5. volume"])),
                })
            self._ok()
            quota.record(self.name)
            df = pd.DataFrame(rows)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            return df
        except Exception as exc:
            self._fail(exc)
            return None


class FMPProvider(BaseProvider):
    """Free tier: 250 requests/day, EOD quotes + fundamentals + news."""

    name = "fmp"
    requires_key = True
    BASE = "https://financialmodelingprep.com/stable"

    def _get(self, path: str, **params) -> dict:
        params["apikey"] = os.environ["FMP_API_KEY"]
        return _http_json(f"{self.BASE}/{path}?{requests.compat.urlencode(params)}")

    def get_quote(self, ticker: str):
        self._guard()
        try:
            data = self._get("quote", symbol=ticker)
            rows = data if isinstance(data, list) else []
            if not rows:
                raise RuntimeError("empty FMP quote")
            q = rows[0]
            self._ok()
            quota.record(self.name)
            return {
                "price": float(q.get("price") or 0),
                "change": float(q.get("change") or 0),
                "change_pct": float(q.get("changesPercentage") or 0),
                "volume": int(float(q.get("volume") or 0)),
                "market_cap": float(q.get("marketCap") or 0),
                "pe_ratio": float(q.get("pe") or 0),
                "name": q.get("name") or ticker,
                "sector": q.get("sector") or "N/A",
                "high": float(q.get("dayHigh") or 0),
                "low": float(q.get("dayLow") or 0),
                "open": float(q.get("open") or 0),
                "prev_close": float(q.get("previousClose") or 0),
            }
        except Exception as exc:
            self._fail(exc)
            return None

    def get_news(self, ticker: str, limit: int = 12):
        self._guard()
        try:
            data = self._get("stock-news", symbol=ticker, limit=min(limit, 30))
            rows = data if isinstance(data, list) else []
            if not rows:
                raise RuntimeError("empty FMP news")
            self._ok()
            quota.record(self.name)
            items = []
            for n in rows[:limit]:
                items.append({
                    "title": n.get("title", ""),
                    "publisher": n.get("site", "Unknown"),
                    "link": n.get("url", ""),
                    "time": int(n.get("publishedDate", 0) or 0),
                    "provider_label": "neutral",
                })
            return items
        except Exception as exc:
            self._fail(exc)
            return None

    def get_history(self, ticker: str, period: str = "1y"):
        self._guard()
        try:
            data = self._get("historical-price-eod", symbol=ticker, period="daily")
            rows = data if isinstance(data, list) else []
            if not rows:
                raise RuntimeError("empty FMP history")
            self._ok()
            quota.record(self.name)
            df = pd.DataFrame([{
                "date": r["date"],
                "open": float(r.get("open") or 0),
                "high": float(r.get("high") or 0),
                "low": float(r.get("low") or 0),
                "close": float(r.get("close") or 0),
                "volume": int(float(r.get("volume") or 0)),
            } for r in rows])
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            return df
        except Exception as exc:
            self._fail(exc)
            return None


class PolygonProvider(BaseProvider):
    """Free tier (Massive 'Basic'): 5 calls/min, 15-min delayed, ~2y history."""

    name = "polygon"
    requires_key = True
    BASE = "https://api.polygon.io"

    def _get(self, path: str, **params) -> dict:
        params["apiKey"] = os.environ["POLYGON_API_KEY"]
        return _http_json(f"{self.BASE}/{path}?{requests.compat.urlencode(params)}")

    def get_quote(self, ticker: str):
        self._guard()
        try:
            data = self._get("v2/aggs/ticker/{ticker}/prev".format(ticker=ticker.replace("^", "X"), ), adjusted="true")
            results = (data or {}).get("results") or []
            if not results:
                raise RuntimeError("empty polygon quote")
            r = results[0]
            prev = float(r.get("o") or 0)
            price = float(r.get("c") or 0)
            self._ok()
            quota.record(self.name)
            return {
                "price": price,
                "change": price - prev,
                "change_pct": ((price - prev) / prev * 100) if prev else 0,
                "volume": int(float(r.get("v") or 0)),
                "market_cap": 0,
                "pe_ratio": 0,
                "name": ticker,
                "sector": "N/A",
                "high": float(r.get("h") or 0),
                "low": float(r.get("l") or 0),
                "open": prev,
                "prev_close": prev,
            }
        except Exception as exc:
            self._fail(exc)
            return None

    def get_news(self, ticker: str, limit: int = 12):
        self._guard()
        try:
            data = self._get("v2/reference/news", ticker=ticker.replace("^", ""), limit=min(limit, 30))
            results = (data or {}).get("results") or []
            if not results:
                raise RuntimeError("empty polygon news")
            self._ok()
            quota.record(self.name)
            items = []
            for n in results[:limit]:
                items.append({
                    "title": n.get("title", ""),
                    "publisher": n.get("publisher", {}).get("name", "Unknown")
                    if isinstance(n.get("publisher"), dict) else n.get("publisher", "Unknown"),
                    "link": n.get("article_url", ""),
                    "time": int(n.get("published_utc", 0) or 0),
                    "provider_label": "neutral",
                })
            return items
        except Exception as exc:
            self._fail(exc)
            return None

    def get_history(self, ticker: str, period: str = "1y"):
        self._guard()
        try:
            from datetime import timedelta
            end = date.today()
            start = end - timedelta(days={"1mo": 45, "6mo": 200, "1y": 400, "2y": 760}.get(period, 400))
            data = self._get("v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}".format(
                ticker=ticker.replace("^", "X"), start=start.isoformat(), end=end.isoformat()), adjusted="true")
            results = (data or {}).get("results") or []
            if not results:
                raise RuntimeError("empty polygon history")
            self._ok()
            quota.record(self.name)
            df = pd.DataFrame([{
                "date": pd.Timestamp(r["t"], unit="ms"),
                "open": float(r.get("o") or 0),
                "high": float(r.get("h") or 0),
                "low": float(r.get("l") or 0),
                "close": float(r.get("c") or 0),
                "volume": int(float(r.get("v") or 0)),
            } for r in results])
            if not df.empty:
                df = df.set_index("date")
            return df
        except Exception as exc:
            self._fail(exc)
            return None


class YFinanceProvider(BaseProvider):
    """Keyless fallback; always available."""

    name = "yfinance"
    requires_key = False

    def get_quote(self, ticker: str):
        if yf is None:
            return None
        try:
            info = yf.Ticker(ticker).info
            self._ok()
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
        except Exception as exc:
            self._fail(exc)
            return None

    def get_news(self, ticker: str, limit: int = 12):
        if yf is None:
            return None
        try:
            t = yf.Ticker(ticker)
            items = t.news if hasattr(t, "news") else []
            self._ok()
            result = []
            for n in items[:limit]:
                content = n.get("content") if isinstance(n, dict) and "content" in n else n
                if isinstance(content, dict):
                    result.append({
                        "title": content.get("title", ""),
                        "publisher": content.get("provider", {}).get("displayName", "Unknown")
                        if isinstance(content.get("provider"), dict) else content.get("publisher", "Unknown"),
                        "link": content.get("canonicalUrl", {}).get("url", "")
                        if isinstance(content.get("canonicalUrl"), dict) else content.get("link", ""),
                        "time": int(content.get("pubDate", 0) or content.get("providerPublishTime", 0) or 0),
                        "provider_label": "neutral",
                    })
                else:
                    result.append({
                        "title": n.get("title", "") if isinstance(n, dict) else "",
                        "publisher": n.get("publisher", "Unknown") if isinstance(n, dict) else "Unknown",
                        "link": n.get("link", "") if isinstance(n, dict) else "",
                        "time": int(n.get("providerPublishTime", 0)) if isinstance(n, dict) else 0,
                        "provider_label": "neutral",
                    })
            return result
        except Exception as exc:
            self._fail(exc)
            return None

    def get_history(self, ticker: str, period: str = "1y"):
        if yf is None:
            return None
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=False)
            if data is None or data.empty:
                raise RuntimeError("empty yfinance history")
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(c).lower() for c in data.columns]
            data.index = pd.to_datetime(data.index)
            self._ok()
            return data
        except Exception as exc:
            self._fail(exc)
            return None


def _instantiate() -> list:
    providers = [AlphaVantageProvider(), FMPProvider(), PolygonProvider(), YFinanceProvider()]
    by_name = {p.name: p for p in providers}
    order = _PROVIDER_NAMES
    env_order = os.environ.get("DATA_PROVIDER_ORDER", "")
    if env_order:
        order = [n.strip().lower() for n in env_order.split(",") if n.strip()]
    trials_order = []
    try:
        with open(TRIALS_FILE) as fh:
            trials = json.load(fh)
        trials_order = trials.get("order", [])
    except Exception:
        pass
    effective = []
    for n in trials_order + order:
        if n in by_name and n not in effective:
            effective.append(n)
    return [by_name[n] for n in effective]


def get_quote_chain(ticker: str):
    """First provider that returns a quote wins; None if all fail."""
    for p in _instantiate():
        if not p.configured:
            continue
        try:
            q = p.get_quote(ticker)
        except Exception:
            q = None
        if q:
            return {"provider": p.name, **q}
    return None


def get_news_chain(ticker: str, limit: int = 12):
    """First provider that returns news wins; None if all fail."""
    for p in _instantiate():
        if not p.configured:
            continue
        try:
            items = p.get_news(ticker, limit)
        except Exception:
            items = None
        if items:
            return {"provider": p.name, "items": items}
    return None


def get_history_chain(ticker: str, period: str = "1y"):
    """First provider that returns history wins; None if all fail."""
    for p in _instantiate():
        if not p.configured:
            continue
        try:
            df = p.get_history(ticker, period)
        except Exception:
            df = None
        if df is not None and not df.empty:
            return df
    return None


def provider_status() -> dict:
    """Report configuration and health of every provider for the /api/providers endpoint."""
    providers = _instantiate()
    entries = []
    for p in providers:
        entries.append({
            "name": p.name,
            "configured": p.configured,
            "key_env": _KEY_ENV.get(p.name),
            "daily_limit": DAILY_LIMITS.get(p.name, 0),
            "used_today": quota.used_today(p.name),
            "hits": p.hits,
            "errors": p.errors,
            "last_ok": p.last_ok,
            "last_error": p.last_error,
        })
    active = next((e["name"] for e in entries if e["configured"]), entries[0]["name"] if entries else None)
    return {"providers": entries, "active": active}
