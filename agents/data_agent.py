"""
Data Ingestion Agent
=====================
Fetch and normalize market, macro, crypto, and sentiment data.

Frequency: Realtime
Failure Policy: Retry → fallback source → mark degraded.

Output Contract:
{
  "event": "DATA_INGESTED",
  "symbol": "AAPL",
  "dataframe": "...",
  "source": "openbb",
  "latency_ms": 120
}
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from agents.base_agent import BaseAgent
from core.event_bus import Event, EventType

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# Integration with core data ingest pipeline
try:
    from core.data_ingest import DataIngestPipeline, DataNormalizer
    CORE_INGEST_AVAILABLE = True
except ImportError:
    CORE_INGEST_AVAILABLE = False

logger = logging.getLogger(__name__)

import time as _time

class _RateLimiter:
    """Simple token-bucket rate limiter for API calls."""
    def __init__(self, min_interval: float = 0.5):
        self._min_interval = min_interval
        self._last_call = 0.0
    
    def wait(self):
        elapsed = _time.time() - self._last_call
        if elapsed < self._min_interval:
            _time.sleep(self._min_interval - elapsed)
        self._last_call = _time.time()

_rate_limiter = _RateLimiter(min_interval=0.3)

# Configurable retry / fallback
MAX_RETRIES = 3
FALLBACK_SOURCES = ["yahoo", "openbb", "fred"]


class DataIngestionAgent(BaseAgent):
    """
    Agent 1: Fetch and normalize market, macro, crypto, and sentiment data.
    
    Integrates with core.data_ingest.DataIngestPipeline for full
    multi-source ingestion (Yahoo, FRED, CoinGecko, OpenBB).
    
    Responsibilities:
    - Pull data from configured providers
    - Normalize schema
    - Timestamp synchronization
    - Push to validation queue
    """
    
    def __init__(self):
        super().__init__(
            name="DataIngestionAgent",
            subscriptions=[]  # This is a source agent — no incoming events
        )
        self._providers: Dict[str, Any] = {}
        self._degraded_sources: set = set()
        self._core_pipeline: Optional[Any] = None
    
    def initialize(self):
        """Register data providers and warm connections."""
        self._providers = {
            "yahoo": self._fetch_yahoo,
            "openbb": self._fetch_openbb,
            "fred": self._fetch_fred,
            "indian": self._fetch_indian,
        }
        # Wire up core pipeline if available
        if CORE_INGEST_AVAILABLE:
            try:
                self._core_pipeline = DataIngestPipeline()
                self._log("Core DataIngestPipeline connected (700+ feeds)")
            except Exception as e:
                self._log(f"Core pipeline init failed: {e}", level="warning")
        self._log("Data providers registered")
    
    def consume(self, event: Event):
        """DataIngestionAgent is source-only; no incoming events consumed."""
        pass
    
    def produce(self) -> Optional[Event]:
        """Not used directly — use ingest() to trigger data pulls."""
        return None
    
    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if not self._degraded_sources else "degraded",
            "degraded_sources": list(self._degraded_sources),
            "available_providers": list(self._providers.keys()),
        }
    
    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────
    
    def ingest(self, symbols: List[str], source: str = "yahoo",
               asset_class: str = "equity", period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Pull data for a list of symbols.
        
        Retry → fallback source → mark degraded.
        """
        results = {}
        for symbol in symbols:
            df = self._fetch_with_retry(symbol, source, asset_class, period)
            if df is not None and not df.empty:
                df = self._normalize(df, symbol, source)
                results[symbol] = df
                
                # Publish DATA_INGESTED event
                self._publish(
                    EventType.DATA_INGESTED.value,
                    {
                        "symbol": symbol,
                        "rows": len(df),
                        "columns": list(df.columns),
                        "source": source,
                        "asset_class": asset_class,
                        "latency_ms": self._metrics.avg_latency_ms,
                    }
                )
            else:
                self._log(f"Failed to ingest {symbol} from all sources", level="error")
        
        return results
    
    def ingest_universe(self, universe: str = "sp500_top50",
                        save: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Ingest a predefined universe using the core DataIngestPipeline.
        
        Universes: sp500_top50, indices, sectors, etfs, crypto, forex
        
        Requires core.data_ingest.DataIngestPipeline.
        """
        if not self._core_pipeline:
            self._log("Core pipeline not available — falling back to direct ingest", level="warning")
            from core.data_ingest import DataIngestPipeline
            universes = DataIngestPipeline.UNIVERSES
            symbols = universes.get(universe, ["AAPL", "MSFT", "GOOGL"])
            return self.ingest(symbols)
        
        try:
            results = self._core_pipeline.ingest_universe(universe, save=save)
            for symbol, df in results.items():
                self._publish(
                    EventType.DATA_INGESTED.value,
                    {
                        "symbol": symbol,
                        "rows": len(df),
                        "columns": list(df.columns),
                        "source": "core_pipeline",
                        "asset_class": universe,
                        "latency_ms": 0,
                    }
                )
            self._log(f"Ingested universe '{universe}': {len(results)} symbols")
            return results
        except Exception as e:
            self._log(f"Universe ingest failed: {e}", level="error")
            return {}
    
    def ingest_macro(self) -> pd.DataFrame:
        """Ingest macro economic data (FRED indicators) via core pipeline."""
        if self._core_pipeline:
            try:
                return self._core_pipeline.ingest_macro()
            except Exception as e:
                self._log(f"Macro ingest failed: {e}", level="error")
        return pd.DataFrame()
    
    def ingest_crypto(self) -> pd.DataFrame:
        """Ingest cryptocurrency data via core pipeline."""
        if self._core_pipeline:
            try:
                return self._core_pipeline.ingest_crypto()
            except Exception as e:
                self._log(f"Crypto ingest failed: {e}", level="error")
        return pd.DataFrame()

    def ingest_indian_market(self, universe: str = "nifty50",
                             start: Optional[str] = None,
                             end: Optional[str] = None,
                             save: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Ingest Indian market (NSE/BSE) data for a predefined universe.

        Universes: nifty50, sensex30, nifty_it, nifty_bank, indian_etf
        """
        if self._core_pipeline:
            try:
                results = self._core_pipeline.ingest_indian(universe, start, end, save=save)
                for symbol, df in results.items():
                    self._publish(
                        EventType.DATA_INGESTED.value,
                        {
                            "symbol": symbol,
                            "rows": len(df),
                            "columns": list(df.columns),
                            "source": "yfinance",
                            "asset_class": f"indian_{universe}",
                            "latency_ms": 0,
                        }
                    )
                self._log(f"Ingested Indian market universe '{universe}': {len(results)} symbols")
                return results
            except Exception as e:
                self._log(f"Indian market ingest failed: {e}", level="error")
                return {}
        else:
            self._log("Core pipeline not available for Indian market ingest", level="warning")
            try:
                from data.indian_markets import IndianMarketDataFetcher
                fetcher = IndianMarketDataFetcher()
                return fetcher.fetch_universe(universe, start, end)
            except ImportError as e:
                self._log(f"Indian markets module not available: {e}", level="error")
                return {}
    
    # ──────────────────────────────────────────────
    # Internal: Fetch with retry + fallback
    # ──────────────────────────────────────────────
    
    def _fetch_with_retry(self, symbol: str, source: str,
                          asset_class: str, period: str) -> Optional[pd.DataFrame]:
        """Retry up to MAX_RETRIES, then try fallback sources."""
        sources_to_try = [source] + [s for s in FALLBACK_SOURCES if s != source]
        
        for src in sources_to_try:
            if src in self._degraded_sources:
                continue
            for attempt in range(1, MAX_RETRIES + 1):
                _rate_limiter.wait()
                try:
                    start_ts = time.time()
                    fetcher = self._providers.get(src)
                    if fetcher is None:
                        break
                    df = fetcher(symbol, period)
                    latency = (time.time() - start_ts) * 1000
                    self._log(f"Fetched {symbol} from {src} in {latency:.0f}ms (attempt {attempt})")
                    return df
                except Exception as e:
                    self._log(f"Attempt {attempt} for {symbol} via {src} failed: {e}", level="warning")
                    time.sleep(0.5 * attempt)
            
            # All retries exhausted for this source — mark degraded
            self._degraded_sources.add(src)
            self._log(f"Source {src} marked degraded", level="warning")
        
        return None
    
    # ──────────────────────────────────────────────
    # Normalization
    # ──────────────────────────────────────────────
    
    def _normalize(self, df: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
        """
        Normalize to standard schema:
        date | open | high | low | close | volume | adj_close | symbol | source
        """
        df = df.copy()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
        
        # Standardize column names
        col_map = {}
        for col in df.columns:
            lower = str(col).lower().strip()
            if 'open' in lower:
                col_map[col] = 'open'
            elif 'high' in lower:
                col_map[col] = 'high'
            elif 'low' in lower:
                col_map[col] = 'low'
            elif 'adj' in lower and 'close' in lower:
                col_map[col] = 'adj_close'
            elif 'close' in lower:
                col_map[col] = 'close'
            elif 'volume' in lower or 'vol' == lower:
                col_map[col] = 'volume'
        
        df.rename(columns=col_map, inplace=True)
        df.index.name = 'date'
        
        # Add metadata columns
        df['symbol'] = symbol
        df['source'] = source
        
        # Ensure adj_close exists
        if 'adj_close' not in df.columns and 'close' in df.columns:
            df['adj_close'] = df['close']
        
        # Timestamp synchronization — ensure UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        
        return df
    
    # ──────────────────────────────────────────────
    # Provider implementations
    # ──────────────────────────────────────────────
    
    def _fetch_yahoo(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch from Yahoo Finance."""
        if not YF_AVAILABLE:
            raise ImportError("yfinance not installed")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        return df
    
    def _fetch_openbb(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch from OpenBB (stub — requires OpenBB SDK)."""
        try:
            from data.openbb_integration import OpenBBIntegration
            obb = OpenBBIntegration()
            return obb.get_stock_data(symbol, period=period)
        except ImportError:
            raise ImportError("OpenBB integration not available")
    
    def _fetch_fred(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch macro data from FRED (stub)."""
        try:
            import pandas_datareader.data as web
            from datetime import timedelta
            end = datetime.now()
            start = end - timedelta(days=365)
            return web.DataReader(symbol, 'fred', start, end)
        except ImportError:
            raise ImportError("pandas-datareader not installed for FRED data")

    def _fetch_indian(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch Indian stock data via yfinance with .NS suffix."""
        if not YF_AVAILABLE:
            raise ImportError("yfinance not installed")
        # Ensure .NS suffix for NSE stocks
        yahoo_symbol = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(period=period)
        return df
