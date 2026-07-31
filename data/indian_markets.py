"""
Indian Stock Market Integration Module (yfinance)
===================================================
Fetches Indian stock data via Yahoo Finance.
Indian stocks use .NS suffix for NSE and .BO for BSE on Yahoo Finance.

Indices: ^NSEI (NIFTY 50), ^BSESN (SENSEX)
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data_warehouse"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Predefined Indian market universes
# ---------------------------------------------------------------------------

NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TATAMOTORS.NS", "WIPRO.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "M&M.NS", "JSWSTEEL.NS",
    "TATASTEEL.NS", "ADANIENT.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "INDUSINDBK.NS", "GRASIM.NS", "HINDALCO.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "CIPLA.NS", "APOLLOHOSP.NS",
    "TATACONSUM.NS", "BPCL.NS", "COALINDIA.NS",
]

SENSEX_30 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TATAMOTORS.NS", "WIPRO.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "M&M.NS", "JSWSTEEL.NS",
    "TATASTEEL.NS", "ADANIENT.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "INDUSINDBK.NS",
]

NIFTY_IT = [
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    "MPHASIS.NS", "LTTS.NS", "PERSISTENT.NS", "COFORGE.NS", "INFOBIP.NS",
]

NIFTY_BANK = [
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "INDUSINDBK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", "PNB.NS",
    "IDFCFIRSTB.NS", "AUBANK.NS", "MUTHOOTFIN.NS",
]

INDIAN_ETF = [
    "NIFTYBEES.NS", "JUNIORBEES.NS", "BANKBEES.NS", "GOLDBEES.NS",
]

INDIAN_UNIVERSES = {
    "nifty50": NIFTY_50,
    "sensex30": SENSEX_30,
    "nifty_it": NIFTY_IT,
    "nifty_bank": NIFTY_BANK,
    "indian_etf": INDIAN_ETF,
}


class IndianMarketDataFetcher:
    """
    Fetch Indian stock data via yfinance.

    Indian tickers on Yahoo Finance:
    - NSE: SYMBOL.NS  (e.g. RELIANCE.NS)
    - BSE: SYMBOL.BO  (e.g. RELIANCE.BO)
    - Indices: ^NSEI (NIFTY 50), ^BSESN (SENSEX)
    """

    INDICES = {
        "NIFTY50": "^NSEI",
        "SENSEX": "^BSESN",
        "NIFTY_BANK": "^NSEBANK",
        "NIFTY_IT": "^CNXIT",
    }

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(exist_ok=True)
        if not YF_AVAILABLE:
            logger.warning("yfinance not installed — IndianMarketDataFetcher will not work")

    # ------------------------------------------------------------------
    # OHLCV helpers
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single Indian stock symbol."""
        if not YF_AVAILABLE:
            logger.error("yfinance not installed")
            return pd.DataFrame()

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index.name = "date"
                df["symbol"] = symbol
                df["ingested_at"] = datetime.now(timezone.utc)
            return df
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def fetch_universe(
        self,
        universe: str = "nifty50",
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV for every symbol in a predefined Indian universe."""
        symbols = INDIAN_UNIVERSES.get(universe, [])
        if not symbols:
            logger.warning(f"Unknown Indian universe: {universe}")
            return {}

        results: Dict[str, pd.DataFrame] = {}
        total = len(symbols)
        for i, sym in enumerate(symbols):
            df = self.fetch_ohlcv(sym, start, end, interval)
            if not df.empty:
                results[sym] = df
                self._save(df, "indian", sym)
            if (i + 1) % 10 == 0:
                logger.info(f"Indian ingest progress: {i + 1}/{total}")

        logger.info(f"Ingested {len(results)}/{total} symbols from '{universe}'")
        return results

    # ------------------------------------------------------------------
    # Index data
    # ------------------------------------------------------------------

    def fetch_index(
        self,
        name: str = "NIFTY50",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch historical data for a major Indian index."""
        symbol = self.INDICES.get(name)
        if symbol is None:
            logger.warning(f"Unknown index: {name}")
            return pd.DataFrame()

        df = self.fetch_ohlcv(symbol, start, end)
        if not df.empty:
            self._save(df, "indian_indices", symbol)
        return df

    def fetch_all_indices(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch all major Indian indices."""
        results: Dict[str, pd.DataFrame] = {}
        for name in self.INDICES:
            df = self.fetch_index(name, start, end)
            if not df.empty:
                results[name] = df
        return results

    # ------------------------------------------------------------------
    # Options chain
    # ------------------------------------------------------------------

    def fetch_options_chain(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch options chain data for an Indian stock.

        Parameters
        ----------
        symbol : str
            Yahoo Finance symbol (e.g. "RELIANCE.NS").

        Returns
        -------
        dict  expiry -> {calls: DataFrame, puts: DataFrame}
        """
        if not YF_AVAILABLE:
            logger.error("yfinance not installed")
            return {}

        try:
            ticker = yf.Ticker(symbol)
            dates = ticker.options
            chains: Dict[str, Any] = {}
            for expiry in dates[:3]:
                opt = ticker.option_chain(expiry)
                chains[expiry] = {
                    "calls": opt.calls,
                    "puts": opt.puts,
                }
            return chains
        except Exception as e:
            logger.error(f"Error fetching options for {symbol}: {e}")
            return {}

    # ------------------------------------------------------------------
    # Ticker info
    # ------------------------------------------------------------------

    def fetch_info(self, symbol: str) -> Dict[str, Any]:
        """Fetch ticker info for an Indian stock."""
        if not YF_AVAILABLE:
            return {}

        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}

    # ------------------------------------------------------------------
    # Batch convenience
    # ------------------------------------------------------------------

    def fetch_all_universes(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch data for every predefined Indian universe."""
        results: Dict[str, Dict[str, pd.DataFrame]] = {}
        for universe in INDIAN_UNIVERSES:
            results[universe] = self.fetch_universe(universe, start, end)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, df: pd.DataFrame, category: str, symbol: str):
        """Persist a DataFrame to parquet."""
        safe_name = symbol.replace("^", "IDX_")
        path = self.data_dir / category / f"{safe_name}.parquet"
        path.parent.mkdir(exist_ok=True)
        df.to_parquet(path)

    def load(self, category: str, symbol: str) -> pd.DataFrame:
        """Load a previously saved parquet file."""
        safe_name = symbol.replace("^", "IDX_")
        path = self.data_dir / category / f"{safe_name}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def quick_ingest_indian(
    symbols: Optional[List[str]] = None,
    universe: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Quick data ingestion for Indian market symbols.

    Parameters
    ----------
    symbols : list, optional
        Explicit list of Yahoo Finance symbols (e.g. ["RELIANCE.NS", "TCS.NS"]).
    universe : str, optional
        Predefined universe name from INDIAN_UNIVERSES.
        Ignored when *symbols* is provided.
    start / end : str, optional
        Date range strings ("YYYY-MM-DD").
    """
    fetcher = IndianMarketDataFetcher()

    if symbols:
        results: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = fetcher.fetch_ohlcv(sym, start, end)
            if not df.empty:
                results[sym] = df
        return results

    target = universe or "nifty50"
    return fetcher.fetch_universe(target, start, end)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fetcher = IndianMarketDataFetcher()

    print("=== Fetching NIFTY 50 (first 5) ===")
    df = fetcher.fetch_ohlcv("RELIANCE.NS", start="2024-01-01")
    if not df.empty:
        print(df.tail(5))

    print("\n=== Fetching NIFTY 50 index ===")
    idx = fetcher.fetch_index("NIFTY50", start="2024-01-01")
    if not idx.empty:
        print(idx.tail(5))

    print("\n=== Fetching options chain for RELIANCE.NS ===")
    chains = fetcher.fetch_options_chain("RELIANCE.NS")
    for expiry, data in list(chains.items())[:1]:
        print(f"Expiry: {expiry}")
        print(f"  Calls: {len(data['calls'])} rows")
        print(f"  Puts:  {len(data['puts'])} rows")
