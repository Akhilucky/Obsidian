"""
Data Ingest Module - OpenBB + Multi-Source Integration
=======================================================
700+ free feeds (stocks, options, crypto, macro) → normalized parquet files

Supports:
- OpenBB SDK for comprehensive market data
- Yahoo Finance as fallback
- FRED for macro data
- CoinGecko for crypto
- Options data from multiple sources
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data_warehouse"
DATA_DIR.mkdir(exist_ok=True)


class DataSource(Enum):
    """Available data sources."""
    YAHOO = "yahoo"
    OPENBB = "openbb"
    FRED = "fred"
    COINGECKO = "coingecko"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    TIINGO = "tiingo"
    NSE = "nse"
    BSE = "bse"


class AssetClass(Enum):
    """Asset classes."""
    EQUITY = "equity"
    OPTIONS = "options"
    CRYPTO = "crypto"
    FOREX = "forex"
    FUTURES = "futures"
    MACRO = "macro"
    FIXED_INCOME = "fixed_income"
    COMMODITIES = "commodities"


@dataclass
class DataConfig:
    """Configuration for data ingestion."""
    source: DataSource = DataSource.YAHOO
    asset_class: AssetClass = AssetClass.EQUITY
    symbols: List[str] = field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    interval: str = "1d"
    save_format: str = "parquet"


class DataNormalizer:
    """Normalize data from various sources to a common format."""
    
    STANDARD_COLUMNS = {
        'ohlcv': ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close'],
        'fundamentals': ['symbol', 'date', 'metric', 'value'],
        'macro': ['date', 'indicator', 'value', 'country'],
        'options': ['date', 'expiry', 'strike', 'type', 'bid', 'ask', 'volume', 'oi', 'iv'],
    }
    
    @staticmethod
    def normalize_ohlcv(df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Normalize OHLCV data to standard format."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Standardize column names
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume',
            'Adj Close': 'adj_close', 'Adj_Close': 'adj_close',
            'open': 'open', 'high': 'high', 'low': 'low',
            'close': 'close', 'volume': 'volume'
        }
        
        df.columns = [column_mapping.get(c, c.lower()) for c in df.columns]
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
        
        df.index.name = 'date'
        
        # Add adj_close if missing
        if 'adj_close' not in df.columns:
            df['adj_close'] = df['close']
        
        # Add source metadata
        df['source'] = source
        df['ingested_at'] = datetime.now()
        
        return df
    
    @staticmethod
    def normalize_fundamentals(data: Dict, symbol: str) -> pd.DataFrame:
        """Normalize fundamental data."""
        records = []
        for metric, value in data.items():
            if isinstance(value, (int, float)):
                records.append({
                    'symbol': symbol,
                    'date': datetime.now().date(),
                    'metric': metric,
                    'value': value
                })
        return pd.DataFrame(records)


class YahooDataFetcher:
    """Fetch data from Yahoo Finance."""
    
    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("yfinance not installed")
    
    def fetch_ohlcv(self, symbol: str, start: str = None, end: str = None, 
                   interval: str = "1d") -> pd.DataFrame:
        """Fetch OHLCV data."""
        if not self.available:
            return pd.DataFrame()
        
        try:
            ticker = self.yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval)
            if not df.empty:
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_info(self, symbol: str) -> Dict:
        """Fetch ticker info."""
        if not self.available:
            return {}
        
        try:
            ticker = self.yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}
    
    def fetch_options(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch options chain."""
        if not self.available:
            return {}
        
        try:
            ticker = self.yf.Ticker(symbol)
            dates = ticker.options
            chains = {}
            for date in dates[:3]:  # Limit to first 3 expiries
                opt = ticker.option_chain(date)
                chains[date] = {
                    'calls': opt.calls,
                    'puts': opt.puts
                }
            return chains
        except Exception as e:
            logger.error(f"Error fetching options for {symbol}: {e}")
            return {}


class FREDDataFetcher:
    """Fetch macro data from FRED."""
    
    MACRO_SERIES = {
        'GDP': 'GDP',
        'UNEMPLOYMENT': 'UNRATE',
        'INFLATION': 'CPIAUCSL',
        'FED_FUNDS': 'FEDFUNDS',
        'T10Y2Y': 'T10Y2Y',
        'VIX': 'VIXCLS',
        'REAL_GDP': 'GDPC1',
        'INDUSTRIAL_PROD': 'INDPRO',
        'HOUSING_STARTS': 'HOUST',
        'RETAIL_SALES': 'RSAFS',
        'M2': 'M2SL',
        'YIELD_10Y': 'DGS10',
        'YIELD_2Y': 'DGS2',
        'CORP_SPREAD': 'BAA10Y',
    }
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        try:
            import pandas_datareader as pdr
            self.pdr = pdr
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("pandas_datareader not installed")
    
    def fetch_series(self, series_id: str, start: str = None, end: str = None) -> pd.DataFrame:
        """Fetch a FRED series."""
        if not self.available:
            return pd.DataFrame()
        
        try:
            start = start or (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
            end = end or datetime.now().strftime('%Y-%m-%d')
            
            df = self.pdr.get_data_fred(series_id, start=start, end=end)
            df.columns = ['value']
            df['indicator'] = series_id
            return df
        except Exception as e:
            logger.error(f"Error fetching FRED series {series_id}: {e}")
            return pd.DataFrame()
    
    def fetch_all_macro(self, start: str = None, end: str = None) -> pd.DataFrame:
        """Fetch all macro indicators."""
        all_data = []
        for name, series_id in self.MACRO_SERIES.items():
            df = self.fetch_series(series_id, start, end)
            if not df.empty:
                df['name'] = name
                all_data.append(df)
        
        if all_data:
            return pd.concat(all_data)
        return pd.DataFrame()


class CryptoDataFetcher:
    """Fetch cryptocurrency data."""
    
    TOP_CRYPTOS = [
        'bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano',
        'ripple', 'polkadot', 'dogecoin', 'avalanche-2', 'chainlink'
    ]
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        try:
            import httpx
            self.httpx = httpx
            self.available = True
        except ImportError:
            try:
                import requests
                self.requests = requests
                self.httpx = None
                self.available = True
            except ImportError:
                self.available = False
    
    def fetch_price_history(self, coin_id: str, days: int = 365) -> pd.DataFrame:
        """Fetch price history for a coin."""
        if not self.available:
            return pd.DataFrame()
        
        try:
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {'vs_currency': 'usd', 'days': days}
            
            if self.httpx:
                response = self.httpx.get(url, params=params, timeout=30)
                data = response.json()
            else:
                response = self.requests.get(url, params=params, timeout=30)
                data = response.json()
            
            if 'prices' in data:
                df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('date', inplace=True)
                df['coin'] = coin_id
                return df[['price', 'coin']]
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching crypto {coin_id}: {e}")
            return pd.DataFrame()
    
    def fetch_all_prices(self) -> pd.DataFrame:
        """Fetch current prices for top cryptos."""
        if not self.available:
            return pd.DataFrame()
        
        try:
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': ','.join(self.TOP_CRYPTOS),
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_market_cap': 'true'
            }
            
            if self.httpx:
                response = self.httpx.get(url, params=params, timeout=30)
                data = response.json()
            else:
                response = self.requests.get(url, params=params, timeout=30)
                data = response.json()
            
            records = []
            for coin, info in data.items():
                records.append({
                    'coin': coin,
                    'price': info.get('usd', 0),
                    'change_24h': info.get('usd_24h_change', 0),
                    'market_cap': info.get('usd_market_cap', 0)
                })
            return pd.DataFrame(records)
        except Exception as e:
            logger.error(f"Error fetching crypto prices: {e}")
            return pd.DataFrame()


class DataIngestPipeline:
    """Main data ingestion pipeline."""
    
    # Universe definitions
    UNIVERSES = {
        'sp500_top50': [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'V', 'XOM', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'LLY', 'ABBV', 'MRK',
            'AVGO', 'PEP', 'KO', 'COST', 'TMO', 'MCD', 'WMT', 'CSCO', 'ACN', 'ABT',
            'CRM', 'DHR', 'LIN', 'ADBE', 'CMCSA', 'NKE', 'TXN', 'VZ', 'NEE', 'PM',
            'ORCL', 'AMD', 'INTC', 'QCOM', 'IBM', 'GE', 'CAT', 'BA', 'HON', 'UPS'
        ],
        'indices': ['^GSPC', '^IXIC', '^DJI', '^RUT', '^VIX', '^TNX'],
        'sectors': [
            'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLP', 'XLY', 'XLU', 'XLRE', 'XLB', 'XLC'
        ],
        'etfs': [
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO', 'EFA', 'EEM',
            'GLD', 'SLV', 'USO', 'TLT', 'HYG', 'LQD', 'AGG', 'BND'
        ],
        'crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD'],
        'forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X'],
        'nifty50': [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS',
            'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS',
            'SUNPHARMA.NS', 'TATAMOTORS.NS', 'WIPRO.NS', 'TITAN.NS', 'ULTRACEMCO.NS',
            'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'M&M.NS', 'JSWSTEEL.NS',
            'TATASTEEL.NS', 'ADANIENT.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS',
            'INDUSINDBK.NS', 'GRASIM.NS', 'HINDALCO.NS', 'DIVISLAB.NS',
            'DRREDDY.NS', 'EICHERMOT.NS', 'CIPLA.NS', 'APOLLOHOSP.NS',
            'TATACONSUM.NS', 'BPCL.NS', 'COALINDIA.NS',
        ],
        'sensex30': [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS',
            'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS',
            'SUNPHARMA.NS', 'TATAMOTORS.NS', 'WIPRO.NS', 'TITAN.NS', 'ULTRACEMCO.NS',
            'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'M&M.NS', 'JSWSTEEL.NS',
            'TATASTEEL.NS', 'ADANIENT.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS',
            'INDUSINDBK.NS',
        ],
        'nifty_it': [
            'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS',
            'MPHASIS.NS', 'LTTS.NS', 'PERSISTENT.NS', 'COFORGE.NS', 'INFOBIP.NS',
        ],
    }
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize fetchers
        self.yahoo = YahooDataFetcher()
        self.fred = FREDDataFetcher()
        self.crypto = CryptoDataFetcher()
        self.normalizer = DataNormalizer()
    
    def ingest_universe(self, universe: str, start: str = None, end: str = None,
                       save: bool = True) -> Dict[str, pd.DataFrame]:
        """Ingest data for a predefined universe."""
        if universe not in self.UNIVERSES:
            raise ValueError(f"Unknown universe: {universe}")
        
        symbols = self.UNIVERSES[universe]
        return self.ingest_symbols(symbols, start, end, save, universe)
    
    def ingest_symbols(self, symbols: List[str], start: str = None, end: str = None,
                      save: bool = True, category: str = "custom") -> Dict[str, pd.DataFrame]:
        """Ingest OHLCV data for multiple symbols."""
        start = start or (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        end = end or datetime.now().strftime('%Y-%m-%d')
        
        results = {}
        logger.info(f"Ingesting {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols):
            try:
                df = self.yahoo.fetch_ohlcv(symbol, start, end)
                if not df.empty:
                    df = self.normalizer.normalize_ohlcv(df, 'yahoo')
                    df['symbol'] = symbol
                    results[symbol] = df
                    
                    if save:
                        self._save_data(df, category, symbol)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i+1}/{len(symbols)}")
            except Exception as e:
                logger.error(f"Failed to ingest {symbol}: {e}")
        
        logger.info(f"Successfully ingested {len(results)}/{len(symbols)} symbols")
        return results
    
    def ingest_macro(self, save: bool = True) -> pd.DataFrame:
        """Ingest macro economic data."""
        logger.info("Ingesting macro data...")
        df = self.fred.fetch_all_macro()
        
        if not df.empty and save:
            path = self.data_dir / "macro" / "fred_indicators.parquet"
            path.parent.mkdir(exist_ok=True)
            df.to_parquet(path)
            logger.info(f"Saved macro data to {path}")
        
        return df
    
    def ingest_crypto(self, save: bool = True) -> pd.DataFrame:
        """Ingest cryptocurrency data."""
        logger.info("Ingesting crypto data...")
        df = self.crypto.fetch_all_prices()
        
        if not df.empty and save:
            path = self.data_dir / "crypto" / "prices.parquet"
            path.parent.mkdir(exist_ok=True)
            df.to_parquet(path)
            logger.info(f"Saved crypto data to {path}")
        
        return df

    def ingest_indian(self, universe: str = "nifty50", start: str = None, end: str = None,
                      save: bool = True) -> Dict[str, pd.DataFrame]:
        """Ingest Indian market (NSE/BSE) data via yfinance."""
        logger.info(f"Ingesting Indian market universe: {universe}")
        try:
            from data.indian_markets import IndianMarketDataFetcher
        except ImportError:
            logger.error("data.indian_markets module not available")
            return {}

        fetcher = IndianMarketDataFetcher(data_dir=self.data_dir)
        results = fetcher.fetch_universe(universe, start, end)

        if save and results:
            for sym, df in results.items():
                self._save_data(df, "indian", sym)

        logger.info(f"Indian ingest complete: {len(results)} symbols")
        return results
    
    def ingest_all(self, save: bool = True) -> Dict[str, Any]:
        """Run full data ingestion pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING FULL DATA INGESTION PIPELINE")
        logger.info("=" * 60)
        
        results = {}
        
        # Ingest all universes
        for universe in self.UNIVERSES:
            logger.info(f"\n--- Ingesting {universe} ---")
            results[universe] = self.ingest_universe(universe, save=save)
        
        # Ingest macro
        results['macro'] = self.ingest_macro(save=save)
        
        # Ingest crypto from CoinGecko
        results['crypto_live'] = self.ingest_crypto(save=save)
        
        # Create manifest
        manifest = {
            'ingested_at': datetime.now().isoformat(),
            'universes': {k: len(v) for k, v in results.items() if isinstance(v, dict)},
            'data_dir': str(self.data_dir)
        }
        
        manifest_path = self.data_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info("\n" + "=" * 60)
        logger.info("DATA INGESTION COMPLETE")
        logger.info("=" * 60)
        
        return results
    
    def _save_data(self, df: pd.DataFrame, category: str, symbol: str):
        """Save data to parquet file."""
        path = self.data_dir / category / f"{symbol.replace('^', 'IDX_')}.parquet"
        path.parent.mkdir(exist_ok=True)
        df.to_parquet(path)
    
    def load_data(self, category: str, symbol: str) -> pd.DataFrame:
        """Load data from parquet file."""
        path = self.data_dir / category / f"{symbol.replace('^', 'IDX_')}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()
    
    def get_combined_data(self, symbols: List[str], column: str = 'close') -> pd.DataFrame:
        """Get combined data for multiple symbols."""
        dfs = []
        for symbol in symbols:
            for category in ['sp500_top50', 'indices', 'sectors', 'etfs', 'crypto', 'forex', 'indian', 'indian_indices']:
                df = self.load_data(category, symbol)
                if not df.empty:
                    dfs.append(df[[column]].rename(columns={column: symbol}))
                    break
        
        if dfs:
            return pd.concat(dfs, axis=1)
        return pd.DataFrame()


# Convenience functions
def quick_ingest(symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Quick data ingestion for a list of symbols."""
    pipeline = DataIngestPipeline()
    if symbols:
        return pipeline.ingest_symbols(symbols)
    return pipeline.ingest_universe('sp500_top50')


def load_universe(universe: str) -> pd.DataFrame:
    """Load a pre-ingested universe."""
    pipeline = DataIngestPipeline()
    symbols = pipeline.UNIVERSES.get(universe, [])
    return pipeline.get_combined_data(symbols)


if __name__ == "__main__":
    # Run full ingestion
    pipeline = DataIngestPipeline()
    
    # Quick test with top 10 stocks
    print("Testing data ingestion with top 10 stocks...")
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'JNJ']
    results = pipeline.ingest_symbols(test_symbols, save=True, category="test")
    
    print(f"\nIngested {len(results)} symbols")
    for symbol, df in list(results.items())[:3]:
        print(f"\n{symbol}: {len(df)} rows")
        print(df.tail(3))
