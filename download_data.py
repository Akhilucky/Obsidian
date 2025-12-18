"""
Download Sample Data for Bloomberg Terminal
============================================
Downloads and caches market data for the dashboard.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data directories
DATA_DIR = Path(__file__).parent / "data_cache"
DATA_DIR.mkdir(exist_ok=True)


def download_market_data():
    """Download market data for all major assets."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return False
    
    logger.info("=" * 60)
    logger.info("DOWNLOADING MARKET DATA")
    logger.info("=" * 60)
    
    # Define universes
    universes = {
        'indices': ['^GSPC', '^IXIC', '^DJI', '^RUT', '^VIX', '^TNX'],
        'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B'],
        'tech': ['AMD', 'INTC', 'CRM', 'ADBE', 'ORCL', 'IBM', 'CSCO', 'QCOM'],
        'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW'],
        'healthcare': ['UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT'],
        'consumer': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE'],
        'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO'],
        'sectors': ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLP', 'XLY', 'XLU', 'XLRE', 'XLB', 'XLC'],
        'etfs': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'GLD', 'TLT', 'HYG'],
        'crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD'],
        'forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X'],
    }
    
    all_symbols = []
    for category, symbols in universes.items():
        all_symbols.extend(symbols)
    
    all_symbols = list(set(all_symbols))
    logger.info(f"Downloading data for {len(all_symbols)} symbols...")
    
    # Download all data at once
    try:
        data = yf.download(all_symbols, period="2y", progress=True, group_by='ticker')
    except Exception as e:
        logger.error(f"Bulk download failed: {e}")
        data = None
    
    # Save individual files
    success_count = 0
    for symbol in all_symbols:
        try:
            if data is not None and symbol in data.columns.get_level_values(0):
                df = data[symbol].copy()
                df = df.dropna()
            else:
                # Download individually
                df = yf.download(symbol, period="2y", progress=False)
            
            if not df.empty:
                # Normalize columns
                df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                df['symbol'] = symbol
                
                # Save
                safe_name = symbol.replace('^', 'IDX_').replace('=', '_')
                path = DATA_DIR / f"{safe_name}.parquet"
                df.to_parquet(path)
                success_count += 1
        except Exception as e:
            logger.warning(f"Failed to save {symbol}: {e}")
    
    logger.info(f"Successfully saved {success_count}/{len(all_symbols)} symbols")
    
    # Save universe definitions
    with open(DATA_DIR / "universes.json", 'w') as f:
        json.dump(universes, f, indent=2)
    
    # Save manifest
    manifest = {
        'downloaded_at': datetime.now().isoformat(),
        'symbols_count': success_count,
        'period': '2y',
        'universes': list(universes.keys())
    }
    with open(DATA_DIR / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return True


def download_fundamental_data():
    """Download fundamental data for stocks."""
    try:
        import yfinance as yf
    except ImportError:
        return False
    
    logger.info("\nDownloading fundamental data...")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'JNJ', 'V',
               'WMT', 'PG', 'XOM', 'UNH', 'HD', 'MA', 'BAC', 'CVX', 'ABBV', 'PFE']
    
    fundamentals = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            fundamentals.append({
                'symbol': symbol,
                'name': info.get('shortName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'profit_margin': info.get('profitMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'roe': info.get('returnOnEquity', 0),
                'roa': info.get('returnOnAssets', 0),
                'revenue': info.get('totalRevenue', 0),
                'gross_profit': info.get('grossProfits', 0),
                'ebitda': info.get('ebitda', 0),
                'free_cash_flow': info.get('freeCashflow', 0),
                'total_debt': info.get('totalDebt', 0),
                'total_cash': info.get('totalCash', 0),
                'beta': info.get('beta', 1),
                '52w_high': info.get('fiftyTwoWeekHigh', 0),
                '52w_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0),
            })
        except Exception as e:
            logger.warning(f"Failed to get fundamentals for {symbol}: {e}")
    
    if fundamentals:
        df = pd.DataFrame(fundamentals)
        df.to_parquet(DATA_DIR / "fundamentals.parquet", index=False)
        logger.info(f"Saved fundamentals for {len(fundamentals)} stocks")
    
    return True


def compute_features():
    """Compute features for cached data."""
    logger.info("\nComputing features...")
    
    try:
        import ta
    except ImportError:
        logger.warning("ta library not installed. Skipping feature computation.")
        return False
    
    parquet_files = list(DATA_DIR.glob("*.parquet"))
    
    for path in parquet_files:
        if path.name in ['fundamentals.parquet', 'features.parquet']:
            continue
        
        try:
            df = pd.read_parquet(path)
            
            if 'close' not in df.columns:
                continue
            
            # Calculate features
            df['return_1d'] = df['close'].pct_change(1)
            df['return_5d'] = df['close'].pct_change(5)
            df['return_20d'] = df['close'].pct_change(20)
            
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            df['volatility_20d'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
            
            # Technical indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            
            # Save
            df.to_parquet(path)
            
        except Exception as e:
            logger.warning(f"Failed to compute features for {path.name}: {e}")
    
    logger.info("Feature computation complete")
    return True


def generate_alpha_signals():
    """Generate alpha signals for the universe."""
    logger.info("\nGenerating alpha signals...")
    
    try:
        from core.signal_generator import AlphaTableGenerator
    except ImportError:
        logger.warning("Signal generator not available")
        return False
    
    # Load data
    data = {}
    for path in DATA_DIR.glob("*.parquet"):
        if path.name in ['fundamentals.parquet', 'alpha_signals.parquet', 'universes.json', 'manifest.json']:
            continue
        
        try:
            df = pd.read_parquet(path)
            if 'close' in df.columns and 'symbol' in df.columns:
                symbol = df['symbol'].iloc[0]
                data[symbol] = df
        except:
            pass
    
    if not data:
        logger.warning("No data available for signal generation")
        return False
    
    # Generate signals
    generator = AlphaTableGenerator()
    alpha_table = generator.generate_alpha_table(data)
    
    if not alpha_table.empty:
        alpha_table.to_parquet(DATA_DIR / "alpha_signals.parquet", index=False)
        logger.info(f"Generated signals for {len(alpha_table)} symbols")
    
    return True


def main():
    """Run full data download and processing."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              BLOOMBERG TERMINAL - DATA DOWNLOAD              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Download market data
    download_market_data()
    
    # Download fundamentals
    download_fundamental_data()
    
    # Compute features
    compute_features()
    
    # Generate signals
    generate_alpha_signals()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    DOWNLOAD COMPLETE!                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Summary
    parquet_count = len(list(DATA_DIR.glob("*.parquet")))
    print(f"    Data cached in: {DATA_DIR}")
    print(f"    Files created: {parquet_count}")
    print("")


if __name__ == "__main__":
    main()
