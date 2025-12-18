"""
Enhanced Market Data & Quotes Module
=====================================

Real-time/delayed quotes, multi-asset support, market depth, and corporate actions
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Supported asset classes"""
    EQUITY = "equity"
    OPTIONS = "options"
    FUTURES = "futures"
    CRYPTO = "crypto"
    FOREX = "forex"
    ETF = "etf"
    MUTUAL_FUND = "mutual_fund"
    BOND = "bond"
    COMMODITY = "commodity"
    INDEX = "index"


class TimeFrame(Enum):
    """Chart timeframes"""
    TICK = "tick"
    SECOND_1 = "1s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


class CorporateActionType(Enum):
    """Types of corporate actions"""
    DIVIDEND = "dividend"
    STOCK_SPLIT = "stock_split"
    REVERSE_SPLIT = "reverse_split"
    BONUS = "bonus"
    RIGHTS = "rights"
    MERGER = "merger"
    ACQUISITION = "acquisition"
    SPINOFF = "spinoff"
    DELISTING = "delisting"
    IPO = "ipo"


@dataclass
class Quote:
    """Real-time quote data"""
    symbol: str
    asset_class: AssetClass
    last_price: float
    change: float
    change_percent: float
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    volume: int
    avg_volume: int
    open_price: float
    high_price: float
    low_price: float
    prev_close: float
    timestamp: datetime
    exchange: str
    # Extended quote data
    vwap: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None


@dataclass
class Level2Quote:
    """Market depth / Level 2 data"""
    symbol: str
    timestamp: datetime
    bids: List[Tuple[float, int]]  # List of (price, size) tuples
    asks: List[Tuple[float, int]]
    total_bid_volume: int
    total_ask_volume: int
    spread: float
    spread_percent: float


@dataclass
class OHLCV:
    """OHLCV candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    trades: Optional[int] = None


@dataclass
class CorporateAction:
    """Corporate action event"""
    symbol: str
    action_type: CorporateActionType
    ex_date: datetime
    record_date: Optional[datetime]
    payment_date: Optional[datetime]
    description: str
    value: Optional[float] = None  # Dividend amount, split ratio, etc.
    currency: str = "USD"


class MarketDataProvider(ABC):
    """Abstract base class for market data providers"""
    
    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[Quote]:
        pass
    
    @abstractmethod
    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        pass
    
    @abstractmethod
    def get_historical(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime
    ) -> List[OHLCV]:
        pass


class UnifiedMarketData:
    """
    Unified market data interface supporting multiple providers and asset classes
    """
    
    def __init__(self):
        self.providers: Dict[str, Any] = {}
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 60  # seconds
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize data providers"""
        # Try OpenBB first
        try:
            from data.openbb_integration import OpenBBIntegration
            self.providers['openbb'] = OpenBBIntegration()
            logger.info("OpenBB provider initialized")
        except Exception as e:
            logger.warning(f"OpenBB not available: {e}")
        
        # yfinance fallback
        try:
            import yfinance as yf
            self.providers['yfinance'] = yf
            logger.info("yfinance provider initialized")
        except ImportError:
            logger.warning("yfinance not available")
        
        # Indian markets
        try:
            from data.indian_markets import NSEDataFetcher, BSEDataFetcher
            self.providers['nse'] = NSEDataFetcher()
            self.providers['bse'] = BSEDataFetcher()
            logger.info("Indian market providers initialized")
        except Exception as e:
            logger.warning(f"Indian market providers not available: {e}")
    
    def get_quote(self, symbol: str, asset_class: AssetClass = AssetClass.EQUITY) -> Optional[Quote]:
        """Get real-time quote for any asset"""
        cache_key = f"quote_{symbol}_{asset_class.value}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_data
        
        quote = None
        
        if asset_class == AssetClass.EQUITY:
            quote = self._get_equity_quote(symbol)
        elif asset_class == AssetClass.CRYPTO:
            quote = self._get_crypto_quote(symbol)
        elif asset_class == AssetClass.FOREX:
            quote = self._get_forex_quote(symbol)
        elif asset_class == AssetClass.OPTIONS:
            quote = self._get_options_quote(symbol)
        elif asset_class == AssetClass.FUTURES:
            quote = self._get_futures_quote(symbol)
        
        if quote:
            self.cache[cache_key] = (quote, datetime.now())
        
        return quote
    
    def _get_equity_quote(self, symbol: str) -> Optional[Quote]:
        """Get equity quote"""
        if 'yfinance' in self.providers:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                return Quote(
                    symbol=symbol,
                    asset_class=AssetClass.EQUITY,
                    last_price=info.get('regularMarketPrice', 0),
                    change=info.get('regularMarketChange', 0),
                    change_percent=info.get('regularMarketChangePercent', 0),
                    bid=info.get('bid', 0),
                    ask=info.get('ask', 0),
                    bid_size=info.get('bidSize', 0),
                    ask_size=info.get('askSize', 0),
                    volume=info.get('regularMarketVolume', 0),
                    avg_volume=info.get('averageVolume', 0),
                    open_price=info.get('regularMarketOpen', 0),
                    high_price=info.get('regularMarketDayHigh', 0),
                    low_price=info.get('regularMarketDayLow', 0),
                    prev_close=info.get('regularMarketPreviousClose', 0),
                    timestamp=datetime.now(),
                    exchange=info.get('exchange', ''),
                    week_52_high=info.get('fiftyTwoWeekHigh'),
                    week_52_low=info.get('fiftyTwoWeekLow'),
                    market_cap=info.get('marketCap'),
                    pe_ratio=info.get('trailingPE'),
                    eps=info.get('trailingEps'),
                    dividend_yield=info.get('dividendYield'),
                    beta=info.get('beta')
                )
            except Exception as e:
                logger.error(f"Error fetching equity quote for {symbol}: {e}")
        
        return self._get_mock_quote(symbol, AssetClass.EQUITY)
    
    def _get_crypto_quote(self, symbol: str) -> Optional[Quote]:
        """Get cryptocurrency quote"""
        try:
            from pycoingecko import CoinGeckoAPI
            cg = CoinGeckoAPI()
            
            # Map common symbols to CoinGecko IDs
            symbol_map = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'BNB': 'binancecoin',
                'SOL': 'solana',
                'XRP': 'ripple',
                'ADA': 'cardano',
                'DOGE': 'dogecoin',
                'DOT': 'polkadot'
            }
            
            coin_id = symbol_map.get(symbol.upper(), symbol.lower())
            data = cg.get_price(
                ids=coin_id,
                vs_currencies='usd',
                include_24hr_change=True,
                include_24hr_vol=True,
                include_market_cap=True
            )
            
            if coin_id in data:
                coin_data = data[coin_id]
                price = coin_data.get('usd', 0)
                change_24h = coin_data.get('usd_24h_change', 0)
                
                return Quote(
                    symbol=symbol.upper(),
                    asset_class=AssetClass.CRYPTO,
                    last_price=price,
                    change=price * change_24h / 100,
                    change_percent=change_24h,
                    bid=price * 0.999,
                    ask=price * 1.001,
                    bid_size=0,
                    ask_size=0,
                    volume=int(coin_data.get('usd_24h_vol', 0)),
                    avg_volume=0,
                    open_price=price - (price * change_24h / 100),
                    high_price=price * 1.02,
                    low_price=price * 0.98,
                    prev_close=price - (price * change_24h / 100),
                    timestamp=datetime.now(),
                    exchange='Crypto',
                    market_cap=coin_data.get('usd_market_cap')
                )
        except Exception as e:
            logger.error(f"Error fetching crypto quote for {symbol}: {e}")
        
        return self._get_mock_quote(symbol, AssetClass.CRYPTO)
    
    def _get_forex_quote(self, symbol: str) -> Optional[Quote]:
        """Get forex quote"""
        # Try yfinance for forex
        if 'yfinance' in self.providers:
            try:
                import yfinance as yf
                # Convert to yfinance format (e.g., EURUSD -> EURUSD=X)
                ticker_symbol = f"{symbol}=X" if not symbol.endswith('=X') else symbol
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info
                
                return Quote(
                    symbol=symbol,
                    asset_class=AssetClass.FOREX,
                    last_price=info.get('regularMarketPrice', 0),
                    change=info.get('regularMarketChange', 0),
                    change_percent=info.get('regularMarketChangePercent', 0),
                    bid=info.get('bid', 0),
                    ask=info.get('ask', 0),
                    bid_size=0,
                    ask_size=0,
                    volume=info.get('regularMarketVolume', 0),
                    avg_volume=0,
                    open_price=info.get('regularMarketOpen', 0),
                    high_price=info.get('regularMarketDayHigh', 0),
                    low_price=info.get('regularMarketDayLow', 0),
                    prev_close=info.get('regularMarketPreviousClose', 0),
                    timestamp=datetime.now(),
                    exchange='Forex'
                )
            except Exception as e:
                logger.error(f"Error fetching forex quote for {symbol}: {e}")
        
        return self._get_mock_quote(symbol, AssetClass.FOREX)
    
    def _get_options_quote(self, symbol: str) -> Optional[Quote]:
        """Get options quote"""
        return self._get_mock_quote(symbol, AssetClass.OPTIONS)
    
    def _get_futures_quote(self, symbol: str) -> Optional[Quote]:
        """Get futures quote"""
        return self._get_mock_quote(symbol, AssetClass.FUTURES)
    
    def _get_mock_quote(self, symbol: str, asset_class: AssetClass) -> Quote:
        """Generate mock quote for testing"""
        import random
        base_price = random.uniform(50, 500)
        change = random.uniform(-10, 10)
        
        return Quote(
            symbol=symbol,
            asset_class=asset_class,
            last_price=base_price,
            change=change,
            change_percent=(change / base_price) * 100,
            bid=base_price - 0.01,
            ask=base_price + 0.01,
            bid_size=random.randint(100, 10000),
            ask_size=random.randint(100, 10000),
            volume=random.randint(100000, 10000000),
            avg_volume=random.randint(500000, 5000000),
            open_price=base_price - random.uniform(-5, 5),
            high_price=base_price + random.uniform(1, 10),
            low_price=base_price - random.uniform(1, 10),
            prev_close=base_price - change,
            timestamp=datetime.now(),
            exchange='MOCK'
        )
    
    def get_level2(self, symbol: str, levels: int = 10) -> Optional[Level2Quote]:
        """Get Level 2 / Market Depth data"""
        import random
        
        # Generate mock Level 2 data
        base_price = random.uniform(100, 200)
        
        bids = []
        asks = []
        
        for i in range(levels):
            bid_price = base_price - (i * 0.01)
            ask_price = base_price + ((i + 1) * 0.01)
            bids.append((round(bid_price, 2), random.randint(100, 5000)))
            asks.append((round(ask_price, 2), random.randint(100, 5000)))
        
        total_bid_vol = sum(b[1] for b in bids)
        total_ask_vol = sum(a[1] for a in asks)
        
        return Level2Quote(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
            total_bid_volume=total_bid_vol,
            total_ask_volume=total_ask_vol,
            spread=asks[0][0] - bids[0][0],
            spread_percent=((asks[0][0] - bids[0][0]) / base_price) * 100
        )
    
    def get_historical(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.DAY_1,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        periods: int = 100
    ) -> List[OHLCV]:
        """Get historical OHLCV data"""
        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=periods)
        
        if 'yfinance' in self.providers:
            try:
                import yfinance as yf
                
                # Map timeframe to yfinance interval
                interval_map = {
                    TimeFrame.MINUTE_1: '1m',
                    TimeFrame.MINUTE_5: '5m',
                    TimeFrame.MINUTE_15: '15m',
                    TimeFrame.MINUTE_30: '30m',
                    TimeFrame.HOUR_1: '1h',
                    TimeFrame.DAY_1: '1d',
                    TimeFrame.WEEK_1: '1wk',
                    TimeFrame.MONTH_1: '1mo'
                }
                
                interval = interval_map.get(timeframe, '1d')
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end, interval=interval)
                
                candles = []
                for idx, row in df.iterrows():
                    candles.append(OHLCV(
                        timestamp=idx.to_pydatetime(),
                        open=row['Open'],
                        high=row['High'],
                        low=row['Low'],
                        close=row['Close'],
                        volume=int(row['Volume'])
                    ))
                
                return candles
            except Exception as e:
                logger.error(f"Error fetching historical data for {symbol}: {e}")
        
        return self._generate_mock_history(symbol, start, end, timeframe)
    
    def _generate_mock_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame
    ) -> List[OHLCV]:
        """Generate mock historical data"""
        import numpy as np
        
        # Determine number of periods based on timeframe
        days = (end - start).days
        
        if timeframe == TimeFrame.DAY_1:
            periods = days
        elif timeframe == TimeFrame.HOUR_1:
            periods = days * 24
        elif timeframe == TimeFrame.MINUTE_5:
            periods = days * 24 * 12
        else:
            periods = days
        
        periods = min(periods, 1000)  # Cap at 1000 periods
        
        base_price = 100
        returns = np.random.normal(0.0005, 0.02, periods)
        prices = base_price * np.cumprod(1 + returns)
        
        candles = []
        current_time = start
        
        for i, close in enumerate(prices):
            volatility = np.random.uniform(0.01, 0.03)
            open_price = close * (1 + np.random.uniform(-volatility, volatility))
            high = max(open_price, close) * (1 + np.random.uniform(0, volatility))
            low = min(open_price, close) * (1 - np.random.uniform(0, volatility))
            
            candles.append(OHLCV(
                timestamp=current_time,
                open=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close, 2),
                volume=np.random.randint(100000, 10000000)
            ))
            
            # Increment time based on timeframe
            if timeframe == TimeFrame.DAY_1:
                current_time += timedelta(days=1)
            elif timeframe == TimeFrame.HOUR_1:
                current_time += timedelta(hours=1)
            elif timeframe == TimeFrame.MINUTE_5:
                current_time += timedelta(minutes=5)
            else:
                current_time += timedelta(days=1)
        
        return candles
    
    def get_corporate_actions(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[CorporateAction]:
        """Get corporate actions for a symbol"""
        if 'yfinance' in self.providers:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                
                actions = []
                
                # Get dividends
                dividends = ticker.dividends
                if not dividends.empty:
                    for date, amount in dividends.items():
                        actions.append(CorporateAction(
                            symbol=symbol,
                            action_type=CorporateActionType.DIVIDEND,
                            ex_date=date.to_pydatetime(),
                            record_date=None,
                            payment_date=None,
                            description=f"Cash Dividend: ${amount:.4f}",
                            value=amount,
                            currency="USD"
                        ))
                
                # Get splits
                splits = ticker.splits
                if not splits.empty:
                    for date, ratio in splits.items():
                        actions.append(CorporateAction(
                            symbol=symbol,
                            action_type=CorporateActionType.STOCK_SPLIT,
                            ex_date=date.to_pydatetime(),
                            record_date=None,
                            payment_date=None,
                            description=f"Stock Split: {ratio:.0f}:1",
                            value=ratio
                        ))
                
                # Filter by date range
                if start:
                    actions = [a for a in actions if a.ex_date >= start]
                if end:
                    actions = [a for a in actions if a.ex_date <= end]
                
                return sorted(actions, key=lambda x: x.ex_date, reverse=True)
            except Exception as e:
                logger.error(f"Error fetching corporate actions for {symbol}: {e}")
        
        return []
    
    def get_multi_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple symbols"""
        quotes = {}
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote:
                quotes[symbol] = quote
        return quotes


class MarketDataStreamer:
    """
    Real-time market data streaming using websockets
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[callable]] = {}
        self.is_running = False
    
    def subscribe(self, symbol: str, callback: callable):
        """Subscribe to real-time updates for a symbol"""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        self.subscribers[symbol].append(callback)
    
    def unsubscribe(self, symbol: str, callback: callable):
        """Unsubscribe from updates"""
        if symbol in self.subscribers:
            self.subscribers[symbol].remove(callback)
    
    async def start_streaming(self):
        """Start the streaming connection"""
        import asyncio
        
        self.is_running = True
        market_data = UnifiedMarketData()
        
        while self.is_running:
            for symbol, callbacks in self.subscribers.items():
                quote = market_data.get_quote(symbol)
                if quote:
                    for callback in callbacks:
                        try:
                            callback(quote)
                        except Exception as e:
                            logger.error(f"Error in callback for {symbol}: {e}")
            
            await asyncio.sleep(1)  # Update every second
    
    def stop_streaming(self):
        """Stop the streaming connection"""
        self.is_running = False


# Example usage
if __name__ == "__main__":
    # Initialize market data
    market = UnifiedMarketData()
    
    # Get equity quote
    print("=== Equity Quote ===")
    quote = market.get_quote("AAPL", AssetClass.EQUITY)
    if quote:
        print(f"{quote.symbol}: ${quote.last_price:.2f} ({quote.change_percent:+.2f}%)")
    
    # Get crypto quote
    print("\n=== Crypto Quote ===")
    crypto = market.get_quote("BTC", AssetClass.CRYPTO)
    if crypto:
        print(f"{crypto.symbol}: ${crypto.last_price:,.2f} ({crypto.change_percent:+.2f}%)")
    
    # Get Level 2 data
    print("\n=== Level 2 Data ===")
    l2 = market.get_level2("AAPL", levels=5)
    if l2:
        print(f"Spread: ${l2.spread:.4f} ({l2.spread_percent:.4f}%)")
        print(f"Top Bids: {l2.bids[:3]}")
        print(f"Top Asks: {l2.asks[:3]}")
    
    # Get historical data
    print("\n=== Historical Data ===")
    history = market.get_historical("AAPL", TimeFrame.DAY_1, periods=5)
    for candle in history[:5]:
        print(f"{candle.timestamp.date()}: O={candle.open:.2f} H={candle.high:.2f} "
              f"L={candle.low:.2f} C={candle.close:.2f}")
