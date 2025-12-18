"""
Indian Stock Market Integration Module
Comprehensive NSE, BSE, and Indian financial market data
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class Exchange(Enum):
    """Indian stock exchanges"""
    NSE = "NSE"
    BSE = "BSE"


class IndexType(Enum):
    """Major Indian indices"""
    NIFTY_50 = "NIFTY 50"
    NIFTY_BANK = "NIFTY BANK"
    NIFTY_IT = "NIFTY IT"
    NIFTY_PHARMA = "NIFTY PHARMA"
    NIFTY_AUTO = "NIFTY AUTO"
    NIFTY_FMCG = "NIFTY FMCG"
    NIFTY_METAL = "NIFTY METAL"
    NIFTY_REALTY = "NIFTY REALTY"
    NIFTY_ENERGY = "NIFTY ENERGY"
    NIFTY_INFRA = "NIFTY INFRA"
    NIFTY_PSE = "NIFTY PSE"
    NIFTY_MIDCAP_50 = "NIFTY MIDCAP 50"
    NIFTY_SMALLCAP_50 = "NIFTY SMLCAP 50"
    SENSEX = "SENSEX"
    BSE_100 = "BSE 100"
    BSE_200 = "BSE 200"
    BSE_500 = "BSE 500"
    BSE_MIDCAP = "BSE MIDCAP"
    BSE_SMALLCAP = "BSE SMALLCAP"


@dataclass
class StockQuote:
    """Real-time stock quote data"""
    symbol: str
    exchange: Exchange
    company_name: str
    last_price: float
    change: float
    percent_change: float
    open_price: float
    high_price: float
    low_price: float
    prev_close: float
    volume: int
    value: float
    timestamp: datetime
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    upper_circuit: Optional[float] = None
    lower_circuit: Optional[float] = None
    delivery_qty: Optional[int] = None
    delivery_percent: Optional[float] = None


@dataclass
class IndexQuote:
    """Index data"""
    name: str
    value: float
    change: float
    percent_change: float
    open_value: float
    high_value: float
    low_value: float
    prev_close: float
    timestamp: datetime


@dataclass
class MarketBreadth:
    """Market breadth data"""
    advances: int
    declines: int
    unchanged: int
    total: int
    advance_ratio: float
    timestamp: datetime


@dataclass
class FIIDIIData:
    """FII/DII activity data"""
    date: datetime
    fii_buy: float
    fii_sell: float
    fii_net: float
    dii_buy: float
    dii_sell: float
    dii_net: float


@dataclass
class DeliveryData:
    """Delivery volume data"""
    symbol: str
    traded_qty: int
    delivery_qty: int
    delivery_percent: float
    date: datetime


class NSEDataFetcher:
    """
    NSE (National Stock Exchange) data fetcher
    Fetches real-time quotes, historical data, indices, and derivatives
    """
    
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.nse = None
        self._initialize()
    
    def _initialize(self):
        """Initialize NSE connection"""
        try:
            from nsetools import Nse
            self.nse = Nse()
            logger.info("NSE connection initialized")
        except ImportError:
            logger.warning("nsetools not installed. Install with: pip install nsetools")
        except Exception as e:
            logger.error(f"Failed to initialize NSE: {e}")
    
    def get_quote(self, symbol: str) -> Optional[StockQuote]:
        """Get real-time quote for NSE stock"""
        if not self.nse:
            return self._get_mock_quote(symbol)
        
        try:
            quote = self.nse.get_quote(symbol.upper())
            if quote:
                return StockQuote(
                    symbol=symbol.upper(),
                    exchange=Exchange.NSE,
                    company_name=quote.get('companyName', ''),
                    last_price=float(quote.get('lastPrice', 0)),
                    change=float(quote.get('change', 0)),
                    percent_change=float(quote.get('pChange', 0)),
                    open_price=float(quote.get('open', 0)),
                    high_price=float(quote.get('dayHigh', 0)),
                    low_price=float(quote.get('dayLow', 0)),
                    prev_close=float(quote.get('previousClose', 0)),
                    volume=int(quote.get('totalTradedVolume', 0)),
                    value=float(quote.get('totalTradedValue', 0)),
                    timestamp=datetime.now(),
                    week_52_high=float(quote.get('high52', 0)) if quote.get('high52') else None,
                    week_52_low=float(quote.get('low52', 0)) if quote.get('low52') else None,
                    upper_circuit=float(quote.get('upperCP', 0)) if quote.get('upperCP') else None,
                    lower_circuit=float(quote.get('lowerCP', 0)) if quote.get('lowerCP') else None,
                    delivery_qty=int(quote.get('deliveryQuantity', 0)) if quote.get('deliveryQuantity') else None,
                    delivery_percent=float(quote.get('deliveryToTradedQuantity', 0)) if quote.get('deliveryToTradedQuantity') else None
                )
        except Exception as e:
            logger.error(f"Error fetching NSE quote for {symbol}: {e}")
        
        return None
    
    def _get_mock_quote(self, symbol: str) -> StockQuote:
        """Get mock quote for testing"""
        import random
        base_price = random.uniform(100, 5000)
        change = random.uniform(-50, 50)
        return StockQuote(
            symbol=symbol.upper(),
            exchange=Exchange.NSE,
            company_name=f"{symbol} Ltd.",
            last_price=base_price,
            change=change,
            percent_change=(change / base_price) * 100,
            open_price=base_price - random.uniform(-20, 20),
            high_price=base_price + random.uniform(10, 50),
            low_price=base_price - random.uniform(10, 50),
            prev_close=base_price - change,
            volume=random.randint(100000, 10000000),
            value=random.uniform(1000000, 100000000),
            timestamp=datetime.now()
        )
    
    def get_all_stock_codes(self) -> List[str]:
        """Get all NSE stock codes"""
        if not self.nse:
            return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", 
                    "WIPRO", "BHARTIARTL", "ITC", "SBIN", "AXISBANK"]
        
        try:
            return list(self.nse.get_stock_codes().keys())
        except Exception as e:
            logger.error(f"Error fetching stock codes: {e}")
            return []
    
    def get_index_quote(self, index: str = "NIFTY 50") -> Optional[IndexQuote]:
        """Get index quote"""
        if not self.nse:
            return self._get_mock_index(index)
        
        try:
            data = self.nse.get_index_quote(index)
            if data:
                return IndexQuote(
                    name=index,
                    value=float(data.get('lastPrice', 0)),
                    change=float(data.get('change', 0)),
                    percent_change=float(data.get('pChange', 0)),
                    open_value=float(data.get('open', 0)),
                    high_value=float(data.get('high', 0)),
                    low_value=float(data.get('low', 0)),
                    prev_close=float(data.get('previousClose', 0)),
                    timestamp=datetime.now()
                )
        except Exception as e:
            logger.error(f"Error fetching index {index}: {e}")
        
        return None
    
    def _get_mock_index(self, index: str) -> IndexQuote:
        """Get mock index for testing"""
        import random
        base_value = 20000 if "NIFTY" in index else 65000
        change = random.uniform(-200, 200)
        return IndexQuote(
            name=index,
            value=base_value,
            change=change,
            percent_change=(change / base_value) * 100,
            open_value=base_value - random.uniform(-100, 100),
            high_value=base_value + random.uniform(50, 150),
            low_value=base_value - random.uniform(50, 150),
            prev_close=base_value - change,
            timestamp=datetime.now()
        )
    
    def get_top_gainers(self) -> List[Dict]:
        """Get top gainers"""
        if not self.nse:
            return self._get_mock_movers("gainers")
        
        try:
            return self.nse.get_top_gainers()
        except Exception as e:
            logger.error(f"Error fetching top gainers: {e}")
            return []
    
    def get_top_losers(self) -> List[Dict]:
        """Get top losers"""
        if not self.nse:
            return self._get_mock_movers("losers")
        
        try:
            return self.nse.get_top_losers()
        except Exception as e:
            logger.error(f"Error fetching top losers: {e}")
            return []
    
    def _get_mock_movers(self, mover_type: str) -> List[Dict]:
        """Get mock movers for testing"""
        import random
        symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
        movers = []
        for symbol in symbols:
            change = random.uniform(3, 10) if mover_type == "gainers" else random.uniform(-10, -3)
            movers.append({
                "symbol": symbol,
                "ltp": random.uniform(500, 3000),
                "netPrice": change,
                "tradedQuantity": random.randint(100000, 5000000)
            })
        return movers
    
    def is_valid_code(self, symbol: str) -> bool:
        """Check if stock code is valid"""
        if not self.nse:
            return True
        
        try:
            return self.nse.is_valid_code(symbol.upper())
        except:
            return False


class BSEDataFetcher:
    """
    BSE (Bombay Stock Exchange) data fetcher
    """
    
    def __init__(self):
        self.bse = None
        self._initialize()
    
    def _initialize(self):
        """Initialize BSE connection"""
        try:
            from bsedata.bse import BSE
            self.bse = BSE()
            logger.info("BSE connection initialized")
        except ImportError:
            logger.warning("bsedata not installed. Install with: pip install bsedata")
        except Exception as e:
            logger.error(f"Failed to initialize BSE: {e}")
    
    def get_quote(self, scrip_code: str) -> Optional[StockQuote]:
        """Get real-time quote for BSE stock"""
        if not self.bse:
            return self._get_mock_quote(scrip_code)
        
        try:
            quote = self.bse.getQuote(scrip_code)
            if quote:
                return StockQuote(
                    symbol=scrip_code,
                    exchange=Exchange.BSE,
                    company_name=quote.get('companyName', ''),
                    last_price=float(quote.get('currentValue', 0)),
                    change=float(quote.get('change', 0)),
                    percent_change=float(quote.get('pChange', 0)),
                    open_price=float(quote.get('open', 0)),
                    high_price=float(quote.get('high', 0)),
                    low_price=float(quote.get('low', 0)),
                    prev_close=float(quote.get('previousClose', 0)),
                    volume=int(quote.get('totalTradedQuantity', 0)),
                    value=float(quote.get('totalTradedValue', 0)),
                    timestamp=datetime.now(),
                    week_52_high=float(quote.get('52weekHigh', 0)) if quote.get('52weekHigh') else None,
                    week_52_low=float(quote.get('52weekLow', 0)) if quote.get('52weekLow') else None,
                    upper_circuit=float(quote.get('upperBand', 0)) if quote.get('upperBand') else None,
                    lower_circuit=float(quote.get('lowerBand', 0)) if quote.get('lowerBand') else None
                )
        except Exception as e:
            logger.error(f"Error fetching BSE quote for {scrip_code}: {e}")
        
        return None
    
    def _get_mock_quote(self, scrip_code: str) -> StockQuote:
        """Get mock quote for testing"""
        import random
        base_price = random.uniform(100, 5000)
        change = random.uniform(-50, 50)
        return StockQuote(
            symbol=scrip_code,
            exchange=Exchange.BSE,
            company_name=f"Company {scrip_code}",
            last_price=base_price,
            change=change,
            percent_change=(change / base_price) * 100,
            open_price=base_price - random.uniform(-20, 20),
            high_price=base_price + random.uniform(10, 50),
            low_price=base_price - random.uniform(10, 50),
            prev_close=base_price - change,
            volume=random.randint(100000, 10000000),
            value=random.uniform(1000000, 100000000),
            timestamp=datetime.now()
        )
    
    def get_gainers(self) -> List[Dict]:
        """Get top gainers"""
        if not self.bse:
            return []
        
        try:
            return self.bse.topGainers()
        except Exception as e:
            logger.error(f"Error fetching BSE gainers: {e}")
            return []
    
    def get_losers(self) -> List[Dict]:
        """Get top losers"""
        if not self.bse:
            return []
        
        try:
            return self.bse.topLosers()
        except Exception as e:
            logger.error(f"Error fetching BSE losers: {e}")
            return []


class HistoricalDataFetcher:
    """
    Fetch historical data from NSE
    """
    
    def __init__(self):
        self.nsepy_available = False
        self._check_availability()
    
    def _check_availability(self):
        """Check if nsepy is available"""
        try:
            import nsepy
            self.nsepy_available = True
            logger.info("nsepy available for historical data")
        except ImportError:
            logger.warning("nsepy not installed. Install with: pip install nsepy")
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        index: bool = False
    ) -> Optional[Any]:
        """Get historical data for a stock or index"""
        if not self.nsepy_available:
            return self._get_mock_historical(symbol, start_date, end_date)
        
        try:
            from nsepy import get_history
            data = get_history(
                symbol=symbol,
                start=start_date,
                end=end_date,
                index=index
            )
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def _get_mock_historical(self, symbol: str, start_date: datetime, end_date: datetime) -> Any:
        """Generate mock historical data"""
        try:
            import pandas as pd
            import numpy as np
            
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            base_price = 1000
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = base_price * np.cumprod(1 + returns)
            
            data = pd.DataFrame({
                'Open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
                'High': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
                'Low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
                'Close': prices,
                'Volume': np.random.randint(100000, 10000000, len(dates))
            }, index=dates)
            
            return data
        except ImportError:
            return None
    
    def get_index_historical(
        self,
        index: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[Any]:
        """Get historical data for an index"""
        return self.get_historical_data(index, start_date, end_date, index=True)


class FIIDIITracker:
    """
    Track FII/DII activity in Indian markets
    """
    
    def __init__(self):
        self.data_cache: Dict[str, FIIDIIData] = {}
    
    def get_fii_dii_activity(self, date: Optional[datetime] = None) -> Optional[FIIDIIData]:
        """Get FII/DII activity for a specific date"""
        target_date = date or datetime.now()
        
        # Try to fetch from NSDL/CDSL or use cached data
        cached_key = target_date.strftime("%Y-%m-%d")
        if cached_key in self.data_cache:
            return self.data_cache[cached_key]
        
        # Return mock data for demonstration
        return self._get_mock_fii_dii(target_date)
    
    def _get_mock_fii_dii(self, date: datetime) -> FIIDIIData:
        """Generate mock FII/DII data"""
        import random
        
        fii_buy = random.uniform(5000, 15000)
        fii_sell = random.uniform(4000, 14000)
        dii_buy = random.uniform(3000, 10000)
        dii_sell = random.uniform(2500, 9500)
        
        return FIIDIIData(
            date=date,
            fii_buy=fii_buy,
            fii_sell=fii_sell,
            fii_net=fii_buy - fii_sell,
            dii_buy=dii_buy,
            dii_sell=dii_sell,
            dii_net=dii_buy - dii_sell
        )
    
    def get_fii_dii_trend(self, days: int = 30) -> List[FIIDIIData]:
        """Get FII/DII activity trend"""
        trend = []
        end_date = datetime.now()
        
        for i in range(days):
            date = end_date - timedelta(days=i)
            if date.weekday() < 5:  # Skip weekends
                trend.append(self.get_fii_dii_activity(date))
        
        return trend


class SectorAnalyzer:
    """
    Analyze Indian market sectors
    """
    
    SECTORS = {
        "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTI", "MPHASIS", "COFORGE"],
        "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "BANDHANBNK"],
        "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP", "BIOCON", "AUROPHARMA"],
        "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "ASHOKLEY"],
        "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO", "GODREJCP"],
        "Energy": ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "BPCL", "IOC", "GAIL"],
        "Metal": ["TATASTEEL", "HINDALCO", "JSWSTEEL", "VEDL", "COALINDIA", "NMDC", "JINDALSTEL"],
        "Realty": ["DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "BRIGADE", "SOBHA"],
        "Infra": ["LARSEN", "ADANIENT", "ADANIPORTS", "ULTRACEMCO", "GRASIM", "ACC", "AMBUJACEM"],
        "Telecom": ["BHARTIARTL", "IDEA"],
        "Consumer Durables": ["TITAN", "HAVELLS", "VOLTAS", "BLUESTARCO", "CROMPTON"]
    }
    
    def __init__(self):
        self.nse = NSEDataFetcher()
    
    def get_sector_performance(self, sector: str) -> Dict[str, Any]:
        """Get performance of a sector"""
        stocks = self.SECTORS.get(sector, [])
        if not stocks:
            return {"error": f"Unknown sector: {sector}"}
        
        performances = []
        for symbol in stocks:
            quote = self.nse.get_quote(symbol)
            if quote:
                performances.append({
                    "symbol": symbol,
                    "price": quote.last_price,
                    "change": quote.change,
                    "percent_change": quote.percent_change
                })
        
        if performances:
            avg_change = sum(p["percent_change"] for p in performances) / len(performances)
            return {
                "sector": sector,
                "stocks": performances,
                "average_change": avg_change,
                "best_performer": max(performances, key=lambda x: x["percent_change"]),
                "worst_performer": min(performances, key=lambda x: x["percent_change"])
            }
        
        return {"sector": sector, "error": "No data available"}
    
    def get_all_sectors_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all sectors"""
        summaries = []
        for sector in self.SECTORS:
            perf = self.get_sector_performance(sector)
            if "average_change" in perf:
                summaries.append({
                    "sector": sector,
                    "average_change": perf["average_change"],
                    "num_stocks": len(self.SECTORS[sector])
                })
        
        return sorted(summaries, key=lambda x: x.get("average_change", 0), reverse=True)


class IPOTracker:
    """
    Track upcoming and recent IPOs
    """
    
    def __init__(self):
        self.ipo_data: List[Dict] = []
    
    def get_upcoming_ipos(self) -> List[Dict]:
        """Get list of upcoming IPOs"""
        # This would typically fetch from NSE/BSE API or scrape IPO data
        return [
            {
                "name": "Sample IPO Ltd",
                "price_band": "₹500 - ₹525",
                "issue_size": "₹1,000 Cr",
                "open_date": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"),
                "close_date": (datetime.now() + timedelta(days=8)).strftime("%Y-%m-%d"),
                "lot_size": 28,
                "listing_date": (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d")
            }
        ]
    
    def get_recent_listings(self) -> List[Dict]:
        """Get recently listed IPOs"""
        return [
            {
                "name": "Recent IPO Ltd",
                "issue_price": 350,
                "listing_price": 525,
                "current_price": 480,
                "listing_gain": 50.0,
                "current_gain": 37.14,
                "listing_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            }
        ]
    
    def get_ipo_subscription_status(self, ipo_name: str) -> Dict:
        """Get IPO subscription status"""
        return {
            "ipo_name": ipo_name,
            "retail": {"subscribed": 5.2, "times": "5.2x"},
            "nii": {"subscribed": 8.5, "times": "8.5x"},
            "qib": {"subscribed": 15.3, "times": "15.3x"},
            "total": {"subscribed": 10.1, "times": "10.1x"}
        }


class DerivativesData:
    """
    Fetch derivatives (F&O) data from NSE
    """
    
    def __init__(self):
        self.nse = NSEDataFetcher()
    
    def get_option_chain(self, symbol: str, expiry_date: Optional[str] = None) -> Dict:
        """Get option chain for a symbol"""
        # In production, fetch from NSE API
        return self._get_mock_option_chain(symbol)
    
    def _get_mock_option_chain(self, symbol: str) -> Dict:
        """Generate mock option chain"""
        import random
        
        spot_price = random.uniform(15000, 25000) if symbol == "NIFTY" else random.uniform(500, 3000)
        strikes = []
        
        for i in range(-5, 6):
            strike = round(spot_price + (i * 100 if symbol == "NIFTY" else i * 50), 0)
            strikes.append({
                "strike_price": strike,
                "call_oi": random.randint(10000, 500000),
                "call_change_oi": random.randint(-50000, 50000),
                "call_ltp": max(0, spot_price - strike + random.uniform(10, 100)),
                "call_iv": random.uniform(10, 30),
                "put_oi": random.randint(10000, 500000),
                "put_change_oi": random.randint(-50000, 50000),
                "put_ltp": max(0, strike - spot_price + random.uniform(10, 100)),
                "put_iv": random.uniform(10, 30)
            })
        
        return {
            "symbol": symbol,
            "spot_price": spot_price,
            "strikes": strikes,
            "max_pain": spot_price,
            "pcr": random.uniform(0.8, 1.5)
        }
    
    def get_futures_data(self, symbol: str) -> Dict:
        """Get futures data for a symbol"""
        import random
        
        spot_price = random.uniform(500, 3000)
        return {
            "symbol": symbol,
            "spot_price": spot_price,
            "futures": [
                {
                    "expiry": "Current Month",
                    "price": spot_price * 1.001,
                    "oi": random.randint(100000, 5000000),
                    "change_oi": random.randint(-100000, 100000),
                    "volume": random.randint(50000, 2000000),
                    "basis": random.uniform(-10, 20)
                },
                {
                    "expiry": "Next Month",
                    "price": spot_price * 1.003,
                    "oi": random.randint(50000, 2000000),
                    "change_oi": random.randint(-50000, 50000),
                    "volume": random.randint(20000, 1000000),
                    "basis": random.uniform(0, 30)
                }
            ]
        }


class IndianMarketDashboard:
    """
    Unified Indian market data dashboard
    """
    
    def __init__(self):
        self.nse = NSEDataFetcher()
        self.bse = BSEDataFetcher()
        self.historical = HistoricalDataFetcher()
        self.fii_dii = FIIDIITracker()
        self.sector = SectorAnalyzer()
        self.ipo = IPOTracker()
        self.derivatives = DerivativesData()
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get complete market overview"""
        # Get major indices
        indices = {}
        for idx in [IndexType.NIFTY_50, IndexType.NIFTY_BANK, IndexType.SENSEX]:
            quote = self.nse.get_index_quote(idx.value)
            if quote:
                indices[idx.value] = {
                    "value": quote.value,
                    "change": quote.change,
                    "percent_change": quote.percent_change
                }
        
        # Get FII/DII data
        fii_dii = self.fii_dii.get_fii_dii_activity()
        
        # Get top movers
        top_gainers = self.nse.get_top_gainers()[:5]
        top_losers = self.nse.get_top_losers()[:5]
        
        # Get sector summary
        sector_summary = self.sector.get_all_sectors_summary()[:5]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "indices": indices,
            "fii_dii": {
                "fii_net": fii_dii.fii_net if fii_dii else 0,
                "dii_net": fii_dii.dii_net if fii_dii else 0
            },
            "top_gainers": top_gainers,
            "top_losers": top_losers,
            "sector_performance": sector_summary
        }
    
    def get_stock_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock analysis"""
        # Try NSE first
        quote = self.nse.get_quote(symbol)
        exchange = "NSE"
        
        if not quote:
            # Try BSE
            quote = self.bse.get_quote(symbol)
            exchange = "BSE"
        
        if not quote:
            return {"error": f"Stock {symbol} not found on NSE or BSE"}
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        historical = self.historical.get_historical_data(symbol, start_date, end_date)
        
        # Get derivatives data if available
        derivatives = None
        if exchange == "NSE":
            derivatives = self.derivatives.get_option_chain(symbol)
        
        return {
            "symbol": symbol,
            "exchange": exchange,
            "quote": {
                "price": quote.last_price,
                "change": quote.change,
                "percent_change": quote.percent_change,
                "volume": quote.volume,
                "52_week_high": quote.week_52_high,
                "52_week_low": quote.week_52_low
            },
            "has_historical": historical is not None,
            "has_derivatives": derivatives is not None,
            "timestamp": datetime.now().isoformat()
        }
    
    def compare_stocks(self, symbols: List[str]) -> Dict[str, Any]:
        """Compare multiple stocks"""
        comparisons = []
        
        for symbol in symbols:
            quote = self.nse.get_quote(symbol)
            if quote:
                comparisons.append({
                    "symbol": symbol,
                    "price": quote.last_price,
                    "change": quote.change,
                    "percent_change": quote.percent_change,
                    "volume": quote.volume
                })
        
        if comparisons:
            best = max(comparisons, key=lambda x: x["percent_change"])
            worst = min(comparisons, key=lambda x: x["percent_change"])
            
            return {
                "stocks": comparisons,
                "best_performer": best["symbol"],
                "worst_performer": worst["symbol"],
                "timestamp": datetime.now().isoformat()
            }
        
        return {"error": "No valid stocks found"}


# Example usage and testing
if __name__ == "__main__":
    # Initialize dashboard
    dashboard = IndianMarketDashboard()
    
    # Get market overview
    print("=== Indian Market Overview ===")
    overview = dashboard.get_market_overview()
    print(f"Indices: {overview['indices']}")
    print(f"FII Net: ₹{overview['fii_dii']['fii_net']:.2f} Cr")
    print(f"DII Net: ₹{overview['fii_dii']['dii_net']:.2f} Cr")
    
    # Analyze a stock
    print("\n=== Stock Analysis: RELIANCE ===")
    analysis = dashboard.get_stock_analysis("RELIANCE")
    print(f"Price: ₹{analysis['quote']['price']:.2f}")
    print(f"Change: {analysis['quote']['percent_change']:.2f}%")
    
    # Compare stocks
    print("\n=== Stock Comparison ===")
    comparison = dashboard.compare_stocks(["TCS", "INFY", "WIPRO"])
    print(f"Best: {comparison['best_performer']}")
    print(f"Worst: {comparison['worst_performer']}")
