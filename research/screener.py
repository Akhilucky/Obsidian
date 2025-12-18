"""
Stock Screener & Discovery Module
==================================

Custom filters, sector analysis, top movers, and unusual activity detection
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class FilterOperator(Enum):
    """Filter comparison operators"""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="
    BETWEEN = "between"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"


class SortOrder(Enum):
    """Sort order"""
    ASC = "asc"
    DESC = "desc"


@dataclass
class ScreenerFilter:
    """Individual screener filter"""
    field: str
    operator: FilterOperator
    value: Any
    secondary_value: Optional[Any] = None  # For BETWEEN operator


@dataclass
class ScreenerResult:
    """Screener result for a stock"""
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    price: float
    change_percent: float
    volume: int
    pe_ratio: Optional[float]
    eps: Optional[float]
    dividend_yield: Optional[float]
    week_52_high: float
    week_52_low: float
    avg_volume: int
    beta: Optional[float]
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SectorPerformance:
    """Sector performance data"""
    sector: str
    change_1d: float
    change_1w: float
    change_1m: float
    change_ytd: float
    market_cap: float
    num_stocks: int
    top_performer: str
    worst_performer: str


@dataclass
class UnusualActivity:
    """Unusual activity alert"""
    symbol: str
    name: str
    activity_type: str  # volume_spike, price_surge, etc.
    current_value: float
    average_value: float
    deviation_percent: float
    timestamp: datetime
    description: str


class StockScreener:
    """
    Advanced stock screening with custom filters
    """
    
    # Available filter fields
    FILTER_FIELDS = {
        # Price & Performance
        "price": {"type": "number", "description": "Current stock price"},
        "change_percent": {"type": "number", "description": "Daily price change %"},
        "change_1w": {"type": "number", "description": "Weekly price change %"},
        "change_1m": {"type": "number", "description": "Monthly price change %"},
        "change_ytd": {"type": "number", "description": "Year-to-date change %"},
        "week_52_high": {"type": "number", "description": "52-week high"},
        "week_52_low": {"type": "number", "description": "52-week low"},
        "from_52w_high": {"type": "number", "description": "% below 52-week high"},
        "from_52w_low": {"type": "number", "description": "% above 52-week low"},
        
        # Volume
        "volume": {"type": "number", "description": "Trading volume"},
        "avg_volume": {"type": "number", "description": "Average volume"},
        "relative_volume": {"type": "number", "description": "Volume vs average"},
        
        # Valuation
        "market_cap": {"type": "number", "description": "Market capitalization"},
        "pe_ratio": {"type": "number", "description": "P/E ratio"},
        "forward_pe": {"type": "number", "description": "Forward P/E ratio"},
        "peg_ratio": {"type": "number", "description": "PEG ratio"},
        "price_to_book": {"type": "number", "description": "Price to book ratio"},
        "price_to_sales": {"type": "number", "description": "Price to sales ratio"},
        "ev_to_ebitda": {"type": "number", "description": "EV/EBITDA ratio"},
        
        # Fundamentals
        "eps": {"type": "number", "description": "Earnings per share"},
        "revenue": {"type": "number", "description": "Annual revenue"},
        "revenue_growth": {"type": "number", "description": "Revenue growth %"},
        "earnings_growth": {"type": "number", "description": "Earnings growth %"},
        "gross_margin": {"type": "number", "description": "Gross margin %"},
        "operating_margin": {"type": "number", "description": "Operating margin %"},
        "net_margin": {"type": "number", "description": "Net margin %"},
        "roe": {"type": "number", "description": "Return on equity %"},
        "roa": {"type": "number", "description": "Return on assets %"},
        
        # Dividends
        "dividend_yield": {"type": "number", "description": "Dividend yield %"},
        "payout_ratio": {"type": "number", "description": "Payout ratio %"},
        
        # Risk
        "beta": {"type": "number", "description": "Beta"},
        "debt_to_equity": {"type": "number", "description": "Debt to equity ratio"},
        "current_ratio": {"type": "number", "description": "Current ratio"},
        
        # Classification
        "sector": {"type": "string", "description": "Sector"},
        "industry": {"type": "string", "description": "Industry"},
        "country": {"type": "string", "description": "Country"},
        "exchange": {"type": "string", "description": "Exchange"},
    }
    
    # Predefined screener templates
    SCREENER_TEMPLATES = {
        "value_stocks": {
            "name": "Value Stocks",
            "description": "Low P/E, high dividend yield",
            "filters": [
                ScreenerFilter("pe_ratio", FilterOperator.LESS_THAN, 15),
                ScreenerFilter("dividend_yield", FilterOperator.GREATER_THAN, 2),
                ScreenerFilter("market_cap", FilterOperator.GREATER_THAN, 1e9)
            ]
        },
        "growth_stocks": {
            "name": "Growth Stocks",
            "description": "High revenue and earnings growth",
            "filters": [
                ScreenerFilter("revenue_growth", FilterOperator.GREATER_THAN, 20),
                ScreenerFilter("earnings_growth", FilterOperator.GREATER_THAN, 20),
                ScreenerFilter("market_cap", FilterOperator.GREATER_THAN, 1e9)
            ]
        },
        "momentum": {
            "name": "Momentum Stocks",
            "description": "Strong price momentum",
            "filters": [
                ScreenerFilter("change_1m", FilterOperator.GREATER_THAN, 10),
                ScreenerFilter("change_1w", FilterOperator.GREATER_THAN, 5),
                ScreenerFilter("relative_volume", FilterOperator.GREATER_THAN, 1.5)
            ]
        },
        "dividend_champions": {
            "name": "Dividend Champions",
            "description": "High yield, sustainable dividends",
            "filters": [
                ScreenerFilter("dividend_yield", FilterOperator.GREATER_THAN, 3),
                ScreenerFilter("payout_ratio", FilterOperator.LESS_THAN, 70),
                ScreenerFilter("market_cap", FilterOperator.GREATER_THAN, 5e9)
            ]
        },
        "undervalued": {
            "name": "Undervalued Stocks",
            "description": "Trading below fair value",
            "filters": [
                ScreenerFilter("peg_ratio", FilterOperator.LESS_THAN, 1),
                ScreenerFilter("price_to_book", FilterOperator.LESS_THAN, 2),
                ScreenerFilter("roe", FilterOperator.GREATER_THAN, 15)
            ]
        },
        "small_cap_growth": {
            "name": "Small Cap Growth",
            "description": "Small caps with high growth",
            "filters": [
                ScreenerFilter("market_cap", FilterOperator.BETWEEN, 300e6, 2e9),
                ScreenerFilter("revenue_growth", FilterOperator.GREATER_THAN, 25),
                ScreenerFilter("eps", FilterOperator.GREATER_THAN, 0)
            ]
        },
        "low_volatility": {
            "name": "Low Volatility",
            "description": "Stable, low beta stocks",
            "filters": [
                ScreenerFilter("beta", FilterOperator.LESS_THAN, 0.8),
                ScreenerFilter("dividend_yield", FilterOperator.GREATER_THAN, 1),
                ScreenerFilter("market_cap", FilterOperator.GREATER_THAN, 10e9)
            ]
        },
        "high_quality": {
            "name": "High Quality",
            "description": "Strong fundamentals",
            "filters": [
                ScreenerFilter("roe", FilterOperator.GREATER_THAN, 20),
                ScreenerFilter("gross_margin", FilterOperator.GREATER_THAN, 40),
                ScreenerFilter("debt_to_equity", FilterOperator.LESS_THAN, 50),
                ScreenerFilter("current_ratio", FilterOperator.GREATER_THAN, 1.5)
            ]
        }
    }
    
    def __init__(self):
        self.stock_universe: List[Dict] = []
        self._load_universe()
    
    def _load_universe(self):
        """Load stock universe"""
        # This would typically load from a database or API
        # Using mock data for demonstration
        import random
        
        sectors = ["Technology", "Healthcare", "Finance", "Consumer", "Energy", 
                   "Materials", "Industrials", "Utilities", "Real Estate", "Communications"]
        
        industries = {
            "Technology": ["Software", "Hardware", "Semiconductors", "IT Services"],
            "Healthcare": ["Pharmaceuticals", "Biotech", "Medical Devices", "Healthcare Services"],
            "Finance": ["Banks", "Insurance", "Asset Management", "Fintech"],
            "Consumer": ["Retail", "E-commerce", "Consumer Goods", "Entertainment"],
            "Energy": ["Oil & Gas", "Renewable Energy", "Utilities", "Energy Services"],
            "Materials": ["Mining", "Chemicals", "Steel", "Construction Materials"],
            "Industrials": ["Aerospace", "Defense", "Machinery", "Transportation"],
            "Utilities": ["Electric", "Gas", "Water", "Multi-Utilities"],
            "Real Estate": ["REITs", "Real Estate Services", "Property Development"],
            "Communications": ["Telecom", "Media", "Internet Services", "Advertising"]
        }
        
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK.B",
                   "JPM", "JNJ", "V", "PG", "HD", "MA", "UNH", "DIS", "PYPL", "ADBE",
                   "NFLX", "CRM", "INTC", "CSCO", "PFE", "MRK", "KO", "PEP", "WMT",
                   "VZ", "T", "XOM", "CVX", "BA", "GE", "CAT", "MMM", "IBM", "GS",
                   "MS", "C", "WFC", "BAC", "AXP", "BLK", "SCHW", "USB", "PNC"]
        
        for symbol in symbols:
            sector = random.choice(sectors)
            industry = random.choice(industries.get(sector, ["Other"]))
            
            price = random.uniform(20, 500)
            change = random.uniform(-5, 5)
            
            self.stock_universe.append({
                "symbol": symbol,
                "name": f"{symbol} Corporation",
                "sector": sector,
                "industry": industry,
                "price": price,
                "change_percent": change,
                "change_1w": random.uniform(-10, 10),
                "change_1m": random.uniform(-20, 20),
                "change_ytd": random.uniform(-30, 50),
                "market_cap": random.uniform(1e9, 3e12),
                "volume": random.randint(100000, 50000000),
                "avg_volume": random.randint(500000, 20000000),
                "relative_volume": random.uniform(0.5, 3),
                "pe_ratio": random.uniform(5, 60) if random.random() > 0.1 else None,
                "forward_pe": random.uniform(5, 50) if random.random() > 0.1 else None,
                "peg_ratio": random.uniform(0.3, 4) if random.random() > 0.2 else None,
                "price_to_book": random.uniform(0.5, 15),
                "price_to_sales": random.uniform(0.5, 20),
                "ev_to_ebitda": random.uniform(3, 25),
                "eps": random.uniform(-2, 20),
                "revenue": random.uniform(1e9, 500e9),
                "revenue_growth": random.uniform(-10, 50),
                "earnings_growth": random.uniform(-20, 60),
                "gross_margin": random.uniform(20, 80),
                "operating_margin": random.uniform(-10, 50),
                "net_margin": random.uniform(-5, 30),
                "roe": random.uniform(-10, 50),
                "roa": random.uniform(-5, 25),
                "dividend_yield": random.uniform(0, 6),
                "payout_ratio": random.uniform(0, 100),
                "beta": random.uniform(0.3, 2.5),
                "debt_to_equity": random.uniform(0, 200),
                "current_ratio": random.uniform(0.5, 4),
                "week_52_high": price * random.uniform(1.05, 1.5),
                "week_52_low": price * random.uniform(0.5, 0.95),
                "country": "USA",
                "exchange": random.choice(["NYSE", "NASDAQ"])
            })
    
    def screen(
        self,
        filters: List[ScreenerFilter],
        sort_by: str = "market_cap",
        sort_order: SortOrder = SortOrder.DESC,
        limit: int = 50
    ) -> List[ScreenerResult]:
        """Run screener with custom filters"""
        results = []
        
        for stock in self.stock_universe:
            if self._matches_filters(stock, filters):
                results.append(ScreenerResult(
                    symbol=stock["symbol"],
                    name=stock["name"],
                    sector=stock["sector"],
                    industry=stock["industry"],
                    market_cap=stock["market_cap"],
                    price=stock["price"],
                    change_percent=stock["change_percent"],
                    volume=stock["volume"],
                    pe_ratio=stock.get("pe_ratio"),
                    eps=stock.get("eps"),
                    dividend_yield=stock.get("dividend_yield"),
                    week_52_high=stock["week_52_high"],
                    week_52_low=stock["week_52_low"],
                    avg_volume=stock["avg_volume"],
                    beta=stock.get("beta"),
                    additional_data={k: v for k, v in stock.items() 
                                   if k not in ["symbol", "name", "sector", "industry"]}
                ))
        
        # Sort results
        if sort_by in self.FILTER_FIELDS:
            reverse = sort_order == SortOrder.DESC
            results.sort(
                key=lambda x: x.additional_data.get(sort_by, 0) or 0,
                reverse=reverse
            )
        
        return results[:limit]
    
    def _matches_filters(self, stock: Dict, filters: List[ScreenerFilter]) -> bool:
        """Check if stock matches all filters"""
        for f in filters:
            value = stock.get(f.field)
            
            if value is None and f.operator not in [FilterOperator.IN, FilterOperator.NOT_IN]:
                return False
            
            if f.operator == FilterOperator.GREATER_THAN:
                if not (value > f.value):
                    return False
            elif f.operator == FilterOperator.LESS_THAN:
                if not (value < f.value):
                    return False
            elif f.operator == FilterOperator.GREATER_EQUAL:
                if not (value >= f.value):
                    return False
            elif f.operator == FilterOperator.LESS_EQUAL:
                if not (value <= f.value):
                    return False
            elif f.operator == FilterOperator.EQUAL:
                if not (value == f.value):
                    return False
            elif f.operator == FilterOperator.NOT_EQUAL:
                if not (value != f.value):
                    return False
            elif f.operator == FilterOperator.BETWEEN:
                if not (f.value <= value <= f.secondary_value):
                    return False
            elif f.operator == FilterOperator.IN:
                if value not in f.value:
                    return False
            elif f.operator == FilterOperator.NOT_IN:
                if value in f.value:
                    return False
            elif f.operator == FilterOperator.CONTAINS:
                if f.value.lower() not in str(value).lower():
                    return False
        
        return True
    
    def use_template(
        self,
        template_name: str,
        additional_filters: Optional[List[ScreenerFilter]] = None
    ) -> List[ScreenerResult]:
        """Use a predefined screener template"""
        if template_name not in self.SCREENER_TEMPLATES:
            return []
        
        template = self.SCREENER_TEMPLATES[template_name]
        filters = template["filters"].copy()
        
        if additional_filters:
            filters.extend(additional_filters)
        
        return self.screen(filters)
    
    def get_top_gainers(self, limit: int = 20) -> List[ScreenerResult]:
        """Get top gaining stocks"""
        return self.screen(
            filters=[],
            sort_by="change_percent",
            sort_order=SortOrder.DESC,
            limit=limit
        )
    
    def get_top_losers(self, limit: int = 20) -> List[ScreenerResult]:
        """Get top losing stocks"""
        return self.screen(
            filters=[],
            sort_by="change_percent",
            sort_order=SortOrder.ASC,
            limit=limit
        )
    
    def get_most_active(self, limit: int = 20) -> List[ScreenerResult]:
        """Get most actively traded stocks"""
        return self.screen(
            filters=[],
            sort_by="volume",
            sort_order=SortOrder.DESC,
            limit=limit
        )
    
    def get_new_highs(self) -> List[ScreenerResult]:
        """Get stocks at 52-week highs"""
        results = []
        for stock in self.stock_universe:
            if stock["price"] >= stock["week_52_high"] * 0.98:  # Within 2% of high
                results.append(ScreenerResult(
                    symbol=stock["symbol"],
                    name=stock["name"],
                    sector=stock["sector"],
                    industry=stock["industry"],
                    market_cap=stock["market_cap"],
                    price=stock["price"],
                    change_percent=stock["change_percent"],
                    volume=stock["volume"],
                    pe_ratio=stock.get("pe_ratio"),
                    eps=stock.get("eps"),
                    dividend_yield=stock.get("dividend_yield"),
                    week_52_high=stock["week_52_high"],
                    week_52_low=stock["week_52_low"],
                    avg_volume=stock["avg_volume"],
                    beta=stock.get("beta")
                ))
        return results
    
    def get_new_lows(self) -> List[ScreenerResult]:
        """Get stocks at 52-week lows"""
        results = []
        for stock in self.stock_universe:
            if stock["price"] <= stock["week_52_low"] * 1.02:  # Within 2% of low
                results.append(ScreenerResult(
                    symbol=stock["symbol"],
                    name=stock["name"],
                    sector=stock["sector"],
                    industry=stock["industry"],
                    market_cap=stock["market_cap"],
                    price=stock["price"],
                    change_percent=stock["change_percent"],
                    volume=stock["volume"],
                    pe_ratio=stock.get("pe_ratio"),
                    eps=stock.get("eps"),
                    dividend_yield=stock.get("dividend_yield"),
                    week_52_high=stock["week_52_high"],
                    week_52_low=stock["week_52_low"],
                    avg_volume=stock["avg_volume"],
                    beta=stock.get("beta")
                ))
        return results


class SectorAnalyzer:
    """
    Analyze sector and industry performance
    """
    
    SECTORS = [
        "Technology", "Healthcare", "Finance", "Consumer", "Energy",
        "Materials", "Industrials", "Utilities", "Real Estate", "Communications"
    ]
    
    def __init__(self):
        self.screener = StockScreener()
    
    def get_sector_performance(self) -> List[SectorPerformance]:
        """Get performance of all sectors"""
        import random
        
        performances = []
        
        for sector in self.SECTORS:
            # Get stocks in sector
            sector_stocks = [s for s in self.screener.stock_universe 
                           if s.get("sector") == sector]
            
            if not sector_stocks:
                continue
            
            # Calculate aggregate metrics
            total_market_cap = sum(s["market_cap"] for s in sector_stocks)
            avg_change_1d = np.mean([s["change_percent"] for s in sector_stocks])
            avg_change_1w = np.mean([s.get("change_1w", 0) for s in sector_stocks])
            avg_change_1m = np.mean([s.get("change_1m", 0) for s in sector_stocks])
            avg_change_ytd = np.mean([s.get("change_ytd", 0) for s in sector_stocks])
            
            # Find top and worst performers
            sorted_stocks = sorted(sector_stocks, key=lambda x: x["change_percent"], reverse=True)
            
            performances.append(SectorPerformance(
                sector=sector,
                change_1d=avg_change_1d,
                change_1w=avg_change_1w,
                change_1m=avg_change_1m,
                change_ytd=avg_change_ytd,
                market_cap=total_market_cap,
                num_stocks=len(sector_stocks),
                top_performer=sorted_stocks[0]["symbol"] if sorted_stocks else "",
                worst_performer=sorted_stocks[-1]["symbol"] if sorted_stocks else ""
            ))
        
        # Sort by daily change
        performances.sort(key=lambda x: x.change_1d, reverse=True)
        
        return performances
    
    def get_sector_rotation(self, days: int = 30) -> Dict[str, Any]:
        """Analyze sector rotation patterns"""
        import random
        
        # Mock sector rotation data
        rotation_data = {}
        
        for sector in self.SECTORS:
            rotation_data[sector] = {
                "momentum_score": random.uniform(-1, 1),
                "relative_strength": random.uniform(-2, 2),
                "money_flow": random.choice(["Inflow", "Outflow", "Neutral"]),
                "trend": random.choice(["Bullish", "Bearish", "Neutral"])
            }
        
        # Determine rotation phase
        leading_sectors = sorted(
            rotation_data.items(),
            key=lambda x: x[1]["momentum_score"],
            reverse=True
        )[:3]
        
        lagging_sectors = sorted(
            rotation_data.items(),
            key=lambda x: x[1]["momentum_score"]
        )[:3]
        
        return {
            "period_days": days,
            "rotation_data": rotation_data,
            "leading_sectors": [s[0] for s in leading_sectors],
            "lagging_sectors": [s[0] for s in lagging_sectors],
            "market_phase": random.choice(["Expansion", "Peak", "Contraction", "Trough"]),
            "recommended_sectors": [s[0] for s in leading_sectors]
        }
    
    def compare_sectors(self, sectors: List[str]) -> Dict[str, Any]:
        """Compare multiple sectors"""
        comparison = {}
        
        for sector in sectors:
            sector_stocks = [s for s in self.screener.stock_universe 
                           if s.get("sector") == sector]
            
            if sector_stocks:
                comparison[sector] = {
                    "num_stocks": len(sector_stocks),
                    "total_market_cap": sum(s["market_cap"] for s in sector_stocks),
                    "avg_pe": np.mean([s["pe_ratio"] for s in sector_stocks if s.get("pe_ratio")]),
                    "avg_dividend_yield": np.mean([s["dividend_yield"] for s in sector_stocks]),
                    "avg_revenue_growth": np.mean([s["revenue_growth"] for s in sector_stocks]),
                    "avg_beta": np.mean([s["beta"] for s in sector_stocks if s.get("beta")])
                }
        
        return comparison


class UnusualActivityDetector:
    """
    Detect unusual trading activity
    """
    
    def __init__(self):
        self.screener = StockScreener()
        self.alerts: List[UnusualActivity] = []
    
    def detect_volume_spikes(self, threshold: float = 3.0) -> List[UnusualActivity]:
        """Detect unusual volume activity"""
        alerts = []
        
        for stock in self.screener.stock_universe:
            relative_volume = stock.get("relative_volume", 1)
            
            if relative_volume >= threshold:
                alerts.append(UnusualActivity(
                    symbol=stock["symbol"],
                    name=stock["name"],
                    activity_type="Volume Spike",
                    current_value=stock["volume"],
                    average_value=stock["avg_volume"],
                    deviation_percent=(relative_volume - 1) * 100,
                    timestamp=datetime.now(),
                    description=f"Trading at {relative_volume:.1f}x normal volume"
                ))
        
        return sorted(alerts, key=lambda x: x.deviation_percent, reverse=True)
    
    def detect_price_surges(self, threshold: float = 5.0) -> List[UnusualActivity]:
        """Detect unusual price movements"""
        alerts = []
        
        for stock in self.screener.stock_universe:
            change = abs(stock["change_percent"])
            
            if change >= threshold:
                direction = "surge" if stock["change_percent"] > 0 else "drop"
                alerts.append(UnusualActivity(
                    symbol=stock["symbol"],
                    name=stock["name"],
                    activity_type=f"Price {direction.capitalize()}",
                    current_value=stock["price"],
                    average_value=stock["price"] / (1 + stock["change_percent"] / 100),
                    deviation_percent=stock["change_percent"],
                    timestamp=datetime.now(),
                    description=f"Price {direction} of {abs(stock['change_percent']):.1f}%"
                ))
        
        return sorted(alerts, key=lambda x: abs(x.deviation_percent), reverse=True)
    
    def detect_unusual_options_activity(self) -> List[UnusualActivity]:
        """Detect unusual options activity"""
        import random
        
        # Mock unusual options activity
        alerts = []
        
        symbols = ["AAPL", "TSLA", "NVDA", "AMD", "META"]
        for symbol in symbols:
            if random.random() > 0.5:
                alerts.append(UnusualActivity(
                    symbol=symbol,
                    name=f"{symbol} Inc.",
                    activity_type="Options Activity",
                    current_value=random.randint(10000, 100000),
                    average_value=random.randint(1000, 10000),
                    deviation_percent=random.uniform(200, 1000),
                    timestamp=datetime.now(),
                    description=f"Unusual call volume detected"
                ))
        
        return alerts
    
    def get_all_alerts(self) -> List[UnusualActivity]:
        """Get all unusual activity alerts"""
        all_alerts = []
        all_alerts.extend(self.detect_volume_spikes())
        all_alerts.extend(self.detect_price_surges())
        all_alerts.extend(self.detect_unusual_options_activity())
        
        # Sort by timestamp (most recent first)
        all_alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return all_alerts


class ScreenerDashboard:
    """
    Unified screener dashboard
    """
    
    def __init__(self):
        self.screener = StockScreener()
        self.sector_analyzer = SectorAnalyzer()
        self.activity_detector = UnusualActivityDetector()
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview"""
        return {
            "top_gainers": self.screener.get_top_gainers(10),
            "top_losers": self.screener.get_top_losers(10),
            "most_active": self.screener.get_most_active(10),
            "sector_performance": self.sector_analyzer.get_sector_performance(),
            "unusual_activity": self.activity_detector.get_all_alerts()[:10],
            "timestamp": datetime.now().isoformat()
        }
    
    def quick_screen(self, screen_type: str) -> List[ScreenerResult]:
        """Quick predefined screens"""
        if screen_type in self.screener.SCREENER_TEMPLATES:
            return self.screener.use_template(screen_type)
        elif screen_type == "gainers":
            return self.screener.get_top_gainers()
        elif screen_type == "losers":
            return self.screener.get_top_losers()
        elif screen_type == "active":
            return self.screener.get_most_active()
        elif screen_type == "new_highs":
            return self.screener.get_new_highs()
        elif screen_type == "new_lows":
            return self.screener.get_new_lows()
        
        return []


# Example usage
if __name__ == "__main__":
    dashboard = ScreenerDashboard()
    
    print("=== Market Overview ===")
    overview = dashboard.get_market_overview()
    print(f"Top 3 Gainers: {[s.symbol for s in overview['top_gainers'][:3]]}")
    print(f"Top 3 Losers: {[s.symbol for s in overview['top_losers'][:3]]}")
    
    print("\n=== Value Stocks Screen ===")
    value_stocks = dashboard.quick_screen("value_stocks")
    for stock in value_stocks[:5]:
        print(f"  {stock.symbol}: P/E={stock.pe_ratio:.2f if stock.pe_ratio else 'N/A'}, "
              f"Div={stock.dividend_yield:.2f}%")
    
    print("\n=== Sector Performance ===")
    sectors = dashboard.sector_analyzer.get_sector_performance()
    for sector in sectors[:5]:
        print(f"  {sector.sector}: {sector.change_1d:+.2f}%")
    
    print("\n=== Unusual Activity ===")
    alerts = dashboard.activity_detector.get_all_alerts()[:5]
    for alert in alerts:
        print(f"  {alert.symbol}: {alert.activity_type} - {alert.description}")
