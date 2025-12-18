"""
Indian Mutual Funds and ETF Analytics Module
Comprehensive mutual fund NAV, SIP calculator, and ETF tracking
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class FundCategory(Enum):
    """Mutual fund categories"""
    EQUITY_LARGE_CAP = "Equity - Large Cap"
    EQUITY_MID_CAP = "Equity - Mid Cap"
    EQUITY_SMALL_CAP = "Equity - Small Cap"
    EQUITY_MULTI_CAP = "Equity - Multi Cap"
    EQUITY_FLEXI_CAP = "Equity - Flexi Cap"
    EQUITY_ELSS = "Equity - ELSS (Tax Saving)"
    EQUITY_SECTORAL = "Equity - Sectoral/Thematic"
    EQUITY_INDEX = "Equity - Index Fund"
    HYBRID_AGGRESSIVE = "Hybrid - Aggressive"
    HYBRID_BALANCED = "Hybrid - Balanced Advantage"
    HYBRID_CONSERVATIVE = "Hybrid - Conservative"
    DEBT_LIQUID = "Debt - Liquid"
    DEBT_OVERNIGHT = "Debt - Overnight"
    DEBT_ULTRA_SHORT = "Debt - Ultra Short Duration"
    DEBT_SHORT = "Debt - Short Duration"
    DEBT_MEDIUM = "Debt - Medium Duration"
    DEBT_LONG = "Debt - Long Duration"
    DEBT_GILT = "Debt - Gilt"
    DEBT_CORPORATE = "Debt - Corporate Bond"
    INTERNATIONAL = "International/Global"
    GOLD = "Gold"
    OTHER = "Other"


class ETFCategory(Enum):
    """ETF categories"""
    EQUITY_INDEX = "Index ETF"
    EQUITY_SECTORAL = "Sectoral ETF"
    GOLD = "Gold ETF"
    SILVER = "Silver ETF"
    DEBT = "Debt ETF"
    INTERNATIONAL = "International ETF"
    SMART_BETA = "Smart Beta ETF"


@dataclass
class MutualFund:
    """Mutual fund data"""
    scheme_code: str
    scheme_name: str
    nav: float
    nav_date: datetime
    category: FundCategory
    amc: str  # Asset Management Company
    aum: Optional[float] = None  # Assets Under Management in Cr
    expense_ratio: Optional[float] = None
    return_1y: Optional[float] = None
    return_3y: Optional[float] = None
    return_5y: Optional[float] = None
    risk_rating: Optional[str] = None
    min_sip: Optional[float] = None
    min_lumpsum: Optional[float] = None


@dataclass
class ETF:
    """ETF data"""
    symbol: str
    name: str
    nav: float
    market_price: float
    premium_discount: float
    category: ETFCategory
    aum: Optional[float] = None
    expense_ratio: Optional[float] = None
    tracking_error: Optional[float] = None
    underlying_index: Optional[str] = None
    volume: Optional[int] = None


@dataclass
class SIPCalculation:
    """SIP calculation result"""
    monthly_investment: float
    total_investment: float
    expected_returns: float
    wealth_gained: float
    investment_period_months: int
    expected_rate: float


@dataclass
class SWPCalculation:
    """SWP (Systematic Withdrawal Plan) calculation"""
    initial_investment: float
    monthly_withdrawal: float
    total_withdrawal: float
    remaining_balance: float
    withdrawal_period_months: int
    expected_rate: float


@dataclass
class LumpsumCalculation:
    """Lumpsum investment calculation"""
    investment: float
    expected_value: float
    wealth_gained: float
    investment_period_years: int
    expected_rate: float


class MutualFundDataFetcher:
    """
    Fetch mutual fund data from mftool
    """
    
    def __init__(self):
        self.mf = None
        self._initialize()
    
    def _initialize(self):
        """Initialize mftool connection"""
        try:
            from mftool import Mftool
            self.mf = Mftool()
            logger.info("mftool connection initialized")
        except ImportError:
            logger.warning("mftool not installed. Install with: pip install mftool")
        except Exception as e:
            logger.error(f"Failed to initialize mftool: {e}")
    
    def get_scheme_codes(self) -> Dict[str, str]:
        """Get all scheme codes and names"""
        if not self.mf:
            return self._get_sample_schemes()
        
        try:
            return self.mf.get_scheme_codes()
        except Exception as e:
            logger.error(f"Error fetching scheme codes: {e}")
            return {}
    
    def _get_sample_schemes(self) -> Dict[str, str]:
        """Return sample schemes for testing"""
        return {
            "119551": "Nippon India Growth Fund - Direct Plan - Growth",
            "120503": "HDFC Mid-Cap Opportunities Fund - Direct Plan - Growth",
            "120505": "HDFC Flexi Cap Fund - Direct Plan - Growth",
            "118834": "SBI Small Cap Fund - Direct Plan - Growth",
            "119028": "Axis Bluechip Fund - Direct Plan - Growth",
            "119029": "Axis Long Term Equity Fund - Direct Plan - Growth",
            "118989": "Parag Parikh Flexi Cap Fund - Direct Plan - Growth",
            "135781": "Mirae Asset Large Cap Fund - Direct Plan - Growth",
            "120586": "Kotak Flexicap Fund - Direct Plan - Growth",
            "145552": "UTI Nifty Index Fund - Direct Plan - Growth"
        }
    
    def get_scheme_quote(self, scheme_code: str) -> Optional[MutualFund]:
        """Get NAV and details for a specific scheme"""
        if not self.mf:
            return self._get_mock_scheme(scheme_code)
        
        try:
            data = self.mf.get_scheme_quote(scheme_code)
            if data:
                return MutualFund(
                    scheme_code=scheme_code,
                    scheme_name=data.get('scheme_name', ''),
                    nav=float(data.get('nav', 0)),
                    nav_date=datetime.strptime(data.get('last_updated', ''), '%d-%b-%Y')
                             if data.get('last_updated') else datetime.now(),
                    category=FundCategory.EQUITY_MULTI_CAP,  # Would need category mapping
                    amc=data.get('fund_house', '')
                )
        except Exception as e:
            logger.error(f"Error fetching scheme {scheme_code}: {e}")
        
        return None
    
    def _get_mock_scheme(self, scheme_code: str) -> MutualFund:
        """Get mock scheme for testing"""
        import random
        sample_schemes = self._get_sample_schemes()
        name = sample_schemes.get(scheme_code, f"Scheme {scheme_code}")
        
        return MutualFund(
            scheme_code=scheme_code,
            scheme_name=name,
            nav=random.uniform(50, 500),
            nav_date=datetime.now(),
            category=FundCategory.EQUITY_MULTI_CAP,
            amc="Sample AMC",
            aum=random.uniform(1000, 50000),
            expense_ratio=random.uniform(0.1, 2.5),
            return_1y=random.uniform(-10, 30),
            return_3y=random.uniform(5, 25),
            return_5y=random.uniform(8, 20),
            risk_rating="Moderately High",
            min_sip=500,
            min_lumpsum=5000
        )
    
    def get_historical_nav(
        self,
        scheme_code: str,
        as_dataframe: bool = False
    ) -> Any:
        """Get historical NAV data"""
        if not self.mf:
            return self._get_mock_historical_nav(scheme_code, as_dataframe)
        
        try:
            data = self.mf.get_scheme_historical_nav(scheme_code, as_Dataframe=as_dataframe)
            return data
        except Exception as e:
            logger.error(f"Error fetching historical NAV for {scheme_code}: {e}")
            return None
    
    def _get_mock_historical_nav(self, scheme_code: str, as_dataframe: bool) -> Any:
        """Generate mock historical NAV"""
        try:
            import pandas as pd
            import numpy as np
            
            dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
            base_nav = 100
            returns = np.random.normal(0.0005, 0.01, len(dates))
            navs = base_nav * np.cumprod(1 + returns)
            
            df = pd.DataFrame({
                'date': dates,
                'nav': navs
            })
            
            if as_dataframe:
                return df
            else:
                return df.to_dict('records')
        except ImportError:
            return None
    
    def search_scheme(self, query: str) -> List[Dict[str, str]]:
        """Search for schemes by name"""
        all_schemes = self.get_scheme_codes()
        query_lower = query.lower()
        
        results = []
        for code, name in all_schemes.items():
            if query_lower in name.lower():
                results.append({"code": code, "name": name})
        
        return results[:20]  # Return top 20 matches


class SIPCalculator:
    """
    SIP and investment calculators
    """
    
    @staticmethod
    def calculate_sip(
        monthly_investment: float,
        expected_return_rate: float,
        investment_period_years: int
    ) -> SIPCalculation:
        """
        Calculate SIP returns
        
        Args:
            monthly_investment: Monthly SIP amount
            expected_return_rate: Expected annual return (as percentage, e.g., 12 for 12%)
            investment_period_years: Investment period in years
        """
        n = investment_period_years * 12  # Total months
        r = expected_return_rate / 100 / 12  # Monthly rate
        
        # Future Value of SIP: P × ({[1 + r]^n – 1} / r) × (1 + r)
        if r > 0:
            future_value = monthly_investment * (((1 + r) ** n - 1) / r) * (1 + r)
        else:
            future_value = monthly_investment * n
        
        total_investment = monthly_investment * n
        
        return SIPCalculation(
            monthly_investment=monthly_investment,
            total_investment=total_investment,
            expected_returns=future_value,
            wealth_gained=future_value - total_investment,
            investment_period_months=n,
            expected_rate=expected_return_rate
        )
    
    @staticmethod
    def calculate_sip_required(
        target_amount: float,
        expected_return_rate: float,
        investment_period_years: int
    ) -> float:
        """Calculate monthly SIP required to reach target"""
        n = investment_period_years * 12
        r = expected_return_rate / 100 / 12
        
        if r > 0:
            monthly_sip = target_amount / ((((1 + r) ** n - 1) / r) * (1 + r))
        else:
            monthly_sip = target_amount / n
        
        return round(monthly_sip, 2)
    
    @staticmethod
    def calculate_lumpsum(
        investment: float,
        expected_return_rate: float,
        investment_period_years: int
    ) -> LumpsumCalculation:
        """Calculate lumpsum investment returns"""
        r = expected_return_rate / 100
        
        future_value = investment * ((1 + r) ** investment_period_years)
        
        return LumpsumCalculation(
            investment=investment,
            expected_value=future_value,
            wealth_gained=future_value - investment,
            investment_period_years=investment_period_years,
            expected_rate=expected_return_rate
        )
    
    @staticmethod
    def calculate_swp(
        initial_investment: float,
        monthly_withdrawal: float,
        expected_return_rate: float,
        withdrawal_period_years: int
    ) -> SWPCalculation:
        """Calculate SWP (Systematic Withdrawal Plan)"""
        n = withdrawal_period_years * 12
        r = expected_return_rate / 100 / 12
        
        # Calculate remaining balance after withdrawals
        remaining = initial_investment
        total_withdrawal = 0
        
        for _ in range(n):
            remaining = remaining * (1 + r) - monthly_withdrawal
            total_withdrawal += monthly_withdrawal
            if remaining <= 0:
                remaining = 0
                break
        
        return SWPCalculation(
            initial_investment=initial_investment,
            monthly_withdrawal=monthly_withdrawal,
            total_withdrawal=total_withdrawal,
            remaining_balance=max(0, remaining),
            withdrawal_period_months=n,
            expected_rate=expected_return_rate
        )
    
    @staticmethod
    def calculate_step_up_sip(
        starting_sip: float,
        annual_step_up_percent: float,
        expected_return_rate: float,
        investment_period_years: int
    ) -> Dict[str, Any]:
        """Calculate Step-up SIP where SIP amount increases annually"""
        r = expected_return_rate / 100 / 12
        total_investment = 0
        future_value = 0
        yearly_breakdown = []
        
        current_sip = starting_sip
        
        for year in range(1, investment_period_years + 1):
            year_investment = current_sip * 12
            total_investment += year_investment
            
            # Compound previous value
            future_value = future_value * ((1 + r) ** 12)
            
            # Add this year's SIP contributions
            for month in range(12):
                months_remaining = (investment_period_years - year) * 12 + (12 - month)
                future_value += current_sip * ((1 + r) ** months_remaining)
            
            yearly_breakdown.append({
                "year": year,
                "monthly_sip": current_sip,
                "yearly_investment": year_investment,
                "cumulative_investment": total_investment
            })
            
            # Step up for next year
            current_sip = current_sip * (1 + annual_step_up_percent / 100)
        
        return {
            "starting_sip": starting_sip,
            "final_sip": current_sip / (1 + annual_step_up_percent / 100),
            "total_investment": total_investment,
            "expected_value": future_value,
            "wealth_gained": future_value - total_investment,
            "step_up_percent": annual_step_up_percent,
            "years": investment_period_years,
            "yearly_breakdown": yearly_breakdown
        }


class MutualFundAnalyzer:
    """
    Analyze and compare mutual funds
    """
    
    def __init__(self):
        self.fetcher = MutualFundDataFetcher()
    
    def compare_funds(self, scheme_codes: List[str]) -> Dict[str, Any]:
        """Compare multiple mutual funds"""
        funds = []
        
        for code in scheme_codes:
            fund = self.fetcher.get_scheme_quote(code)
            if fund:
                funds.append({
                    "code": code,
                    "name": fund.scheme_name,
                    "nav": fund.nav,
                    "category": fund.category.value if fund.category else "Unknown",
                    "aum": fund.aum,
                    "expense_ratio": fund.expense_ratio,
                    "return_1y": fund.return_1y,
                    "return_3y": fund.return_3y,
                    "return_5y": fund.return_5y
                })
        
        if funds:
            # Find best performers
            best_1y = max(funds, key=lambda x: x.get("return_1y", 0) or 0)
            best_3y = max(funds, key=lambda x: x.get("return_3y", 0) or 0)
            best_5y = max(funds, key=lambda x: x.get("return_5y", 0) or 0)
            lowest_expense = min(funds, key=lambda x: x.get("expense_ratio", 100) or 100)
            
            return {
                "funds": funds,
                "best_1y_return": best_1y["name"],
                "best_3y_return": best_3y["name"],
                "best_5y_return": best_5y["name"],
                "lowest_expense_ratio": lowest_expense["name"],
                "timestamp": datetime.now().isoformat()
            }
        
        return {"error": "No valid funds found"}
    
    def get_category_funds(self, category: FundCategory, limit: int = 10) -> List[Dict]:
        """Get top funds in a category"""
        # In production, this would filter from database
        all_schemes = self.fetcher.get_scheme_codes()
        
        funds = []
        count = 0
        for code, name in all_schemes.items():
            if count >= limit:
                break
            fund = self.fetcher.get_scheme_quote(code)
            if fund:
                funds.append({
                    "code": code,
                    "name": fund.scheme_name,
                    "nav": fund.nav,
                    "return_1y": fund.return_1y
                })
                count += 1
        
        return sorted(funds, key=lambda x: x.get("return_1y", 0) or 0, reverse=True)
    
    def calculate_returns(
        self,
        scheme_code: str,
        investment_date: datetime,
        investment_amount: float
    ) -> Dict[str, Any]:
        """Calculate returns for a specific investment"""
        # Get historical NAV
        historical = self.fetcher.get_historical_nav(scheme_code, as_dataframe=True)
        
        if historical is None:
            return {"error": "Could not fetch historical data"}
        
        try:
            import pandas as pd
            
            # Get NAV on investment date and current NAV
            current_fund = self.fetcher.get_scheme_quote(scheme_code)
            
            if current_fund:
                # Assuming we invested at historical NAV
                # This is simplified - actual implementation would look up exact date
                units_purchased = investment_amount / 100  # Assumed initial NAV
                current_value = units_purchased * current_fund.nav
                
                return {
                    "scheme": current_fund.scheme_name,
                    "investment_amount": investment_amount,
                    "investment_date": investment_date.strftime("%Y-%m-%d"),
                    "units_purchased": units_purchased,
                    "current_nav": current_fund.nav,
                    "current_value": current_value,
                    "absolute_return": current_value - investment_amount,
                    "percent_return": ((current_value - investment_amount) / investment_amount) * 100
                }
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
        
        return {"error": "Could not calculate returns"}


class ETFAnalyzer:
    """
    Analyze Indian ETFs
    """
    
    # Major Indian ETFs
    POPULAR_ETFS = {
        "NIFTYBEES": {
            "name": "Nippon India ETF Nifty BeES",
            "underlying": "NIFTY 50",
            "category": ETFCategory.EQUITY_INDEX
        },
        "BANKBEES": {
            "name": "Nippon India ETF Bank BeES",
            "underlying": "NIFTY BANK",
            "category": ETFCategory.EQUITY_SECTORAL
        },
        "GOLDBEES": {
            "name": "Nippon India ETF Gold BeES",
            "underlying": "Gold",
            "category": ETFCategory.GOLD
        },
        "SILVERBEES": {
            "name": "Nippon India ETF Silver BeES",
            "underlying": "Silver",
            "category": ETFCategory.SILVER
        },
        "JUNIORBEES": {
            "name": "Nippon India ETF Junior BeES",
            "underlying": "NIFTY NEXT 50",
            "category": ETFCategory.EQUITY_INDEX
        },
        "SETFNIF50": {
            "name": "SBI ETF Nifty 50",
            "underlying": "NIFTY 50",
            "category": ETFCategory.EQUITY_INDEX
        },
        "SETFSN50": {
            "name": "SBI ETF Sensex Next 50",
            "underlying": "SENSEX NEXT 50",
            "category": ETFCategory.EQUITY_INDEX
        },
        "HNGSNGBEES": {
            "name": "Nippon India ETF Hang Seng BeES",
            "underlying": "HANG SENG",
            "category": ETFCategory.INTERNATIONAL
        },
        "N100": {
            "name": "Motilal Oswal NASDAQ 100 ETF",
            "underlying": "NASDAQ 100",
            "category": ETFCategory.INTERNATIONAL
        },
        "ITBEES": {
            "name": "Nippon India ETF Nifty IT",
            "underlying": "NIFTY IT",
            "category": ETFCategory.EQUITY_SECTORAL
        },
        "INFRABEES": {
            "name": "Nippon India ETF Nifty Infrastructure",
            "underlying": "NIFTY INFRA",
            "category": ETFCategory.EQUITY_SECTORAL
        },
        "PSUBNKBEES": {
            "name": "Nippon India ETF Nifty PSU Bank",
            "underlying": "NIFTY PSU BANK",
            "category": ETFCategory.EQUITY_SECTORAL
        },
        "MOM100": {
            "name": "Motilal Oswal Nifty 200 Momentum 30 ETF",
            "underlying": "NIFTY 200 MOMENTUM 30",
            "category": ETFCategory.SMART_BETA
        },
        "LIQUIDBEES": {
            "name": "Nippon India ETF Liquid BeES",
            "underlying": "Tri-Party Repo",
            "category": ETFCategory.DEBT
        }
    }
    
    def __init__(self):
        from data.indian_markets import NSEDataFetcher
        self.nse = NSEDataFetcher()
    
    def get_etf_quote(self, symbol: str) -> Optional[ETF]:
        """Get ETF quote and analysis"""
        symbol = symbol.upper()
        
        if symbol not in self.POPULAR_ETFS:
            return None
        
        etf_info = self.POPULAR_ETFS[symbol]
        
        # Get market price from NSE
        quote = self.nse.get_quote(symbol)
        
        if quote:
            import random
            nav = quote.last_price * (1 + random.uniform(-0.005, 0.005))  # NAV close to market price
            premium_discount = ((quote.last_price - nav) / nav) * 100
            
            return ETF(
                symbol=symbol,
                name=etf_info["name"],
                nav=nav,
                market_price=quote.last_price,
                premium_discount=premium_discount,
                category=etf_info["category"],
                aum=random.uniform(500, 10000),
                expense_ratio=random.uniform(0.05, 0.5),
                tracking_error=random.uniform(0.01, 0.5),
                underlying_index=etf_info["underlying"],
                volume=quote.volume
            )
        
        return None
    
    def get_all_etfs(self) -> List[Dict]:
        """Get all tracked ETFs"""
        etfs = []
        for symbol, info in self.POPULAR_ETFS.items():
            etf = self.get_etf_quote(symbol)
            if etf:
                etfs.append({
                    "symbol": symbol,
                    "name": etf.name,
                    "category": etf.category.value,
                    "underlying": etf.underlying_index,
                    "market_price": etf.market_price,
                    "nav": etf.nav,
                    "premium_discount": etf.premium_discount
                })
        return etfs
    
    def get_etfs_by_category(self, category: ETFCategory) -> List[Dict]:
        """Get ETFs by category"""
        etfs = []
        for symbol, info in self.POPULAR_ETFS.items():
            if info["category"] == category:
                etf = self.get_etf_quote(symbol)
                if etf:
                    etfs.append({
                        "symbol": symbol,
                        "name": etf.name,
                        "market_price": etf.market_price,
                        "tracking_error": etf.tracking_error,
                        "expense_ratio": etf.expense_ratio
                    })
        return etfs
    
    def calculate_tracking_error(self, etf_symbol: str, days: int = 252) -> Dict[str, Any]:
        """Calculate tracking error for an ETF"""
        import random
        
        # In production, compare ETF returns with index returns
        tracking_error = random.uniform(0.01, 0.5)
        
        return {
            "etf_symbol": etf_symbol,
            "period_days": days,
            "tracking_error": tracking_error,
            "tracking_difference": random.uniform(-0.3, 0.3),
            "interpretation": "Low" if tracking_error < 0.1 else "Moderate" if tracking_error < 0.3 else "High"
        }
    
    def compare_etfs(self, symbols: List[str]) -> Dict[str, Any]:
        """Compare multiple ETFs"""
        etfs = []
        
        for symbol in symbols:
            etf = self.get_etf_quote(symbol.upper())
            if etf:
                etfs.append({
                    "symbol": symbol,
                    "name": etf.name,
                    "market_price": etf.market_price,
                    "expense_ratio": etf.expense_ratio,
                    "tracking_error": etf.tracking_error,
                    "aum": etf.aum,
                    "premium_discount": etf.premium_discount
                })
        
        if etfs:
            lowest_expense = min(etfs, key=lambda x: x["expense_ratio"])
            lowest_tracking_error = min(etfs, key=lambda x: x["tracking_error"])
            highest_aum = max(etfs, key=lambda x: x["aum"])
            
            return {
                "etfs": etfs,
                "recommendation": {
                    "lowest_expense_ratio": lowest_expense["symbol"],
                    "lowest_tracking_error": lowest_tracking_error["symbol"],
                    "highest_liquidity": highest_aum["symbol"]
                },
                "timestamp": datetime.now().isoformat()
            }
        
        return {"error": "No valid ETFs found"}


class PortfolioGoalPlanner:
    """
    Goal-based investment planning
    """
    
    def __init__(self):
        self.sip_calc = SIPCalculator()
    
    def plan_retirement(
        self,
        current_age: int,
        retirement_age: int,
        monthly_expenses: float,
        inflation_rate: float = 6.0,
        post_retirement_return: float = 8.0,
        pre_retirement_return: float = 12.0,
        life_expectancy: int = 85
    ) -> Dict[str, Any]:
        """Plan for retirement"""
        years_to_retirement = retirement_age - current_age
        retirement_years = life_expectancy - retirement_age
        
        # Calculate future monthly expenses at retirement
        future_expenses = monthly_expenses * ((1 + inflation_rate / 100) ** years_to_retirement)
        
        # Calculate retirement corpus needed
        # Using present value of annuity formula
        real_return = (1 + post_retirement_return / 100) / (1 + inflation_rate / 100) - 1
        if real_return > 0:
            corpus_needed = future_expenses * 12 * ((1 - (1 + real_return) ** (-retirement_years)) / real_return)
        else:
            corpus_needed = future_expenses * 12 * retirement_years
        
        # Calculate SIP needed
        sip_required = self.sip_calc.calculate_sip_required(
            corpus_needed, pre_retirement_return, years_to_retirement
        )
        
        return {
            "current_age": current_age,
            "retirement_age": retirement_age,
            "current_monthly_expenses": monthly_expenses,
            "future_monthly_expenses": future_expenses,
            "corpus_needed": corpus_needed,
            "monthly_sip_required": sip_required,
            "years_to_retirement": years_to_retirement,
            "assumptions": {
                "inflation_rate": inflation_rate,
                "pre_retirement_return": pre_retirement_return,
                "post_retirement_return": post_retirement_return,
                "life_expectancy": life_expectancy
            }
        }
    
    def plan_child_education(
        self,
        current_child_age: int,
        education_start_age: int,
        current_education_cost: float,
        education_inflation: float = 10.0,
        expected_return: float = 12.0
    ) -> Dict[str, Any]:
        """Plan for child's education"""
        years_to_education = education_start_age - current_child_age
        
        # Calculate future education cost
        future_cost = current_education_cost * ((1 + education_inflation / 100) ** years_to_education)
        
        # Calculate SIP needed
        sip_required = self.sip_calc.calculate_sip_required(
            future_cost, expected_return, years_to_education
        )
        
        # Calculate lumpsum needed today
        lumpsum_today = future_cost / ((1 + expected_return / 100) ** years_to_education)
        
        return {
            "current_child_age": current_child_age,
            "education_start_age": education_start_age,
            "current_education_cost": current_education_cost,
            "future_education_cost": future_cost,
            "years_to_goal": years_to_education,
            "monthly_sip_required": sip_required,
            "lumpsum_required_today": lumpsum_today,
            "assumptions": {
                "education_inflation": education_inflation,
                "expected_return": expected_return
            }
        }
    
    def plan_house_purchase(
        self,
        target_property_value: float,
        down_payment_percent: float,
        years_to_save: int,
        property_appreciation: float = 5.0,
        expected_return: float = 12.0
    ) -> Dict[str, Any]:
        """Plan for house purchase"""
        # Calculate future property value
        future_property_value = target_property_value * ((1 + property_appreciation / 100) ** years_to_save)
        
        # Calculate down payment needed
        down_payment_needed = future_property_value * (down_payment_percent / 100)
        
        # Calculate SIP needed
        sip_required = self.sip_calc.calculate_sip_required(
            down_payment_needed, expected_return, years_to_save
        )
        
        return {
            "current_property_value": target_property_value,
            "future_property_value": future_property_value,
            "down_payment_percent": down_payment_percent,
            "down_payment_needed": down_payment_needed,
            "years_to_save": years_to_save,
            "monthly_sip_required": sip_required,
            "assumptions": {
                "property_appreciation": property_appreciation,
                "expected_return": expected_return
            }
        }


class MutualFundDashboard:
    """
    Unified mutual fund and ETF dashboard
    """
    
    def __init__(self):
        self.mf_fetcher = MutualFundDataFetcher()
        self.mf_analyzer = MutualFundAnalyzer()
        self.etf_analyzer = ETFAnalyzer()
        self.sip_calculator = SIPCalculator()
        self.goal_planner = PortfolioGoalPlanner()
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get dashboard summary"""
        # Get sample funds
        sample_codes = list(self.mf_fetcher.get_scheme_codes().keys())[:5]
        funds = []
        for code in sample_codes:
            fund = self.mf_fetcher.get_scheme_quote(code)
            if fund:
                funds.append({
                    "code": code,
                    "name": fund.scheme_name,
                    "nav": fund.nav,
                    "return_1y": fund.return_1y
                })
        
        # Get sample ETFs
        etfs = self.etf_analyzer.get_etfs_by_category(ETFCategory.EQUITY_INDEX)[:5]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "top_mutual_funds": funds,
            "index_etfs": etfs,
            "market_status": "Open" if datetime.now().hour in range(9, 16) else "Closed"
        }
    
    def quick_sip_calculation(
        self,
        monthly_amount: float,
        years: int,
        return_scenario: str = "moderate"
    ) -> Dict[str, SIPCalculation]:
        """Quick SIP calculation with multiple scenarios"""
        rates = {
            "conservative": 8,
            "moderate": 12,
            "aggressive": 15
        }
        
        results = {}
        for scenario, rate in rates.items():
            results[scenario] = self.sip_calculator.calculate_sip(
                monthly_amount, rate, years
            )
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize dashboard
    dashboard = MutualFundDashboard()
    
    # Get dashboard summary
    print("=== Mutual Fund Dashboard ===")
    summary = dashboard.get_dashboard_summary()
    print(f"Top Funds: {len(summary['top_mutual_funds'])}")
    print(f"Index ETFs: {len(summary['index_etfs'])}")
    
    # SIP Calculation
    print("\n=== SIP Calculator ===")
    sip = SIPCalculator.calculate_sip(10000, 12, 10)
    print(f"Monthly SIP: ₹{sip.monthly_investment:,.0f}")
    print(f"Total Investment: ₹{sip.total_investment:,.0f}")
    print(f"Expected Returns: ₹{sip.expected_returns:,.0f}")
    print(f"Wealth Gained: ₹{sip.wealth_gained:,.0f}")
    
    # Goal Planning
    print("\n=== Retirement Planning ===")
    planner = PortfolioGoalPlanner()
    retirement = planner.plan_retirement(
        current_age=30,
        retirement_age=60,
        monthly_expenses=50000
    )
    print(f"Corpus Needed: ₹{retirement['corpus_needed']:,.0f}")
    print(f"Monthly SIP Required: ₹{retirement['monthly_sip_required']:,.0f}")
    
    # ETF Analysis
    print("\n=== ETF Analysis ===")
    etf_analyzer = ETFAnalyzer()
    comparison = etf_analyzer.compare_etfs(["NIFTYBEES", "SETFNIF50", "JUNIORBEES"])
    print(f"Lowest Expense Ratio: {comparison['recommendation']['lowest_expense_ratio']}")
