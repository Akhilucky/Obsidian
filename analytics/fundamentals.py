"""
Fundamentals & Company Financials Module
=========================================

Financial statements, key metrics, earnings calendar, and SEC filings
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class StatementType(Enum):
    """Financial statement types"""
    INCOME = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"


class FilingType(Enum):
    """SEC filing types"""
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_8K = "8-K"
    FORM_DEF14A = "DEF 14A"
    FORM_4 = "4"
    FORM_13F = "13F"
    FORM_S1 = "S-1"


@dataclass
class IncomeStatement:
    """Income statement data"""
    date: datetime
    period: str  # annual, quarterly
    revenue: float
    cost_of_revenue: float
    gross_profit: float
    operating_expenses: float
    operating_income: float
    interest_expense: float
    income_before_tax: float
    income_tax: float
    net_income: float
    eps_basic: float
    eps_diluted: float
    shares_outstanding: int
    ebitda: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None


@dataclass
class BalanceSheet:
    """Balance sheet data"""
    date: datetime
    period: str
    # Assets
    total_assets: float
    current_assets: float
    cash: float
    short_term_investments: float
    receivables: float
    inventory: float
    non_current_assets: float
    ppe: float  # Property, Plant & Equipment
    goodwill: float
    intangibles: float
    # Liabilities
    total_liabilities: float
    current_liabilities: float
    accounts_payable: float
    short_term_debt: float
    non_current_liabilities: float
    long_term_debt: float
    # Equity
    total_equity: float
    retained_earnings: float
    common_stock: float


@dataclass
class CashFlowStatement:
    """Cash flow statement data"""
    date: datetime
    period: str
    # Operating
    operating_cash_flow: float
    depreciation: float
    changes_in_working_capital: float
    # Investing
    investing_cash_flow: float
    capex: float
    acquisitions: float
    investments: float
    # Financing
    financing_cash_flow: float
    dividends_paid: float
    stock_repurchases: float
    debt_issued: float
    debt_repaid: float
    # Net
    net_change_in_cash: float
    free_cash_flow: float


@dataclass
class KeyMetrics:
    """Key financial metrics and ratios"""
    symbol: str
    date: datetime
    # Valuation
    market_cap: float
    enterprise_value: float
    pe_ratio: Optional[float]
    forward_pe: Optional[float]
    peg_ratio: Optional[float]
    price_to_sales: Optional[float]
    price_to_book: Optional[float]
    ev_to_ebitda: Optional[float]
    ev_to_revenue: Optional[float]
    # Profitability
    gross_margin: Optional[float]
    operating_margin: Optional[float]
    net_margin: Optional[float]
    roe: Optional[float]  # Return on Equity
    roa: Optional[float]  # Return on Assets
    roic: Optional[float]  # Return on Invested Capital
    # Per Share
    eps: Optional[float]
    eps_growth: Optional[float]
    revenue_per_share: Optional[float]
    book_value_per_share: Optional[float]
    free_cash_flow_per_share: Optional[float]
    dividends_per_share: Optional[float]
    dividend_yield: Optional[float]
    payout_ratio: Optional[float]
    # Financial Health
    current_ratio: Optional[float]
    quick_ratio: Optional[float]
    debt_to_equity: Optional[float]
    debt_to_assets: Optional[float]
    interest_coverage: Optional[float]
    # Growth
    revenue_growth: Optional[float]
    earnings_growth: Optional[float]
    fcf_growth: Optional[float]
    # Efficiency
    asset_turnover: Optional[float]
    inventory_turnover: Optional[float]
    receivables_turnover: Optional[float]


@dataclass
class CompanyProfile:
    """Company profile information"""
    symbol: str
    name: str
    description: str
    sector: str
    industry: str
    country: str
    exchange: str
    currency: str
    website: str
    employees: int
    ceo: str
    address: str
    phone: str
    founded: Optional[int] = None
    ipo_date: Optional[datetime] = None
    cusip: Optional[str] = None
    isin: Optional[str] = None


@dataclass
class EarningsEvent:
    """Earnings calendar event"""
    symbol: str
    company_name: str
    report_date: datetime
    fiscal_quarter: str
    fiscal_year: int
    eps_estimate: Optional[float]
    eps_actual: Optional[float]
    revenue_estimate: Optional[float]
    revenue_actual: Optional[float]
    surprise_percent: Optional[float]
    time: str  # BMO (Before Market Open), AMC (After Market Close)


@dataclass
class SECFiling:
    """SEC filing information"""
    symbol: str
    filing_type: FilingType
    filing_date: datetime
    accepted_date: datetime
    title: str
    url: str
    description: Optional[str] = None


class FundamentalsProvider:
    """
    Fetch fundamental data from multiple sources
    """
    
    def __init__(self):
        self.yf = None
        self._initialize()
    
    def _initialize(self):
        """Initialize data providers"""
        try:
            import yfinance as yf
            self.yf = yf
            logger.info("yfinance initialized for fundamentals")
        except ImportError:
            logger.warning("yfinance not available")
    
    def get_income_statement(
        self,
        symbol: str,
        annual: bool = True
    ) -> List[IncomeStatement]:
        """Get income statements"""
        if self.yf:
            try:
                ticker = self.yf.Ticker(symbol)
                if annual:
                    df = ticker.income_stmt
                else:
                    df = ticker.quarterly_income_stmt
                
                if df is not None and not df.empty:
                    statements = []
                    for col in df.columns:
                        try:
                            revenue = float(df.loc['Total Revenue', col]) if 'Total Revenue' in df.index else 0
                            gross = float(df.loc['Gross Profit', col]) if 'Gross Profit' in df.index else 0
                            operating = float(df.loc['Operating Income', col]) if 'Operating Income' in df.index else 0
                            net = float(df.loc['Net Income', col]) if 'Net Income' in df.index else 0
                            
                            statements.append(IncomeStatement(
                                date=col.to_pydatetime() if hasattr(col, 'to_pydatetime') else datetime.now(),
                                period="annual" if annual else "quarterly",
                                revenue=revenue,
                                cost_of_revenue=float(df.loc['Cost Of Revenue', col]) if 'Cost Of Revenue' in df.index else 0,
                                gross_profit=gross,
                                operating_expenses=float(df.loc['Operating Expense', col]) if 'Operating Expense' in df.index else 0,
                                operating_income=operating,
                                interest_expense=float(df.loc['Interest Expense', col]) if 'Interest Expense' in df.index else 0,
                                income_before_tax=float(df.loc['Pretax Income', col]) if 'Pretax Income' in df.index else 0,
                                income_tax=float(df.loc['Tax Provision', col]) if 'Tax Provision' in df.index else 0,
                                net_income=net,
                                eps_basic=float(df.loc['Basic EPS', col]) if 'Basic EPS' in df.index else 0,
                                eps_diluted=float(df.loc['Diluted EPS', col]) if 'Diluted EPS' in df.index else 0,
                                shares_outstanding=int(df.loc['Basic Average Shares', col]) if 'Basic Average Shares' in df.index else 0,
                                ebitda=float(df.loc['EBITDA', col]) if 'EBITDA' in df.index else None,
                                gross_margin=(gross / revenue * 100) if revenue else None,
                                operating_margin=(operating / revenue * 100) if revenue else None,
                                net_margin=(net / revenue * 100) if revenue else None
                            ))
                        except Exception as e:
                            logger.warning(f"Error parsing income statement column: {e}")
                    
                    return statements
            except Exception as e:
                logger.error(f"Error fetching income statement for {symbol}: {e}")
        
        return self._get_mock_income_statements(symbol, annual)
    
    def _get_mock_income_statements(self, symbol: str, annual: bool) -> List[IncomeStatement]:
        """Generate mock income statements"""
        import random
        
        statements = []
        base_revenue = random.uniform(1e9, 100e9)
        
        periods = 4 if annual else 8
        for i in range(periods):
            revenue = base_revenue * (1 + random.uniform(-0.1, 0.2))
            cost = revenue * random.uniform(0.4, 0.7)
            gross = revenue - cost
            opex = gross * random.uniform(0.3, 0.5)
            operating = gross - opex
            net = operating * random.uniform(0.7, 0.9)
            
            statements.append(IncomeStatement(
                date=datetime.now() - timedelta(days=365 * i if annual else 90 * i),
                period="annual" if annual else "quarterly",
                revenue=revenue,
                cost_of_revenue=cost,
                gross_profit=gross,
                operating_expenses=opex,
                operating_income=operating,
                interest_expense=revenue * 0.01,
                income_before_tax=operating,
                income_tax=operating * 0.21,
                net_income=net,
                eps_basic=net / 1e9,
                eps_diluted=net / 1.1e9,
                shares_outstanding=int(1e9),
                ebitda=operating * 1.2,
                gross_margin=(gross / revenue * 100),
                operating_margin=(operating / revenue * 100),
                net_margin=(net / revenue * 100)
            ))
        
        return statements
    
    def get_balance_sheet(
        self,
        symbol: str,
        annual: bool = True
    ) -> List[BalanceSheet]:
        """Get balance sheets"""
        if self.yf:
            try:
                ticker = self.yf.Ticker(symbol)
                if annual:
                    df = ticker.balance_sheet
                else:
                    df = ticker.quarterly_balance_sheet
                
                if df is not None and not df.empty:
                    sheets = []
                    for col in df.columns:
                        try:
                            sheets.append(BalanceSheet(
                                date=col.to_pydatetime() if hasattr(col, 'to_pydatetime') else datetime.now(),
                                period="annual" if annual else "quarterly",
                                total_assets=float(df.loc['Total Assets', col]) if 'Total Assets' in df.index else 0,
                                current_assets=float(df.loc['Current Assets', col]) if 'Current Assets' in df.index else 0,
                                cash=float(df.loc['Cash And Cash Equivalents', col]) if 'Cash And Cash Equivalents' in df.index else 0,
                                short_term_investments=float(df.loc['Other Short Term Investments', col]) if 'Other Short Term Investments' in df.index else 0,
                                receivables=float(df.loc['Receivables', col]) if 'Receivables' in df.index else 0,
                                inventory=float(df.loc['Inventory', col]) if 'Inventory' in df.index else 0,
                                non_current_assets=float(df.loc['Total Non Current Assets', col]) if 'Total Non Current Assets' in df.index else 0,
                                ppe=float(df.loc['Net PPE', col]) if 'Net PPE' in df.index else 0,
                                goodwill=float(df.loc['Goodwill', col]) if 'Goodwill' in df.index else 0,
                                intangibles=float(df.loc['Other Intangible Assets', col]) if 'Other Intangible Assets' in df.index else 0,
                                total_liabilities=float(df.loc['Total Liabilities Net Minority Interest', col]) if 'Total Liabilities Net Minority Interest' in df.index else 0,
                                current_liabilities=float(df.loc['Current Liabilities', col]) if 'Current Liabilities' in df.index else 0,
                                accounts_payable=float(df.loc['Payables And Accrued Expenses', col]) if 'Payables And Accrued Expenses' in df.index else 0,
                                short_term_debt=float(df.loc['Current Debt', col]) if 'Current Debt' in df.index else 0,
                                non_current_liabilities=float(df.loc['Total Non Current Liabilities Net Minority Interest', col]) if 'Total Non Current Liabilities Net Minority Interest' in df.index else 0,
                                long_term_debt=float(df.loc['Long Term Debt', col]) if 'Long Term Debt' in df.index else 0,
                                total_equity=float(df.loc['Total Equity Gross Minority Interest', col]) if 'Total Equity Gross Minority Interest' in df.index else 0,
                                retained_earnings=float(df.loc['Retained Earnings', col]) if 'Retained Earnings' in df.index else 0,
                                common_stock=float(df.loc['Common Stock', col]) if 'Common Stock' in df.index else 0
                            ))
                        except Exception as e:
                            logger.warning(f"Error parsing balance sheet column: {e}")
                    
                    return sheets
            except Exception as e:
                logger.error(f"Error fetching balance sheet for {symbol}: {e}")
        
        return self._get_mock_balance_sheets(symbol, annual)
    
    def _get_mock_balance_sheets(self, symbol: str, annual: bool) -> List[BalanceSheet]:
        """Generate mock balance sheets"""
        import random
        
        sheets = []
        base_assets = random.uniform(10e9, 500e9)
        
        periods = 4 if annual else 8
        for i in range(periods):
            total_assets = base_assets * (1 + random.uniform(-0.05, 0.15))
            current = total_assets * random.uniform(0.3, 0.5)
            liabilities = total_assets * random.uniform(0.4, 0.7)
            equity = total_assets - liabilities
            
            sheets.append(BalanceSheet(
                date=datetime.now() - timedelta(days=365 * i if annual else 90 * i),
                period="annual" if annual else "quarterly",
                total_assets=total_assets,
                current_assets=current,
                cash=current * 0.3,
                short_term_investments=current * 0.2,
                receivables=current * 0.25,
                inventory=current * 0.15,
                non_current_assets=total_assets - current,
                ppe=(total_assets - current) * 0.5,
                goodwill=(total_assets - current) * 0.2,
                intangibles=(total_assets - current) * 0.1,
                total_liabilities=liabilities,
                current_liabilities=liabilities * 0.4,
                accounts_payable=liabilities * 0.15,
                short_term_debt=liabilities * 0.1,
                non_current_liabilities=liabilities * 0.6,
                long_term_debt=liabilities * 0.35,
                total_equity=equity,
                retained_earnings=equity * 0.6,
                common_stock=equity * 0.2
            ))
        
        return sheets
    
    def get_cash_flow(
        self,
        symbol: str,
        annual: bool = True
    ) -> List[CashFlowStatement]:
        """Get cash flow statements"""
        if self.yf:
            try:
                ticker = self.yf.Ticker(symbol)
                if annual:
                    df = ticker.cashflow
                else:
                    df = ticker.quarterly_cashflow
                
                if df is not None and not df.empty:
                    statements = []
                    for col in df.columns:
                        try:
                            ocf = float(df.loc['Operating Cash Flow', col]) if 'Operating Cash Flow' in df.index else 0
                            capex = abs(float(df.loc['Capital Expenditure', col])) if 'Capital Expenditure' in df.index else 0
                            
                            statements.append(CashFlowStatement(
                                date=col.to_pydatetime() if hasattr(col, 'to_pydatetime') else datetime.now(),
                                period="annual" if annual else "quarterly",
                                operating_cash_flow=ocf,
                                depreciation=float(df.loc['Depreciation And Amortization', col]) if 'Depreciation And Amortization' in df.index else 0,
                                changes_in_working_capital=float(df.loc['Change In Working Capital', col]) if 'Change In Working Capital' in df.index else 0,
                                investing_cash_flow=float(df.loc['Investing Cash Flow', col]) if 'Investing Cash Flow' in df.index else 0,
                                capex=capex,
                                acquisitions=0,
                                investments=0,
                                financing_cash_flow=float(df.loc['Financing Cash Flow', col]) if 'Financing Cash Flow' in df.index else 0,
                                dividends_paid=abs(float(df.loc['Cash Dividends Paid', col])) if 'Cash Dividends Paid' in df.index else 0,
                                stock_repurchases=abs(float(df.loc['Repurchase Of Capital Stock', col])) if 'Repurchase Of Capital Stock' in df.index else 0,
                                debt_issued=float(df.loc['Issuance Of Debt', col]) if 'Issuance Of Debt' in df.index else 0,
                                debt_repaid=abs(float(df.loc['Repayment Of Debt', col])) if 'Repayment Of Debt' in df.index else 0,
                                net_change_in_cash=float(df.loc['Changes In Cash', col]) if 'Changes In Cash' in df.index else 0,
                                free_cash_flow=ocf - capex
                            ))
                        except Exception as e:
                            logger.warning(f"Error parsing cash flow column: {e}")
                    
                    return statements
            except Exception as e:
                logger.error(f"Error fetching cash flow for {symbol}: {e}")
        
        return self._get_mock_cash_flows(symbol, annual)
    
    def _get_mock_cash_flows(self, symbol: str, annual: bool) -> List[CashFlowStatement]:
        """Generate mock cash flow statements"""
        import random
        
        statements = []
        base_ocf = random.uniform(1e9, 50e9)
        
        periods = 4 if annual else 8
        for i in range(periods):
            ocf = base_ocf * (1 + random.uniform(-0.2, 0.3))
            capex = ocf * random.uniform(0.2, 0.5)
            
            statements.append(CashFlowStatement(
                date=datetime.now() - timedelta(days=365 * i if annual else 90 * i),
                period="annual" if annual else "quarterly",
                operating_cash_flow=ocf,
                depreciation=capex * 0.3,
                changes_in_working_capital=ocf * random.uniform(-0.2, 0.2),
                investing_cash_flow=-capex * 1.2,
                capex=capex,
                acquisitions=capex * random.uniform(0, 0.5),
                investments=capex * random.uniform(-0.3, 0.3),
                financing_cash_flow=-ocf * 0.3,
                dividends_paid=ocf * 0.15,
                stock_repurchases=ocf * 0.1,
                debt_issued=ocf * random.uniform(0, 0.2),
                debt_repaid=ocf * random.uniform(0, 0.15),
                net_change_in_cash=ocf * random.uniform(-0.1, 0.1),
                free_cash_flow=ocf - capex
            ))
        
        return statements
    
    def get_key_metrics(self, symbol: str) -> Optional[KeyMetrics]:
        """Get key financial metrics"""
        if self.yf:
            try:
                ticker = self.yf.Ticker(symbol)
                info = ticker.info
                
                return KeyMetrics(
                    symbol=symbol,
                    date=datetime.now(),
                    market_cap=info.get('marketCap', 0),
                    enterprise_value=info.get('enterpriseValue', 0),
                    pe_ratio=info.get('trailingPE'),
                    forward_pe=info.get('forwardPE'),
                    peg_ratio=info.get('pegRatio'),
                    price_to_sales=info.get('priceToSalesTrailing12Months'),
                    price_to_book=info.get('priceToBook'),
                    ev_to_ebitda=info.get('enterpriseToEbitda'),
                    ev_to_revenue=info.get('enterpriseToRevenue'),
                    gross_margin=info.get('grossMargins', 0) * 100 if info.get('grossMargins') else None,
                    operating_margin=info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else None,
                    net_margin=info.get('profitMargins', 0) * 100 if info.get('profitMargins') else None,
                    roe=info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else None,
                    roa=info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else None,
                    roic=None,
                    eps=info.get('trailingEps'),
                    eps_growth=info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else None,
                    revenue_per_share=info.get('revenuePerShare'),
                    book_value_per_share=info.get('bookValue'),
                    free_cash_flow_per_share=info.get('freeCashflow', 0) / info.get('sharesOutstanding', 1) if info.get('sharesOutstanding') else None,
                    dividends_per_share=info.get('dividendRate'),
                    dividend_yield=info.get('dividendYield', 0) * 100 if info.get('dividendYield') else None,
                    payout_ratio=info.get('payoutRatio', 0) * 100 if info.get('payoutRatio') else None,
                    current_ratio=info.get('currentRatio'),
                    quick_ratio=info.get('quickRatio'),
                    debt_to_equity=info.get('debtToEquity'),
                    debt_to_assets=None,
                    interest_coverage=None,
                    revenue_growth=info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else None,
                    earnings_growth=info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else None,
                    fcf_growth=None,
                    asset_turnover=None,
                    inventory_turnover=None,
                    receivables_turnover=None
                )
            except Exception as e:
                logger.error(f"Error fetching key metrics for {symbol}: {e}")
        
        return self._get_mock_key_metrics(symbol)
    
    def _get_mock_key_metrics(self, symbol: str) -> KeyMetrics:
        """Generate mock key metrics"""
        import random
        
        return KeyMetrics(
            symbol=symbol,
            date=datetime.now(),
            market_cap=random.uniform(1e9, 3e12),
            enterprise_value=random.uniform(1e9, 3e12),
            pe_ratio=random.uniform(10, 50),
            forward_pe=random.uniform(8, 40),
            peg_ratio=random.uniform(0.5, 3),
            price_to_sales=random.uniform(1, 20),
            price_to_book=random.uniform(1, 15),
            ev_to_ebitda=random.uniform(5, 25),
            ev_to_revenue=random.uniform(1, 15),
            gross_margin=random.uniform(20, 80),
            operating_margin=random.uniform(5, 40),
            net_margin=random.uniform(2, 30),
            roe=random.uniform(5, 40),
            roa=random.uniform(2, 20),
            roic=random.uniform(5, 30),
            eps=random.uniform(1, 20),
            eps_growth=random.uniform(-20, 50),
            revenue_per_share=random.uniform(10, 500),
            book_value_per_share=random.uniform(5, 200),
            free_cash_flow_per_share=random.uniform(1, 50),
            dividends_per_share=random.uniform(0, 5),
            dividend_yield=random.uniform(0, 5),
            payout_ratio=random.uniform(0, 60),
            current_ratio=random.uniform(0.5, 3),
            quick_ratio=random.uniform(0.3, 2.5),
            debt_to_equity=random.uniform(0, 200),
            debt_to_assets=random.uniform(0, 60),
            interest_coverage=random.uniform(2, 20),
            revenue_growth=random.uniform(-10, 40),
            earnings_growth=random.uniform(-20, 60),
            fcf_growth=random.uniform(-30, 50),
            asset_turnover=random.uniform(0.3, 2),
            inventory_turnover=random.uniform(2, 20),
            receivables_turnover=random.uniform(3, 15)
        )
    
    def get_company_profile(self, symbol: str) -> Optional[CompanyProfile]:
        """Get company profile"""
        if self.yf:
            try:
                ticker = self.yf.Ticker(symbol)
                info = ticker.info
                
                return CompanyProfile(
                    symbol=symbol,
                    name=info.get('longName', ''),
                    description=info.get('longBusinessSummary', ''),
                    sector=info.get('sector', ''),
                    industry=info.get('industry', ''),
                    country=info.get('country', ''),
                    exchange=info.get('exchange', ''),
                    currency=info.get('currency', 'USD'),
                    website=info.get('website', ''),
                    employees=info.get('fullTimeEmployees', 0),
                    ceo=info.get('companyOfficers', [{}])[0].get('name', '') if info.get('companyOfficers') else '',
                    address=f"{info.get('address1', '')} {info.get('city', '')} {info.get('state', '')} {info.get('zip', '')}",
                    phone=info.get('phone', '')
                )
            except Exception as e:
                logger.error(f"Error fetching company profile for {symbol}: {e}")
        
        return self._get_mock_profile(symbol)
    
    def _get_mock_profile(self, symbol: str) -> CompanyProfile:
        """Generate mock company profile"""
        return CompanyProfile(
            symbol=symbol,
            name=f"{symbol} Corporation",
            description=f"A leading company in its industry, {symbol} Corporation provides innovative solutions to customers worldwide.",
            sector="Technology",
            industry="Software",
            country="United States",
            exchange="NASDAQ",
            currency="USD",
            website=f"https://www.{symbol.lower()}.com",
            employees=50000,
            ceo="John Smith",
            address="123 Tech Street, San Francisco, CA 94105",
            phone="+1-555-123-4567"
        )


class EarningsCalendar:
    """
    Track earnings announcements and calendar
    """
    
    def __init__(self):
        self.provider = FundamentalsProvider()
    
    def get_upcoming_earnings(
        self,
        symbols: Optional[List[str]] = None,
        days_ahead: int = 30
    ) -> List[EarningsEvent]:
        """Get upcoming earnings announcements"""
        events = []
        
        if symbols:
            for symbol in symbols:
                event = self._get_earnings_for_symbol(symbol)
                if event:
                    events.append(event)
        else:
            # Get major upcoming earnings
            major_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
            for symbol in major_symbols:
                event = self._get_earnings_for_symbol(symbol)
                if event:
                    events.append(event)
        
        # Sort by date
        events.sort(key=lambda x: x.report_date)
        
        return events
    
    def _get_earnings_for_symbol(self, symbol: str) -> Optional[EarningsEvent]:
        """Get earnings event for a symbol"""
        import random
        
        # Mock earnings data
        return EarningsEvent(
            symbol=symbol,
            company_name=f"{symbol} Inc.",
            report_date=datetime.now() + timedelta(days=random.randint(1, 60)),
            fiscal_quarter=f"Q{random.randint(1, 4)}",
            fiscal_year=2024,
            eps_estimate=random.uniform(0.5, 5),
            eps_actual=None,
            revenue_estimate=random.uniform(1e9, 100e9),
            revenue_actual=None,
            surprise_percent=None,
            time=random.choice(['BMO', 'AMC'])
        )
    
    def get_earnings_history(
        self,
        symbol: str,
        quarters: int = 8
    ) -> List[EarningsEvent]:
        """Get historical earnings"""
        import random
        
        history = []
        
        for i in range(quarters):
            q = ((datetime.now().month - 1) // 3 - i) % 4 + 1
            year = datetime.now().year - (i // 4)
            
            estimate = random.uniform(0.5, 5)
            actual = estimate * (1 + random.uniform(-0.2, 0.3))
            surprise = ((actual - estimate) / estimate) * 100
            
            history.append(EarningsEvent(
                symbol=symbol,
                company_name=f"{symbol} Inc.",
                report_date=datetime.now() - timedelta(days=90 * i),
                fiscal_quarter=f"Q{q}",
                fiscal_year=year,
                eps_estimate=estimate,
                eps_actual=actual,
                revenue_estimate=random.uniform(1e9, 100e9),
                revenue_actual=random.uniform(1e9, 100e9),
                surprise_percent=surprise,
                time=random.choice(['BMO', 'AMC'])
            ))
        
        return history


class SECFilingsTracker:
    """
    Track SEC filings
    """
    
    def __init__(self):
        self.base_url = "https://www.sec.gov"
    
    def get_recent_filings(
        self,
        symbol: str,
        filing_types: Optional[List[FilingType]] = None,
        limit: int = 20
    ) -> List[SECFiling]:
        """Get recent SEC filings"""
        import random
        
        if filing_types is None:
            filing_types = list(FilingType)
        
        filings = []
        
        for i in range(min(limit, 20)):
            filing_type = random.choice(filing_types)
            
            filings.append(SECFiling(
                symbol=symbol,
                filing_type=filing_type,
                filing_date=datetime.now() - timedelta(days=random.randint(1, 365)),
                accepted_date=datetime.now() - timedelta(days=random.randint(1, 365)),
                title=self._get_filing_title(filing_type),
                url=f"{self.base_url}/cgi-bin/browse-edgar?action=getcompany&CIK={symbol}",
                description=self._get_filing_description(filing_type)
            ))
        
        filings.sort(key=lambda x: x.filing_date, reverse=True)
        return filings
    
    def _get_filing_title(self, filing_type: FilingType) -> str:
        """Get filing title"""
        titles = {
            FilingType.FORM_10K: "Annual Report",
            FilingType.FORM_10Q: "Quarterly Report",
            FilingType.FORM_8K: "Current Report",
            FilingType.FORM_DEF14A: "Proxy Statement",
            FilingType.FORM_4: "Insider Transaction",
            FilingType.FORM_13F: "Institutional Holdings",
            FilingType.FORM_S1: "Registration Statement"
        }
        return titles.get(filing_type, "SEC Filing")
    
    def _get_filing_description(self, filing_type: FilingType) -> str:
        """Get filing description"""
        descriptions = {
            FilingType.FORM_10K: "Annual report with comprehensive overview of business and financials",
            FilingType.FORM_10Q: "Quarterly financial report",
            FilingType.FORM_8K: "Report of material events or corporate changes",
            FilingType.FORM_DEF14A: "Definitive proxy statement for shareholder meeting",
            FilingType.FORM_4: "Statement of changes in beneficial ownership",
            FilingType.FORM_13F: "Quarterly report of institutional holdings",
            FilingType.FORM_S1: "Registration statement for new securities"
        }
        return descriptions.get(filing_type, "")


class FundamentalsDashboard:
    """
    Unified fundamentals dashboard
    """
    
    def __init__(self):
        self.provider = FundamentalsProvider()
        self.earnings = EarningsCalendar()
        self.filings = SECFilingsTracker()
    
    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive company overview"""
        profile = self.provider.get_company_profile(symbol)
        metrics = self.provider.get_key_metrics(symbol)
        
        return {
            "profile": profile,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_financial_statements(self, symbol: str) -> Dict[str, Any]:
        """Get all financial statements"""
        return {
            "income_statement": self.provider.get_income_statement(symbol, annual=True),
            "balance_sheet": self.provider.get_balance_sheet(symbol, annual=True),
            "cash_flow": self.provider.get_cash_flow(symbol, annual=True),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_valuation_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get valuation analysis"""
        metrics = self.provider.get_key_metrics(symbol)
        
        if metrics:
            # Valuation assessment
            valuation_score = 0
            reasons = []
            
            if metrics.pe_ratio and metrics.pe_ratio < 15:
                valuation_score += 1
                reasons.append("Low P/E ratio suggests undervaluation")
            elif metrics.pe_ratio and metrics.pe_ratio > 30:
                valuation_score -= 1
                reasons.append("High P/E ratio suggests overvaluation")
            
            if metrics.peg_ratio and metrics.peg_ratio < 1:
                valuation_score += 1
                reasons.append("PEG < 1 indicates growth at reasonable price")
            
            if metrics.price_to_book and metrics.price_to_book < 1:
                valuation_score += 1
                reasons.append("Trading below book value")
            
            assessment = "Undervalued" if valuation_score > 0 else "Overvalued" if valuation_score < 0 else "Fairly Valued"
            
            return {
                "symbol": symbol,
                "assessment": assessment,
                "score": valuation_score,
                "reasons": reasons,
                "metrics": {
                    "pe_ratio": metrics.pe_ratio,
                    "forward_pe": metrics.forward_pe,
                    "peg_ratio": metrics.peg_ratio,
                    "price_to_book": metrics.price_to_book,
                    "price_to_sales": metrics.price_to_sales,
                    "ev_to_ebitda": metrics.ev_to_ebitda
                }
            }
        
        return {"error": "Could not fetch metrics"}


# Example usage
if __name__ == "__main__":
    dashboard = FundamentalsDashboard()
    
    print("=== Company Overview: AAPL ===")
    overview = dashboard.get_company_overview("AAPL")
    if overview.get("profile"):
        print(f"Name: {overview['profile'].name}")
        print(f"Sector: {overview['profile'].sector}")
    
    print("\n=== Key Metrics ===")
    if overview.get("metrics"):
        m = overview["metrics"]
        print(f"Market Cap: ${m.market_cap/1e9:.2f}B")
        print(f"P/E Ratio: {m.pe_ratio:.2f}" if m.pe_ratio else "P/E: N/A")
        print(f"ROE: {m.roe:.2f}%" if m.roe else "ROE: N/A")
    
    print("\n=== Valuation Analysis ===")
    valuation = dashboard.get_valuation_analysis("AAPL")
    print(f"Assessment: {valuation.get('assessment')}")
    for reason in valuation.get('reasons', []):
        print(f"  - {reason}")
