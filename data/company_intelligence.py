"""
Company Intelligence Module
=============================
Comprehensive company research including:
- Ownership & subsidiaries
- Institutional investors
- Supply chain relationships
- Competitor analysis
- News aggregation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import requests
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class RelationshipType(Enum):
    """Types of company relationships"""
    SUBSIDIARY = "subsidiary"
    PARENT = "parent"
    SUPPLIER = "supplier"
    CUSTOMER = "customer"
    PARTNER = "partner"
    COMPETITOR = "competitor"
    INVESTOR = "investor"
    INVESTEE = "investee"


@dataclass
class CompanyRelationship:
    """Represents a relationship between two companies"""
    source_company: str
    target_company: str
    relationship_type: RelationshipType
    description: str = ""
    percentage: Optional[float] = None  # For ownership
    value: Optional[float] = None  # For investments/contracts
    start_date: Optional[datetime] = None
    source: str = ""


@dataclass
class NewsArticle:
    """Represents a news article"""
    title: str
    summary: str
    source: str
    url: str
    published_at: datetime
    sentiment: Optional[float] = None
    relevance: float = 1.0
    symbols: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)


@dataclass
class InstitutionalHolder:
    """Institutional investor holding"""
    name: str
    shares: int
    value: float
    percent_held: float
    change_shares: int = 0
    change_percent: float = 0
    report_date: Optional[datetime] = None


@dataclass
class CompanyProfile:
    """Complete company profile"""
    symbol: str
    name: str
    sector: str
    industry: str
    country: str
    market_cap: float
    description: str
    website: str
    employees: int = 0
    ceo: str = ""
    founded: str = ""
    headquarters: str = ""
    
    # Ownership
    subsidiaries: List[str] = field(default_factory=list)
    parent_company: Optional[str] = None
    major_shareholders: List[InstitutionalHolder] = field(default_factory=list)
    
    # Relationships
    suppliers: List[str] = field(default_factory=list)
    customers: List[str] = field(default_factory=list)
    competitors: List[str] = field(default_factory=list)
    partners: List[str] = field(default_factory=list)


class CompanyIntelligence:
    """
    Company Intelligence & Research
    =================================
    Get comprehensive company information including:
    - Company profile and fundamentals
    - Ownership structure
    - Institutional holders
    - Supply chain relationships
    - News and sentiment
    """
    
    def __init__(self, newsapi_key: Optional[str] = None, 
                 alphavantage_key: Optional[str] = None):
        self.newsapi_key = newsapi_key
        self.alphavantage_key = alphavantage_key
        self._cache: Dict[str, Any] = {}
    
    def get_company_profile(self, symbol: str) -> Optional[CompanyProfile]:
        """Get comprehensive company profile"""
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            profile = CompanyProfile(
                symbol=symbol,
                name=info.get('longName', info.get('shortName', symbol)),
                sector=info.get('sector', 'Unknown'),
                industry=info.get('industry', 'Unknown'),
                country=info.get('country', 'Unknown'),
                market_cap=info.get('marketCap', 0),
                description=info.get('longBusinessSummary', ''),
                website=info.get('website', ''),
                employees=info.get('fullTimeEmployees', 0),
                headquarters=f"{info.get('city', '')}, {info.get('state', '')}, {info.get('country', '')}"
            )
            
            # Get institutional holders
            try:
                holders = ticker.institutional_holders
                if holders is not None and not holders.empty:
                    for _, row in holders.iterrows():
                        holder = InstitutionalHolder(
                            name=row.get('Holder', ''),
                            shares=int(row.get('Shares', 0)),
                            value=float(row.get('Value', 0)),
                            percent_held=float(row.get('% Out', 0)) if '% Out' in row else 0,
                            report_date=row.get('Date Reported')
                        )
                        profile.major_shareholders.append(holder)
            except:
                pass
            
            # Identify competitors (same industry)
            profile.competitors = self._get_industry_peers(symbol, info.get('industry', ''))
            
            return profile
            
        except Exception as e:
            print(f"Error fetching company profile: {e}")
            return None
    
    def _get_industry_peers(self, symbol: str, industry: str) -> List[str]:
        """Get companies in same industry"""
        # Common industry peer mappings
        industry_peers = {
            'Consumer Electronics': ['AAPL', 'SAMSUNG.KS', 'SONY', 'LG.KS', 'XIAOMI'],
            'Internet Content & Information': ['GOOGL', 'META', 'SNAP', 'PINS', 'TWTR'],
            'Software—Infrastructure': ['MSFT', 'ORCL', 'CRM', 'NOW', 'SNOW'],
            'Auto Manufacturers': ['TSLA', 'F', 'GM', 'TM', 'HMC', 'RIVN'],
            'Semiconductors': ['NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TSM'],
            'E-Commerce': ['AMZN', 'BABA', 'JD', 'EBAY', 'ETSY', 'SHOP'],
            'Banks—Diversified': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS'],
            # Indian companies
            'IT Services': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS'],
            'Banks—Regional—Asia': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS'],
            'Oil & Gas': ['RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS', 'HINDPETRO.NS']
        }
        
        peers = industry_peers.get(industry, [])
        return [p for p in peers if p != symbol][:5]
    
    def get_ownership_structure(self, symbol: str) -> Dict[str, Any]:
        """Get detailed ownership structure"""
        if not YFINANCE_AVAILABLE:
            return {}
        
        try:
            ticker = yf.Ticker(symbol)
            
            result = {
                'institutional_holders': [],
                'mutual_fund_holders': [],
                'insider_holders': [],
                'major_shareholders': [],
                'ownership_breakdown': {}
            }
            
            # Institutional holders
            try:
                inst_holders = ticker.institutional_holders
                if inst_holders is not None and not inst_holders.empty:
                    result['institutional_holders'] = inst_holders.to_dict('records')
            except:
                pass
            
            # Mutual fund holders
            try:
                mf_holders = ticker.mutualfund_holders
                if mf_holders is not None and not mf_holders.empty:
                    result['mutual_fund_holders'] = mf_holders.to_dict('records')
            except:
                pass
            
            # Major holders
            try:
                major = ticker.major_holders
                if major is not None and not major.empty:
                    result['ownership_breakdown'] = dict(zip(major[1], major[0]))
            except:
                pass
            
            return result
            
        except Exception as e:
            print(f"Error fetching ownership: {e}")
            return {}
    
    def get_supply_chain(self, symbol: str) -> Dict[str, List[str]]:
        """
        Get supply chain relationships
        Note: This data is typically from specialized providers.
        Here we use known relationships for major companies.
        """
        # Known supply chain relationships (sample data)
        supply_chains = {
            'AAPL': {
                'suppliers': ['TSM', 'QCOM', 'AVGO', 'TXN', 'ADI', 'SWKS', 'CRUS', 'HON.NS'],
                'customers': ['AMZN', 'BBY', 'TGT', 'WMT'],
                'partners': ['GOOGL', 'IBM', 'CSCO']
            },
            'TSLA': {
                'suppliers': ['PCRFY', 'ALB', 'LAC', 'LTHM', 'NIO'],
                'customers': [],  # Direct to consumer
                'partners': ['PANASONIC', 'LG ENERGY']
            },
            'AMZN': {
                'suppliers': ['UPS', 'FDX', 'CHRW'],
                'customers': [],  # B2C
                'partners': ['MSFT', 'GOOGL', 'INTC']
            },
            # Indian companies
            'RELIANCE.NS': {
                'suppliers': ['ONGC.NS', 'GAIL.NS'],
                'customers': ['IOC.NS', 'BPCL.NS'],
                'partners': ['GOOGL', 'META', 'BP']
            },
            'TCS.NS': {
                'suppliers': ['MSFT', 'AWS', 'GOOGL'],
                'customers': ['JPM', 'BAC', 'WFC', 'C'],
                'partners': ['IBM', 'ORCL', 'SAP']
            },
            'INFY.NS': {
                'suppliers': ['MSFT', 'AWS', 'GOOGL'],
                'customers': ['GS', 'MS', 'AAPL'],
                'partners': ['GOOGL', 'MSFT', 'SAP']
            }
        }
        
        return supply_chains.get(symbol, {
            'suppliers': [],
            'customers': [],
            'partners': []
        })
    
    def get_subsidiaries(self, symbol: str) -> Dict[str, Any]:
        """Get company subsidiaries and corporate structure"""
        # Known subsidiary relationships
        corporate_structures = {
            'GOOGL': {
                'parent': 'Alphabet Inc.',
                'subsidiaries': [
                    {'name': 'Google LLC', 'ownership': 100, 'type': 'Technology'},
                    {'name': 'YouTube', 'ownership': 100, 'type': 'Media'},
                    {'name': 'Waymo', 'ownership': 100, 'type': 'Autonomous Vehicles'},
                    {'name': 'DeepMind', 'ownership': 100, 'type': 'AI Research'},
                    {'name': 'Verily', 'ownership': 100, 'type': 'Life Sciences'},
                    {'name': 'Fitbit', 'ownership': 100, 'type': 'Wearables'}
                ]
            },
            'META': {
                'parent': 'Meta Platforms Inc.',
                'subsidiaries': [
                    {'name': 'Facebook', 'ownership': 100, 'type': 'Social Media'},
                    {'name': 'Instagram', 'ownership': 100, 'type': 'Social Media'},
                    {'name': 'WhatsApp', 'ownership': 100, 'type': 'Messaging'},
                    {'name': 'Oculus', 'ownership': 100, 'type': 'VR/AR'},
                    {'name': 'Reality Labs', 'ownership': 100, 'type': 'Metaverse'}
                ]
            },
            'AMZN': {
                'parent': 'Amazon.com Inc.',
                'subsidiaries': [
                    {'name': 'Amazon Web Services (AWS)', 'ownership': 100, 'type': 'Cloud'},
                    {'name': 'Whole Foods Market', 'ownership': 100, 'type': 'Retail'},
                    {'name': 'Ring', 'ownership': 100, 'type': 'Smart Home'},
                    {'name': 'Twitch', 'ownership': 100, 'type': 'Streaming'},
                    {'name': 'MGM Studios', 'ownership': 100, 'type': 'Entertainment'},
                    {'name': 'Audible', 'ownership': 100, 'type': 'Audiobooks'}
                ]
            },
            # Indian conglomerates
            'RELIANCE.NS': {
                'parent': 'Reliance Industries Limited',
                'subsidiaries': [
                    {'name': 'Jio Platforms', 'ownership': 100, 'type': 'Telecom'},
                    {'name': 'Reliance Retail', 'ownership': 100, 'type': 'Retail'},
                    {'name': 'Reliance Petroleum', 'ownership': 100, 'type': 'Oil & Gas'},
                    {'name': 'Network18', 'ownership': 75, 'type': 'Media'},
                    {'name': 'Reliance Digital', 'ownership': 100, 'type': 'Electronics Retail'}
                ]
            },
            'TCS.NS': {
                'parent': 'Tata Consultancy Services',
                'holding_company': 'Tata Sons',
                'subsidiaries': [
                    {'name': 'TCS China', 'ownership': 100, 'type': 'IT Services'},
                    {'name': 'CMC Limited', 'ownership': 100, 'type': 'IT Services'},
                    {'name': 'Tata Elxsi', 'ownership': 100, 'type': 'Design Services'}
                ],
                'sister_companies': ['TATAMOTORS.NS', 'TATASTEEL.NS', 'TATAPOWER.NS', 'TITAN.NS']
            },
            'HDFCBANK.NS': {
                'parent': 'HDFC Bank Limited',
                'subsidiaries': [
                    {'name': 'HDB Financial Services', 'ownership': 95, 'type': 'NBFC'},
                    {'name': 'HDFC Securities', 'ownership': 100, 'type': 'Brokerage'}
                ],
                'merged_entity': 'HDFC Ltd (merged 2023)'
            }
        }
        
        return corporate_structures.get(symbol, {
            'parent': symbol,
            'subsidiaries': []
        })
    
    def get_investments(self, symbol: str) -> Dict[str, Any]:
        """Get company's strategic investments"""
        # Known investment relationships
        investments = {
            'GOOGL': {
                'investments': [
                    {'company': 'SpaceX', 'amount': 1000000000, 'stake': 7.5, 'type': 'Private'},
                    {'company': 'Uber', 'amount': 258000000, 'stake': 5.2, 'type': 'IPO'},
                    {'company': 'CrowdStrike', 'amount': 100000000, 'stake': 2.0, 'type': 'Public'}
                ],
                'acquisitions': [
                    {'company': 'Fitbit', 'amount': 2100000000, 'year': 2021},
                    {'company': 'Mandiant', 'amount': 5400000000, 'year': 2022}
                ]
            },
            'MSFT': {
                'investments': [
                    {'company': 'OpenAI', 'amount': 13000000000, 'stake': 49, 'type': 'Private'},
                    {'company': 'Cruise', 'amount': 2000000000, 'stake': 0, 'type': 'Partnership'}
                ],
                'acquisitions': [
                    {'company': 'Activision Blizzard', 'amount': 69000000000, 'year': 2023},
                    {'company': 'Nuance', 'amount': 19700000000, 'year': 2022},
                    {'company': 'LinkedIn', 'amount': 26200000000, 'year': 2016}
                ]
            },
            # Indian companies
            'RELIANCE.NS': {
                'investments': [
                    {'company': 'Dunzo', 'amount': 200000000, 'stake': 25.8, 'type': 'Private'},
                    {'company': 'Urban Ladder', 'amount': 0, 'stake': 96, 'type': 'Acquisition'},
                    {'company': 'Just Dial', 'amount': 530000000, 'stake': 66.4, 'type': 'Public'}
                ],
                'foreign_investments': [
                    {'investor': 'Google', 'amount': 4500000000, 'stake': 7.7, 'in': 'Jio Platforms'},
                    {'investor': 'Meta', 'amount': 5700000000, 'stake': 9.9, 'in': 'Jio Platforms'},
                    {'investor': 'Intel', 'amount': 253000000, 'stake': 0.39, 'in': 'Jio Platforms'}
                ]
            }
        }
        
        return investments.get(symbol, {'investments': [], 'acquisitions': []})


class NewsAggregator:
    """
    Financial News Aggregation
    ===========================
    Aggregate news from multiple sources for stocks.
    """
    
    def __init__(self, newsapi_key: Optional[str] = None,
                 alphavantage_key: Optional[str] = None):
        self.newsapi_key = newsapi_key
        self.alphavantage_key = alphavantage_key
    
    def get_stock_news(self, symbol: str, days: int = 7) -> List[NewsArticle]:
        """Get news articles for a stock"""
        articles = []
        
        # Try yfinance news
        if YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                news = ticker.news
                
                for item in news[:20]:
                    article = NewsArticle(
                        title=item.get('title', ''),
                        summary=item.get('summary', '')[:500] if item.get('summary') else '',
                        source=item.get('publisher', 'Unknown'),
                        url=item.get('link', ''),
                        published_at=datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                        symbols=[symbol],
                        categories=item.get('relatedTickers', [])
                    )
                    articles.append(article)
            except Exception as e:
                print(f"Error fetching yfinance news: {e}")
        
        # Try NewsAPI if available
        if self.newsapi_key:
            try:
                company_name = self._get_company_name(symbol)
                newsapi_articles = self._fetch_newsapi(company_name, days)
                articles.extend(newsapi_articles)
            except Exception as e:
                print(f"Error fetching NewsAPI: {e}")
        
        # Sort by date
        articles.sort(key=lambda x: x.published_at, reverse=True)
        
        return articles
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name from symbol"""
        if YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                return ticker.info.get('shortName', symbol)
            except:
                pass
        return symbol
    
    def _fetch_newsapi(self, query: str, days: int) -> List[NewsArticle]:
        """Fetch from NewsAPI"""
        if not self.newsapi_key:
            return []
        
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'publishedAt',
                'apiKey': self.newsapi_key,
                'language': 'en',
                'pageSize': 20
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            articles = []
            for item in data.get('articles', []):
                article = NewsArticle(
                    title=item.get('title', ''),
                    summary=item.get('description', ''),
                    source=item.get('source', {}).get('name', 'Unknown'),
                    url=item.get('url', ''),
                    published_at=datetime.fromisoformat(
                        item.get('publishedAt', '').replace('Z', '+00:00')
                    ) if item.get('publishedAt') else datetime.now()
                )
                articles.append(article)
            
            return articles
            
        except Exception as e:
            print(f"NewsAPI error: {e}")
            return []
    
    def get_sector_news(self, sector: str) -> List[NewsArticle]:
        """Get news for a sector"""
        sector_keywords = {
            'Technology': ['tech', 'software', 'AI', 'semiconductor'],
            'Financial': ['banking', 'finance', 'fintech', 'interest rates'],
            'Healthcare': ['pharma', 'biotech', 'healthcare', 'FDA'],
            'Energy': ['oil', 'gas', 'renewable', 'energy'],
            'Consumer': ['retail', 'e-commerce', 'consumer']
        }
        
        keywords = sector_keywords.get(sector, [sector.lower()])
        query = ' OR '.join(keywords)
        
        return self._fetch_newsapi(query, days=3) if self.newsapi_key else []
    
    def get_market_news(self, market: str = 'US') -> List[NewsArticle]:
        """Get general market news"""
        market_queries = {
            'US': 'stock market OR S&P 500 OR Nasdaq OR Federal Reserve',
            'India': 'Nifty OR Sensex OR RBI OR Indian stock market',
            'Crypto': 'Bitcoin OR Ethereum OR cryptocurrency',
            'Global': 'global markets OR world economy'
        }
        
        query = market_queries.get(market, market_queries['US'])
        return self._fetch_newsapi(query, days=2) if self.newsapi_key else []


class CompanyResearch:
    """
    Unified Company Research Interface
    ====================================
    One-stop interface for all company intelligence.
    """
    
    def __init__(self, newsapi_key: Optional[str] = None,
                 alphavantage_key: Optional[str] = None):
        self.intelligence = CompanyIntelligence(newsapi_key, alphavantage_key)
        self.news = NewsAggregator(newsapi_key, alphavantage_key)
    
    def full_research(self, symbol: str) -> Dict[str, Any]:
        """Get complete research on a company"""
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }
        
        # Company profile
        profile = self.intelligence.get_company_profile(symbol)
        if profile:
            result['profile'] = {
                'name': profile.name,
                'sector': profile.sector,
                'industry': profile.industry,
                'country': profile.country,
                'market_cap': profile.market_cap,
                'employees': profile.employees,
                'description': profile.description[:500] if profile.description else '',
                'website': profile.website,
                'competitors': profile.competitors
            }
        
        # Ownership
        result['ownership'] = self.intelligence.get_ownership_structure(symbol)
        
        # Supply chain
        result['supply_chain'] = self.intelligence.get_supply_chain(symbol)
        
        # Corporate structure
        result['corporate_structure'] = self.intelligence.get_subsidiaries(symbol)
        
        # Investments
        result['investments'] = self.intelligence.get_investments(symbol)
        
        # News
        news_articles = self.news.get_stock_news(symbol)
        result['news'] = [
            {
                'title': a.title,
                'source': a.source,
                'published': a.published_at.isoformat() if a.published_at else None,
                'url': a.url
            }
            for a in news_articles[:10]
        ]
        
        return result
    
    def compare_companies(self, symbols: List[str]) -> pd.DataFrame:
        """Compare multiple companies"""
        data = []
        
        for symbol in symbols:
            profile = self.intelligence.get_company_profile(symbol)
            if profile:
                data.append({
                    'Symbol': symbol,
                    'Name': profile.name,
                    'Sector': profile.sector,
                    'Industry': profile.industry,
                    'Market Cap': profile.market_cap,
                    'Employees': profile.employees,
                    'Country': profile.country
                })
        
        return pd.DataFrame(data)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COMPANY INTELLIGENCE DEMO")
    print("=" * 60)
    
    research = CompanyResearch()
    
    # Get company research
    print("\n1. Full Research on AAPL:")
    result = research.full_research('AAPL')
    
    if 'profile' in result:
        print(f"   Name: {result['profile']['name']}")
        print(f"   Sector: {result['profile']['sector']}")
        print(f"   Market Cap: ${result['profile']['market_cap']:,.0f}")
        print(f"   Competitors: {result['profile']['competitors']}")
    
    print("\n2. Supply Chain:")
    supply_chain = result.get('supply_chain', {})
    print(f"   Suppliers: {supply_chain.get('suppliers', [])}")
    print(f"   Partners: {supply_chain.get('partners', [])}")
    
    print("\n3. Corporate Structure:")
    structure = result.get('corporate_structure', {})
    if 'subsidiaries' in structure:
        for sub in structure['subsidiaries'][:3]:
            print(f"   - {sub['name']} ({sub['type']})")
    
    print("\n4. Recent News:")
    for news in result.get('news', [])[:3]:
        print(f"   - {news['title'][:60]}...")
    
    # Indian company
    print("\n" + "=" * 60)
    print("INDIAN COMPANY RESEARCH: RELIANCE.NS")
    print("=" * 60)
    
    reliance = research.full_research('RELIANCE.NS')
    if 'corporate_structure' in reliance:
        print("\nSubsidiaries:")
        for sub in reliance['corporate_structure'].get('subsidiaries', []):
            print(f"   - {sub['name']} ({sub['type']}) - {sub['ownership']}% owned")
