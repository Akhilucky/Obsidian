"""
Alternative Data Integration Module
Integration of non-traditional data sources including weather data,
satellite imagery analysis, web scraping, GitHub activity, and social signals.
"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse
import hashlib

import numpy as np
import pandas as pd

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Types of alternative data sources."""
    WEATHER = "weather"
    SATELLITE = "satellite"
    WEB_SCRAPING = "web_scraping"
    GITHUB = "github"
    SOCIAL_MEDIA = "social_media"
    JOB_POSTINGS = "job_postings"
    PATENT_FILINGS = "patent_filings"
    GOVERNMENT = "government"
    SHIPPING = "shipping"
    CONSUMER = "consumer"


class SignalStrength(Enum):
    """Alternative data signal strength."""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class AlternativeDataSignal:
    """Signal derived from alternative data."""
    source: DataSourceType
    symbol: str
    signal: SignalStrength
    confidence: float
    description: str
    data_points: int
    timestamp: datetime
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class WeatherData:
    """Weather data point."""
    location: str
    date: datetime
    temperature: float
    precipitation: float
    wind_speed: float
    humidity: float
    conditions: str
    impact_sector: str = ""
    impact_score: float = 0.0


@dataclass
class GitHubMetrics:
    """GitHub repository metrics."""
    repo: str
    stars: int
    forks: int
    open_issues: int
    contributors: int
    commits_30d: int
    stars_growth: float
    fork_growth: float
    activity_score: float
    last_commit: datetime


@dataclass
class SentimentData:
    """Social sentiment data."""
    symbol: str
    platform: str
    mentions: int
    positive: float
    negative: float
    neutral: float
    trending_score: float
    timestamp: datetime


class WeatherDataProvider:
    """
    Weather data integration for sector analysis.
    Weather impacts retail, agriculture, energy, and more.
    """
    
    # Sector weather sensitivity
    SECTOR_WEATHER_IMPACT = {
        'Retail': {
            'extreme_cold': -0.15,
            'extreme_heat': -0.10,
            'heavy_rain': -0.08,
            'snow': -0.12,
            'pleasant': 0.05
        },
        'Energy': {
            'extreme_cold': 0.20,
            'extreme_heat': 0.15,
            'mild': -0.10,
            'normal': 0.0
        },
        'Agriculture': {
            'drought': -0.25,
            'flood': -0.30,
            'frost': -0.20,
            'optimal': 0.10
        },
        'Construction': {
            'rain': -0.15,
            'snow': -0.25,
            'extreme_cold': -0.20,
            'pleasant': 0.10
        },
        'Airlines': {
            'storm': -0.20,
            'snow': -0.15,
            'fog': -0.10,
            'clear': 0.05
        },
        'Hospitality': {
            'rain': -0.10,
            'extreme_heat': -0.08,
            'pleasant': 0.12,
            'holiday_weather': 0.15
        }
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize weather provider."""
        self.api_key = api_key
        self.cache = {}
        self.cache_duration = timedelta(hours=1)
    
    async def get_weather_data(
        self,
        locations: List[str],
        days_ahead: int = 7
    ) -> List[WeatherData]:
        """
        Get weather forecast data.
        
        Args:
            locations: List of location names or coordinates
            days_ahead: Days of forecast
            
        Returns:
            List of WeatherData objects
        """
        weather_data = []
        
        for location in locations:
            # Check cache
            cache_key = f"{location}_{days_ahead}"
            if cache_key in self.cache:
                cached_time, cached_data = self.cache[cache_key]
                if datetime.now() - cached_time < self.cache_duration:
                    weather_data.extend(cached_data)
                    continue
            
            # Generate simulated weather data (replace with actual API calls)
            location_weather = self._simulate_weather(location, days_ahead)
            weather_data.extend(location_weather)
            
            # Cache results
            self.cache[cache_key] = (datetime.now(), location_weather)
        
        return weather_data
    
    def _simulate_weather(
        self,
        location: str,
        days: int
    ) -> List[WeatherData]:
        """Simulate weather data for demo purposes."""
        weather = []
        base_temp = 70 + np.random.randn() * 15
        
        conditions = ['Clear', 'Partly Cloudy', 'Cloudy', 'Rain', 'Storm', 'Snow']
        condition_probs = [0.3, 0.25, 0.2, 0.15, 0.05, 0.05]
        
        for d in range(days):
            date = datetime.now() + timedelta(days=d)
            temp = base_temp + np.random.randn() * 10 + 5 * np.sin(d * 0.5)
            precip = max(0, np.random.randn() * 0.5)
            wind = max(0, 10 + np.random.randn() * 8)
            humidity = min(100, max(20, 60 + np.random.randn() * 20))
            
            condition = np.random.choice(conditions, p=condition_probs)
            
            weather.append(WeatherData(
                location=location,
                date=date,
                temperature=temp,
                precipitation=precip,
                wind_speed=wind,
                humidity=humidity,
                conditions=condition
            ))
        
        return weather
    
    def analyze_sector_impact(
        self,
        weather_data: List[WeatherData],
        sector: str
    ) -> Dict[str, Any]:
        """
        Analyze weather impact on a sector.
        
        Args:
            weather_data: Weather forecast data
            sector: Sector to analyze
            
        Returns:
            Impact analysis
        """
        if sector not in self.SECTOR_WEATHER_IMPACT:
            return {'impact': 0, 'description': 'No weather sensitivity data'}
        
        sector_sensitivity = self.SECTOR_WEATHER_IMPACT[sector]
        
        total_impact = 0
        impact_days = 0
        descriptions = []
        
        for wd in weather_data:
            # Classify weather condition
            if wd.temperature < 30:
                condition_type = 'extreme_cold'
            elif wd.temperature > 95:
                condition_type = 'extreme_heat'
            elif wd.conditions in ['Storm', 'Heavy Rain']:
                condition_type = 'storm'
            elif wd.conditions == 'Snow':
                condition_type = 'snow'
            elif wd.conditions == 'Rain':
                condition_type = 'rain'
            elif 65 <= wd.temperature <= 80 and wd.conditions in ['Clear', 'Partly Cloudy']:
                condition_type = 'pleasant'
            else:
                condition_type = 'normal'
            
            impact = sector_sensitivity.get(condition_type, 0)
            if impact != 0:
                total_impact += impact
                impact_days += 1
                descriptions.append(f"{wd.date.strftime('%m/%d')}: {condition_type} ({impact:+.1%})")
        
        avg_impact = total_impact / len(weather_data) if weather_data else 0
        
        return {
            'sector': sector,
            'average_impact': avg_impact,
            'impact_days': impact_days,
            'total_forecast_days': len(weather_data),
            'signal': self._impact_to_signal(avg_impact),
            'descriptions': descriptions[:5]
        }
    
    def _impact_to_signal(self, impact: float) -> SignalStrength:
        """Convert impact score to signal strength."""
        if impact > 0.1:
            return SignalStrength.VERY_BULLISH
        elif impact > 0.03:
            return SignalStrength.BULLISH
        elif impact < -0.1:
            return SignalStrength.VERY_BEARISH
        elif impact < -0.03:
            return SignalStrength.BEARISH
        else:
            return SignalStrength.NEUTRAL


class GitHubActivityAnalyzer:
    """
    Analyze GitHub activity for tech companies.
    Developer activity can signal company momentum and innovation.
    """
    
    # Company to GitHub org mapping
    COMPANY_REPOS = {
        'MSFT': ['microsoft/vscode', 'microsoft/TypeScript', 'Azure/azure-sdk-for-python'],
        'GOOGL': ['google/jax', 'tensorflow/tensorflow', 'google/material-design-icons'],
        'META': ['facebook/react', 'facebook/react-native', 'pytorch/pytorch'],
        'AMZN': ['aws/aws-cli', 'aws/aws-cdk', 'aws/amazon-sagemaker-examples'],
        'AAPL': ['apple/swift', 'apple/foundationdb', 'apple/ml-stable-diffusion'],
        'NVDA': ['NVIDIA/cuda-samples', 'NVIDIA/TensorRT', 'NVIDIA/DeepLearningExamples'],
        'TSLA': ['teslamotors/linux', 'teslamotors/buildroot'],
        'CRM': ['salesforce/design-system-react', 'salesforce/lwc'],
        'IBM': ['IBM/watson-developer-cloud', 'IBM/quantum-api'],
        'ORCL': ['oracle/graal', 'oracle/opengdk']
    }
    
    def __init__(self, api_token: Optional[str] = None):
        """Initialize GitHub analyzer."""
        self.api_token = api_token
        self.base_url = "https://api.github.com"
        self.cache = {}
    
    async def get_repo_metrics(self, repo: str) -> GitHubMetrics:
        """
        Get metrics for a GitHub repository.
        
        Args:
            repo: Repository in format 'owner/repo'
            
        Returns:
            GitHubMetrics object
        """
        # Check cache
        if repo in self.cache:
            cached_time, cached_data = self.cache[repo]
            if datetime.now() - cached_time < timedelta(hours=1):
                return cached_data
        
        # Simulate API response (replace with actual API calls)
        metrics = self._simulate_repo_metrics(repo)
        
        self.cache[repo] = (datetime.now(), metrics)
        return metrics
    
    def _simulate_repo_metrics(self, repo: str) -> GitHubMetrics:
        """Simulate GitHub metrics for demo."""
        # Generate realistic-looking metrics
        base_stars = hash(repo) % 50000 + 1000
        
        return GitHubMetrics(
            repo=repo,
            stars=base_stars,
            forks=int(base_stars * 0.15),
            open_issues=int(base_stars * 0.02),
            contributors=int(base_stars * 0.05),
            commits_30d=50 + hash(repo) % 200,
            stars_growth=(hash(repo) % 20 - 5) / 100,
            fork_growth=(hash(repo) % 15 - 3) / 100,
            activity_score=0.5 + (hash(repo) % 50) / 100,
            last_commit=datetime.now() - timedelta(hours=hash(repo) % 48)
        )
    
    async def analyze_company(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze GitHub activity for a company.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            GitHub activity analysis
        """
        repos = self.COMPANY_REPOS.get(symbol, [])
        
        if not repos:
            return {
                'symbol': symbol,
                'available': False,
                'message': 'No GitHub repos mapped for this company'
            }
        
        # Get metrics for all repos
        all_metrics = []
        for repo in repos:
            metrics = await self.get_repo_metrics(repo)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        total_stars = sum(m.stars for m in all_metrics)
        total_forks = sum(m.forks for m in all_metrics)
        total_commits = sum(m.commits_30d for m in all_metrics)
        avg_growth = np.mean([m.stars_growth for m in all_metrics])
        avg_activity = np.mean([m.activity_score for m in all_metrics])
        
        # Generate signal
        signal = SignalStrength.NEUTRAL
        if avg_growth > 0.05 and avg_activity > 0.7:
            signal = SignalStrength.VERY_BULLISH
        elif avg_growth > 0.02 and avg_activity > 0.5:
            signal = SignalStrength.BULLISH
        elif avg_growth < -0.03:
            signal = SignalStrength.BEARISH
        elif avg_growth < -0.05:
            signal = SignalStrength.VERY_BEARISH
        
        return {
            'symbol': symbol,
            'available': True,
            'repos_analyzed': len(repos),
            'total_stars': total_stars,
            'total_forks': total_forks,
            'commits_30d': total_commits,
            'avg_stars_growth': avg_growth,
            'avg_activity_score': avg_activity,
            'signal': signal.value,
            'repos': [
                {
                    'name': m.repo,
                    'stars': m.stars,
                    'commits_30d': m.commits_30d,
                    'growth': m.stars_growth
                }
                for m in all_metrics
            ]
        }


class WebScrapingEngine:
    """
    Web scraping engine for alternative data collection.
    Extracts data from public websites for analysis.
    """
    
    def __init__(self, respect_robots: bool = True):
        """Initialize web scraper."""
        self.respect_robots = respect_robots
        self.session = None
        self.user_agent = "AlternativeDataBot/1.0"
        self.rate_limit = 1.0  # seconds between requests
        self.last_request = datetime.now() - timedelta(seconds=10)
    
    async def _rate_limit_wait(self):
        """Wait for rate limit."""
        elapsed = (datetime.now() - self.last_request).total_seconds()
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self.last_request = datetime.now()
    
    async def scrape_job_postings(
        self,
        company: str,
        keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Scrape job posting data for a company.
        Growth in job postings can indicate expansion.
        
        Args:
            company: Company name
            keywords: Keywords to filter
            
        Returns:
            Job posting analysis
        """
        # Simulate job posting data
        np.random.seed(hash(company) % 10000)
        
        total_jobs = 100 + np.random.randint(0, 2000)
        growth_30d = (np.random.random() - 0.3) * 0.5  # -15% to +35%
        
        departments = {
            'Engineering': 0.35 + np.random.random() * 0.15,
            'Sales': 0.15 + np.random.random() * 0.10,
            'Marketing': 0.10 + np.random.random() * 0.05,
            'Operations': 0.10 + np.random.random() * 0.05,
            'HR': 0.05 + np.random.random() * 0.03,
            'Finance': 0.05 + np.random.random() * 0.03,
            'Other': 0.10
        }
        
        # Normalize
        total = sum(departments.values())
        departments = {k: v/total for k, v in departments.items()}
        
        # Signal based on job growth
        if growth_30d > 0.20:
            signal = SignalStrength.VERY_BULLISH
        elif growth_30d > 0.05:
            signal = SignalStrength.BULLISH
        elif growth_30d < -0.10:
            signal = SignalStrength.BEARISH
        else:
            signal = SignalStrength.NEUTRAL
        
        return {
            'company': company,
            'total_openings': total_jobs,
            'growth_30d': growth_30d,
            'department_breakdown': {
                k: int(v * total_jobs) for k, v in departments.items()
            },
            'signal': signal.value,
            'interpretation': f"Job postings {'grew' if growth_30d > 0 else 'declined'} "
                            f"{abs(growth_30d)*100:.1f}% in the last 30 days"
        }
    
    async def scrape_patent_filings(
        self,
        company: str,
        years: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze patent filing trends.
        Patent activity indicates R&D investment and innovation.
        
        Args:
            company: Company name
            years: Years of history
            
        Returns:
            Patent analysis
        """
        np.random.seed(hash(company) % 10000)
        
        # Simulate patent data
        yearly_patents = []
        base = 50 + np.random.randint(0, 200)
        
        for y in range(years):
            patents = int(base * (1 + (np.random.random() - 0.3) * 0.4))
            yearly_patents.append({
                'year': datetime.now().year - years + y + 1,
                'patents_filed': patents,
                'patents_granted': int(patents * 0.7)
            })
            base = patents
        
        # Calculate trend
        if len(yearly_patents) >= 2:
            growth = (yearly_patents[-1]['patents_filed'] - yearly_patents[0]['patents_filed']) / yearly_patents[0]['patents_filed']
        else:
            growth = 0
        
        # Categories
        categories = {
            'AI/ML': 0.25 + np.random.random() * 0.15,
            'Cloud': 0.20 + np.random.random() * 0.10,
            'Hardware': 0.15 + np.random.random() * 0.10,
            'Security': 0.10 + np.random.random() * 0.08,
            'Other': 0.20
        }
        total = sum(categories.values())
        categories = {k: v/total for k, v in categories.items()}
        
        return {
            'company': company,
            'yearly_filings': yearly_patents,
            'total_patents': sum(yp['patents_filed'] for yp in yearly_patents),
            'growth_rate': growth,
            'top_categories': categories,
            'innovation_score': min(1.0, 0.5 + growth * 0.5),
            'signal': SignalStrength.BULLISH.value if growth > 0.1 else SignalStrength.NEUTRAL.value
        }
    
    async def scrape_consumer_reviews(
        self,
        product_or_company: str,
        source: str = "general"
    ) -> Dict[str, Any]:
        """
        Analyze consumer review sentiment.
        
        Args:
            product_or_company: Product or company name
            source: Review source
            
        Returns:
            Review analysis
        """
        np.random.seed(hash(product_or_company) % 10000)
        
        total_reviews = 1000 + np.random.randint(0, 50000)
        avg_rating = 3.0 + np.random.random() * 2.0
        
        # Rating distribution
        distribution = {
            5: 0.3 + np.random.random() * 0.2,
            4: 0.25 + np.random.random() * 0.1,
            3: 0.15 + np.random.random() * 0.1,
            2: 0.10 + np.random.random() * 0.05,
            1: 0.10 + np.random.random() * 0.05
        }
        total = sum(distribution.values())
        distribution = {k: v/total for k, v in distribution.items()}
        
        # Trend
        rating_trend = (np.random.random() - 0.4) * 0.3  # -10% to +20%
        
        # Key themes
        themes = ['Quality', 'Value', 'Service', 'Shipping', 'Features']
        positive_themes = np.random.choice(themes, size=2, replace=False).tolist()
        negative_themes = np.random.choice(
            [t for t in themes if t not in positive_themes],
            size=1
        ).tolist()
        
        return {
            'product': product_or_company,
            'total_reviews': total_reviews,
            'average_rating': avg_rating,
            'rating_distribution': distribution,
            'rating_trend_30d': rating_trend,
            'positive_themes': positive_themes,
            'negative_themes': negative_themes,
            'sentiment_score': avg_rating / 5.0,
            'signal': SignalStrength.BULLISH.value if avg_rating > 4.0 else SignalStrength.NEUTRAL.value
        }


class SatelliteDataAnalyzer:
    """
    Satellite and geospatial data analysis.
    Tracks parking lot traffic, shipping activity, oil storage, etc.
    """
    
    # Simulated satellite data sources
    DATA_SOURCES = {
        'retail_traffic': {
            'description': 'Parking lot car counts at retail locations',
            'update_frequency': 'daily',
            'lag_days': 2
        },
        'oil_storage': {
            'description': 'Oil tank fill levels',
            'update_frequency': 'weekly',
            'lag_days': 5
        },
        'shipping_activity': {
            'description': 'Port and shipping traffic',
            'update_frequency': 'daily',
            'lag_days': 1
        },
        'construction': {
            'description': 'Construction site activity',
            'update_frequency': 'weekly',
            'lag_days': 7
        },
        'agriculture': {
            'description': 'Crop health and growth indices',
            'update_frequency': 'weekly',
            'lag_days': 3
        }
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize satellite analyzer."""
        self.api_key = api_key
    
    async def get_retail_traffic(
        self,
        company: str,
        lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Get retail parking lot traffic data.
        
        Args:
            company: Retail company name
            lookback_days: Days of history
            
        Returns:
            Traffic analysis
        """
        np.random.seed(hash(company) % 10000)
        
        # Generate simulated traffic data
        dates = pd.date_range(
            end=datetime.now() - timedelta(days=2),  # 2-day lag
            periods=lookback_days,
            freq='D'
        )
        
        # Base traffic with weekly seasonality
        base_traffic = 1000 + np.random.randint(0, 5000)
        daily_traffic = []
        
        for i, date in enumerate(dates):
            # Weekly pattern (weekend higher)
            day_factor = 1.3 if date.dayofweek >= 5 else 1.0
            
            # Random variation
            variation = 1 + (np.random.random() - 0.5) * 0.3
            
            # Trend
            trend = 1 + (np.random.random() - 0.4) * 0.001 * i
            
            traffic = int(base_traffic * day_factor * variation * trend)
            daily_traffic.append(traffic)
        
        traffic_series = pd.Series(daily_traffic, index=dates)
        
        # Calculate metrics
        avg_traffic = traffic_series.mean()
        recent_avg = traffic_series.tail(14).mean()
        historical_avg = traffic_series.head(30).mean()
        yoy_change = (recent_avg - historical_avg) / historical_avg
        
        return {
            'company': company,
            'average_daily_traffic': avg_traffic,
            'recent_14d_avg': recent_avg,
            'yoy_change': yoy_change,
            'weekly_pattern': {
                'weekday_avg': traffic_series[traffic_series.index.dayofweek < 5].mean(),
                'weekend_avg': traffic_series[traffic_series.index.dayofweek >= 5].mean()
            },
            'signal': SignalStrength.BULLISH.value if yoy_change > 0.05 else 
                     SignalStrength.BEARISH.value if yoy_change < -0.05 else
                     SignalStrength.NEUTRAL.value,
            'data_source': 'satellite_parking_analysis',
            'last_update': dates[-1].isoformat()
        }
    
    async def get_oil_storage(
        self,
        region: str = "cushing"
    ) -> Dict[str, Any]:
        """
        Get oil storage tank levels.
        
        Args:
            region: Storage region
            
        Returns:
            Oil storage analysis
        """
        np.random.seed(hash(region) % 10000)
        
        # Simulated storage data
        capacity = 80000000  # barrels
        current_fill = capacity * (0.4 + np.random.random() * 0.4)
        weekly_change = (np.random.random() - 0.5) * 0.05
        
        return {
            'region': region,
            'capacity_barrels': capacity,
            'current_storage': current_fill,
            'utilization': current_fill / capacity,
            'weekly_change': weekly_change,
            'signal': SignalStrength.BEARISH.value if weekly_change > 0.03 else
                     SignalStrength.BULLISH.value if weekly_change < -0.03 else
                     SignalStrength.NEUTRAL.value,
            'interpretation': f"Storage {'building' if weekly_change > 0 else 'drawing'} "
                            f"at {abs(weekly_change)*100:.1f}% weekly rate"
        }
    
    async def get_shipping_activity(
        self,
        port: str = "los_angeles"
    ) -> Dict[str, Any]:
        """
        Get port shipping activity.
        
        Args:
            port: Port name
            
        Returns:
            Shipping analysis
        """
        np.random.seed(hash(port) % 10000)
        
        ships_in_port = 20 + np.random.randint(0, 50)
        ships_waiting = 5 + np.random.randint(0, 30)
        avg_wait_days = 2 + np.random.random() * 10
        
        weekly_throughput = 100 + np.random.randint(0, 200)
        throughput_change = (np.random.random() - 0.4) * 0.2
        
        return {
            'port': port,
            'ships_in_port': ships_in_port,
            'ships_waiting': ships_waiting,
            'average_wait_days': avg_wait_days,
            'weekly_throughput_teu': weekly_throughput * 1000,
            'throughput_change': throughput_change,
            'congestion_score': min(1.0, ships_waiting / 30),
            'signal': SignalStrength.BULLISH.value if throughput_change > 0.1 else
                     SignalStrength.BEARISH.value if throughput_change < -0.1 else
                     SignalStrength.NEUTRAL.value
        }


class SocialSentimentAnalyzer:
    """
    Social media sentiment analysis.
    Tracks mentions and sentiment across platforms.
    """
    
    PLATFORMS = ['twitter', 'reddit', 'stocktwits', 'discord']
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """Initialize social analyzer."""
        self.api_keys = api_keys or {}
    
    async def get_social_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get social media sentiment for a symbol.
        
        Args:
            symbol: Stock symbol
            lookback_hours: Hours of data
            
        Returns:
            Social sentiment analysis
        """
        np.random.seed(hash(symbol + str(datetime.now().date())) % 10000)
        
        platform_data = {}
        total_mentions = 0
        weighted_sentiment = 0
        
        for platform in self.PLATFORMS:
            mentions = 10 + np.random.randint(0, 1000)
            positive = 0.3 + np.random.random() * 0.4
            negative = 0.1 + np.random.random() * 0.2
            neutral = 1 - positive - negative
            
            platform_data[platform] = {
                'mentions': mentions,
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'sentiment_score': positive - negative
            }
            
            total_mentions += mentions
            weighted_sentiment += mentions * (positive - negative)
        
        avg_sentiment = weighted_sentiment / total_mentions if total_mentions > 0 else 0
        
        # Trending score based on mention velocity
        trending_score = min(1.0, total_mentions / 5000)
        
        # Determine signal
        if avg_sentiment > 0.3 and trending_score > 0.5:
            signal = SignalStrength.VERY_BULLISH
        elif avg_sentiment > 0.15:
            signal = SignalStrength.BULLISH
        elif avg_sentiment < -0.2:
            signal = SignalStrength.BEARISH
        elif avg_sentiment < -0.3 and trending_score > 0.5:
            signal = SignalStrength.VERY_BEARISH
        else:
            signal = SignalStrength.NEUTRAL
        
        return {
            'symbol': symbol,
            'total_mentions': total_mentions,
            'average_sentiment': avg_sentiment,
            'trending_score': trending_score,
            'platforms': platform_data,
            'signal': signal.value,
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_trending_tickers(
        self,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get currently trending tickers."""
        np.random.seed(hash(str(datetime.now().date())) % 10000)
        
        # Simulate trending tickers
        sample_tickers = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'GOOGL', 'META', 
                         'AMZN', 'GME', 'AMC', 'SPY', 'QQQ', 'PLTR', 'COIN', 'SOFI']
        
        trending = []
        for ticker in np.random.choice(sample_tickers, size=min(limit, len(sample_tickers)), replace=False):
            mentions = 100 + np.random.randint(0, 5000)
            sentiment = (np.random.random() - 0.3) * 0.8
            
            trending.append({
                'symbol': ticker,
                'mentions_24h': mentions,
                'sentiment': sentiment,
                'change_vs_7d': (np.random.random() - 0.3) * 200,
                'rank': len(trending) + 1
            })
        
        # Sort by mentions
        trending.sort(key=lambda x: x['mentions_24h'], reverse=True)
        for i, t in enumerate(trending):
            t['rank'] = i + 1
        
        return trending


class AlternativeDataAggregator:
    """
    Aggregates signals from all alternative data sources.
    """
    
    def __init__(
        self,
        weather_api_key: Optional[str] = None,
        github_token: Optional[str] = None,
        social_api_keys: Optional[Dict[str, str]] = None
    ):
        """Initialize aggregator with API keys."""
        self.weather = WeatherDataProvider(weather_api_key)
        self.github = GitHubActivityAnalyzer(github_token)
        self.web_scraper = WebScrapingEngine()
        self.satellite = SatelliteDataAnalyzer()
        self.social = SocialSentimentAnalyzer(social_api_keys)
    
    async def get_comprehensive_signals(
        self,
        symbol: str,
        company_name: Optional[str] = None,
        sector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive alternative data signals.
        
        Args:
            symbol: Stock symbol
            company_name: Company name for web searches
            sector: Company sector
            
        Returns:
            Comprehensive alternative data analysis
        """
        company_name = company_name or symbol
        
        signals = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': {}
        }
        
        # Gather data from all sources
        try:
            # Social sentiment
            social = await self.social.get_social_sentiment(symbol)
            signals['sources']['social_sentiment'] = social
        except Exception as e:
            logger.warning(f"Social sentiment error: {e}")
        
        try:
            # GitHub activity
            github = await self.github.analyze_company(symbol)
            signals['sources']['github_activity'] = github
        except Exception as e:
            logger.warning(f"GitHub analysis error: {e}")
        
        try:
            # Job postings
            jobs = await self.web_scraper.scrape_job_postings(company_name)
            signals['sources']['job_postings'] = jobs
        except Exception as e:
            logger.warning(f"Job postings error: {e}")
        
        try:
            # Consumer reviews
            reviews = await self.web_scraper.scrape_consumer_reviews(company_name)
            signals['sources']['consumer_reviews'] = reviews
        except Exception as e:
            logger.warning(f"Consumer reviews error: {e}")
        
        # Weather impact (if sector provided)
        if sector:
            try:
                weather_data = await self.weather.get_weather_data(
                    ['New York', 'Los Angeles', 'Chicago'],
                    days_ahead=7
                )
                weather_impact = self.weather.analyze_sector_impact(weather_data, sector)
                signals['sources']['weather_impact'] = weather_impact
            except Exception as e:
                logger.warning(f"Weather analysis error: {e}")
        
        # Calculate composite signal
        signals['composite_signal'] = self._calculate_composite_signal(signals['sources'])
        
        return signals
    
    def _calculate_composite_signal(
        self,
        sources: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate composite signal from all sources."""
        signal_values = {
            SignalStrength.VERY_BULLISH.value: 2,
            SignalStrength.BULLISH.value: 1,
            SignalStrength.NEUTRAL.value: 0,
            SignalStrength.BEARISH.value: -1,
            SignalStrength.VERY_BEARISH.value: -2
        }
        
        total_score = 0
        source_count = 0
        source_signals = {}
        
        for source_name, source_data in sources.items():
            if 'signal' in source_data:
                signal = source_data['signal']
                score = signal_values.get(signal, 0)
                total_score += score
                source_count += 1
                source_signals[source_name] = signal
        
        if source_count == 0:
            return {
                'signal': SignalStrength.NEUTRAL.value,
                'confidence': 0.0,
                'source_count': 0
            }
        
        avg_score = total_score / source_count
        
        # Convert back to signal
        if avg_score >= 1.5:
            composite = SignalStrength.VERY_BULLISH
        elif avg_score >= 0.5:
            composite = SignalStrength.BULLISH
        elif avg_score <= -1.5:
            composite = SignalStrength.VERY_BEARISH
        elif avg_score <= -0.5:
            composite = SignalStrength.BEARISH
        else:
            composite = SignalStrength.NEUTRAL
        
        # Confidence based on agreement
        signals_list = list(source_signals.values())
        if len(signals_list) > 0:
            most_common = max(set(signals_list), key=signals_list.count)
            agreement = signals_list.count(most_common) / len(signals_list)
        else:
            agreement = 0
        
        return {
            'signal': composite.value,
            'score': avg_score,
            'confidence': agreement,
            'source_count': source_count,
            'source_signals': source_signals
        }
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """Generate formatted alternative data report."""
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║           ALTERNATIVE DATA INTELLIGENCE REPORT                   ║
╠══════════════════════════════════════════════════════════════════╣
║ Symbol: {analysis['symbol']:<55}║
║ Generated: {analysis['timestamp'][:19]:<51}║
╚══════════════════════════════════════════════════════════════════╝

"""
        
        if 'composite_signal' in analysis:
            cs = analysis['composite_signal']
            report += f"""
🎯 COMPOSITE SIGNAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signal: {cs['signal'].upper()}
Score: {cs.get('score', 0):+.2f}
Confidence: {cs.get('confidence', 0)*100:.0f}%
Sources Analyzed: {cs.get('source_count', 0)}

"""
        
        sources = analysis.get('sources', {})
        
        if 'social_sentiment' in sources:
            ss = sources['social_sentiment']
            report += f"""
📱 SOCIAL SENTIMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Mentions (24h): {ss.get('total_mentions', 0):,}
Average Sentiment: {ss.get('average_sentiment', 0):+.2f}
Trending Score: {ss.get('trending_score', 0):.2f}
Signal: {ss.get('signal', 'N/A')}

"""
        
        if 'github_activity' in sources and sources['github_activity'].get('available'):
            gh = sources['github_activity']
            report += f"""
💻 GITHUB ACTIVITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Repos Analyzed: {gh.get('repos_analyzed', 0)}
Total Stars: {gh.get('total_stars', 0):,}
30-Day Commits: {gh.get('commits_30d', 0):,}
Stars Growth: {gh.get('avg_stars_growth', 0)*100:+.1f}%
Activity Score: {gh.get('avg_activity_score', 0):.2f}
Signal: {gh.get('signal', 'N/A')}

"""
        
        if 'job_postings' in sources:
            jp = sources['job_postings']
            report += f"""
💼 JOB POSTINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Openings: {jp.get('total_openings', 0):,}
30-Day Growth: {jp.get('growth_30d', 0)*100:+.1f}%
Signal: {jp.get('signal', 'N/A')}

"""
        
        if 'consumer_reviews' in sources:
            cr = sources['consumer_reviews']
            report += f"""
⭐ CONSUMER REVIEWS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Average Rating: {cr.get('average_rating', 0):.1f}/5.0
Total Reviews: {cr.get('total_reviews', 0):,}
Sentiment Score: {cr.get('sentiment_score', 0):.2f}
Signal: {cr.get('signal', 'N/A')}

"""
        
        if 'weather_impact' in sources:
            wi = sources['weather_impact']
            report += f"""
🌤️ WEATHER IMPACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sector: {wi.get('sector', 'N/A')}
Average Impact: {wi.get('average_impact', 0)*100:+.1f}%
Impact Days: {wi.get('impact_days', 0)} of {wi.get('total_forecast_days', 0)}
Signal: {wi.get('signal', SignalStrength.NEUTRAL).value if isinstance(wi.get('signal'), SignalStrength) else wi.get('signal', 'N/A')}

"""
        
        report += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  Alternative data signals are supplementary and should not be
    the sole basis for investment decisions.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return report


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize aggregator
        aggregator = AlternativeDataAggregator()
        
        # Get comprehensive signals for a stock
        analysis = await aggregator.get_comprehensive_signals(
            symbol="AAPL",
            company_name="Apple",
            sector="Technology"
        )
        
        # Generate report
        report = aggregator.generate_report(analysis)
        print(report)
        
        # Get trending tickers
        print("\n📈 TRENDING TICKERS")
        print("=" * 50)
        trending = await aggregator.social.get_trending_tickers(10)
        for t in trending:
            sentiment_icon = "🟢" if t['sentiment'] > 0.1 else "🔴" if t['sentiment'] < -0.1 else "⚪"
            print(f"{t['rank']:2}. {t['symbol']:<6} | {t['mentions_24h']:,} mentions | {sentiment_icon} {t['sentiment']:+.2f}")
    
    # Run async main
    asyncio.run(main())
