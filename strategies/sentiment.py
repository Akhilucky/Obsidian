"""
Sentiment-Based Trading Strategies
===================================
Trading strategies based on news sentiment, social media signals,
and options flow analysis.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Set
from collections import deque, Counter
from enum import Enum
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SentimentLevel(Enum):
    """Sentiment classification levels."""
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class SentimentScore:
    """Sentiment score with metadata."""
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    source: str
    timestamp: datetime
    text_snippet: str = ""
    
    @property
    def level(self) -> SentimentLevel:
        if self.score <= -0.6:
            return SentimentLevel.VERY_BEARISH
        elif self.score <= -0.2:
            return SentimentLevel.BEARISH
        elif self.score >= 0.6:
            return SentimentLevel.VERY_BULLISH
        elif self.score >= 0.2:
            return SentimentLevel.BULLISH
        return SentimentLevel.NEUTRAL


@dataclass
class NewsArticle:
    """News article representation."""
    title: str
    content: str
    source: str
    published_at: datetime
    url: str = ""
    symbols: List[str] = field(default_factory=list)
    sentiment: Optional[SentimentScore] = None


@dataclass
class SocialPost:
    """Social media post representation."""
    text: str
    author: str
    platform: str
    timestamp: datetime
    likes: int = 0
    shares: int = 0
    comments: int = 0
    symbols: List[str] = field(default_factory=list)
    sentiment: Optional[SentimentScore] = None


@dataclass
class OptionsFlow:
    """Options flow data."""
    symbol: str
    timestamp: datetime
    option_type: str  # 'call' or 'put'
    strike: float
    expiration: datetime
    premium: float
    volume: int
    open_interest: int
    is_unusual: bool = False
    trade_side: str = ""  # 'buy' or 'sell'


@dataclass
class SentimentConfig:
    """Configuration for sentiment strategies."""
    # Sentiment thresholds
    bullish_threshold: float = 0.3
    bearish_threshold: float = -0.3
    strong_threshold: float = 0.6
    
    # Time windows
    sentiment_window_hours: int = 24
    momentum_window_hours: int = 4
    
    # Minimum requirements
    min_articles: int = 3
    min_social_posts: int = 10
    min_confidence: float = 0.5
    
    # Weighting
    news_weight: float = 0.4
    social_weight: float = 0.3
    options_weight: float = 0.3
    
    # Trading parameters
    position_size: float = 1.0
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.05


class TextAnalyzer:
    """
    Text sentiment analysis using lexicon-based approach.
    Can be extended with ML models.
    """
    
    def __init__(self):
        # Financial sentiment lexicon
        self.positive_words = {
            'bullish', 'buy', 'long', 'upgrade', 'beat', 'exceeds', 'strong',
            'growth', 'profit', 'gain', 'positive', 'outperform', 'rally',
            'surge', 'soar', 'jump', 'boom', 'breakthrough', 'innovative',
            'opportunity', 'upside', 'momentum', 'breakout', 'moon', 'rocket',
            'diamond', 'hodl', 'alpha', 'winner', 'crush', 'smash', 'explode'
        }
        
        self.negative_words = {
            'bearish', 'sell', 'short', 'downgrade', 'miss', 'below', 'weak',
            'decline', 'loss', 'drop', 'negative', 'underperform', 'crash',
            'plunge', 'sink', 'fall', 'bust', 'failure', 'risk', 'danger',
            'downside', 'reversal', 'breakdown', 'dump', 'baghold', 'rekt',
            'loser', 'collapse', 'warning', 'concern', 'fear', 'uncertain'
        }
        
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'highly': 1.5, 'incredibly': 1.8,
            'absolutely': 2.0, 'definitely': 1.3, 'certainly': 1.3,
            'massive': 1.8, 'huge': 1.6, 'tiny': 0.5, 'slight': 0.5
        }
        
        self.negators = {'not', 'no', "n't", 'never', 'neither', 'nobody', 'nothing'}
    
    def analyze(self, text: str) -> SentimentScore:
        """
        Analyze sentiment of text.
        
        Returns SentimentScore with score between -1 and 1.
        """
        words = self._tokenize(text.lower())
        
        positive_count = 0
        negative_count = 0
        total_words = len(words)
        
        if total_words == 0:
            return SentimentScore(
                score=0.0,
                confidence=0.0,
                source="text_analysis",
                timestamp=datetime.now(),
                text_snippet=text[:100]
            )
        
        i = 0
        while i < len(words):
            word = words[i]
            multiplier = 1.0
            
            # Check for intensifiers
            if i > 0 and words[i-1] in self.intensifiers:
                multiplier = self.intensifiers[words[i-1]]
            
            # Check for negation
            if i > 0 and words[i-1] in self.negators:
                multiplier *= -1
            
            if word in self.positive_words:
                positive_count += multiplier
            elif word in self.negative_words:
                negative_count += multiplier
            
            i += 1
        
        # Calculate score
        total_sentiment = positive_count - negative_count
        max_possible = max(positive_count + negative_count, 1)
        score = total_sentiment / max_possible
        
        # Calculate confidence based on number of sentiment words
        sentiment_word_ratio = (positive_count + negative_count) / total_words
        confidence = min(sentiment_word_ratio * 5, 1.0)  # Scale up
        
        return SentimentScore(
            score=np.clip(score, -1, 1),
            confidence=confidence,
            source="text_analysis",
            timestamp=datetime.now(),
            text_snippet=text[:100]
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Remove special characters but keep important ones
        text = re.sub(r'[^a-zA-Z\s\'\-]', ' ', text)
        return text.split()
    
    def extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text."""
        # Look for $SYMBOL pattern
        dollar_symbols = re.findall(r'\$([A-Z]{1,5})\b', text.upper())
        
        # Look for common stock mention patterns
        cashtag_pattern = re.findall(r'#([A-Z]{1,5})\b', text.upper())
        
        return list(set(dollar_symbols + cashtag_pattern))


class NewsSentimentStrategy:
    """
    News-based sentiment trading strategy.
    
    Analyzes news articles for sentiment and generates signals
    based on aggregated sentiment scores.
    """
    
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        self.analyzer = TextAnalyzer()
        self.news_cache: Dict[str, List[NewsArticle]] = {}
        self.sentiment_history: Dict[str, List[SentimentScore]] = {}
    
    def add_article(self, article: NewsArticle):
        """Add a news article to the cache."""
        # Analyze sentiment if not already done
        if article.sentiment is None:
            combined_text = f"{article.title}. {article.content}"
            article.sentiment = self.analyzer.analyze(combined_text)
            article.sentiment.source = article.source
        
        # Extract symbols if not provided
        if not article.symbols:
            article.symbols = self.analyzer.extract_symbols(
                f"{article.title} {article.content}"
            )
        
        # Cache by symbol
        for symbol in article.symbols:
            if symbol not in self.news_cache:
                self.news_cache[symbol] = []
            self.news_cache[symbol].append(article)
            
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = []
            self.sentiment_history[symbol].append(article.sentiment)
    
    def get_aggregate_sentiment(self, 
                                 symbol: str,
                                 hours: int = None) -> Optional[SentimentScore]:
        """
        Calculate aggregate sentiment for a symbol.
        
        Uses time-weighted average of recent article sentiments.
        """
        hours = hours or self.config.sentiment_window_hours
        cutoff = datetime.now() - timedelta(hours=hours)
        
        if symbol not in self.news_cache:
            return None
        
        recent_articles = [
            a for a in self.news_cache[symbol]
            if a.published_at >= cutoff and a.sentiment
        ]
        
        if len(recent_articles) < self.config.min_articles:
            return None
        
        # Time-weighted average
        now = datetime.now()
        weighted_scores = []
        weights = []
        
        for article in recent_articles:
            # More recent = higher weight
            hours_ago = (now - article.published_at).total_seconds() / 3600
            weight = 1 / (1 + hours_ago * 0.1)  # Decay factor
            
            # Confidence-weighted
            weight *= article.sentiment.confidence
            
            weighted_scores.append(article.sentiment.score * weight)
            weights.append(weight)
        
        if sum(weights) == 0:
            return None
        
        avg_score = sum(weighted_scores) / sum(weights)
        avg_confidence = np.mean([a.sentiment.confidence for a in recent_articles])
        
        return SentimentScore(
            score=avg_score,
            confidence=avg_confidence,
            source="news_aggregate",
            timestamp=datetime.now(),
            text_snippet=f"{len(recent_articles)} articles analyzed"
        )
    
    def get_sentiment_momentum(self, symbol: str) -> float:
        """
        Calculate sentiment momentum (change in sentiment over time).
        """
        hours = self.config.momentum_window_hours
        cutoff = datetime.now() - timedelta(hours=hours)
        
        if symbol not in self.sentiment_history:
            return 0.0
        
        recent = [s for s in self.sentiment_history[symbol] 
                 if s.timestamp >= cutoff]
        
        if len(recent) < 2:
            return 0.0
        
        # Compare first half to second half
        mid = len(recent) // 2
        first_half = np.mean([s.score for s in recent[:mid]])
        second_half = np.mean([s.score for s in recent[mid:]])
        
        return second_half - first_half
    
    def generate_signal(self, symbol: str) -> Dict:
        """
        Generate trading signal based on news sentiment.
        """
        sentiment = self.get_aggregate_sentiment(symbol)
        if sentiment is None:
            return {'signal': 0, 'strength': SignalStrength.WEAK, 'reason': 'Insufficient data'}
        
        momentum = self.get_sentiment_momentum(symbol)
        
        signal = 0
        strength = SignalStrength.WEAK
        reason = ""
        
        # Check sentiment level
        if sentiment.score >= self.config.strong_threshold:
            signal = 1
            strength = SignalStrength.VERY_STRONG
            reason = "Very bullish news sentiment"
        elif sentiment.score >= self.config.bullish_threshold:
            signal = 1
            strength = SignalStrength.MODERATE
            reason = "Bullish news sentiment"
        elif sentiment.score <= -self.config.strong_threshold:
            signal = -1
            strength = SignalStrength.VERY_STRONG
            reason = "Very bearish news sentiment"
        elif sentiment.score <= self.config.bearish_threshold:
            signal = -1
            strength = SignalStrength.MODERATE
            reason = "Bearish news sentiment"
        
        # Adjust for momentum
        if signal != 0 and np.sign(momentum) == np.sign(signal):
            if strength == SignalStrength.MODERATE:
                strength = SignalStrength.STRONG
            reason += f" (momentum: {momentum:+.2f})"
        
        # Reduce confidence if sentiment contradicts momentum
        if signal != 0 and np.sign(momentum) != np.sign(signal):
            if strength.value > 1:
                strength = SignalStrength(strength.value - 1)
            reason += " [momentum divergence]"
        
        return {
            'signal': signal,
            'strength': strength,
            'sentiment_score': sentiment.score,
            'confidence': sentiment.confidence,
            'momentum': momentum,
            'reason': reason
        }


class SocialSentimentStrategy:
    """
    Social media sentiment strategy.
    
    Analyzes Reddit, Twitter, and other social media for
    sentiment signals with volume/engagement weighting.
    """
    
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        self.analyzer = TextAnalyzer()
        self.posts_cache: Dict[str, List[SocialPost]] = {}
        self.trending_symbols: Counter = Counter()
        self.sentiment_velocity: Dict[str, List[Tuple[datetime, float]]] = {}
    
    def add_post(self, post: SocialPost):
        """Add a social media post."""
        # Analyze sentiment
        if post.sentiment is None:
            post.sentiment = self.analyzer.analyze(post.text)
            post.sentiment.source = post.platform
        
        # Extract symbols
        if not post.symbols:
            post.symbols = self.analyzer.extract_symbols(post.text)
        
        # Update caches
        for symbol in post.symbols:
            if symbol not in self.posts_cache:
                self.posts_cache[symbol] = []
            self.posts_cache[symbol].append(post)
            
            # Update velocity tracking
            if symbol not in self.sentiment_velocity:
                self.sentiment_velocity[symbol] = []
            self.sentiment_velocity[symbol].append(
                (post.timestamp, post.sentiment.score)
            )
            
            # Update trending
            self.trending_symbols[symbol] += 1
    
    def calculate_engagement_weight(self, post: SocialPost) -> float:
        """
        Calculate weight based on engagement metrics.
        """
        # Log-scale to prevent outliers from dominating
        likes_weight = np.log1p(post.likes)
        shares_weight = np.log1p(post.shares) * 2  # Shares more valuable
        comments_weight = np.log1p(post.comments) * 1.5
        
        return likes_weight + shares_weight + comments_weight + 1  # +1 base weight
    
    def get_aggregate_sentiment(self,
                                 symbol: str,
                                 hours: int = None) -> Optional[SentimentScore]:
        """
        Calculate engagement-weighted sentiment for a symbol.
        """
        hours = hours or self.config.sentiment_window_hours
        cutoff = datetime.now() - timedelta(hours=hours)
        
        if symbol not in self.posts_cache:
            return None
        
        recent_posts = [
            p for p in self.posts_cache[symbol]
            if p.timestamp >= cutoff and p.sentiment
        ]
        
        if len(recent_posts) < self.config.min_social_posts:
            return None
        
        # Engagement and time weighted
        now = datetime.now()
        weighted_scores = []
        weights = []
        
        for post in recent_posts:
            # Time decay
            hours_ago = (now - post.timestamp).total_seconds() / 3600
            time_weight = 1 / (1 + hours_ago * 0.2)
            
            # Engagement weight
            engagement = self.calculate_engagement_weight(post)
            
            # Combined weight
            weight = time_weight * engagement * post.sentiment.confidence
            
            weighted_scores.append(post.sentiment.score * weight)
            weights.append(weight)
        
        if sum(weights) == 0:
            return None
        
        avg_score = sum(weighted_scores) / sum(weights)
        
        return SentimentScore(
            score=avg_score,
            confidence=min(len(recent_posts) / 50, 1.0),  # More posts = more confidence
            source="social_aggregate",
            timestamp=datetime.now(),
            text_snippet=f"{len(recent_posts)} posts analyzed"
        )
    
    def get_volume_spike(self, symbol: str, baseline_hours: int = 168) -> float:
        """
        Detect unusual mention volume (potential catalyst).
        
        Returns ratio of recent volume to baseline.
        """
        if symbol not in self.posts_cache:
            return 1.0
        
        now = datetime.now()
        recent_cutoff = now - timedelta(hours=4)
        baseline_cutoff = now - timedelta(hours=baseline_hours)
        
        recent_count = len([
            p for p in self.posts_cache[symbol]
            if p.timestamp >= recent_cutoff
        ])
        
        baseline_posts = [
            p for p in self.posts_cache[symbol]
            if baseline_cutoff <= p.timestamp < recent_cutoff
        ]
        
        if not baseline_posts:
            return 1.0
        
        baseline_hourly = len(baseline_posts) / (baseline_hours - 4)
        recent_hourly = recent_count / 4
        
        return recent_hourly / max(baseline_hourly, 0.1)
    
    def get_sentiment_velocity(self, symbol: str) -> float:
        """
        Calculate rate of change of sentiment.
        """
        if symbol not in self.sentiment_velocity:
            return 0.0
        
        cutoff = datetime.now() - timedelta(hours=4)
        recent = [(t, s) for t, s in self.sentiment_velocity[symbol] if t >= cutoff]
        
        if len(recent) < 5:
            return 0.0
        
        # Linear regression for velocity
        times = [(t - recent[0][0]).total_seconds() / 3600 for t, s in recent]
        scores = [s for t, s in recent]
        
        if len(set(times)) < 2:
            return 0.0
        
        # Simple slope calculation
        x_mean = np.mean(times)
        y_mean = np.mean(scores)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(times, scores))
        denominator = sum((x - x_mean) ** 2 for x in times)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def get_trending(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get top trending symbols by mention count."""
        return self.trending_symbols.most_common(top_n)
    
    def generate_signal(self, symbol: str) -> Dict:
        """
        Generate trading signal based on social sentiment.
        """
        sentiment = self.get_aggregate_sentiment(symbol)
        if sentiment is None:
            return {'signal': 0, 'strength': SignalStrength.WEAK, 'reason': 'Insufficient data'}
        
        volume_spike = self.get_volume_spike(symbol)
        velocity = self.get_sentiment_velocity(symbol)
        
        signal = 0
        strength = SignalStrength.WEAK
        reason = ""
        
        # Base signal from sentiment
        if sentiment.score >= self.config.bullish_threshold:
            signal = 1
            strength = SignalStrength.MODERATE
            reason = "Bullish social sentiment"
        elif sentiment.score <= self.config.bearish_threshold:
            signal = -1
            strength = SignalStrength.MODERATE
            reason = "Bearish social sentiment"
        
        # Strengthen on volume spike
        if volume_spike > 3:
            if strength.value < 4:
                strength = SignalStrength(strength.value + 1)
            reason += f" (volume spike: {volume_spike:.1f}x)"
        
        # Strengthen on positive velocity alignment
        if signal != 0 and np.sign(velocity) == np.sign(signal) and abs(velocity) > 0.1:
            if strength.value < 4:
                strength = SignalStrength(strength.value + 1)
            reason += f" (accelerating)"
        
        return {
            'signal': signal,
            'strength': strength,
            'sentiment_score': sentiment.score,
            'confidence': sentiment.confidence,
            'volume_spike': volume_spike,
            'velocity': velocity,
            'reason': reason
        }


class OptionsFlowStrategy:
    """
    Options flow analysis strategy.
    
    Analyzes unusual options activity to detect institutional
    positioning and generate trading signals.
    """
    
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        self.flow_data: Dict[str, List[OptionsFlow]] = {}
        self.unusual_activity: Dict[str, List[OptionsFlow]] = {}
    
    def add_flow(self, flow: OptionsFlow):
        """Add options flow data."""
        if flow.symbol not in self.flow_data:
            self.flow_data[flow.symbol] = []
        self.flow_data[flow.symbol].append(flow)
        
        if flow.is_unusual:
            if flow.symbol not in self.unusual_activity:
                self.unusual_activity[flow.symbol] = []
            self.unusual_activity[flow.symbol].append(flow)
    
    def detect_unusual_activity(self, flow: OptionsFlow) -> bool:
        """
        Detect if flow is unusual based on volume and OI.
        """
        # Unusual if volume > 50% of open interest
        if flow.open_interest > 0:
            vol_oi_ratio = flow.volume / flow.open_interest
            if vol_oi_ratio > 0.5:
                return True
        
        # Unusual premium threshold (e.g., > $100k)
        if flow.premium > 100000:
            return True
        
        return False
    
    def calculate_put_call_ratio(self, 
                                  symbol: str,
                                  hours: int = 24) -> float:
        """
        Calculate put/call ratio for a symbol.
        
        < 0.7: Bullish (more calls)
        0.7-1.3: Neutral
        > 1.3: Bearish (more puts)
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        if symbol not in self.flow_data:
            return 1.0
        
        recent = [f for f in self.flow_data[symbol] if f.timestamp >= cutoff]
        
        call_volume = sum(f.volume for f in recent if f.option_type == 'call')
        put_volume = sum(f.volume for f in recent if f.option_type == 'put')
        
        if call_volume == 0:
            return float('inf') if put_volume > 0 else 1.0
        
        return put_volume / call_volume
    
    def calculate_premium_flow(self, 
                                symbol: str,
                                hours: int = 24) -> Dict:
        """
        Calculate net premium flow (bullish vs bearish).
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        if symbol not in self.flow_data:
            return {'bullish': 0, 'bearish': 0, 'net': 0}
        
        recent = [f for f in self.flow_data[symbol] if f.timestamp >= cutoff]
        
        bullish_premium = 0
        bearish_premium = 0
        
        for flow in recent:
            if flow.option_type == 'call':
                if flow.trade_side == 'buy':
                    bullish_premium += flow.premium
                else:
                    bearish_premium += flow.premium
            else:  # put
                if flow.trade_side == 'buy':
                    bearish_premium += flow.premium
                else:
                    bullish_premium += flow.premium
        
        return {
            'bullish': bullish_premium,
            'bearish': bearish_premium,
            'net': bullish_premium - bearish_premium
        }
    
    def get_unusual_activity_summary(self, symbol: str) -> Dict:
        """
        Summarize unusual options activity.
        """
        if symbol not in self.unusual_activity:
            return {'count': 0, 'total_premium': 0, 'call_pct': 0}
        
        cutoff = datetime.now() - timedelta(hours=24)
        recent = [f for f in self.unusual_activity[symbol] if f.timestamp >= cutoff]
        
        if not recent:
            return {'count': 0, 'total_premium': 0, 'call_pct': 0}
        
        total_premium = sum(f.premium for f in recent)
        call_count = sum(1 for f in recent if f.option_type == 'call')
        call_pct = call_count / len(recent) if recent else 0
        
        # Categorize by strike distance (ITM, ATM, OTM)
        # Would need current price for proper calculation
        
        return {
            'count': len(recent),
            'total_premium': total_premium,
            'call_pct': call_pct,
            'avg_premium': total_premium / len(recent),
            'largest_trade': max(recent, key=lambda x: x.premium).premium
        }
    
    def detect_smart_money(self, symbol: str) -> Optional[str]:
        """
        Detect potential smart money positioning.
        
        Returns: 'bullish', 'bearish', or None
        """
        unusual = self.get_unusual_activity_summary(symbol)
        pc_ratio = self.calculate_put_call_ratio(symbol)
        premium_flow = self.calculate_premium_flow(symbol)
        
        # Smart money indicators
        bullish_signals = 0
        bearish_signals = 0
        
        # Unusual call activity
        if unusual['count'] > 5 and unusual['call_pct'] > 0.7:
            bullish_signals += 2
        elif unusual['count'] > 5 and unusual['call_pct'] < 0.3:
            bearish_signals += 2
        
        # Low P/C ratio (many calls)
        if pc_ratio < 0.5:
            bullish_signals += 1
        elif pc_ratio > 2.0:
            bearish_signals += 1
        
        # Premium flow
        if premium_flow['net'] > 500000:  # > $500k bullish
            bullish_signals += 2
        elif premium_flow['net'] < -500000:
            bearish_signals += 2
        
        if bullish_signals >= 3:
            return 'bullish'
        elif bearish_signals >= 3:
            return 'bearish'
        
        return None
    
    def generate_signal(self, symbol: str) -> Dict:
        """
        Generate trading signal based on options flow.
        """
        pc_ratio = self.calculate_put_call_ratio(symbol)
        premium_flow = self.calculate_premium_flow(symbol)
        unusual = self.get_unusual_activity_summary(symbol)
        smart_money = self.detect_smart_money(symbol)
        
        signal = 0
        strength = SignalStrength.WEAK
        reason = ""
        
        # Smart money detection is strongest signal
        if smart_money == 'bullish':
            signal = 1
            strength = SignalStrength.STRONG
            reason = "Smart money accumulating calls"
        elif smart_money == 'bearish':
            signal = -1
            strength = SignalStrength.STRONG
            reason = "Smart money accumulating puts"
        
        # P/C ratio signal
        elif pc_ratio < 0.5:
            signal = 1
            strength = SignalStrength.MODERATE
            reason = f"Bullish P/C ratio: {pc_ratio:.2f}"
        elif pc_ratio > 2.0:
            signal = -1
            strength = SignalStrength.MODERATE
            reason = f"Bearish P/C ratio: {pc_ratio:.2f}"
        
        # Premium flow can strengthen signal
        if signal != 0 and np.sign(premium_flow['net']) == np.sign(signal):
            if abs(premium_flow['net']) > 1000000:
                if strength.value < 4:
                    strength = SignalStrength(strength.value + 1)
                reason += f" (premium flow: ${premium_flow['net']/1e6:.1f}M)"
        
        # Unusual activity adds confidence
        if unusual['count'] > 10:
            if strength.value < 4:
                strength = SignalStrength(strength.value + 1)
        
        return {
            'signal': signal,
            'strength': strength,
            'pc_ratio': pc_ratio,
            'premium_flow': premium_flow,
            'unusual_count': unusual['count'],
            'smart_money': smart_money,
            'reason': reason
        }


class SentimentSuite:
    """
    Unified interface for all sentiment-based strategies.
    """
    
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        self.news_strategy = NewsSentimentStrategy(self.config)
        self.social_strategy = SocialSentimentStrategy(self.config)
        self.options_strategy = OptionsFlowStrategy(self.config)
    
    def add_news(self, article: NewsArticle):
        """Add news article."""
        self.news_strategy.add_article(article)
    
    def add_social_post(self, post: SocialPost):
        """Add social media post."""
        self.social_strategy.add_post(post)
    
    def add_options_flow(self, flow: OptionsFlow):
        """Add options flow data."""
        self.options_strategy.add_flow(flow)
    
    def get_combined_signal(self, symbol: str) -> Dict:
        """
        Get combined sentiment signal from all sources.
        
        Weights each source according to config and combines signals.
        """
        news_signal = self.news_strategy.generate_signal(symbol)
        social_signal = self.social_strategy.generate_signal(symbol)
        options_signal = self.options_strategy.generate_signal(symbol)
        
        # Calculate weighted signal
        weighted_signal = 0
        total_weight = 0
        
        # News component
        if news_signal['signal'] != 0:
            weight = self.config.news_weight * news_signal['strength'].value
            weighted_signal += news_signal['signal'] * weight
            total_weight += weight
        
        # Social component
        if social_signal['signal'] != 0:
            weight = self.config.social_weight * social_signal['strength'].value
            weighted_signal += social_signal['signal'] * weight
            total_weight += weight
        
        # Options component
        if options_signal['signal'] != 0:
            weight = self.config.options_weight * options_signal['strength'].value
            weighted_signal += options_signal['signal'] * weight
            total_weight += weight
        
        # Determine final signal
        if total_weight == 0:
            final_signal = 0
            final_strength = SignalStrength.WEAK
        else:
            avg_signal = weighted_signal / total_weight
            final_signal = 1 if avg_signal > 0.3 else (-1 if avg_signal < -0.3 else 0)
            
            # Determine strength from agreement
            signals = [s['signal'] for s in [news_signal, social_signal, options_signal]]
            agreement = sum(1 for s in signals if s == final_signal and s != 0)
            
            if agreement >= 3:
                final_strength = SignalStrength.VERY_STRONG
            elif agreement >= 2:
                final_strength = SignalStrength.STRONG
            else:
                final_strength = SignalStrength.MODERATE
        
        return {
            'signal': final_signal,
            'strength': final_strength,
            'news': news_signal,
            'social': social_signal,
            'options': options_signal,
            'weighted_score': weighted_signal / total_weight if total_weight > 0 else 0,
            'source_agreement': sum(1 for s in [news_signal, social_signal, options_signal] 
                                   if s['signal'] == final_signal and final_signal != 0)
        }
    
    def get_sentiment_dashboard(self, symbol: str) -> Dict:
        """
        Get comprehensive sentiment dashboard for a symbol.
        """
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'combined_signal': self.get_combined_signal(symbol),
            'news': {
                'sentiment': self.news_strategy.get_aggregate_sentiment(symbol),
                'momentum': self.news_strategy.get_sentiment_momentum(symbol)
            },
            'social': {
                'sentiment': self.social_strategy.get_aggregate_sentiment(symbol),
                'volume_spike': self.social_strategy.get_volume_spike(symbol),
                'velocity': self.social_strategy.get_sentiment_velocity(symbol)
            },
            'options': {
                'pc_ratio': self.options_strategy.calculate_put_call_ratio(symbol),
                'premium_flow': self.options_strategy.calculate_premium_flow(symbol),
                'unusual_activity': self.options_strategy.get_unusual_activity_summary(symbol),
                'smart_money': self.options_strategy.detect_smart_money(symbol)
            }
        }
    
    def get_trending_symbols(self) -> List[Dict]:
        """Get trending symbols across all sources."""
        trending = self.social_strategy.get_trending(20)
        
        results = []
        for symbol, count in trending:
            signal = self.get_combined_signal(symbol)
            results.append({
                'symbol': symbol,
                'mention_count': count,
                'signal': signal['signal'],
                'strength': signal['strength'].name
            })
        
        return results


# Factory function
def create_sentiment_strategy(
    strategy_type: str,
    config: SentimentConfig = None
) -> object:
    """
    Factory function to create sentiment strategies.
    
    Args:
        strategy_type: 'news', 'social', 'options', or 'combined'
        config: Strategy configuration
    """
    strategies = {
        'news': NewsSentimentStrategy,
        'social': SocialSentimentStrategy,
        'options': OptionsFlowStrategy,
        'combined': SentimentSuite
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_type}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_type](config)


if __name__ == "__main__":
    print("=== Sentiment Strategy Demo ===\n")
    
    # Create suite
    suite = SentimentSuite()
    
    # Add sample news
    articles = [
        NewsArticle(
            title="Apple Reports Record Breaking Quarter, Beats All Expectations",
            content="Apple Inc. reported outstanding earnings, with revenue growth exceeding analyst expectations. The company showed strong iPhone sales and services growth.",
            source="Reuters",
            published_at=datetime.now() - timedelta(hours=2),
            symbols=["AAPL"]
        ),
        NewsArticle(
            title="Apple's New Product Launch Drives Bullish Sentiment",
            content="Investors are extremely optimistic about Apple's new product lineup. The innovation could drive significant upside potential.",
            source="Bloomberg",
            published_at=datetime.now() - timedelta(hours=4),
            symbols=["AAPL"]
        ),
        NewsArticle(
            title="Tech Sector Rally Led by Apple",
            content="Apple shares surge as the company outperforms expectations. Strong growth momentum continues.",
            source="CNBC",
            published_at=datetime.now() - timedelta(hours=6),
            symbols=["AAPL"]
        )
    ]
    
    for article in articles:
        suite.add_news(article)
    
    # Add sample social posts
    posts = [
        SocialPost(
            text="$AAPL to the moon! 🚀 Diamond hands, this is going to explode!",
            author="trader123",
            platform="reddit",
            timestamp=datetime.now() - timedelta(hours=1),
            likes=150,
            shares=20,
            symbols=["AAPL"]
        ),
        SocialPost(
            text="Just bought more $AAPL calls. The earnings were incredible, bullish AF",
            author="optionsguy",
            platform="twitter",
            timestamp=datetime.now() - timedelta(minutes=30),
            likes=500,
            shares=50,
            symbols=["AAPL"]
        )
    ] * 10  # Simulate more posts
    
    for post in posts:
        suite.add_social_post(post)
    
    # Add sample options flow
    flows = [
        OptionsFlow(
            symbol="AAPL",
            timestamp=datetime.now() - timedelta(hours=1),
            option_type="call",
            strike=180,
            expiration=datetime.now() + timedelta(days=30),
            premium=500000,
            volume=5000,
            open_interest=2000,
            is_unusual=True,
            trade_side="buy"
        ),
        OptionsFlow(
            symbol="AAPL",
            timestamp=datetime.now() - timedelta(hours=2),
            option_type="call",
            strike=185,
            expiration=datetime.now() + timedelta(days=45),
            premium=300000,
            volume=3000,
            open_interest=1500,
            is_unusual=True,
            trade_side="buy"
        )
    ] * 5
    
    for flow in flows:
        suite.add_options_flow(flow)
    
    # Get combined signal
    print("--- AAPL Combined Sentiment Analysis ---\n")
    signal = suite.get_combined_signal("AAPL")
    
    print(f"Combined Signal: {'BUY' if signal['signal'] > 0 else 'SELL' if signal['signal'] < 0 else 'NEUTRAL'}")
    print(f"Strength: {signal['strength'].name}")
    print(f"Source Agreement: {signal['source_agreement']}/3")
    
    print("\n--- Individual Source Signals ---")
    print(f"News: {signal['news']['reason']}")
    print(f"Social: {signal['social']['reason']}")
    print(f"Options: {signal['options']['reason']}")
    
    print("\n--- Options Flow Details ---")
    print(f"Put/Call Ratio: {signal['options']['pc_ratio']:.2f}")
    print(f"Smart Money: {signal['options']['smart_money']}")
