"""
Sentiment Analysis Engine
=========================

Advanced NLP-based sentiment analysis for financial markets:
- News sentiment scoring
- Social media sentiment (Twitter/Reddit)
- SEC filings sentiment
- Earnings call transcript analysis

Institutional-grade sentiment scoring used by quant funds.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


class FinancialLexicon:
    """
    Financial-specific sentiment lexicon.
    Contains words and phrases specific to financial markets.
    """
    
    # Bullish (positive) terms
    BULLISH_WORDS = {
        'upgrade': 3, 'buy': 2, 'outperform': 3, 'bullish': 3,
        'growth': 2, 'surge': 3, 'rally': 3, 'breakout': 3,
        'profit': 2, 'beat': 2, 'exceed': 2, 'strong': 2,
        'positive': 2, 'opportunity': 2, 'upside': 2, 'gain': 2,
        'record': 2, 'momentum': 2, 'recovery': 2, 'innovation': 2,
        'acquisition': 1, 'dividend': 2, 'expansion': 2, 'partnership': 1,
        'breakthrough': 3, 'optimistic': 2, 'confident': 2, 'robust': 2,
        'accelerate': 2, 'soar': 3, 'spike': 2, 'jump': 2
    }
    
    # Bearish (negative) terms
    BEARISH_WORDS = {
        'downgrade': -3, 'sell': -2, 'underperform': -3, 'bearish': -3,
        'decline': -2, 'plunge': -3, 'crash': -4, 'correction': -2,
        'loss': -2, 'miss': -2, 'disappoint': -2, 'weak': -2,
        'negative': -2, 'risk': -1, 'downside': -2, 'drop': -2,
        'bankruptcy': -4, 'lawsuit': -2, 'investigation': -2, 'fraud': -4,
        'recession': -3, 'layoff': -2, 'restructuring': -1, 'debt': -1,
        'warning': -2, 'concern': -1, 'volatile': -1, 'uncertainty': -2,
        'fall': -2, 'tumble': -3, 'sink': -2, 'collapse': -4
    }
    
    # Neutral/context-dependent terms
    NEUTRAL_WORDS = {
        'hold', 'maintain', 'unchanged', 'stable', 'steady',
        'expect', 'forecast', 'estimate', 'predict', 'anticipate'
    }
    
    @classmethod
    def get_financial_score(cls, text):
        """Calculate financial-specific sentiment score."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        score = 0
        matches = {'bullish': [], 'bearish': []}
        
        for word in words:
            if word in cls.BULLISH_WORDS:
                score += cls.BULLISH_WORDS[word]
                matches['bullish'].append(word)
            elif word in cls.BEARISH_WORDS:
                score += cls.BEARISH_WORDS[word]
                matches['bearish'].append(word)
        
        # Normalize by text length
        normalized_score = score / max(len(words), 1) * 10
        
        return {
            'raw_score': score,
            'normalized_score': normalized_score,
            'matches': matches,
            'sentiment': 'bullish' if normalized_score > 0.5 else 'bearish' if normalized_score < -0.5 else 'neutral'
        }


class VaderSentimentAnalyzer:
    """
    VADER-based sentiment analysis optimized for financial text.
    VADER is specifically tuned for social media and financial news.
    """
    
    def __init__(self):
        if not VADER_AVAILABLE:
            raise ImportError("vaderSentiment required. Install: pip install vaderSentiment")
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Add financial terms to VADER lexicon
        financial_terms = {**FinancialLexicon.BULLISH_WORDS, **FinancialLexicon.BEARISH_WORDS}
        for word, score in financial_terms.items():
            self.analyzer.lexicon[word] = score
    
    def analyze(self, text):
        """Analyze sentiment of text."""
        scores = self.analyzer.polarity_scores(text)
        
        # Combine with financial lexicon
        fin_scores = FinancialLexicon.get_financial_score(text)
        
        # Weighted combination
        combined_score = 0.6 * scores['compound'] + 0.4 * (fin_scores['normalized_score'] / 10)
        
        return {
            'compound': combined_score,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'financial_score': fin_scores['normalized_score'],
            'sentiment': 'bullish' if combined_score > 0.1 else 'bearish' if combined_score < -0.1 else 'neutral'
        }
    
    def analyze_batch(self, texts):
        """Analyze multiple texts."""
        return [self.analyze(text) for text in texts]


class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analysis.
    State-of-the-art NLP model trained specifically on financial text.
    """
    
    def __init__(self, model_name="ProsusAI/finbert"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required. Install: pip install transformers torch")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
    
    def analyze(self, text, max_length=512):
        """Analyze sentiment using FinBERT."""
        # Truncate text if necessary
        if len(text) > max_length:
            text = text[:max_length]
        
        result = self.pipeline(text)[0]
        
        # Map labels to scores
        label_map = {'positive': 1, 'negative': -1, 'neutral': 0}
        sentiment_score = label_map.get(result['label'], 0) * result['score']
        
        return {
            'label': result['label'],
            'confidence': result['score'],
            'sentiment_score': sentiment_score
        }
    
    def analyze_batch(self, texts, batch_size=16):
        """Analyze multiple texts in batches."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = self.pipeline(batch)
            results.extend(batch_results)
        return results


class NewsSentimentAnalyzer:
    """
    Comprehensive news sentiment analyzer.
    Processes financial news headlines and articles.
    """
    
    def __init__(self, use_finbert=False):
        self.use_finbert = use_finbert
        
        if use_finbert and TRANSFORMERS_AVAILABLE:
            self.analyzer = FinBERTSentimentAnalyzer()
        elif VADER_AVAILABLE:
            self.analyzer = VaderSentimentAnalyzer()
        else:
            self.analyzer = None
    
    def analyze_news(self, news_items):
        """
        Analyze a list of news items.
        
        Args:
            news_items: List of dicts with 'title', 'content', 'date' keys
        
        Returns:
            DataFrame with sentiment scores
        """
        results = []
        
        for item in news_items:
            title = item.get('title', '')
            content = item.get('content', '')
            date = item.get('date', datetime.now())
            
            # Analyze title and content
            text = f"{title}. {content}"
            
            if self.analyzer:
                sentiment = self.analyzer.analyze(text)
            else:
                # Fallback to basic lexicon analysis
                sentiment = FinancialLexicon.get_financial_score(text)
                sentiment['compound'] = sentiment['normalized_score'] / 10
            
            results.append({
                'date': date,
                'title': title,
                'sentiment_score': sentiment.get('compound', sentiment.get('sentiment_score', 0)),
                'sentiment': sentiment.get('sentiment', sentiment.get('label', 'neutral'))
            })
        
        return pd.DataFrame(results)
    
    def get_aggregate_sentiment(self, news_df, lookback_days=7):
        """Calculate aggregate sentiment over a period."""
        if news_df.empty:
            return {'score': 0, 'sentiment': 'neutral', 'count': 0}
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        if 'date' in news_df.columns:
            news_df = news_df[news_df['date'] >= cutoff_date]
        
        if news_df.empty:
            return {'score': 0, 'sentiment': 'neutral', 'count': 0}
        
        avg_score = news_df['sentiment_score'].mean()
        
        return {
            'score': avg_score,
            'sentiment': 'bullish' if avg_score > 0.1 else 'bearish' if avg_score < -0.1 else 'neutral',
            'count': len(news_df),
            'bullish_pct': (news_df['sentiment'] == 'bullish').mean() * 100,
            'bearish_pct': (news_df['sentiment'] == 'bearish').mean() * 100
        }


class SocialMediaSentiment:
    """
    Social media sentiment analyzer.
    Processes Twitter, Reddit, StockTwits data.
    """
    
    def __init__(self):
        if VADER_AVAILABLE:
            self.analyzer = VaderSentimentAnalyzer()
        else:
            self.analyzer = None
    
    def parse_cashtags(self, text):
        """Extract stock tickers from cashtags."""
        return re.findall(r'\$([A-Z]{1,5})', text.upper())
    
    def parse_hashtags(self, text):
        """Extract hashtags from text."""
        return re.findall(r'#(\w+)', text)
    
    def analyze_post(self, text, ticker=None):
        """Analyze a social media post."""
        # Extract tickers mentioned
        tickers = self.parse_cashtags(text)
        if ticker and ticker.upper() not in tickers:
            tickers.append(ticker.upper())
        
        # Get sentiment
        if self.analyzer:
            sentiment = self.analyzer.analyze(text)
        else:
            sentiment = FinancialLexicon.get_financial_score(text)
            sentiment['compound'] = sentiment['normalized_score'] / 10
        
        # Detect urgency/emotion indicators
        urgency = len(re.findall(r'!+', text)) > 0
        all_caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        return {
            'tickers': tickers,
            'sentiment_score': sentiment.get('compound', 0),
            'sentiment': sentiment.get('sentiment', 'neutral'),
            'urgency': urgency,
            'emotional': all_caps_ratio > 0.5
        }
    
    def calculate_social_score(self, posts, ticker):
        """Calculate aggregate social sentiment for a ticker."""
        ticker = ticker.upper()
        relevant_posts = []
        
        for post in posts:
            analysis = self.analyze_post(post.get('text', ''), ticker)
            if ticker in analysis['tickers']:
                relevant_posts.append(analysis)
        
        if not relevant_posts:
            return {'score': 0, 'sentiment': 'neutral', 'volume': 0}
        
        scores = [p['sentiment_score'] for p in relevant_posts]
        avg_score = np.mean(scores)
        
        # Volume-weighted sentiment (more posts = stronger signal)
        volume_factor = min(len(relevant_posts) / 100, 1)  # Cap at 100 posts
        
        return {
            'score': avg_score,
            'weighted_score': avg_score * (0.5 + 0.5 * volume_factor),
            'sentiment': 'bullish' if avg_score > 0.1 else 'bearish' if avg_score < -0.1 else 'neutral',
            'volume': len(relevant_posts),
            'urgency_pct': np.mean([p['urgency'] for p in relevant_posts]) * 100,
            'emotional_pct': np.mean([p['emotional'] for p in relevant_posts]) * 100
        }


class EarningsCallAnalyzer:
    """
    Analyze earnings call transcripts for sentiment and key topics.
    Institutional-grade analysis of management tone and guidance.
    """
    
    def __init__(self):
        if VADER_AVAILABLE:
            self.analyzer = VaderSentimentAnalyzer()
        else:
            self.analyzer = None
        
        # Key topics to track
        self.topics = {
            'guidance': ['guidance', 'outlook', 'expect', 'forecast', 'anticipate'],
            'growth': ['growth', 'expand', 'increase', 'scale', 'momentum'],
            'margins': ['margin', 'profitability', 'cost', 'efficiency', 'expense'],
            'competition': ['competition', 'market share', 'competitive', 'rival'],
            'innovation': ['innovation', 'product', 'launch', 'develop', 'R&D'],
            'risks': ['risk', 'challenge', 'headwind', 'concern', 'uncertainty']
        }
    
    def analyze_transcript(self, transcript):
        """Analyze an earnings call transcript."""
        # Split into sentences
        sentences = re.split(r'[.!?]', transcript)
        
        # Overall sentiment
        overall_sentiment = []
        topic_sentiments = defaultdict(list)
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
            
            # Get sentiment
            if self.analyzer:
                sent = self.analyzer.analyze(sentence)
                score = sent['compound']
            else:
                sent = FinancialLexicon.get_financial_score(sentence)
                score = sent['normalized_score'] / 10
            
            overall_sentiment.append(score)
            
            # Categorize by topic
            sentence_lower = sentence.lower()
            for topic, keywords in self.topics.items():
                if any(kw in sentence_lower for kw in keywords):
                    topic_sentiments[topic].append(score)
        
        # Calculate results
        results = {
            'overall_sentiment': np.mean(overall_sentiment) if overall_sentiment else 0,
            'sentiment_std': np.std(overall_sentiment) if overall_sentiment else 0,
            'topic_sentiments': {}
        }
        
        for topic, scores in topic_sentiments.items():
            results['topic_sentiments'][topic] = {
                'score': np.mean(scores) if scores else 0,
                'mentions': len(scores)
            }
        
        # Confidence assessment
        results['tone'] = 'positive' if results['overall_sentiment'] > 0.1 else 'negative' if results['overall_sentiment'] < -0.1 else 'neutral'
        results['consistency'] = 1 - min(results['sentiment_std'], 1)  # Lower std = more consistent
        
        return results


class SentimentAggregator:
    """
    Master sentiment aggregator combining all sources.
    Creates a unified sentiment signal for trading decisions.
    """
    
    def __init__(self):
        self.news_analyzer = NewsSentimentAnalyzer()
        self.social_analyzer = SocialMediaSentiment()
        
        # Source weights
        self.weights = {
            'news': 0.4,
            'social': 0.3,
            'price_action': 0.3
        }
    
    def calculate_price_sentiment(self, price_data):
        """Derive sentiment from price action."""
        if price_data is None or len(price_data) < 20:
            return {'score': 0, 'sentiment': 'neutral'}
        
        returns = price_data['Close'].pct_change()
        
        # Short-term momentum
        short_term = returns.tail(5).mean()
        
        # Trend strength
        sma_20 = price_data['Close'].rolling(20).mean()
        trend = (price_data['Close'].iloc[-1] / sma_20.iloc[-1]) - 1
        
        # Combine
        score = 0.5 * (short_term * 100) + 0.5 * trend
        score = np.clip(score, -1, 1)
        
        return {
            'score': score,
            'sentiment': 'bullish' if score > 0.02 else 'bearish' if score < -0.02 else 'neutral',
            'momentum': short_term,
            'trend': trend
        }
    
    def get_composite_sentiment(self, ticker, news=None, social_posts=None, price_data=None):
        """
        Calculate composite sentiment from all sources.
        
        Returns:
            dict with composite score and breakdown
        """
        sentiments = {}
        
        # News sentiment
        if news:
            news_df = self.news_analyzer.analyze_news(news)
            sentiments['news'] = self.news_analyzer.get_aggregate_sentiment(news_df)
        else:
            sentiments['news'] = {'score': 0, 'sentiment': 'neutral'}
        
        # Social sentiment
        if social_posts:
            sentiments['social'] = self.social_analyzer.calculate_social_score(social_posts, ticker)
        else:
            sentiments['social'] = {'score': 0, 'sentiment': 'neutral'}
        
        # Price action sentiment
        sentiments['price_action'] = self.calculate_price_sentiment(price_data)
        
        # Composite score
        composite = sum(
            sentiments[source]['score'] * weight 
            for source, weight in self.weights.items()
        )
        
        return {
            'ticker': ticker,
            'composite_score': composite,
            'composite_sentiment': 'bullish' if composite > 0.1 else 'bearish' if composite < -0.1 else 'neutral',
            'breakdown': sentiments,
            'signal_strength': abs(composite),
            'recommendation': self._get_recommendation(composite)
        }
    
    def _get_recommendation(self, score):
        """Get trading recommendation based on sentiment score."""
        if score > 0.3:
            return 'STRONG BUY'
        elif score > 0.1:
            return 'BUY'
        elif score > -0.1:
            return 'HOLD'
        elif score > -0.3:
            return 'SELL'
        else:
            return 'STRONG SELL'


if __name__ == "__main__":
    print("=" * 60)
    print("Sentiment Analysis Engine")
    print("=" * 60)
    
    # Test financial lexicon
    print("\n--- Financial Lexicon Test ---")
    test_texts = [
        "Apple stock surges on strong earnings beat, bullish momentum expected",
        "Tesla shares plunge amid concerns over declining sales and increased competition",
        "Microsoft maintains steady growth with cloud services expansion"
    ]
    
    for text in test_texts:
        result = FinancialLexicon.get_financial_score(text)
        print(f"\nText: {text[:50]}...")
        print(f"Sentiment: {result['sentiment']} (score: {result['normalized_score']:.2f})")
    
    # Test VADER if available
    if VADER_AVAILABLE:
        print("\n--- VADER Analyzer Test ---")
        vader = VaderSentimentAnalyzer()
        for text in test_texts:
            result = vader.analyze(text)
            print(f"\nText: {text[:50]}...")
            print(f"Sentiment: {result['sentiment']} (compound: {result['compound']:.2f})")
    else:
        print("\n[VADER not installed. Install: pip install vaderSentiment]")
    
    # Test aggregator with price data
    print("\n--- Composite Sentiment Test ---")
    from data.openbb_integration import OpenBBIntegration
    
    openbb = OpenBBIntegration()
    price_data = openbb.fetch_stock_data('AAPL', '2024-01-01', '2024-12-31')
    
    if price_data is not None and not price_data.empty:
        aggregator = SentimentAggregator()
        
        # Mock news
        mock_news = [
            {'title': 'Apple reports record iPhone sales', 'content': 'Strong demand drives growth', 'date': datetime.now()},
            {'title': 'Apple faces regulatory challenges in EU', 'content': 'Concerns over app store policies', 'date': datetime.now()}
        ]
        
        result = aggregator.get_composite_sentiment('AAPL', news=mock_news, price_data=price_data)
        
        print(f"\nTicker: {result['ticker']}")
        print(f"Composite Score: {result['composite_score']:.3f}")
        print(f"Sentiment: {result['composite_sentiment']}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"\nBreakdown:")
        for source, data in result['breakdown'].items():
            print(f"  {source}: {data['sentiment']} ({data['score']:.3f})")
    
    print("\n" + "=" * 60)
    print("Sentiment Analysis Complete!")
    print("=" * 60)
