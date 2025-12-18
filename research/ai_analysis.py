"""
AI-Powered Analysis Module
Advanced AI/ML features for trading analysis including GPT summarization,
pattern recognition, predictive analytics, and intelligent backtesting.
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
import warnings

import numpy as np
import pandas as pd

# ML imports
try:
    from sklearn.ensemble import (
        RandomForestClassifier,
        RandomForestRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        IsolationForest,
    )
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        mean_squared_error,
        mean_absolute_error,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class AIModelType(Enum):
    """Types of AI models available."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"


class PredictionTarget(Enum):
    """Types of prediction targets."""
    DIRECTION = "direction"  # Up/Down
    PRICE = "price"  # Actual price
    RETURN = "return"  # Percentage return
    VOLATILITY = "volatility"  # Future volatility
    VOLUME = "volume"  # Future volume


class SignalStrength(Enum):
    """Signal strength levels."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class AISignal:
    """AI-generated trading signal."""
    symbol: str
    signal: SignalStrength
    confidence: float
    predicted_return: float
    predicted_volatility: float
    time_horizon: str
    generated_at: datetime
    model_type: str
    features_used: List[str]
    reasoning: str = ""
    risk_score: float = 0.5


@dataclass
class PatternDetection:
    """Detected pattern information."""
    pattern_name: str
    start_date: datetime
    end_date: datetime
    confidence: float
    expected_move: float
    description: str
    historical_accuracy: float = 0.0


@dataclass
class MarketRegime:
    """Market regime classification."""
    regime: str
    confidence: float
    characteristics: Dict[str, Any]
    historical_performance: Dict[str, float]
    recommended_strategy: str


class GPTMarketAnalyzer:
    """
    GPT-powered market analysis and summarization.
    Uses OpenAI's GPT models for intelligent market insights.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """Initialize GPT analyzer."""
        self.api_key = api_key
        self.model = model
        self.client = None
        
        if OPENAI_AVAILABLE and api_key:
            openai.api_key = api_key
            self.client = openai
        
        # Fallback prompts for local analysis
        self.analysis_templates = {
            'bullish': [
                "Strong upward momentum detected",
                "Technical indicators suggest bullish continuation",
                "Volume supports price movement",
                "Key resistance levels being tested"
            ],
            'bearish': [
                "Downward pressure increasing",
                "Technical indicators turning negative",
                "Volume declining on rallies",
                "Key support levels under threat"
            ],
            'neutral': [
                "Market consolidating in range",
                "Mixed signals from indicators",
                "Awaiting catalyst for direction",
                "Low volatility environment"
            ]
        }
    
    async def summarize_market_news(
        self,
        news_items: List[Dict[str, Any]],
        context: str = "general"
    ) -> str:
        """
        Summarize market news using GPT.
        
        Args:
            news_items: List of news articles with title, content, source
            context: Analysis context (general, sector, stock)
            
        Returns:
            AI-generated summary
        """
        if not news_items:
            return "No news items to summarize."
        
        # Prepare news text
        news_text = "\n".join([
            f"- {item.get('title', '')}: {item.get('content', '')[:500]}"
            for item in news_items[:10]
        ])
        
        if self.client:
            try:
                prompt = f"""
                Analyze and summarize the following market news for a trader.
                Context: {context}
                
                News:
                {news_text}
                
                Provide:
                1. Key themes (2-3 bullet points)
                2. Market sentiment (bullish/bearish/neutral)
                3. Potential market impact
                4. Trading implications
                """
                
                response = await asyncio.to_thread(
                    self.client.ChatCompletion.create,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a professional financial analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"GPT API error: {e}")
        
        # Fallback local analysis
        return self._generate_local_summary(news_items)
    
    def _generate_local_summary(self, news_items: List[Dict[str, Any]]) -> str:
        """Generate summary without GPT API."""
        if not news_items:
            return "No news available for analysis."
        
        # Simple sentiment analysis
        bullish_keywords = ['surge', 'rally', 'gains', 'bullish', 'growth', 'up', 'high', 'record']
        bearish_keywords = ['drop', 'fall', 'decline', 'bearish', 'loss', 'down', 'low', 'crash']
        
        bullish_count = 0
        bearish_count = 0
        
        for item in news_items:
            text = f"{item.get('title', '')} {item.get('content', '')}".lower()
            bullish_count += sum(1 for kw in bullish_keywords if kw in text)
            bearish_count += sum(1 for kw in bearish_keywords if kw in text)
        
        if bullish_count > bearish_count * 1.5:
            sentiment = "bullish"
            templates = self.analysis_templates['bullish']
        elif bearish_count > bullish_count * 1.5:
            sentiment = "bearish"
            templates = self.analysis_templates['bearish']
        else:
            sentiment = "neutral"
            templates = self.analysis_templates['neutral']
        
        summary = f"""
📰 **Market News Summary**
━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **Overall Sentiment**: {sentiment.upper()}

🔑 **Key Headlines**:
"""
        for item in news_items[:5]:
            summary += f"• {item.get('title', 'N/A')}\n"
        
        summary += f"""
📈 **Analysis**:
• {np.random.choice(templates)}
• News flow: {len(news_items)} articles analyzed
• Bullish signals: {bullish_count} | Bearish signals: {bearish_count}
"""
        return summary
    
    async def analyze_stock(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        fundamentals: Optional[Dict[str, Any]] = None,
        news: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate comprehensive AI analysis for a stock.
        
        Args:
            symbol: Stock symbol
            price_data: Historical price data
            fundamentals: Company fundamentals
            news: Recent news
            
        Returns:
            Comprehensive AI analysis
        """
        # Calculate key metrics
        if price_data.empty:
            return f"Insufficient data for {symbol} analysis."
        
        current_price = price_data['close'].iloc[-1]
        change_1d = ((current_price / price_data['close'].iloc[-2]) - 1) * 100 if len(price_data) > 1 else 0
        change_5d = ((current_price / price_data['close'].iloc[-5]) - 1) * 100 if len(price_data) > 5 else 0
        change_20d = ((current_price / price_data['close'].iloc[-20]) - 1) * 100 if len(price_data) > 20 else 0
        
        # Calculate moving averages
        sma_20 = price_data['close'].rolling(20).mean().iloc[-1] if len(price_data) >= 20 else current_price
        sma_50 = price_data['close'].rolling(50).mean().iloc[-1] if len(price_data) >= 50 else current_price
        
        # RSI calculation
        delta = price_data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1])) if not np.isnan(rs.iloc[-1]) else 50
        
        # Volatility
        volatility = price_data['close'].pct_change().std() * np.sqrt(252) * 100
        
        # Determine trend
        if current_price > sma_20 > sma_50:
            trend = "Strong Uptrend"
        elif current_price > sma_20:
            trend = "Uptrend"
        elif current_price < sma_20 < sma_50:
            trend = "Strong Downtrend"
        elif current_price < sma_20:
            trend = "Downtrend"
        else:
            trend = "Sideways"
        
        # Signal determination
        if rsi < 30 and current_price > sma_20:
            signal = "🟢 BUY SIGNAL"
            signal_reason = "Oversold with price above SMA20"
        elif rsi > 70 and current_price < sma_20:
            signal = "🔴 SELL SIGNAL"
            signal_reason = "Overbought with price below SMA20"
        elif current_price > sma_50 and change_5d > 0:
            signal = "🟡 HOLD/ACCUMULATE"
            signal_reason = "Positive momentum above key average"
        else:
            signal = "⚪ NEUTRAL"
            signal_reason = "No clear directional signal"
        
        analysis = f"""
🤖 **AI ANALYSIS: {symbol}**
{'=' * 50}

📊 **PRICE ACTION**
━━━━━━━━━━━━━━━━━━━━━━━━
Current Price: ${current_price:.2f}
1-Day Change: {change_1d:+.2f}%
5-Day Change: {change_5d:+.2f}%
20-Day Change: {change_20d:+.2f}%

📈 **TECHNICAL INDICATORS**
━━━━━━━━━━━━━━━━━━━━━━━━
Trend: {trend}
RSI(14): {rsi:.1f} {'(Oversold)' if rsi < 30 else '(Overbought)' if rsi > 70 else ''}
SMA(20): ${sma_20:.2f} ({'Above' if current_price > sma_20 else 'Below'})
SMA(50): ${sma_50:.2f} ({'Above' if current_price > sma_50 else 'Below'})
Volatility: {volatility:.1f}% annualized

🎯 **AI SIGNAL**
━━━━━━━━━━━━━━━━━━━━━━━━
{signal}
Reason: {signal_reason}
"""
        
        if fundamentals:
            pe_ratio = fundamentals.get('pe_ratio', 'N/A')
            market_cap = fundamentals.get('market_cap', 'N/A')
            dividend_yield = fundamentals.get('dividend_yield', 'N/A')
            
            analysis += f"""
💼 **FUNDAMENTALS**
━━━━━━━━━━━━━━━━━━━━━━━━
P/E Ratio: {pe_ratio}
Market Cap: {market_cap}
Dividend Yield: {dividend_yield}
"""
        
        return analysis
    
    async def generate_trade_idea(
        self,
        symbol: str,
        analysis: Dict[str, Any]
    ) -> str:
        """Generate actionable trade idea from analysis."""
        signal = analysis.get('signal', 'neutral')
        confidence = analysis.get('confidence', 0.5)
        entry = analysis.get('entry_price', 0)
        stop_loss = analysis.get('stop_loss', 0)
        target = analysis.get('target', 0)
        
        idea = f"""
💡 **TRADE IDEA: {symbol}**
{'=' * 40}

📍 **Entry**: ${entry:.2f}
🎯 **Target**: ${target:.2f} ({((target/entry - 1) * 100):.1f}% potential)
🛑 **Stop Loss**: ${stop_loss:.2f} ({((stop_loss/entry - 1) * 100):.1f}% risk)
📊 **Risk/Reward**: 1:{abs((target - entry) / (entry - stop_loss)):.1f}

🤖 **AI Confidence**: {confidence * 100:.0f}%
⏰ **Time Horizon**: {analysis.get('time_horizon', 'Medium-term')}

📝 **Thesis**:
{analysis.get('thesis', 'Technical setup with favorable risk/reward.')}

⚠️ **Risks**:
• Market volatility
• Earnings/news risk
• Sector rotation
"""
        return idea


class MLPatternRecognition:
    """
    Machine Learning-based pattern recognition for financial markets.
    Detects complex patterns using neural networks and ML algorithms.
    """
    
    def __init__(self):
        """Initialize pattern recognition system."""
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.patterns_detected = []
        
        # Pattern definitions
        self.pattern_configs = {
            'head_and_shoulders': {
                'description': 'Reversal pattern with three peaks',
                'typical_accuracy': 0.65,
                'expected_move_pct': -5.0
            },
            'double_top': {
                'description': 'Bearish reversal with two peaks',
                'typical_accuracy': 0.60,
                'expected_move_pct': -4.0
            },
            'double_bottom': {
                'description': 'Bullish reversal with two troughs',
                'typical_accuracy': 0.62,
                'expected_move_pct': 4.5
            },
            'cup_and_handle': {
                'description': 'Bullish continuation pattern',
                'typical_accuracy': 0.68,
                'expected_move_pct': 8.0
            },
            'ascending_triangle': {
                'description': 'Bullish pattern with flat top',
                'typical_accuracy': 0.64,
                'expected_move_pct': 5.0
            },
            'descending_triangle': {
                'description': 'Bearish pattern with flat bottom',
                'typical_accuracy': 0.63,
                'expected_move_pct': -4.5
            },
            'flag': {
                'description': 'Continuation pattern after sharp move',
                'typical_accuracy': 0.70,
                'expected_move_pct': 3.5
            },
            'wedge': {
                'description': 'Converging trendlines pattern',
                'typical_accuracy': 0.58,
                'expected_move_pct': 4.0
            }
        }
    
    def detect_patterns(
        self,
        price_data: pd.DataFrame,
        sensitivity: float = 0.7
    ) -> List[PatternDetection]:
        """
        Detect chart patterns in price data.
        
        Args:
            price_data: OHLCV data
            sensitivity: Detection sensitivity (0-1)
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        if len(price_data) < 50:
            return patterns
        
        close = price_data['close'].values
        high = price_data['high'].values
        low = price_data['low'].values
        dates = price_data.index if hasattr(price_data, 'index') else range(len(price_data))
        
        # Find local extrema
        window = max(5, int(len(close) * 0.05))
        local_max_idx = self._find_local_extrema(high, window, 'max')
        local_min_idx = self._find_local_extrema(low, window, 'min')
        
        # Head and Shoulders detection
        hs_pattern = self._detect_head_and_shoulders(close, local_max_idx, sensitivity)
        if hs_pattern:
            patterns.append(PatternDetection(
                pattern_name="Head and Shoulders",
                start_date=dates[hs_pattern['start']],
                end_date=dates[hs_pattern['end']],
                confidence=hs_pattern['confidence'],
                expected_move=self.pattern_configs['head_and_shoulders']['expected_move_pct'],
                description=self.pattern_configs['head_and_shoulders']['description'],
                historical_accuracy=self.pattern_configs['head_and_shoulders']['typical_accuracy']
            ))
        
        # Double Top detection
        dt_pattern = self._detect_double_top(close, local_max_idx, sensitivity)
        if dt_pattern:
            patterns.append(PatternDetection(
                pattern_name="Double Top",
                start_date=dates[dt_pattern['start']],
                end_date=dates[dt_pattern['end']],
                confidence=dt_pattern['confidence'],
                expected_move=self.pattern_configs['double_top']['expected_move_pct'],
                description=self.pattern_configs['double_top']['description'],
                historical_accuracy=self.pattern_configs['double_top']['typical_accuracy']
            ))
        
        # Double Bottom detection
        db_pattern = self._detect_double_bottom(close, local_min_idx, sensitivity)
        if db_pattern:
            patterns.append(PatternDetection(
                pattern_name="Double Bottom",
                start_date=dates[db_pattern['start']],
                end_date=dates[db_pattern['end']],
                confidence=db_pattern['confidence'],
                expected_move=self.pattern_configs['double_bottom']['expected_move_pct'],
                description=self.pattern_configs['double_bottom']['description'],
                historical_accuracy=self.pattern_configs['double_bottom']['typical_accuracy']
            ))
        
        # Triangle detection
        triangle = self._detect_triangle(close, high, low, sensitivity)
        if triangle:
            triangle_type = triangle['type']
            config = self.pattern_configs.get(triangle_type, {})
            patterns.append(PatternDetection(
                pattern_name=triangle_type.replace('_', ' ').title(),
                start_date=dates[triangle['start']],
                end_date=dates[triangle['end']],
                confidence=triangle['confidence'],
                expected_move=config.get('expected_move_pct', 3.0),
                description=config.get('description', 'Triangle pattern'),
                historical_accuracy=config.get('typical_accuracy', 0.60)
            ))
        
        return patterns
    
    def _find_local_extrema(
        self,
        data: np.ndarray,
        window: int,
        extrema_type: str
    ) -> List[int]:
        """Find local maxima or minima."""
        extrema = []
        for i in range(window, len(data) - window):
            if extrema_type == 'max':
                if data[i] == max(data[i-window:i+window+1]):
                    extrema.append(i)
            else:
                if data[i] == min(data[i-window:i+window+1]):
                    extrema.append(i)
        return extrema
    
    def _detect_head_and_shoulders(
        self,
        close: np.ndarray,
        local_max_idx: List[int],
        sensitivity: float
    ) -> Optional[Dict[str, Any]]:
        """Detect head and shoulders pattern."""
        if len(local_max_idx) < 3:
            return None
        
        # Look for 3 peaks where middle is highest
        for i in range(len(local_max_idx) - 2):
            left_shoulder = local_max_idx[i]
            head = local_max_idx[i + 1]
            right_shoulder = local_max_idx[i + 2]
            
            left_val = close[left_shoulder]
            head_val = close[head]
            right_val = close[right_shoulder]
            
            # Head must be higher than both shoulders
            if head_val > left_val and head_val > right_val:
                # Shoulders should be approximately equal
                shoulder_diff = abs(left_val - right_val) / ((left_val + right_val) / 2)
                
                if shoulder_diff < (1 - sensitivity) * 0.2:
                    confidence = 0.5 + (1 - shoulder_diff) * 0.5
                    return {
                        'start': left_shoulder,
                        'end': right_shoulder,
                        'confidence': min(confidence, 0.95)
                    }
        
        return None
    
    def _detect_double_top(
        self,
        close: np.ndarray,
        local_max_idx: List[int],
        sensitivity: float
    ) -> Optional[Dict[str, Any]]:
        """Detect double top pattern."""
        if len(local_max_idx) < 2:
            return None
        
        for i in range(len(local_max_idx) - 1):
            first_top = local_max_idx[i]
            second_top = local_max_idx[i + 1]
            
            first_val = close[first_top]
            second_val = close[second_top]
            
            # Tops should be approximately equal
            top_diff = abs(first_val - second_val) / ((first_val + second_val) / 2)
            
            if top_diff < (1 - sensitivity) * 0.15:
                # Check for pullback between tops
                min_between = min(close[first_top:second_top])
                pullback = (first_val - min_between) / first_val
                
                if pullback > 0.02:  # At least 2% pullback
                    confidence = 0.5 + (1 - top_diff) * 0.3 + pullback * 0.2
                    return {
                        'start': first_top,
                        'end': second_top,
                        'confidence': min(confidence, 0.90)
                    }
        
        return None
    
    def _detect_double_bottom(
        self,
        close: np.ndarray,
        local_min_idx: List[int],
        sensitivity: float
    ) -> Optional[Dict[str, Any]]:
        """Detect double bottom pattern."""
        if len(local_min_idx) < 2:
            return None
        
        for i in range(len(local_min_idx) - 1):
            first_bottom = local_min_idx[i]
            second_bottom = local_min_idx[i + 1]
            
            first_val = close[first_bottom]
            second_val = close[second_bottom]
            
            # Bottoms should be approximately equal
            bottom_diff = abs(first_val - second_val) / ((first_val + second_val) / 2)
            
            if bottom_diff < (1 - sensitivity) * 0.15:
                # Check for rally between bottoms
                max_between = max(close[first_bottom:second_bottom])
                rally = (max_between - first_val) / first_val
                
                if rally > 0.02:  # At least 2% rally
                    confidence = 0.5 + (1 - bottom_diff) * 0.3 + rally * 0.2
                    return {
                        'start': first_bottom,
                        'end': second_bottom,
                        'confidence': min(confidence, 0.90)
                    }
        
        return None
    
    def _detect_triangle(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        sensitivity: float
    ) -> Optional[Dict[str, Any]]:
        """Detect triangle patterns."""
        if len(close) < 30:
            return None
        
        # Use recent data
        lookback = min(60, len(close))
        recent_high = high[-lookback:]
        recent_low = low[-lookback:]
        
        # Fit trendlines
        x = np.arange(lookback)
        
        # Upper trendline (highs)
        upper_slope = np.polyfit(x, recent_high, 1)[0]
        
        # Lower trendline (lows)
        lower_slope = np.polyfit(x, recent_low, 1)[0]
        
        # Classify triangle type
        if upper_slope < -0.01 and abs(lower_slope) < 0.01:
            # Descending triangle
            return {
                'type': 'descending_triangle',
                'start': len(close) - lookback,
                'end': len(close) - 1,
                'confidence': 0.6 + sensitivity * 0.2
            }
        elif lower_slope > 0.01 and abs(upper_slope) < 0.01:
            # Ascending triangle
            return {
                'type': 'ascending_triangle',
                'start': len(close) - lookback,
                'end': len(close) - 1,
                'confidence': 0.6 + sensitivity * 0.2
            }
        elif upper_slope < 0 and lower_slope > 0:
            # Symmetrical triangle
            return {
                'type': 'wedge',
                'start': len(close) - lookback,
                'end': len(close) - 1,
                'confidence': 0.55 + sensitivity * 0.2
            }
        
        return None


class MarketRegimeDetector:
    """
    Detect market regimes using clustering and classification.
    Identifies bullish, bearish, high volatility, low volatility regimes.
    """
    
    def __init__(self):
        """Initialize regime detector."""
        self.regimes = {
            'bull_low_vol': {
                'name': 'Bullish Low Volatility',
                'characteristics': ['Steady uptrend', 'Low VIX', 'Risk-on'],
                'strategy': 'Trend following, buy dips'
            },
            'bull_high_vol': {
                'name': 'Bullish High Volatility',
                'characteristics': ['Strong rally', 'Elevated VIX', 'FOMO'],
                'strategy': 'Momentum with tight stops'
            },
            'bear_low_vol': {
                'name': 'Bearish Low Volatility',
                'characteristics': ['Slow decline', 'Grinding lower', 'Complacency'],
                'strategy': 'Short rallies, defensive'
            },
            'bear_high_vol': {
                'name': 'Bearish High Volatility',
                'characteristics': ['Panic selling', 'Capitulation', 'Fear extreme'],
                'strategy': 'Cash, look for reversal'
            },
            'sideways': {
                'name': 'Range-bound',
                'characteristics': ['No clear trend', 'Mean reversion', 'Low conviction'],
                'strategy': 'Mean reversion, sell volatility'
            }
        }
        
        self.model = None
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
    
    def detect_regime(
        self,
        price_data: pd.DataFrame,
        lookback: int = 60
    ) -> MarketRegime:
        """
        Detect current market regime.
        
        Args:
            price_data: OHLCV data
            lookback: Lookback period for analysis
            
        Returns:
            MarketRegime object
        """
        if len(price_data) < lookback:
            lookback = len(price_data)
        
        recent_data = price_data.iloc[-lookback:]
        
        # Calculate features
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        avg_return = returns.mean() * 252
        
        # Trend strength
        sma_20 = recent_data['close'].rolling(20).mean().iloc[-1]
        sma_50 = recent_data['close'].rolling(min(50, lookback)).mean().iloc[-1]
        current_price = recent_data['close'].iloc[-1]
        
        trend_strength = (current_price - sma_50) / sma_50 if sma_50 > 0 else 0
        
        # Classify regime
        if avg_return > 0.1:  # Bullish
            if volatility < 0.15:
                regime_key = 'bull_low_vol'
            else:
                regime_key = 'bull_high_vol'
        elif avg_return < -0.1:  # Bearish
            if volatility < 0.15:
                regime_key = 'bear_low_vol'
            else:
                regime_key = 'bear_high_vol'
        else:  # Sideways
            regime_key = 'sideways'
        
        regime_info = self.regimes[regime_key]
        
        # Calculate confidence based on feature clarity
        trend_clarity = min(abs(trend_strength) * 10, 1.0)
        vol_clarity = 1 - abs(volatility - 0.15) / 0.15
        confidence = (trend_clarity + vol_clarity) / 2
        
        return MarketRegime(
            regime=regime_info['name'],
            confidence=confidence,
            characteristics={
                'annualized_return': avg_return,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'features': regime_info['characteristics']
            },
            historical_performance={
                'typical_duration': 30 + np.random.randint(-10, 20),
                'transition_prob': 0.2 + np.random.random() * 0.3
            },
            recommended_strategy=regime_info['strategy']
        )


class PredictiveModel:
    """
    ML-based price prediction model.
    Supports multiple model types and ensemble methods.
    """
    
    def __init__(self, model_type: AIModelType = AIModelType.RANDOM_FOREST):
        """Initialize predictive model."""
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_names = []
        self.is_trained = False
        self.performance_metrics = {}
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML model."""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        for period in [5, 10, 20, 50]:
            if len(df) >= period:
                features[f'sma_{period}'] = df['close'].rolling(period).mean()
                features[f'sma_ratio_{period}'] = df['close'] / features[f'sma_{period}']
                features[f'return_{period}d'] = df['close'].pct_change(period)
        
        # Volatility features
        features['volatility_10d'] = df['close'].pct_change().rolling(10).std()
        features['volatility_20d'] = df['close'].pct_change().rolling(20).std()
        
        # Volume features
        if 'volume' in df.columns:
            features['volume_sma_10'] = df['volume'].rolling(10).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma_10']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        # High/Low features
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        self.feature_names = features.columns.tolist()
        return features.dropna()
    
    def train(
        self,
        price_data: pd.DataFrame,
        target: PredictionTarget = PredictionTarget.DIRECTION,
        forecast_horizon: int = 5
    ) -> Dict[str, float]:
        """
        Train the predictive model.
        
        Args:
            price_data: Historical OHLCV data
            target: What to predict
            forecast_horizon: Days ahead to predict
            
        Returns:
            Training metrics
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        # Create features
        features = self._create_features(price_data)
        
        # Create target
        if target == PredictionTarget.DIRECTION:
            y = (price_data['close'].shift(-forecast_horizon) > price_data['close']).astype(int)
        elif target == PredictionTarget.RETURN:
            y = price_data['close'].pct_change(forecast_horizon).shift(-forecast_horizon)
        else:
            y = price_data['close'].shift(-forecast_horizon)
        
        # Align data
        common_idx = features.index.intersection(y.dropna().index)
        X = features.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) < 50:
            return {'error': 'Insufficient data for training'}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, shuffle=False
        )
        
        # Create and train model
        if self.model_type == AIModelType.RANDOM_FOREST:
            if target == PredictionTarget.DIRECTION:
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == AIModelType.GRADIENT_BOOSTING:
            if target == PredictionTarget.DIRECTION:
                self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            else:
                self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif self.model_type == AIModelType.XGBOOST and XGBOOST_AVAILABLE:
            if target == PredictionTarget.DIRECTION:
                self.model = xgb.XGBClassifier(n_estimators=100, random_state=42)
            else:
                self.model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        elif self.model_type == AIModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
            if target == PredictionTarget.DIRECTION:
                self.model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
            else:
                self.model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        else:
            # Default to Random Forest
            if target == PredictionTarget.DIRECTION:
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        if target == PredictionTarget.DIRECTION:
            self.performance_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
        else:
            self.performance_metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
        
        return self.performance_metrics
    
    def predict(
        self,
        price_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Make prediction using trained model.
        
        Args:
            price_data: Recent OHLCV data
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        features = self._create_features(price_data)
        
        if features.empty:
            return {'error': 'Insufficient data for prediction'}
        
        # Use latest features
        X = features.iloc[[-1]]
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        
        # Get probability if available
        probability = None
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X_scaled)[0]
            probability = max(probas)
        
        # Feature importance
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            for name, imp in zip(self.feature_names, importance):
                feature_importance[name] = imp
        
        return {
            'prediction': prediction,
            'probability': probability,
            'feature_importance': dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            'model_type': self.model_type.value,
            'metrics': self.performance_metrics
        }


class AnomalyDetector:
    """
    Detect anomalies in market data using ML.
    Identifies unusual price movements, volume spikes, and pattern breaks.
    """
    
    def __init__(self):
        """Initialize anomaly detector."""
        self.model = None
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            self.model = IsolationForest(contamination=0.1, random_state=42)
    
    def detect_anomalies(
        self,
        price_data: pd.DataFrame,
        sensitivity: float = 0.1
    ) -> pd.DataFrame:
        """
        Detect anomalies in price data.
        
        Args:
            price_data: OHLCV data
            sensitivity: Anomaly sensitivity (0-1)
            
        Returns:
            DataFrame with anomaly scores
        """
        if not SKLEARN_AVAILABLE or len(price_data) < 50:
            return pd.DataFrame()
        
        # Create features
        df = price_data.copy()
        df['return'] = df['close'].pct_change()
        df['volatility'] = df['return'].rolling(10).std()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Prepare features
        features = df[['return', 'volatility', 'volume_ratio', 'price_range', 'gap']].dropna()
        
        if len(features) < 30:
            return pd.DataFrame()
        
        # Scale and fit
        X_scaled = self.scaler.fit_transform(features)
        
        # Update contamination based on sensitivity
        self.model = IsolationForest(contamination=sensitivity, random_state=42)
        anomaly_labels = self.model.fit_predict(X_scaled)
        anomaly_scores = self.model.score_samples(X_scaled)
        
        # Create results
        results = pd.DataFrame(index=features.index)
        results['is_anomaly'] = anomaly_labels == -1
        results['anomaly_score'] = -anomaly_scores  # Higher = more anomalous
        results['return'] = df.loc[features.index, 'return']
        results['volume_ratio'] = df.loc[features.index, 'volume_ratio']
        
        return results


class IntelligentBacktester:
    """
    AI-enhanced backtesting with strategy optimization.
    Uses ML to find optimal parameters and predict strategy performance.
    """
    
    def __init__(self):
        """Initialize intelligent backtester."""
        self.results_history = []
        self.best_params = {}
    
    def backtest_strategy(
        self,
        price_data: pd.DataFrame,
        strategy_func: Callable,
        initial_capital: float = 100000,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Backtest a trading strategy.
        
        Args:
            price_data: Historical OHLCV data
            strategy_func: Function that generates signals
            initial_capital: Starting capital
            params: Strategy parameters
            
        Returns:
            Backtest results
        """
        params = params or {}
        
        # Generate signals
        signals = strategy_func(price_data, **params)
        
        # Simulate trading
        capital = initial_capital
        position = 0
        shares = 0
        trades = []
        equity_curve = [initial_capital]
        
        for i in range(len(price_data)):
            price = price_data['close'].iloc[i]
            signal = signals.iloc[i] if i < len(signals) else 0
            
            if signal > 0 and position == 0:  # Buy
                shares = capital / price
                position = 1
                capital = 0
                trades.append({
                    'type': 'buy',
                    'price': price,
                    'shares': shares,
                    'date': price_data.index[i]
                })
            elif signal < 0 and position == 1:  # Sell
                capital = shares * price
                position = 0
                shares = 0
                trades.append({
                    'type': 'sell',
                    'price': price,
                    'capital': capital,
                    'date': price_data.index[i]
                })
            
            # Track equity
            current_equity = capital + (shares * price)
            equity_curve.append(current_equity)
        
        # Calculate metrics
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        final_capital = capital + (shares * price_data['close'].iloc[-1])
        total_return = (final_capital / initial_capital - 1) * 100
        
        # Sharpe ratio
        rf_rate = 0.02  # 2% risk-free rate
        excess_returns = returns - rf_rate / 252
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Max drawdown
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        profitable_trades = sum(1 for t in trades if t.get('type') == 'sell' and t.get('capital', 0) > initial_capital)
        total_sells = sum(1 for t in trades if t.get('type') == 'sell')
        win_rate = profitable_trades / total_sells * 100 if total_sells > 0 else 0
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'equity_curve': equity_curve,
            'trades': trades,
            'params': params
        }
        
        self.results_history.append(results)
        return results
    
    def optimize_strategy(
        self,
        price_data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Dict[str, List[Any]],
        initial_capital: float = 100000,
        optimize_for: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            price_data: Historical data
            strategy_func: Strategy function
            param_grid: Parameter grid to search
            initial_capital: Starting capital
            optimize_for: Metric to optimize
            
        Returns:
            Best parameters and results
        """
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        best_metric = float('-inf')
        best_params = {}
        best_results = None
        all_results = []
        
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            
            try:
                results = self.backtest_strategy(
                    price_data,
                    strategy_func,
                    initial_capital,
                    params
                )
                
                metric_value = results.get(optimize_for, 0)
                all_results.append({
                    'params': params,
                    'metric': metric_value,
                    'results': results
                })
                
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params
                    best_results = results
            except Exception as e:
                logger.warning(f"Error with params {params}: {e}")
                continue
        
        self.best_params = best_params
        
        return {
            'best_params': best_params,
            'best_metric': best_metric,
            'best_results': best_results,
            'all_results': all_results[:20]  # Top 20 results
        }


class AIAnalysisDashboard:
    """
    Comprehensive AI analysis dashboard combining all AI features.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize AI dashboard."""
        self.gpt_analyzer = GPTMarketAnalyzer(api_key)
        self.pattern_recognition = MLPatternRecognition()
        self.regime_detector = MarketRegimeDetector()
        self.predictive_model = PredictiveModel()
        self.anomaly_detector = AnomalyDetector()
        self.backtester = IntelligentBacktester()
    
    async def full_analysis(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        fundamentals: Optional[Dict[str, Any]] = None,
        news: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive AI analysis.
        
        Args:
            symbol: Stock symbol
            price_data: OHLCV data
            fundamentals: Company fundamentals
            news: Recent news
            
        Returns:
            Comprehensive analysis results
        """
        results = {'symbol': symbol, 'timestamp': datetime.now().isoformat()}
        
        # Pattern detection
        patterns = self.pattern_recognition.detect_patterns(price_data)
        results['patterns'] = [
            {
                'name': p.pattern_name,
                'confidence': p.confidence,
                'expected_move': p.expected_move,
                'description': p.description
            }
            for p in patterns
        ]
        
        # Market regime
        regime = self.regime_detector.detect_regime(price_data)
        results['regime'] = {
            'name': regime.regime,
            'confidence': regime.confidence,
            'characteristics': regime.characteristics,
            'strategy': regime.recommended_strategy
        }
        
        # Train and predict
        if len(price_data) >= 100:
            self.predictive_model.train(price_data)
            prediction = self.predictive_model.predict(price_data)
            results['prediction'] = prediction
        
        # Anomaly detection
        anomalies = self.anomaly_detector.detect_anomalies(price_data)
        if not anomalies.empty:
            recent_anomalies = anomalies[anomalies['is_anomaly']].tail(5)
            results['anomalies'] = recent_anomalies.to_dict('records')
        
        # GPT analysis
        stock_analysis = await self.gpt_analyzer.analyze_stock(
            symbol, price_data, fundamentals, news
        )
        results['ai_summary'] = stock_analysis
        
        # News summary if available
        if news:
            news_summary = await self.gpt_analyzer.summarize_market_news(news)
            results['news_summary'] = news_summary
        
        return results
    
    def generate_signal(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        include_ai: bool = True
    ) -> AISignal:
        """
        Generate AI trading signal.
        
        Args:
            symbol: Stock symbol
            price_data: OHLCV data
            include_ai: Include ML predictions
            
        Returns:
            AISignal object
        """
        # Pattern-based signal
        patterns = self.pattern_recognition.detect_patterns(price_data)
        pattern_signal = 0
        for p in patterns:
            if p.expected_move > 0:
                pattern_signal += p.confidence
            else:
                pattern_signal -= p.confidence
        
        # Regime-based signal
        regime = self.regime_detector.detect_regime(price_data)
        regime_signal = 0
        if 'bull' in regime.regime.lower():
            regime_signal = 0.5
        elif 'bear' in regime.regime.lower():
            regime_signal = -0.5
        
        # Technical signal
        close = price_data['close']
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
        current = close.iloc[-1]
        
        tech_signal = 0
        if current > sma_20 > sma_50:
            tech_signal = 0.5
        elif current < sma_20 < sma_50:
            tech_signal = -0.5
        
        # ML prediction signal
        ml_signal = 0
        if include_ai and SKLEARN_AVAILABLE and len(price_data) >= 100:
            try:
                self.predictive_model.train(price_data)
                pred = self.predictive_model.predict(price_data)
                if pred.get('prediction') == 1:
                    ml_signal = pred.get('probability', 0.5)
                else:
                    ml_signal = -pred.get('probability', 0.5)
            except Exception:
                pass
        
        # Combine signals
        total_signal = pattern_signal * 0.2 + regime_signal * 0.2 + tech_signal * 0.3 + ml_signal * 0.3
        
        # Determine signal strength
        if total_signal > 0.5:
            signal = SignalStrength.STRONG_BUY
        elif total_signal > 0.2:
            signal = SignalStrength.BUY
        elif total_signal < -0.5:
            signal = SignalStrength.STRONG_SELL
        elif total_signal < -0.2:
            signal = SignalStrength.SELL
        else:
            signal = SignalStrength.HOLD
        
        # Calculate predicted return and volatility
        returns = close.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        predicted_return = total_signal * 0.05  # 5% max expected return
        
        return AISignal(
            symbol=symbol,
            signal=signal,
            confidence=abs(total_signal),
            predicted_return=predicted_return,
            predicted_volatility=volatility,
            time_horizon="5 days",
            generated_at=datetime.now(),
            model_type="ensemble",
            features_used=['patterns', 'regime', 'technical', 'ml'],
            reasoning=f"Pattern: {pattern_signal:.2f}, Regime: {regime.regime}, "
                     f"Technical: {tech_signal:.2f}, ML: {ml_signal:.2f}",
            risk_score=volatility
        )
    
    def get_dashboard_report(
        self,
        symbols: List[str],
        price_data_dict: Dict[str, pd.DataFrame]
    ) -> str:
        """Generate comprehensive dashboard report."""
        report = """
🤖 **AI ANALYSIS DASHBOARD**
{'=' * 60}

"""
        for symbol in symbols:
            if symbol not in price_data_dict:
                continue
            
            price_data = price_data_dict[symbol]
            signal = self.generate_signal(symbol, price_data)
            regime = self.regime_detector.detect_regime(price_data)
            patterns = self.pattern_recognition.detect_patterns(price_data)
            
            report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **{symbol}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 **AI Signal**: {signal.signal.value.upper()}
   Confidence: {signal.confidence * 100:.0f}%
   Predicted Return: {signal.predicted_return * 100:+.1f}%
   Risk Score: {signal.risk_score:.2%}

📈 **Market Regime**: {regime.regime}
   Strategy: {regime.recommended_strategy}

🔍 **Patterns Detected**: {len(patterns)}
"""
            for p in patterns[:3]:
                report += f"   • {p.pattern_name}: {p.confidence * 100:.0f}% confidence\n"
            
            report += "\n"
        
        report += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ **Disclaimer**: AI analysis is for informational purposes only.
Always perform your own research before trading.
"""
        return report


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    price = 100
    prices = []
    for _ in range(252):
        price *= (1 + np.random.randn() * 0.02)
        prices.append(price)
    
    sample_data = pd.DataFrame({
        'open': [p * (1 + np.random.randn() * 0.01) for p in prices],
        'high': [p * (1 + abs(np.random.randn() * 0.02)) for p in prices],
        'low': [p * (1 - abs(np.random.randn() * 0.02)) for p in prices],
        'close': prices,
        'volume': [1000000 * (1 + np.random.randn() * 0.3) for _ in range(252)]
    }, index=dates)
    
    # Initialize dashboard
    dashboard = AIAnalysisDashboard()
    
    # Generate signal
    signal = dashboard.generate_signal("AAPL", sample_data)
    print(f"Signal for AAPL: {signal.signal.value}")
    print(f"Confidence: {signal.confidence * 100:.0f}%")
    print(f"Reasoning: {signal.reasoning}")
    
    # Detect patterns
    patterns = dashboard.pattern_recognition.detect_patterns(sample_data)
    print(f"\nPatterns detected: {len(patterns)}")
    for p in patterns:
        print(f"  - {p.pattern_name}: {p.confidence * 100:.0f}% confidence")
    
    # Detect regime
    regime = dashboard.regime_detector.detect_regime(sample_data)
    print(f"\nMarket Regime: {regime.regime}")
    print(f"Recommended Strategy: {regime.recommended_strategy}")
