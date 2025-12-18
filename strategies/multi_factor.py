"""
Multi-Factor Trading Strategies
================================
Combines fundamental, technical, and alternative factors for
comprehensive stock selection and portfolio optimization.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Set, Union
from enum import Enum
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from scipy import optimize
    from scipy.stats import zscore
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class FactorCategory(Enum):
    """Categories of factors."""
    VALUE = "value"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    SIZE = "size"
    VOLATILITY = "volatility"
    GROWTH = "growth"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"


class RebalanceFrequency(Enum):
    """Portfolio rebalance frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class FactorDefinition:
    """Definition of a single factor."""
    name: str
    category: FactorCategory
    description: str
    higher_is_better: bool = True
    weight: float = 1.0
    min_history_days: int = 20


@dataclass
class StockData:
    """Data container for a single stock."""
    symbol: str
    price_data: pd.DataFrame  # OHLCV
    fundamental_data: Dict = field(default_factory=dict)
    alternative_data: Dict = field(default_factory=dict)


@dataclass
class FactorScore:
    """Score for a single factor."""
    factor_name: str
    raw_value: float
    zscore: float
    percentile: float
    weight: float = 1.0
    
    @property
    def weighted_score(self) -> float:
        return self.zscore * self.weight


@dataclass
class MultiFactorConfig:
    """Configuration for multi-factor strategy."""
    # Factor weights by category
    value_weight: float = 0.2
    momentum_weight: float = 0.2
    quality_weight: float = 0.2
    size_weight: float = 0.1
    volatility_weight: float = 0.1
    growth_weight: float = 0.1
    technical_weight: float = 0.1
    
    # Portfolio construction
    top_n_stocks: int = 20
    min_factor_exposure: float = 0.5
    max_single_stock_weight: float = 0.1
    min_single_stock_weight: float = 0.01
    
    # Risk management
    max_sector_weight: float = 0.3
    target_volatility: float = 0.15
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    
    # Optimization
    use_optimization: bool = True
    risk_aversion: float = 1.0


class FactorCalculator:
    """
    Calculates various factors for stocks.
    """
    
    def __init__(self):
        self.factor_definitions = self._define_factors()
    
    def _define_factors(self) -> Dict[str, FactorDefinition]:
        """Define all available factors."""
        return {
            # Value factors
            'pe_ratio': FactorDefinition(
                'pe_ratio', FactorCategory.VALUE,
                'Price to Earnings ratio (inverted)',
                higher_is_better=False
            ),
            'pb_ratio': FactorDefinition(
                'pb_ratio', FactorCategory.VALUE,
                'Price to Book ratio (inverted)',
                higher_is_better=False
            ),
            'ev_ebitda': FactorDefinition(
                'ev_ebitda', FactorCategory.VALUE,
                'Enterprise Value to EBITDA (inverted)',
                higher_is_better=False
            ),
            'dividend_yield': FactorDefinition(
                'dividend_yield', FactorCategory.VALUE,
                'Dividend Yield',
                higher_is_better=True
            ),
            'fcf_yield': FactorDefinition(
                'fcf_yield', FactorCategory.VALUE,
                'Free Cash Flow Yield',
                higher_is_better=True
            ),
            
            # Momentum factors
            'momentum_1m': FactorDefinition(
                'momentum_1m', FactorCategory.MOMENTUM,
                '1-month price momentum',
                higher_is_better=True, min_history_days=21
            ),
            'momentum_3m': FactorDefinition(
                'momentum_3m', FactorCategory.MOMENTUM,
                '3-month price momentum',
                higher_is_better=True, min_history_days=63
            ),
            'momentum_6m': FactorDefinition(
                'momentum_6m', FactorCategory.MOMENTUM,
                '6-month price momentum',
                higher_is_better=True, min_history_days=126
            ),
            'momentum_12m': FactorDefinition(
                'momentum_12m', FactorCategory.MOMENTUM,
                '12-month price momentum (skip last month)',
                higher_is_better=True, min_history_days=252
            ),
            
            # Quality factors
            'roe': FactorDefinition(
                'roe', FactorCategory.QUALITY,
                'Return on Equity',
                higher_is_better=True
            ),
            'roa': FactorDefinition(
                'roa', FactorCategory.QUALITY,
                'Return on Assets',
                higher_is_better=True
            ),
            'profit_margin': FactorDefinition(
                'profit_margin', FactorCategory.QUALITY,
                'Net Profit Margin',
                higher_is_better=True
            ),
            'debt_to_equity': FactorDefinition(
                'debt_to_equity', FactorCategory.QUALITY,
                'Debt to Equity ratio (inverted)',
                higher_is_better=False
            ),
            'current_ratio': FactorDefinition(
                'current_ratio', FactorCategory.QUALITY,
                'Current Ratio',
                higher_is_better=True
            ),
            
            # Size factors
            'market_cap': FactorDefinition(
                'market_cap', FactorCategory.SIZE,
                'Market Capitalization (inverted for small-cap tilt)',
                higher_is_better=False
            ),
            
            # Volatility factors
            'volatility_20d': FactorDefinition(
                'volatility_20d', FactorCategory.VOLATILITY,
                '20-day volatility (inverted for low-vol)',
                higher_is_better=False, min_history_days=20
            ),
            'beta': FactorDefinition(
                'beta', FactorCategory.VOLATILITY,
                'Market beta (inverted for low-beta)',
                higher_is_better=False, min_history_days=252
            ),
            
            # Growth factors
            'earnings_growth': FactorDefinition(
                'earnings_growth', FactorCategory.GROWTH,
                'Earnings growth rate',
                higher_is_better=True
            ),
            'revenue_growth': FactorDefinition(
                'revenue_growth', FactorCategory.GROWTH,
                'Revenue growth rate',
                higher_is_better=True
            ),
            
            # Technical factors
            'rsi': FactorDefinition(
                'rsi', FactorCategory.TECHNICAL,
                'RSI (mean-reversion, inverted)',
                higher_is_better=False, min_history_days=14
            ),
            'ma_cross': FactorDefinition(
                'ma_cross', FactorCategory.TECHNICAL,
                'Moving average crossover signal',
                higher_is_better=True, min_history_days=50
            ),
            'volume_trend': FactorDefinition(
                'volume_trend', FactorCategory.TECHNICAL,
                'Volume trend indicator',
                higher_is_better=True, min_history_days=20
            ),
        }
    
    def calculate_momentum_factors(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate momentum factors from price series."""
        factors = {}
        
        if len(prices) >= 21:
            factors['momentum_1m'] = (prices.iloc[-1] / prices.iloc[-21] - 1) * 100
        
        if len(prices) >= 63:
            factors['momentum_3m'] = (prices.iloc[-1] / prices.iloc[-63] - 1) * 100
        
        if len(prices) >= 126:
            factors['momentum_6m'] = (prices.iloc[-1] / prices.iloc[-126] - 1) * 100
        
        if len(prices) >= 252:
            # 12-month momentum, skipping last month (momentum crash avoidance)
            factors['momentum_12m'] = (prices.iloc[-21] / prices.iloc[-252] - 1) * 100
        
        return factors
    
    def calculate_volatility_factors(self, 
                                      prices: pd.Series,
                                      market_returns: pd.Series = None) -> Dict[str, float]:
        """Calculate volatility factors."""
        factors = {}
        returns = prices.pct_change().dropna()
        
        if len(returns) >= 20:
            factors['volatility_20d'] = returns.iloc[-20:].std() * np.sqrt(252) * 100
        
        # Beta calculation requires market returns
        if market_returns is not None and len(returns) >= 252:
            common_idx = returns.index.intersection(market_returns.index)
            if len(common_idx) >= 252:
                stock_ret = returns.loc[common_idx].iloc[-252:]
                mkt_ret = market_returns.loc[common_idx].iloc[-252:]
                
                cov = np.cov(stock_ret, mkt_ret)[0, 1]
                mkt_var = np.var(mkt_ret)
                factors['beta'] = cov / mkt_var if mkt_var > 0 else 1.0
        
        return factors
    
    def calculate_technical_factors(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical factors."""
        factors = {}
        prices = data['close'] if 'close' in data.columns else data['Close']
        
        # RSI
        if len(prices) >= 14:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            factors['rsi'] = rsi.iloc[-1]
        
        # Moving average crossover
        if len(prices) >= 50:
            sma_20 = prices.rolling(20).mean()
            sma_50 = prices.rolling(50).mean()
            factors['ma_cross'] = (sma_20.iloc[-1] / sma_50.iloc[-1] - 1) * 100
        
        # Volume trend
        if 'volume' in data.columns and len(data) >= 20:
            vol = data['volume']
            vol_sma = vol.rolling(20).mean()
            factors['volume_trend'] = (vol.iloc[-5:].mean() / vol_sma.iloc[-1] - 1) * 100
        
        return factors
    
    def calculate_all_factors(self,
                               stock: StockData,
                               market_returns: pd.Series = None) -> Dict[str, float]:
        """
        Calculate all available factors for a stock.
        """
        factors = {}
        
        # Price-based factors
        prices = stock.price_data['close'] if 'close' in stock.price_data.columns else stock.price_data['Close']
        
        # Momentum
        factors.update(self.calculate_momentum_factors(prices))
        
        # Volatility
        factors.update(self.calculate_volatility_factors(prices, market_returns))
        
        # Technical
        factors.update(self.calculate_technical_factors(stock.price_data))
        
        # Fundamental factors from fundamental_data
        for key in ['pe_ratio', 'pb_ratio', 'ev_ebitda', 'dividend_yield', 
                    'fcf_yield', 'roe', 'roa', 'profit_margin', 
                    'debt_to_equity', 'current_ratio', 'market_cap',
                    'earnings_growth', 'revenue_growth']:
            if key in stock.fundamental_data:
                factors[key] = stock.fundamental_data[key]
        
        return factors


class FactorRanker:
    """
    Ranks stocks based on factor scores.
    """
    
    def __init__(self, calculator: FactorCalculator = None):
        self.calculator = calculator or FactorCalculator()
    
    def calculate_factor_scores(self,
                                 stocks: List[StockData],
                                 market_returns: pd.Series = None) -> pd.DataFrame:
        """
        Calculate factor scores for all stocks.
        
        Returns DataFrame with stocks as rows and factors as columns.
        """
        all_factors = {}
        
        for stock in stocks:
            factors = self.calculator.calculate_all_factors(stock, market_returns)
            all_factors[stock.symbol] = factors
        
        df = pd.DataFrame(all_factors).T
        return df
    
    def rank_by_factor(self,
                        factor_df: pd.DataFrame,
                        factor_name: str,
                        ascending: bool = False) -> pd.Series:
        """
        Rank stocks by a single factor.
        """
        if factor_name not in factor_df.columns:
            return pd.Series(index=factor_df.index, dtype=float)
        
        values = factor_df[factor_name].dropna()
        
        # Rank (1 = best)
        ranks = values.rank(ascending=ascending)
        
        # Normalize to 0-1
        normalized = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else pd.Series(0.5, index=ranks.index)
        
        return normalized
    
    def calculate_z_scores(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate z-scores for all factors.
        """
        z_scores = pd.DataFrame(index=factor_df.index)
        
        for col in factor_df.columns:
            values = factor_df[col].dropna()
            if len(values) > 1:
                mean = values.mean()
                std = values.std()
                if std > 0:
                    z_scores[col] = (factor_df[col] - mean) / std
                else:
                    z_scores[col] = 0
            else:
                z_scores[col] = 0
        
        return z_scores
    
    def calculate_composite_score(self,
                                   factor_df: pd.DataFrame,
                                   config: MultiFactorConfig) -> pd.Series:
        """
        Calculate composite score combining all factors.
        """
        z_scores = self.calculate_z_scores(factor_df)
        
        # Define weights by category
        category_weights = {
            FactorCategory.VALUE: config.value_weight,
            FactorCategory.MOMENTUM: config.momentum_weight,
            FactorCategory.QUALITY: config.quality_weight,
            FactorCategory.SIZE: config.size_weight,
            FactorCategory.VOLATILITY: config.volatility_weight,
            FactorCategory.GROWTH: config.growth_weight,
            FactorCategory.TECHNICAL: config.technical_weight,
        }
        
        # Get factor definitions
        factor_defs = self.calculator.factor_definitions
        
        composite = pd.Series(0.0, index=z_scores.index)
        total_weight = 0
        
        for factor_name in z_scores.columns:
            if factor_name in factor_defs:
                factor_def = factor_defs[factor_name]
                category_weight = category_weights.get(factor_def.category, 0.1)
                
                # Adjust sign based on higher_is_better
                sign = 1 if factor_def.higher_is_better else -1
                
                # Handle NaN
                factor_values = z_scores[factor_name].fillna(0)
                
                composite += sign * factor_values * category_weight * factor_def.weight
                total_weight += category_weight * factor_def.weight
        
        if total_weight > 0:
            composite /= total_weight
        
        return composite
    
    def get_top_stocks(self,
                        stocks: List[StockData],
                        config: MultiFactorConfig,
                        market_returns: pd.Series = None) -> List[Tuple[str, float]]:
        """
        Get top-ranked stocks based on composite score.
        
        Returns list of (symbol, score) tuples.
        """
        factor_df = self.calculate_factor_scores(stocks, market_returns)
        composite = self.calculate_composite_score(factor_df, config)
        
        # Sort and get top N
        sorted_scores = composite.sort_values(ascending=False)
        top_n = min(config.top_n_stocks, len(sorted_scores))
        
        return [(symbol, score) for symbol, score in sorted_scores.head(top_n).items()]


class PortfolioOptimizer:
    """
    Portfolio optimization for multi-factor strategy.
    """
    
    def __init__(self, config: MultiFactorConfig = None):
        self.config = config or MultiFactorConfig()
    
    def calculate_covariance_matrix(self,
                                     returns: pd.DataFrame,
                                     method: str = 'sample') -> pd.DataFrame:
        """
        Calculate covariance matrix of returns.
        
        Methods:
            - sample: Sample covariance
            - shrinkage: Ledoit-Wolf shrinkage
            - exponential: Exponentially weighted
        """
        if method == 'sample':
            return returns.cov()
        elif method == 'exponential':
            # Exponentially weighted with 60-day half-life
            return returns.ewm(halflife=60).cov().iloc[-len(returns.columns):]
        elif method == 'shrinkage':
            # Simple shrinkage toward diagonal
            sample_cov = returns.cov()
            diag = np.diag(np.diag(sample_cov))
            shrinkage = 0.2
            return pd.DataFrame(
                (1 - shrinkage) * sample_cov.values + shrinkage * diag,
                index=sample_cov.index,
                columns=sample_cov.columns
            )
        return returns.cov()
    
    def optimize_mean_variance(self,
                                expected_returns: pd.Series,
                                cov_matrix: pd.DataFrame,
                                target_return: float = None) -> pd.Series:
        """
        Mean-variance optimization.
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            target_return: Target portfolio return (optional)
        
        Returns:
            Optimal weights as Series
        """
        if not SCIPY_AVAILABLE:
            # Equal weight fallback
            n = len(expected_returns)
            return pd.Series(1/n, index=expected_returns.index)
        
        n = len(expected_returns)
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
        
        def portfolio_return(weights):
            return np.dot(weights, expected_returns.values)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: portfolio_return(w) - target_return
            })
        
        # Bounds
        bounds = [(self.config.min_single_stock_weight, 
                  self.config.max_single_stock_weight)] * n
        
        # Initial guess
        init_weights = np.ones(n) / n
        
        # Optimize
        if target_return is not None:
            # Minimize volatility for target return
            result = optimize.minimize(
                portfolio_volatility,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        else:
            # Maximize Sharpe-like ratio
            def neg_sharpe(weights):
                ret = portfolio_return(weights)
                vol = portfolio_volatility(weights)
                return -(ret / vol) if vol > 0 else 0
            
            result = optimize.minimize(
                neg_sharpe,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        
        if result.success:
            weights = result.x
            # Normalize to sum to 1
            weights = weights / np.sum(weights)
            return pd.Series(weights, index=expected_returns.index)
        
        # Fallback to equal weight
        return pd.Series(1/n, index=expected_returns.index)
    
    def optimize_risk_parity(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Risk parity optimization (equal risk contribution).
        """
        if not SCIPY_AVAILABLE:
            n = len(cov_matrix)
            return pd.Series(1/n, index=cov_matrix.index)
        
        n = len(cov_matrix)
        
        def risk_contribution(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
            marginal_contrib = np.dot(cov_matrix.values, weights) / port_vol
            risk_contrib = weights * marginal_contrib
            return risk_contrib
        
        def objective(weights):
            contrib = risk_contribution(weights)
            target_contrib = 1 / n
            return np.sum((contrib - target_contrib) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.01, 0.5)] * n
        init_weights = np.ones(n) / n
        
        result = optimize.minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return pd.Series(result.x / np.sum(result.x), index=cov_matrix.index)
        
        return pd.Series(1/n, index=cov_matrix.index)
    
    def optimize_max_diversification(self,
                                      cov_matrix: pd.DataFrame,
                                      volatilities: pd.Series) -> pd.Series:
        """
        Maximum diversification optimization.
        """
        if not SCIPY_AVAILABLE:
            n = len(cov_matrix)
            return pd.Series(1/n, index=cov_matrix.index)
        
        n = len(cov_matrix)
        
        def diversification_ratio(weights):
            weighted_vol = np.dot(weights, volatilities.values)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
            return -weighted_vol / port_vol if port_vol > 0 else 0
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.01, 0.3)] * n
        init_weights = np.ones(n) / n
        
        result = optimize.minimize(
            diversification_ratio,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return pd.Series(result.x / np.sum(result.x), index=cov_matrix.index)
        
        return pd.Series(1/n, index=cov_matrix.index)
    
    def apply_sector_constraints(self,
                                  weights: pd.Series,
                                  sectors: Dict[str, str]) -> pd.Series:
        """
        Apply sector weight constraints.
        """
        adjusted = weights.copy()
        
        # Group by sector
        sector_weights = {}
        for symbol in weights.index:
            sector = sectors.get(symbol, 'Unknown')
            if sector not in sector_weights:
                sector_weights[sector] = []
            sector_weights[sector].append(symbol)
        
        # Check and adjust sector weights
        for sector, symbols in sector_weights.items():
            total_weight = adjusted[symbols].sum()
            if total_weight > self.config.max_sector_weight:
                # Scale down proportionally
                scale = self.config.max_sector_weight / total_weight
                adjusted[symbols] *= scale
        
        # Renormalize
        adjusted /= adjusted.sum()
        
        return adjusted


class MultiFactorStrategy:
    """
    Complete multi-factor stock selection and portfolio strategy.
    """
    
    def __init__(self, config: MultiFactorConfig = None):
        self.config = config or MultiFactorConfig()
        self.calculator = FactorCalculator()
        self.ranker = FactorRanker(self.calculator)
        self.optimizer = PortfolioOptimizer(self.config)
        
        self.current_portfolio: Dict[str, float] = {}
        self.portfolio_history: List[Dict] = []
        self.factor_exposures: Dict[str, float] = {}
    
    def select_stocks(self,
                       universe: List[StockData],
                       market_returns: pd.Series = None) -> List[Tuple[str, float]]:
        """
        Select stocks from universe based on factor scores.
        """
        return self.ranker.get_top_stocks(universe, self.config, market_returns)
    
    def construct_portfolio(self,
                             selected_stocks: List[Tuple[str, float]],
                             price_data: Dict[str, pd.DataFrame],
                             optimization_method: str = 'equal') -> Dict[str, float]:
        """
        Construct portfolio with optimal weights.
        
        Methods:
            - equal: Equal weight
            - score: Weight by factor score
            - mv: Mean-variance optimization
            - rp: Risk parity
            - md: Maximum diversification
        """
        symbols = [s[0] for s in selected_stocks]
        scores = {s[0]: s[1] for s in selected_stocks}
        
        if not symbols:
            return {}
        
        if optimization_method == 'equal':
            weight = 1 / len(symbols)
            weights = {s: weight for s in symbols}
        
        elif optimization_method == 'score':
            # Weight proportional to score (shifted to positive)
            min_score = min(scores.values())
            shifted = {s: v - min_score + 0.1 for s, v in scores.items()}
            total = sum(shifted.values())
            weights = {s: v / total for s, v in shifted.items()}
        
        elif optimization_method in ['mv', 'rp', 'md']:
            # Need returns data
            returns_df = pd.DataFrame()
            for symbol in symbols:
                if symbol in price_data:
                    prices = price_data[symbol]['close'] if 'close' in price_data[symbol].columns else price_data[symbol]['Close']
                    returns_df[symbol] = prices.pct_change()
            
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 20:
                # Not enough data, use equal weight
                weight = 1 / len(symbols)
                weights = {s: weight for s in symbols}
            else:
                cov_matrix = self.optimizer.calculate_covariance_matrix(returns_df)
                
                if optimization_method == 'mv':
                    expected_returns = pd.Series(scores)
                    expected_returns = expected_returns.loc[returns_df.columns]
                    opt_weights = self.optimizer.optimize_mean_variance(expected_returns, cov_matrix)
                elif optimization_method == 'rp':
                    opt_weights = self.optimizer.optimize_risk_parity(cov_matrix)
                else:  # md
                    vols = returns_df.std() * np.sqrt(252)
                    opt_weights = self.optimizer.optimize_max_diversification(cov_matrix, vols)
                
                weights = opt_weights.to_dict()
        
        else:
            weight = 1 / len(symbols)
            weights = {s: weight for s in symbols}
        
        # Apply constraints
        for symbol in weights:
            weights[symbol] = max(weights[symbol], self.config.min_single_stock_weight)
            weights[symbol] = min(weights[symbol], self.config.max_single_stock_weight)
        
        # Renormalize
        total = sum(weights.values())
        weights = {s: w / total for s, w in weights.items()}
        
        self.current_portfolio = weights
        return weights
    
    def calculate_factor_exposures(self,
                                    stocks: List[StockData],
                                    weights: Dict[str, float],
                                    market_returns: pd.Series = None) -> Dict[str, float]:
        """
        Calculate portfolio's factor exposures.
        """
        factor_df = self.ranker.calculate_factor_scores(stocks, market_returns)
        z_scores = self.ranker.calculate_z_scores(factor_df)
        
        exposures = {}
        
        for factor in z_scores.columns:
            weighted_exposure = 0
            for symbol, weight in weights.items():
                if symbol in z_scores.index:
                    weighted_exposure += z_scores.loc[symbol, factor] * weight
            exposures[factor] = weighted_exposure
        
        self.factor_exposures = exposures
        return exposures
    
    def backtest(self,
                  universe: List[StockData],
                  start_date: datetime,
                  end_date: datetime,
                  initial_capital: float = 100000,
                  transaction_cost: float = 0.001) -> Dict:
        """
        Backtest the multi-factor strategy.
        """
        # Get all price data
        price_data = {}
        for stock in universe:
            prices = stock.price_data['close'] if 'close' in stock.price_data.columns else stock.price_data['Close']
            prices = prices.loc[start_date:end_date]
            if len(prices) > 0:
                price_data[stock.symbol] = prices
        
        # Get common dates
        all_dates = None
        for symbol, prices in price_data.items():
            if all_dates is None:
                all_dates = set(prices.index)
            else:
                all_dates = all_dates.intersection(set(prices.index))
        
        if not all_dates:
            return {'error': 'No common dates found'}
        
        dates = sorted(list(all_dates))
        
        # Determine rebalance dates
        if self.config.rebalance_frequency == RebalanceFrequency.DAILY:
            rebalance_dates = dates
        elif self.config.rebalance_frequency == RebalanceFrequency.WEEKLY:
            rebalance_dates = [d for d in dates if d.weekday() == 0]  # Mondays
        elif self.config.rebalance_frequency == RebalanceFrequency.MONTHLY:
            rebalance_dates = []
            current_month = None
            for d in dates:
                if d.month != current_month:
                    rebalance_dates.append(d)
                    current_month = d.month
        else:  # Quarterly
            rebalance_dates = []
            current_quarter = None
            for d in dates:
                quarter = (d.month - 1) // 3
                if quarter != current_quarter:
                    rebalance_dates.append(d)
                    current_quarter = quarter
        
        # Run backtest
        portfolio_value = [initial_capital]
        holdings: Dict[str, float] = {}  # symbol -> shares
        cash = initial_capital
        
        for i, date in enumerate(dates):
            # Check for rebalance
            if date in rebalance_dates or not holdings:
                # Select stocks and construct portfolio
                # (Using simplified approach - in practice would use data up to date)
                selected = self.select_stocks(universe)
                target_weights = self.construct_portfolio(
                    selected,
                    {s: pd.DataFrame({'close': price_data[s]}) for s in price_data},
                    'score'
                )
                
                # Calculate current portfolio value
                current_value = cash
                for symbol, shares in holdings.items():
                    if symbol in price_data:
                        current_value += shares * price_data[symbol].loc[date]
                
                # Rebalance
                new_holdings = {}
                total_cost = 0
                
                for symbol, weight in target_weights.items():
                    if symbol in price_data:
                        target_value = current_value * weight
                        price = price_data[symbol].loc[date]
                        shares = target_value / price
                        new_holdings[symbol] = shares
                        
                        # Transaction cost
                        old_shares = holdings.get(symbol, 0)
                        trade_value = abs(shares - old_shares) * price
                        total_cost += trade_value * transaction_cost
                
                cash = current_value - sum(s * price_data[sym].loc[date] 
                                           for sym, s in new_holdings.items()) - total_cost
                holdings = new_holdings
            
            # Calculate end of day value
            port_value = cash
            for symbol, shares in holdings.items():
                if symbol in price_data and date in price_data[symbol].index:
                    port_value += shares * price_data[symbol].loc[date]
            
            portfolio_value.append(port_value)
        
        # Calculate metrics
        returns = np.diff(portfolio_value) / portfolio_value[:-1]
        
        # Benchmark (equal weight)
        benchmark_returns = []
        for i in range(1, len(dates)):
            daily_returns = []
            for symbol in price_data:
                if dates[i-1] in price_data[symbol].index and dates[i] in price_data[symbol].index:
                    ret = (price_data[symbol].loc[dates[i]] / price_data[symbol].loc[dates[i-1]] - 1)
                    daily_returns.append(ret)
            benchmark_returns.append(np.mean(daily_returns) if daily_returns else 0)
        
        return {
            'total_return': (portfolio_value[-1] / portfolio_value[0] - 1),
            'annualized_return': ((portfolio_value[-1] / portfolio_value[0]) ** (252 / len(dates)) - 1),
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'max_drawdown': np.min(np.array(portfolio_value) / np.maximum.accumulate(portfolio_value) - 1),
            'volatility': np.std(returns) * np.sqrt(252),
            'benchmark_return': (1 + np.array(benchmark_returns)).prod() - 1,
            'alpha': np.mean(returns) - np.mean(benchmark_returns),
            'portfolio_values': portfolio_value,
            'dates': dates,
            'final_holdings': holdings
        }
    
    def get_factor_attribution(self) -> Dict[str, Dict]:
        """
        Get factor attribution for current portfolio.
        """
        factor_defs = self.calculator.factor_definitions
        
        attribution = {}
        for factor, exposure in self.factor_exposures.items():
            if factor in factor_defs:
                factor_def = factor_defs[factor]
                attribution[factor] = {
                    'exposure': exposure,
                    'category': factor_def.category.value,
                    'direction': 'positive' if (exposure > 0) == factor_def.higher_is_better else 'negative',
                    'description': factor_def.description
                }
        
        return attribution
    
    def generate_report(self, stocks: List[StockData]) -> Dict:
        """
        Generate comprehensive strategy report.
        """
        # Get rankings
        factor_df = self.ranker.calculate_factor_scores(stocks)
        composite = self.ranker.calculate_composite_score(factor_df, self.config)
        
        # Top and bottom stocks
        sorted_composite = composite.sort_values(ascending=False)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'universe_size': len(stocks),
            'top_stocks': sorted_composite.head(10).to_dict(),
            'bottom_stocks': sorted_composite.tail(10).to_dict(),
            'factor_coverage': {
                col: factor_df[col].notna().sum() / len(factor_df)
                for col in factor_df.columns
            },
            'current_portfolio': self.current_portfolio,
            'factor_exposures': self.factor_exposures,
            'config': {
                'top_n': self.config.top_n_stocks,
                'rebalance': self.config.rebalance_frequency.value,
                'weights': {
                    'value': self.config.value_weight,
                    'momentum': self.config.momentum_weight,
                    'quality': self.config.quality_weight,
                    'growth': self.config.growth_weight
                }
            }
        }


class MultiFactorSuite:
    """
    Unified interface for multi-factor strategies.
    """
    
    def __init__(self, config: MultiFactorConfig = None):
        self.config = config or MultiFactorConfig()
        self.strategy = MultiFactorStrategy(self.config)
    
    def screen_universe(self,
                         stocks: List[StockData],
                         min_market_cap: float = None,
                         min_liquidity: float = None) -> List[StockData]:
        """
        Pre-screen universe based on basic criteria.
        """
        filtered = []
        
        for stock in stocks:
            # Market cap filter
            if min_market_cap:
                mc = stock.fundamental_data.get('market_cap', 0)
                if mc < min_market_cap:
                    continue
            
            # Liquidity filter
            if min_liquidity and 'volume' in stock.price_data.columns:
                avg_volume = stock.price_data['volume'].iloc[-20:].mean()
                avg_price = stock.price_data['close'].iloc[-20:].mean()
                liquidity = avg_volume * avg_price
                if liquidity < min_liquidity:
                    continue
            
            filtered.append(stock)
        
        return filtered
    
    def run_strategy(self,
                      universe: List[StockData],
                      optimization: str = 'score') -> Dict:
        """
        Run full strategy: screen, select, and construct portfolio.
        """
        # Screen
        screened = self.screen_universe(universe)
        
        # Select
        selected = self.strategy.select_stocks(screened)
        
        # Construct
        price_data = {s.symbol: s.price_data for s in screened}
        weights = self.strategy.construct_portfolio(selected, price_data, optimization)
        
        # Factor exposures
        exposures = self.strategy.calculate_factor_exposures(screened, weights)
        
        return {
            'universe_size': len(universe),
            'screened_size': len(screened),
            'selected_stocks': selected,
            'portfolio_weights': weights,
            'factor_exposures': exposures
        }
    
    def compare_strategies(self,
                            universe: List[StockData],
                            start_date: datetime,
                            end_date: datetime) -> Dict:
        """
        Compare different portfolio construction methods.
        """
        methods = ['equal', 'score', 'mv', 'rp']
        results = {}
        
        for method in methods:
            # Create fresh strategy
            strategy = MultiFactorStrategy(self.config)
            
            # Would need to implement full backtest for each method
            # This is a simplified comparison
            selected = strategy.select_stocks(universe)
            price_data = {s.symbol: s.price_data for s in universe}
            weights = strategy.construct_portfolio(selected, price_data, method)
            
            results[method] = {
                'num_stocks': len(weights),
                'max_weight': max(weights.values()) if weights else 0,
                'min_weight': min(weights.values()) if weights else 0,
                'concentration': sum(w**2 for w in weights.values()) if weights else 0
            }
        
        return results


# Factory function
def create_multi_factor_strategy(
    config: MultiFactorConfig = None,
    preset: str = None
) -> MultiFactorStrategy:
    """
    Factory function to create multi-factor strategy.
    
    Presets:
        - value: Heavy value tilt
        - momentum: Heavy momentum tilt
        - quality: Heavy quality tilt
        - balanced: Equal weight all factors
    """
    if config is None:
        config = MultiFactorConfig()
    
    if preset == 'value':
        config.value_weight = 0.4
        config.momentum_weight = 0.1
        config.quality_weight = 0.2
    elif preset == 'momentum':
        config.value_weight = 0.1
        config.momentum_weight = 0.4
        config.quality_weight = 0.2
    elif preset == 'quality':
        config.value_weight = 0.2
        config.momentum_weight = 0.1
        config.quality_weight = 0.4
    elif preset == 'balanced':
        config.value_weight = 0.2
        config.momentum_weight = 0.2
        config.quality_weight = 0.2
        config.size_weight = 0.1
        config.volatility_weight = 0.1
        config.growth_weight = 0.1
        config.technical_weight = 0.1
    
    return MultiFactorStrategy(config)


if __name__ == "__main__":
    print("=== Multi-Factor Strategy Demo ===\n")
    
    # Create sample stock data
    np.random.seed(42)
    
    def create_sample_stock(symbol: str, trend: float = 0.0001) -> StockData:
        dates = pd.date_range('2022-01-01', periods=500, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.02 + trend))
        
        price_data = pd.DataFrame({
            'open': prices * (1 + np.random.randn(500) * 0.01),
            'high': prices * (1 + np.abs(np.random.randn(500)) * 0.02),
            'low': prices * (1 - np.abs(np.random.randn(500)) * 0.02),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 500)
        }, index=dates)
        
        fundamental_data = {
            'pe_ratio': np.random.uniform(10, 30),
            'pb_ratio': np.random.uniform(1, 5),
            'roe': np.random.uniform(0.05, 0.25),
            'profit_margin': np.random.uniform(0.05, 0.20),
            'market_cap': np.random.uniform(1e9, 100e9),
            'debt_to_equity': np.random.uniform(0.1, 1.5),
            'earnings_growth': np.random.uniform(-0.1, 0.3),
            'revenue_growth': np.random.uniform(-0.05, 0.25),
            'dividend_yield': np.random.uniform(0, 0.04)
        }
        
        return StockData(
            symbol=symbol,
            price_data=price_data,
            fundamental_data=fundamental_data
        )
    
    # Create universe
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ',
               'WMT', 'PG', 'XOM', 'CVX', 'BAC', 'DIS', 'NFLX', 'INTC', 'AMD', 'CRM']
    
    universe = [create_sample_stock(s, trend=np.random.uniform(-0.0001, 0.0002)) for s in symbols]
    
    # Create strategy
    config = MultiFactorConfig(
        top_n_stocks=10,
        value_weight=0.25,
        momentum_weight=0.25,
        quality_weight=0.25,
        growth_weight=0.25
    )
    
    strategy = MultiFactorStrategy(config)
    
    # Select stocks
    print("--- Stock Selection ---")
    selected = strategy.select_stocks(universe)
    print(f"Top {len(selected)} stocks by composite score:")
    for symbol, score in selected[:5]:
        print(f"  {symbol}: {score:.3f}")
    
    # Construct portfolio
    print("\n--- Portfolio Construction ---")
    price_data = {s.symbol: s.price_data for s in universe}
    weights = strategy.construct_portfolio(selected, price_data, 'score')
    print("Portfolio weights:")
    for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {symbol}: {weight:.1%}")
    
    # Factor exposures
    print("\n--- Factor Exposures ---")
    exposures = strategy.calculate_factor_exposures(universe, weights)
    for factor, exposure in sorted(exposures.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {factor}: {exposure:+.2f}")
    
    # Factor attribution
    print("\n--- Factor Attribution ---")
    attribution = strategy.get_factor_attribution()
    for factor, info in list(attribution.items())[:3]:
        print(f"  {factor}: {info['exposure']:+.2f} ({info['category']}, {info['direction']})")
