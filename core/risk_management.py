"""
Risk & Portfolio Management Module
===================================
Professional-grade portfolio construction and risk management.

Features:
- Covariance shrinkage (Ledoit-Wolf, Oracle Approximating Shrinkage)
- Black-Litterman model with view confidence
- Hierarchical Risk Parity (HRP)
- Sector neutralization
- 2/20 fee model calculation
- VaR, CVaR, and stress testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import warnings

warnings.filterwarnings('ignore')

try:
    from scipy import stats
    from scipy.optimize import minimize
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    HRP = "hrp"
    BLACK_LITTERMAN = "black_litterman"
    MAX_DIVERSIFICATION = "max_diversification"
    EQUAL_WEIGHT = "equal_weight"


@dataclass
class PortfolioConstraints:
    """Portfolio constraints for optimization."""
    min_weight: float = 0.0
    max_weight: float = 1.0
    long_only: bool = True
    sector_neutral: bool = False
    max_sector_weight: float = 0.30
    max_position_count: int = 50
    min_position_weight: float = 0.01
    turnover_limit: float = 1.0  # Max portfolio turnover


@dataclass
class FeeModel:
    """2/20 hedge fund fee model."""
    management_fee: float = 0.02  # 2% annual
    performance_fee: float = 0.20  # 20% of profits
    hurdle_rate: float = 0.0  # Hurdle rate for performance fee
    high_water_mark: bool = True
    hwm_value: float = 100.0  # Starting NAV
    
    def calculate_fees(self, nav_start: float, nav_end: float, 
                      days: int = 365) -> Dict[str, float]:
        """Calculate management and performance fees."""
        # Management fee (prorated)
        mgmt_fee = nav_start * self.management_fee * (days / 365)
        
        # Performance fee
        gross_return = (nav_end - nav_start) / nav_start
        excess_return = max(0, gross_return - self.hurdle_rate * (days / 365))
        
        if self.high_water_mark:
            # Only charge on gains above HWM
            if nav_end > self.hwm_value:
                profit = nav_end - max(nav_start, self.hwm_value)
                perf_fee = profit * self.performance_fee
                self.hwm_value = nav_end  # Update HWM
            else:
                perf_fee = 0.0
        else:
            perf_fee = nav_start * excess_return * self.performance_fee
        
        total_fees = mgmt_fee + perf_fee
        nav_after_fees = nav_end - total_fees
        
        return {
            'management_fee': mgmt_fee,
            'performance_fee': perf_fee,
            'total_fees': total_fees,
            'nav_before_fees': nav_end,
            'nav_after_fees': nav_after_fees,
            'gross_return': gross_return * 100,
            'net_return': ((nav_after_fees - nav_start) / nav_start) * 100
        }


class CovarianceEstimator:
    """Advanced covariance matrix estimation with shrinkage."""
    
    @staticmethod
    def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
        """Simple sample covariance matrix."""
        return returns.cov()
    
    @staticmethod
    def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> pd.DataFrame:
        """Ledoit-Wolf shrinkage estimator."""
        n, p = returns.shape
        sample_cov = returns.cov().values
        
        # Shrinkage target: diagonal matrix with average variance
        mu = np.trace(sample_cov) / p
        delta = sample_cov - mu * np.eye(p)
        
        # Compute optimal shrinkage intensity
        X = returns.values - returns.mean().values
        
        # Sum of squared off-diagonal elements
        sum_sq = (delta ** 2).sum()
        
        # Compute shrinkage intensity
        gamma = 0.0
        for i in range(n):
            gamma += (np.outer(X[i], X[i]) - sample_cov).flatten() @ (np.outer(X[i], X[i]) - sample_cov).flatten()
        gamma /= (n ** 2)
        
        # Optimal shrinkage
        kappa = (gamma - sum_sq / n) / ((n - 1) * sum_sq / n + 1e-10)
        shrinkage = max(0, min(1, kappa))
        
        # Shrunk covariance
        shrunk_cov = shrinkage * mu * np.eye(p) + (1 - shrinkage) * sample_cov
        
        return pd.DataFrame(shrunk_cov, index=returns.columns, columns=returns.columns)
    
    @staticmethod
    def exponential_weighted(returns: pd.DataFrame, halflife: int = 60) -> pd.DataFrame:
        """Exponentially weighted covariance matrix."""
        return returns.ewm(halflife=halflife).cov().iloc[-len(returns.columns):]


class RiskMetrics:
    """Calculate various risk metrics."""
    
    @staticmethod
    def volatility(returns: pd.Series, annualize: bool = True) -> float:
        """Calculate volatility."""
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(252)
        return vol
    
    @staticmethod
    def var(returns: pd.Series, confidence: float = 0.95, method: str = 'historical') -> float:
        """Calculate Value at Risk."""
        if method == 'historical':
            return -np.percentile(returns, (1 - confidence) * 100)
        elif method == 'parametric':
            mu = returns.mean()
            sigma = returns.std()
            z = stats.norm.ppf(1 - confidence) if SCIPY_AVAILABLE else 1.645
            return -(mu + sigma * z)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    @staticmethod
    def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        var = RiskMetrics.var(returns, confidence)
        return -returns[returns <= -var].mean()
    
    @staticmethod
    def max_drawdown(returns: pd.Series) -> Tuple[float, int, int]:
        """Calculate maximum drawdown and its duration."""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        
        max_dd = drawdowns.min()
        end_idx = drawdowns.idxmin()
        
        # Find peak
        peak_idx = cum_returns.loc[:end_idx].idxmax()
        
        # Duration in days
        duration = (end_idx - peak_idx).days if hasattr(end_idx, 'days') else 0
        
        return max_dd, peak_idx, end_idx
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free / 252
        return np.sqrt(252) * excess_returns.mean() / (returns.std() + 1e-10)
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        excess_returns = returns - risk_free / 252
        downside = returns[returns < 0].std()
        return np.sqrt(252) * excess_returns.mean() / (downside + 1e-10)
    
    @staticmethod
    def calmar_ratio(returns: pd.Series) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        annual_return = returns.mean() * 252
        max_dd, _, _ = RiskMetrics.max_drawdown(returns)
        return annual_return / (abs(max_dd) + 1e-10)
    
    @staticmethod
    def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio."""
        excess = returns - benchmark_returns
        tracking_error = excess.std() * np.sqrt(252)
        return (excess.mean() * 252) / (tracking_error + 1e-10)


class PortfolioOptimizer:
    """Portfolio optimization with multiple methods."""
    
    def __init__(self, returns: pd.DataFrame, constraints: PortfolioConstraints = None):
        self.returns = returns
        self.constraints = constraints or PortfolioConstraints()
        self.n_assets = len(returns.columns)
        self.assets = returns.columns.tolist()
        
        # Compute covariance
        self.cov_matrix = CovarianceEstimator.ledoit_wolf_shrinkage(returns)
        self.expected_returns = returns.mean() * 252
    
    def optimize(self, method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
                **kwargs) -> Dict[str, float]:
        """Run optimization using specified method."""
        if method == OptimizationMethod.EQUAL_WEIGHT:
            return self._equal_weight()
        elif method == OptimizationMethod.MIN_VARIANCE:
            return self._minimum_variance()
        elif method == OptimizationMethod.MAX_SHARPE:
            return self._maximum_sharpe()
        elif method == OptimizationMethod.RISK_PARITY:
            return self._risk_parity()
        elif method == OptimizationMethod.HRP:
            return self._hrp()
        elif method == OptimizationMethod.BLACK_LITTERMAN:
            views = kwargs.get('views', {})
            confidences = kwargs.get('confidences', {})
            return self._black_litterman(views, confidences)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _equal_weight(self) -> Dict[str, float]:
        """Equal weight portfolio."""
        weight = 1.0 / self.n_assets
        return {asset: weight for asset in self.assets}
    
    def _minimum_variance(self) -> Dict[str, float]:
        """Minimum variance portfolio."""
        if not SCIPY_AVAILABLE:
            return self._equal_weight()
        
        cov = self.cov_matrix.values
        
        def objective(w):
            return w @ cov @ w
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(self.constraints.min_weight, self.constraints.max_weight)] * self.n_assets
        
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = result.x
        return {asset: max(0, w) for asset, w in zip(self.assets, weights)}
    
    def _maximum_sharpe(self, risk_free: float = 0.0) -> Dict[str, float]:
        """Maximum Sharpe ratio portfolio."""
        if not SCIPY_AVAILABLE:
            return self._equal_weight()
        
        mu = self.expected_returns.values
        cov = self.cov_matrix.values
        
        def neg_sharpe(w):
            ret = w @ mu
            vol = np.sqrt(w @ cov @ w)
            return -(ret - risk_free) / (vol + 1e-10)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(self.constraints.min_weight, self.constraints.max_weight)] * self.n_assets
        
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = result.x
        return {asset: max(0, w) for asset, w in zip(self.assets, weights)}
    
    def _risk_parity(self) -> Dict[str, float]:
        """Risk parity (equal risk contribution)."""
        if not SCIPY_AVAILABLE:
            return self._equal_weight()
        
        cov = self.cov_matrix.values
        
        def risk_contrib_error(w):
            w = np.array(w)
            vol = np.sqrt(w @ cov @ w)
            mrc = cov @ w / vol  # Marginal risk contribution
            rc = w * mrc  # Risk contribution
            target_rc = vol / self.n_assets
            return np.sum((rc - target_rc) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.01, self.constraints.max_weight)] * self.n_assets
        
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(risk_contrib_error, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = result.x
        return {asset: max(0, w) for asset, w in zip(self.assets, weights)}
    
    def _hrp(self) -> Dict[str, float]:
        """Hierarchical Risk Parity."""
        if not SCIPY_AVAILABLE:
            return self._equal_weight()
        
        # Correlation matrix
        corr = self.returns.corr()
        
        # Distance matrix
        dist = np.sqrt((1 - corr) / 2)
        
        # Hierarchical clustering
        link = linkage(squareform(dist.values), method='single')
        sort_ix = leaves_list(link)
        
        # Reorder covariance matrix
        sorted_assets = [self.assets[i] for i in sort_ix]
        
        # Recursive bisection
        def recursive_bisection(cov, assets):
            if len(assets) == 1:
                return {assets[0]: 1.0}
            
            mid = len(assets) // 2
            left_assets = assets[:mid]
            right_assets = assets[mid:]
            
            left_cov = cov.loc[left_assets, left_assets]
            right_cov = cov.loc[right_assets, right_assets]
            
            # Inverse variance weights
            left_var = np.diag(left_cov.values).sum()
            right_var = np.diag(right_cov.values).sum()
            
            alpha = 1 - left_var / (left_var + right_var)
            
            left_weights = recursive_bisection(cov, left_assets)
            right_weights = recursive_bisection(cov, right_assets)
            
            weights = {}
            for asset, w in left_weights.items():
                weights[asset] = w * alpha
            for asset, w in right_weights.items():
                weights[asset] = w * (1 - alpha)
            
            return weights
        
        return recursive_bisection(self.cov_matrix, sorted_assets)
    
    def _black_litterman(self, views: Dict[str, float], 
                        confidences: Dict[str, float]) -> Dict[str, float]:
        """Black-Litterman model with views."""
        if not views:
            return self._maximum_sharpe()
        
        # Market cap weights (use equal weight as proxy)
        market_weights = np.ones(self.n_assets) / self.n_assets
        
        # Risk aversion parameter
        delta = 2.5
        
        # Implied equilibrium returns
        pi = delta * self.cov_matrix.values @ market_weights
        
        # View matrix
        P = np.zeros((len(views), self.n_assets))
        Q = np.zeros(len(views))
        omega_diag = []
        
        for i, (asset, view_return) in enumerate(views.items()):
            if asset in self.assets:
                idx = self.assets.index(asset)
                P[i, idx] = 1
                Q[i] = view_return
                conf = confidences.get(asset, 0.5)
                # Omega: uncertainty in views
                omega_diag.append((1 - conf) * self.cov_matrix.iloc[idx, idx])
        
        if not omega_diag:
            return self._maximum_sharpe()
        
        Omega = np.diag(omega_diag)
        tau = 0.05  # Scaling factor
        
        # Black-Litterman formula
        cov = self.cov_matrix.values
        inv_cov = np.linalg.pinv(tau * cov)
        inv_omega = np.linalg.pinv(Omega)
        
        bl_returns = np.linalg.pinv(inv_cov + P.T @ inv_omega @ P) @ \
                    (inv_cov @ pi + P.T @ inv_omega @ Q)
        
        # Optimize with BL returns
        self.expected_returns = pd.Series(bl_returns, index=self.assets) * 252
        return self._maximum_sharpe()
    
    def portfolio_stats(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio statistics for given weights."""
        w = np.array([weights.get(a, 0) for a in self.assets])
        
        ret = w @ self.expected_returns.values
        vol = np.sqrt(w @ self.cov_matrix.values @ w) * np.sqrt(252)
        sharpe = ret / (vol + 1e-10)
        
        # Calculate other metrics
        portfolio_returns = (self.returns * w).sum(axis=1)
        
        return {
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'var_95': RiskMetrics.var(portfolio_returns),
            'cvar_95': RiskMetrics.cvar(portfolio_returns),
            'max_drawdown': RiskMetrics.max_drawdown(portfolio_returns)[0],
            'sortino_ratio': RiskMetrics.sortino_ratio(portfolio_returns)
        }


class SectorNeutralizer:
    """Sector neutralization for portfolios."""
    
    def __init__(self, sector_mapping: Dict[str, str]):
        """
        Args:
            sector_mapping: Dict mapping symbol -> sector
        """
        self.sector_mapping = sector_mapping
    
    def neutralize(self, weights: Dict[str, float], 
                  target_sector_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Neutralize sector exposures.
        
        Args:
            weights: Current portfolio weights
            target_sector_weights: Target sector weights (None for equal weight)
        """
        # Group by sector
        sector_weights = {}
        for symbol, weight in weights.items():
            sector = self.sector_mapping.get(symbol, 'Unknown')
            if sector not in sector_weights:
                sector_weights[sector] = {'symbols': [], 'weights': [], 'total': 0}
            sector_weights[sector]['symbols'].append(symbol)
            sector_weights[sector]['weights'].append(weight)
            sector_weights[sector]['total'] += weight
        
        # Calculate target sector weights
        if target_sector_weights is None:
            n_sectors = len(sector_weights)
            target_sector_weights = {s: 1.0 / n_sectors for s in sector_weights}
        
        # Rescale within each sector
        neutralized = {}
        for sector, data in sector_weights.items():
            target = target_sector_weights.get(sector, 0)
            current = data['total']
            
            if current > 0:
                scale = target / current
                for symbol, weight in zip(data['symbols'], data['weights']):
                    neutralized[symbol] = weight * scale
            else:
                for symbol in data['symbols']:
                    neutralized[symbol] = 0
        
        return neutralized


class PortfolioBacktester:
    """Backtest portfolio strategies."""
    
    def __init__(self, returns: pd.DataFrame, fee_model: FeeModel = None):
        self.returns = returns
        self.fee_model = fee_model or FeeModel()
    
    def backtest(self, weights_history: Dict[pd.Timestamp, Dict[str, float]],
                rebalance_freq: str = 'M') -> pd.DataFrame:
        """
        Backtest a portfolio strategy.
        
        Args:
            weights_history: Dict mapping date -> weights
            rebalance_freq: Rebalancing frequency ('D', 'W', 'M')
        """
        dates = sorted(weights_history.keys())
        
        portfolio_values = [100.0]  # Start with 100
        portfolio_returns = []
        
        current_weights = None
        
        for i, date in enumerate(self.returns.index):
            # Check if we need to rebalance
            if date in weights_history:
                current_weights = weights_history[date]
            
            if current_weights is None:
                continue
            
            # Calculate daily return
            daily_return = 0
            for symbol, weight in current_weights.items():
                if symbol in self.returns.columns:
                    daily_return += weight * self.returns.loc[date, symbol]
            
            portfolio_returns.append(daily_return)
            portfolio_values.append(portfolio_values[-1] * (1 + daily_return))
        
        # Create results DataFrame
        results = pd.DataFrame({
            'value': portfolio_values[1:],
            'return': portfolio_returns
        }, index=self.returns.index[:len(portfolio_returns)])
        
        return results
    
    def calculate_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        returns = results['return']
        
        total_return = (results['value'].iloc[-1] / results['value'].iloc[0]) - 1
        
        # Apply fees
        fees = self.fee_model.calculate_fees(
            100, results['value'].iloc[-1], 
            len(results)
        )
        
        return {
            'total_return': total_return * 100,
            'annualized_return': (1 + total_return) ** (252 / len(results)) - 1,
            'volatility': RiskMetrics.volatility(returns),
            'sharpe_ratio': RiskMetrics.sharpe_ratio(returns),
            'sortino_ratio': RiskMetrics.sortino_ratio(returns),
            'max_drawdown': RiskMetrics.max_drawdown(returns)[0],
            'calmar_ratio': RiskMetrics.calmar_ratio(returns),
            'var_95': RiskMetrics.var(returns),
            'cvar_95': RiskMetrics.cvar(returns),
            **fees
        }


if __name__ == "__main__":
    import yfinance as yf
    
    print("Testing Risk & Portfolio Management Module...")
    
    # Fetch sample data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'JNJ', 'V', 'PG']
    
    data = yf.download(symbols, period="2y", progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(returns)
    
    # Test different methods
    print("\n=== Portfolio Optimization Results ===")
    
    methods = [
        OptimizationMethod.EQUAL_WEIGHT,
        OptimizationMethod.MIN_VARIANCE,
        OptimizationMethod.MAX_SHARPE,
        OptimizationMethod.RISK_PARITY,
        OptimizationMethod.HRP
    ]
    
    for method in methods:
        weights = optimizer.optimize(method)
        stats = optimizer.portfolio_stats(weights)
        
        print(f"\n{method.value.upper()}:")
        print(f"  Expected Return: {stats['expected_return']*100:.2f}%")
        print(f"  Volatility: {stats['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {stats['max_drawdown']*100:.2f}%")
        
        # Top 3 weights
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top 3: {[(s, f'{w*100:.1f}%') for s, w in sorted_weights]}")
    
    # Test fee model
    print("\n=== Fee Model (2/20) ===")
    fee_model = FeeModel()
    fees = fee_model.calculate_fees(100, 120, 365)
    print(f"  Gross Return: {fees['gross_return']:.2f}%")
    print(f"  Management Fee: ${fees['management_fee']:.2f}")
    print(f"  Performance Fee: ${fees['performance_fee']:.2f}")
    print(f"  Net Return: {fees['net_return']:.2f}%")
