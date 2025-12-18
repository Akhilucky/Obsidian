"""
Advanced Portfolio Optimization
================================

Institutional-grade portfolio optimization algorithms:
- Mean-Variance Optimization (Markowitz)
- Black-Litterman Model
- Risk Parity
- Hierarchical Risk Parity (HRP)
- Maximum Diversification
- Minimum Variance
- Kelly Criterion
- Factor-based optimization

These are the same techniques used by top asset managers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: Optional[float] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CovarianceEstimator:
    """
    Advanced covariance matrix estimation methods.
    Critical for robust portfolio optimization.
    """
    
    @staticmethod
    def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
        """Standard sample covariance matrix."""
        return returns.cov()
    
    @staticmethod
    def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> pd.DataFrame:
        """
        Ledoit-Wolf shrinkage estimator.
        Shrinks sample covariance toward a structured target.
        """
        n, p = returns.shape
        sample_cov = returns.cov()
        
        # Calculate shrinkage target (scaled identity)
        mu = np.trace(sample_cov) / p
        target = mu * np.eye(p)
        
        # Calculate shrinkage intensity
        X = returns.values - returns.mean().values
        
        # Compute shrinkage parameters
        delta = sample_cov.values
        
        # Sum of squared off-diagonal elements
        sum_sq_off_diag = np.sum(np.square(delta)) - np.sum(np.square(np.diag(delta)))
        
        # Frobenius norm squared
        delta_sq_sum = np.sum(np.square(delta - target))
        
        # Compute gamma (shrinkage intensity)
        if delta_sq_sum == 0:
            shrinkage = 0
        else:
            # Simplified Ledoit-Wolf formula
            shrinkage = min(1, max(0, (1/n * sum_sq_off_diag) / delta_sq_sum))
        
        # Shrunk covariance
        shrunk_cov = shrinkage * target + (1 - shrinkage) * delta
        
        return pd.DataFrame(shrunk_cov, index=sample_cov.index, columns=sample_cov.columns)
    
    @staticmethod
    def exponential_weighted(returns: pd.DataFrame, span: int = 60) -> pd.DataFrame:
        """Exponentially weighted covariance (more recent data weighted higher)."""
        return returns.ewm(span=span).cov().iloc[-len(returns.columns):]
    
    @staticmethod
    def denoised_covariance(returns: pd.DataFrame, num_factors: int = 5) -> pd.DataFrame:
        """
        Denoised covariance using random matrix theory.
        Removes noise from eigenvalues.
        """
        cov = returns.cov()
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov.values)
        
        # Keep top eigenvalues (signal), denoise rest
        n = len(eigenvalues)
        avg_small = np.mean(eigenvalues[:n-num_factors])
        
        # Replace small eigenvalues with average
        denoised_eigenvalues = eigenvalues.copy()
        denoised_eigenvalues[:n-num_factors] = avg_small
        
        # Reconstruct covariance
        denoised_cov = eigenvectors @ np.diag(denoised_eigenvalues) @ eigenvectors.T
        
        return pd.DataFrame(denoised_cov, index=cov.index, columns=cov.columns)


class MeanVarianceOptimizer:
    """
    Classic Markowitz Mean-Variance Optimization.
    Foundation of modern portfolio theory.
    """
    
    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate
    
    def optimize(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                 target: str = 'max_sharpe', target_return: float = None,
                 target_volatility: float = None,
                 constraints: Dict = None) -> OptimizationResult:
        """
        Optimize portfolio weights.
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            target: 'max_sharpe', 'min_variance', 'target_return', 'target_volatility'
            target_return: Target return for 'target_return' optimization
            target_volatility: Target volatility for 'target_volatility' optimization
            constraints: Additional constraints
        
        Returns:
            OptimizationResult with optimal weights
        """
        n = len(expected_returns)
        assets = expected_returns.index.tolist()
        
        # Initial guess
        x0 = np.ones(n) / n
        
        # Bounds (0 to 1 for long-only)
        bounds = [(0, 1) for _ in range(n)]
        
        # Weight sum constraint
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Add target constraints
        if target == 'target_return' and target_return is not None:
            cons.append({
                'type': 'eq',
                'fun': lambda x: np.dot(x, expected_returns.values) - target_return
            })
        
        if target == 'target_volatility' and target_volatility is not None:
            cons.append({
                'type': 'eq',
                'fun': lambda x: np.sqrt(x @ cov_matrix.values @ x) - target_volatility
            })
        
        # Define objective based on target
        if target == 'max_sharpe':
            def objective(x):
                ret = np.dot(x, expected_returns.values)
                vol = np.sqrt(x @ cov_matrix.values @ x)
                return -(ret - self.risk_free_rate) / vol  # Negative for maximization
        elif target == 'min_variance':
            def objective(x):
                return x @ cov_matrix.values @ x
        elif target == 'target_return':
            def objective(x):
                return x @ cov_matrix.values @ x  # Minimize variance
        elif target == 'target_volatility':
            def objective(x):
                return -np.dot(x, expected_returns.values)  # Maximize return
        else:
            raise ValueError(f"Unknown target: {target}")
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if not result.success:
            warnings.warn(f"Optimization may not have converged: {result.message}")
        
        weights = dict(zip(assets, result.x))
        
        # Calculate metrics
        exp_ret = np.dot(result.x, expected_returns.values)
        vol = np.sqrt(result.x @ cov_matrix.values @ result.x)
        sharpe = (exp_ret - self.risk_free_rate) / vol
        
        return OptimizationResult(
            weights=weights,
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            metadata={'method': 'mean_variance', 'target': target}
        )
    
    def efficient_frontier(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                           num_points: int = 50) -> pd.DataFrame:
        """Generate efficient frontier."""
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        
        target_returns = np.linspace(min_ret, max_ret, num_points)
        
        frontier = []
        for target_ret in target_returns:
            try:
                result = self.optimize(
                    expected_returns, cov_matrix,
                    target='target_return',
                    target_return=target_ret
                )
                frontier.append({
                    'return': result.expected_return,
                    'volatility': result.volatility,
                    'sharpe': result.sharpe_ratio
                })
            except:
                continue
        
        return pd.DataFrame(frontier)


class BlackLittermanOptimizer:
    """
    Black-Litterman Model.
    Combines market equilibrium with investor views.
    Used by Goldman Sachs and major asset managers.
    """
    
    def __init__(self, risk_free_rate: float = 0.03, tau: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.tau = tau  # Uncertainty in prior
    
    def implied_returns(self, cov_matrix: pd.DataFrame, 
                        market_weights: pd.Series,
                        risk_aversion: float = 2.5) -> pd.Series:
        """Calculate equilibrium implied returns from market weights."""
        implied_ret = risk_aversion * cov_matrix @ market_weights
        return implied_ret
    
    def optimize(self, cov_matrix: pd.DataFrame,
                 market_weights: pd.Series,
                 views: Dict[str, float] = None,
                 view_confidences: Dict[str, float] = None,
                 risk_aversion: float = 2.5) -> OptimizationResult:
        """
        Black-Litterman optimization.
        
        Args:
            cov_matrix: Covariance matrix
            market_weights: Market capitalization weights
            views: Dict of asset -> expected return view
            view_confidences: Dict of asset -> confidence (0 to 1)
            risk_aversion: Risk aversion parameter
        
        Returns:
            OptimizationResult
        """
        assets = cov_matrix.columns.tolist()
        n = len(assets)
        
        # Prior (equilibrium returns)
        prior = self.implied_returns(cov_matrix, market_weights, risk_aversion)
        
        if views is None or len(views) == 0:
            # No views - return market portfolio
            return OptimizationResult(
                weights=market_weights.to_dict(),
                expected_return=float(market_weights @ prior),
                volatility=float(np.sqrt(market_weights @ cov_matrix @ market_weights)),
                sharpe_ratio=float((market_weights @ prior - self.risk_free_rate) / 
                                   np.sqrt(market_weights @ cov_matrix @ market_weights)),
                metadata={'method': 'black_litterman', 'views': 'none'}
            )
        
        # Build views matrix P and views vector Q
        view_assets = list(views.keys())
        k = len(views)  # Number of views
        
        P = np.zeros((k, n))
        Q = np.zeros(k)
        
        for i, (asset, view_return) in enumerate(views.items()):
            if asset in assets:
                P[i, assets.index(asset)] = 1
                Q[i] = view_return
        
        # View uncertainty matrix Omega
        if view_confidences is None:
            view_confidences = {a: 0.5 for a in views}
        
        omega_diag = []
        for asset in views.keys():
            conf = view_confidences.get(asset, 0.5)
            # Higher confidence = lower uncertainty
            uncertainty = (1 - conf) * self.tau * cov_matrix.loc[asset, asset]
            omega_diag.append(uncertainty)
        
        Omega = np.diag(omega_diag)
        
        # Black-Litterman formula
        Sigma = cov_matrix.values
        tau_Sigma = self.tau * Sigma
        
        # Posterior precision
        inv_tau_Sigma = np.linalg.inv(tau_Sigma)
        inv_Omega = np.linalg.inv(Omega)
        
        # Posterior mean
        M1 = inv_tau_Sigma + P.T @ inv_Omega @ P
        M2 = inv_tau_Sigma @ prior.values + P.T @ inv_Omega @ Q
        
        posterior_mean = np.linalg.solve(M1, M2)
        
        # Posterior covariance
        posterior_cov = np.linalg.inv(M1)
        
        # Optimal weights (unconstrained)
        weights_raw = np.linalg.solve(risk_aversion * Sigma, posterior_mean)
        
        # Normalize to sum to 1
        weights = weights_raw / np.sum(weights_raw)
        weights = np.clip(weights, 0, 1)  # Long-only
        weights = weights / np.sum(weights)  # Renormalize
        
        # Calculate metrics
        exp_ret = float(weights @ posterior_mean)
        vol = float(np.sqrt(weights @ Sigma @ weights))
        sharpe = (exp_ret - self.risk_free_rate) / vol
        
        return OptimizationResult(
            weights=dict(zip(assets, weights)),
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            metadata={
                'method': 'black_litterman',
                'views': views,
                'prior_returns': prior.to_dict(),
                'posterior_returns': dict(zip(assets, posterior_mean))
            }
        )


class RiskParityOptimizer:
    """
    Risk Parity / Risk Contribution Optimization.
    Equal risk contribution from all assets.
    Used by Bridgewater's All Weather fund.
    """
    
    def optimize(self, cov_matrix: pd.DataFrame,
                 target_risk_contrib: pd.Series = None) -> OptimizationResult:
        """
        Risk parity optimization.
        
        Args:
            cov_matrix: Covariance matrix
            target_risk_contrib: Target risk contribution for each asset (default: equal)
        
        Returns:
            OptimizationResult
        """
        assets = cov_matrix.columns.tolist()
        n = len(assets)
        Sigma = cov_matrix.values
        
        if target_risk_contrib is None:
            target_risk_contrib = pd.Series([1/n] * n, index=assets)
        
        # Objective: minimize squared difference from target risk contributions
        def risk_contrib(w):
            sigma_p = np.sqrt(w @ Sigma @ w)
            marginal_contrib = Sigma @ w / sigma_p
            risk_contrib = w * marginal_contrib
            return risk_contrib / sigma_p
        
        def objective(w):
            rc = risk_contrib(w)
            return np.sum((rc - target_risk_contrib.values) ** 2)
        
        # Constraints and bounds
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0.01, 1) for _ in range(n)]  # Minimum 1% weight
        
        x0 = np.ones(n) / n
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        weights = dict(zip(assets, result.x))
        
        # Calculate metrics (assume 0 expected return for risk parity)
        vol = float(np.sqrt(result.x @ Sigma @ result.x))
        
        return OptimizationResult(
            weights=weights,
            expected_return=0,  # Risk parity doesn't optimize for return
            volatility=vol,
            sharpe_ratio=0,
            metadata={
                'method': 'risk_parity',
                'risk_contributions': dict(zip(assets, risk_contrib(result.x)))
            }
        )


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (HRP).
    Machine learning-based portfolio optimization.
    More robust than traditional methods.
    """
    
    def __init__(self):
        pass
    
    def _correlation_to_distance(self, corr: pd.DataFrame) -> np.ndarray:
        """Convert correlation to distance matrix."""
        return np.sqrt(0.5 * (1 - corr))
    
    def _get_quasi_diagonal(self, link: np.ndarray) -> List[int]:
        """Extract quasi-diagonal order from hierarchical clustering."""
        link = link.astype(int)
        n = link.shape[0] + 1
        
        # Get the order of leaves
        order = []
        nodes = [n + len(link) - 1]  # Start with root
        
        while len(nodes) > 0:
            node = nodes.pop()
            if node < n:
                order.append(node)
            else:
                idx = node - n
                nodes.append(int(link[idx, 0]))
                nodes.append(int(link[idx, 1]))
        
        return order
    
    def _recursive_bisection(self, cov: pd.DataFrame, 
                              sorted_idx: List[int]) -> pd.Series:
        """Recursive bisection for HRP weights."""
        weights = pd.Series(1.0, index=cov.index)
        cluster_items = [sorted_idx]
        
        while len(cluster_items) > 0:
            # Bisect each cluster
            new_clusters = []
            for items in cluster_items:
                if len(items) > 1:
                    half = len(items) // 2
                    left = items[:half]
                    right = items[half:]
                    
                    # Calculate variance of each half
                    cov_left = cov.iloc[left, left]
                    cov_right = cov.iloc[right, right]
                    
                    # Inverse variance weights
                    w_left = self._get_cluster_variance(cov_left)
                    w_right = self._get_cluster_variance(cov_right)
                    
                    alpha = 1 - w_left / (w_left + w_right)
                    
                    # Assign weights
                    weights.iloc[left] *= alpha
                    weights.iloc[right] *= (1 - alpha)
                    
                    new_clusters.extend([left, right])
            
            cluster_items = [c for c in new_clusters if len(c) > 1]
        
        return weights
    
    def _get_cluster_variance(self, cov: pd.DataFrame) -> float:
        """Calculate variance of equally-weighted portfolio."""
        n = len(cov)
        weights = np.ones(n) / n
        return float(weights @ cov.values @ weights)
    
    def optimize(self, returns: pd.DataFrame) -> OptimizationResult:
        """
        HRP optimization.
        
        Args:
            returns: Historical returns DataFrame
        
        Returns:
            OptimizationResult
        """
        assets = returns.columns.tolist()
        
        # Covariance and correlation
        cov = returns.cov()
        corr = returns.corr()
        
        # Distance matrix
        dist = self._correlation_to_distance(corr)
        dist_condensed = squareform(dist.values, checks=False)
        
        # Hierarchical clustering
        link = linkage(dist_condensed, method='single')
        
        # Get quasi-diagonal order
        sorted_idx = self._get_quasi_diagonal(link)
        
        # Recursive bisection
        weights = self._recursive_bisection(cov.iloc[sorted_idx, sorted_idx], 
                                             list(range(len(sorted_idx))))
        
        # Reorder weights to original order
        final_weights = pd.Series(index=cov.columns)
        for i, idx in enumerate(sorted_idx):
            final_weights.iloc[idx] = weights.iloc[i]
        
        weights_dict = final_weights.to_dict()
        
        # Calculate metrics
        w = final_weights.values
        vol = float(np.sqrt(w @ cov.values @ w))
        exp_ret = float(w @ returns.mean().values * 252)  # Annualized
        sharpe = exp_ret / vol if vol > 0 else 0
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=exp_ret,
            volatility=vol * np.sqrt(252),  # Annualized
            sharpe_ratio=sharpe,
            metadata={'method': 'hierarchical_risk_parity'}
        )


class MaxDiversificationOptimizer:
    """
    Maximum Diversification Optimization.
    Maximizes the diversification ratio.
    """
    
    def optimize(self, cov_matrix: pd.DataFrame) -> OptimizationResult:
        """
        Maximize diversification ratio.
        
        DR = weighted average volatility / portfolio volatility
        """
        assets = cov_matrix.columns.tolist()
        n = len(assets)
        Sigma = cov_matrix.values
        
        # Individual volatilities
        vols = np.sqrt(np.diag(Sigma))
        
        def neg_diversification_ratio(w):
            port_vol = np.sqrt(w @ Sigma @ w)
            weighted_vol = np.dot(w, vols)
            return -weighted_vol / port_vol  # Negative for minimization
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0, 1) for _ in range(n)]
        x0 = np.ones(n) / n
        
        result = minimize(neg_diversification_ratio, x0, method='SLSQP',
                         bounds=bounds, constraints=cons)
        
        weights = dict(zip(assets, result.x))
        vol = float(np.sqrt(result.x @ Sigma @ result.x))
        div_ratio = -result.fun
        
        return OptimizationResult(
            weights=weights,
            expected_return=0,
            volatility=vol,
            sharpe_ratio=0,
            metadata={
                'method': 'max_diversification',
                'diversification_ratio': div_ratio
            }
        )


class MinimumVarianceOptimizer:
    """
    Minimum Variance / Global Minimum Variance portfolio.
    """
    
    def optimize(self, cov_matrix: pd.DataFrame,
                 use_cvxpy: bool = True) -> OptimizationResult:
        """Find minimum variance portfolio."""
        assets = cov_matrix.columns.tolist()
        n = len(assets)
        Sigma = cov_matrix.values
        
        if use_cvxpy and CVXPY_AVAILABLE:
            w = cp.Variable(n)
            objective = cp.Minimize(cp.quad_form(w, Sigma))
            constraints = [
                cp.sum(w) == 1,
                w >= 0
            ]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            
            weights = w.value
        else:
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = [(0, 1) for _ in range(n)]
            x0 = np.ones(n) / n
            
            result = minimize(lambda w: w @ Sigma @ w, x0, method='SLSQP',
                            bounds=bounds, constraints=cons)
            weights = result.x
        
        weights_dict = dict(zip(assets, weights))
        vol = float(np.sqrt(weights @ Sigma @ weights))
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=0,
            volatility=vol,
            sharpe_ratio=0,
            metadata={'method': 'minimum_variance'}
        )


class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing.
    Maximizes long-term growth rate.
    """
    
    def __init__(self, max_leverage: float = 1.0, fraction: float = 0.5):
        """
        Args:
            max_leverage: Maximum leverage (1.0 = no leverage)
            fraction: Kelly fraction (0.5 = half Kelly, safer)
        """
        self.max_leverage = max_leverage
        self.fraction = fraction
    
    def calculate_kelly(self, expected_return: float, volatility: float) -> float:
        """
        Calculate Kelly fraction for a single asset.
        
        Kelly = (expected_return) / (volatility^2)
        """
        if volatility == 0:
            return 0
        
        kelly = expected_return / (volatility ** 2)
        
        # Apply fractional Kelly
        kelly *= self.fraction
        
        # Apply leverage constraint
        return min(kelly, self.max_leverage)
    
    def optimize(self, expected_returns: pd.Series, 
                 cov_matrix: pd.DataFrame) -> OptimizationResult:
        """
        Multi-asset Kelly optimization.
        
        Kelly weights = Sigma^(-1) @ expected_returns
        """
        assets = expected_returns.index.tolist()
        n = len(assets)
        
        try:
            Sigma_inv = np.linalg.inv(cov_matrix.values)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse for singular matrices
            Sigma_inv = np.linalg.pinv(cov_matrix.values)
        
        # Unconstrained Kelly weights
        kelly_weights = Sigma_inv @ expected_returns.values
        
        # Apply fractional Kelly
        kelly_weights *= self.fraction
        
        # Normalize and constrain
        if np.sum(kelly_weights) > self.max_leverage:
            kelly_weights = kelly_weights / np.sum(kelly_weights) * self.max_leverage
        
        # Handle negative weights (short positions)
        if np.any(kelly_weights < 0):
            kelly_weights = np.maximum(kelly_weights, 0)
            kelly_weights = kelly_weights / np.sum(kelly_weights)
        
        weights_dict = dict(zip(assets, kelly_weights))
        
        # Calculate metrics
        exp_ret = float(kelly_weights @ expected_returns.values)
        vol = float(np.sqrt(kelly_weights @ cov_matrix.values @ kelly_weights))
        sharpe = exp_ret / vol if vol > 0 else 0
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            metadata={
                'method': 'kelly_criterion',
                'fraction': self.fraction,
                'max_leverage': self.max_leverage
            }
        )


class PortfolioOptimizer:
    """
    Unified interface for all optimization methods.
    """
    
    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate
        
        # Initialize optimizers
        self.mean_variance = MeanVarianceOptimizer(risk_free_rate)
        self.black_litterman = BlackLittermanOptimizer(risk_free_rate)
        self.risk_parity = RiskParityOptimizer()
        self.hrp = HierarchicalRiskParity()
        self.max_div = MaxDiversificationOptimizer()
        self.min_var = MinimumVarianceOptimizer()
        self.kelly = KellyCriterion()
        self.cov_estimator = CovarianceEstimator()
    
    def optimize(self, returns: pd.DataFrame, method: str = 'mean_variance',
                 **kwargs) -> OptimizationResult:
        """
        Optimize portfolio using specified method.
        
        Args:
            returns: Historical returns DataFrame
            method: Optimization method
            **kwargs: Method-specific parameters
        
        Returns:
            OptimizationResult
        """
        # Estimate covariance
        cov_method = kwargs.pop('cov_method', 'ledoit_wolf')
        
        if cov_method == 'ledoit_wolf':
            cov = self.cov_estimator.ledoit_wolf_shrinkage(returns)
        elif cov_method == 'exponential':
            cov = self.cov_estimator.exponential_weighted(returns)
        elif cov_method == 'denoised':
            cov = self.cov_estimator.denoised_covariance(returns)
        else:
            cov = self.cov_estimator.sample_covariance(returns)
        
        # Expected returns
        expected_returns = returns.mean() * 252  # Annualized
        
        # Optimize based on method
        if method == 'mean_variance':
            target = kwargs.get('target', 'max_sharpe')
            return self.mean_variance.optimize(expected_returns, cov, target=target)
        
        elif method == 'black_litterman':
            market_weights = kwargs.get('market_weights')
            if market_weights is None:
                market_weights = pd.Series(1/len(returns.columns), index=returns.columns)
            views = kwargs.get('views', {})
            return self.black_litterman.optimize(cov, market_weights, views)
        
        elif method == 'risk_parity':
            return self.risk_parity.optimize(cov)
        
        elif method == 'hrp':
            return self.hrp.optimize(returns)
        
        elif method == 'max_diversification':
            return self.max_div.optimize(cov)
        
        elif method == 'min_variance':
            return self.min_var.optimize(cov)
        
        elif method == 'kelly':
            return self.kelly.optimize(expected_returns, cov)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compare_methods(self, returns: pd.DataFrame,
                        methods: List[str] = None) -> pd.DataFrame:
        """Compare different optimization methods."""
        if methods is None:
            methods = ['mean_variance', 'risk_parity', 'hrp', 
                      'max_diversification', 'min_variance']
        
        results = []
        for method in methods:
            try:
                result = self.optimize(returns, method=method)
                results.append({
                    'method': method,
                    'expected_return': result.expected_return,
                    'volatility': result.volatility,
                    'sharpe_ratio': result.sharpe_ratio,
                    **result.weights
                })
            except Exception as e:
                print(f"Method {method} failed: {e}")
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 60)
    print("Advanced Portfolio Optimization")
    print("=" * 60)
    
    # Generate sample returns
    np.random.seed(42)
    n_assets = 5
    n_days = 500
    
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    
    # Correlated returns
    mean_returns = np.array([0.0008, 0.0007, 0.0006, 0.0009, 0.0005])
    cov = np.array([
        [0.0004, 0.0002, 0.0002, 0.0003, 0.0002],
        [0.0002, 0.0005, 0.0002, 0.0002, 0.0002],
        [0.0002, 0.0002, 0.0003, 0.0002, 0.0002],
        [0.0003, 0.0002, 0.0002, 0.0006, 0.0002],
        [0.0002, 0.0002, 0.0002, 0.0002, 0.0005]
    ])
    
    returns = pd.DataFrame(
        np.random.multivariate_normal(mean_returns, cov, n_days),
        columns=assets
    )
    
    print(f"\nAssets: {assets}")
    print(f"Days of data: {n_days}")
    
    # Test all optimizers
    optimizer = PortfolioOptimizer()
    
    print("\n--- Mean-Variance (Max Sharpe) ---")
    result = optimizer.optimize(returns, method='mean_variance', target='max_sharpe')
    print(f"Expected Return: {result.expected_return*100:.2f}%")
    print(f"Volatility: {result.volatility*100:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print("Weights:")
    for asset, weight in result.weights.items():
        print(f"  {asset}: {weight*100:.1f}%")
    
    print("\n--- Risk Parity ---")
    result = optimizer.optimize(returns, method='risk_parity')
    print(f"Volatility: {result.volatility*100:.2f}%")
    print("Weights:")
    for asset, weight in result.weights.items():
        print(f"  {asset}: {weight*100:.1f}%")
    
    print("\n--- Hierarchical Risk Parity ---")
    result = optimizer.optimize(returns, method='hrp')
    print(f"Volatility: {result.volatility*100:.2f}%")
    print("Weights:")
    for asset, weight in result.weights.items():
        print(f"  {asset}: {weight*100:.1f}%")
    
    print("\n--- Black-Litterman (with views) ---")
    views = {'AAPL': 0.15, 'MSFT': 0.12}  # Bullish on AAPL and MSFT
    result = optimizer.optimize(returns, method='black_litterman', views=views)
    print(f"Expected Return: {result.expected_return*100:.2f}%")
    print("Weights:")
    for asset, weight in result.weights.items():
        print(f"  {asset}: {weight*100:.1f}%")
    
    print("\n--- Method Comparison ---")
    comparison = optimizer.compare_methods(returns)
    print(comparison[['method', 'expected_return', 'volatility', 'sharpe_ratio']].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Portfolio Optimization Complete!")
    print("=" * 60)
