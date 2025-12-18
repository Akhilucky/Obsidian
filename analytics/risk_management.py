"""
Risk Management Suite
Comprehensive risk management tools including stress testing, VaR calculations,
Monte Carlo simulations, and portfolio risk analytics.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskMetricType(Enum):
    """Types of risk metrics."""
    VAR = "value_at_risk"
    CVAR = "conditional_var"
    VOLATILITY = "volatility"
    BETA = "beta"
    SHARPE = "sharpe_ratio"
    SORTINO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    TRACKING_ERROR = "tracking_error"
    INFORMATION_RATIO = "information_ratio"


class StressScenario(Enum):
    """Pre-defined stress scenarios."""
    MARKET_CRASH_2008 = "market_crash_2008"
    COVID_CRASH_2020 = "covid_crash_2020"
    DOT_COM_BUST = "dot_com_bust"
    FLASH_CRASH = "flash_crash"
    BLACK_MONDAY = "black_monday"
    INTEREST_RATE_SHOCK = "interest_rate_shock"
    INFLATION_SURGE = "inflation_surge"
    GEOPOLITICAL_CRISIS = "geopolitical_crisis"
    CUSTOM = "custom"


@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    volatility: float = 0.0
    annualized_volatility: float = 0.0
    beta: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0


@dataclass
class StressTestResult:
    """Result from stress test."""
    scenario: str
    portfolio_impact: float
    impact_percentage: float
    worst_asset: str
    worst_asset_impact: float
    best_asset: str
    best_asset_impact: float
    recovery_time_estimate: int  # days
    risk_mitigation: List[str]


@dataclass
class MonteCarloResult:
    """Result from Monte Carlo simulation."""
    simulations: int
    time_horizon: int
    mean_final_value: float
    median_final_value: float
    std_final_value: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    probability_of_loss: float
    probability_of_target: float
    target_return: float
    paths: Optional[np.ndarray] = None


class ValueAtRisk:
    """
    Value at Risk (VaR) calculations using multiple methods.
    Supports historical, parametric, and Monte Carlo VaR.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize VaR calculator.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def historical_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 1000000
    ) -> float:
        """
        Calculate historical VaR.
        
        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            
        Returns:
            VaR in dollar terms
        """
        if returns.empty:
            return 0.0
        
        var_pct = np.percentile(returns.dropna(), self.alpha * 100)
        return abs(var_pct * portfolio_value)
    
    def parametric_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 1000000
    ) -> float:
        """
        Calculate parametric (Variance-Covariance) VaR.
        Assumes normal distribution.
        
        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            
        Returns:
            VaR in dollar terms
        """
        if returns.empty or not SCIPY_AVAILABLE:
            return 0.0
        
        mean = returns.mean()
        std = returns.std()
        
        z_score = stats.norm.ppf(self.alpha)
        var_pct = mean + z_score * std
        
        return abs(var_pct * portfolio_value)
    
    def monte_carlo_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 1000000,
        num_simulations: int = 10000,
        time_horizon: int = 1
    ) -> float:
        """
        Calculate Monte Carlo VaR.
        
        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            num_simulations: Number of simulation paths
            time_horizon: Days ahead to simulate
            
        Returns:
            VaR in dollar terms
        """
        if returns.empty:
            return 0.0
        
        mean = returns.mean()
        std = returns.std()
        
        # Simulate returns
        simulated_returns = np.random.normal(
            mean * time_horizon,
            std * np.sqrt(time_horizon),
            num_simulations
        )
        
        simulated_values = portfolio_value * (1 + simulated_returns)
        losses = portfolio_value - simulated_values
        
        var = np.percentile(losses, self.confidence_level * 100)
        return max(var, 0)
    
    def conditional_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 1000000
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        Average of losses beyond VaR.
        
        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            
        Returns:
            CVaR in dollar terms
        """
        if returns.empty:
            return 0.0
        
        var_pct = np.percentile(returns.dropna(), self.alpha * 100)
        cvar_pct = returns[returns <= var_pct].mean()
        
        return abs(cvar_pct * portfolio_value)
    
    def marginal_var(
        self,
        portfolio_returns: pd.Series,
        position_returns: pd.Series,
        portfolio_value: float = 1000000,
        position_weight: float = 0.1
    ) -> float:
        """
        Calculate Marginal VaR for a position.
        
        Args:
            portfolio_returns: Portfolio returns
            position_returns: Position returns
            portfolio_value: Total portfolio value
            position_weight: Position weight in portfolio
            
        Returns:
            Marginal VaR
        """
        if portfolio_returns.empty or position_returns.empty:
            return 0.0
        
        # Calculate correlation and volatilities
        corr = portfolio_returns.corr(position_returns)
        port_vol = portfolio_returns.std()
        pos_vol = position_returns.std()
        
        # Beta of position to portfolio
        beta = corr * pos_vol / port_vol if port_vol > 0 else 0
        
        # Marginal VaR = Beta * Portfolio VaR
        portfolio_var = self.historical_var(portfolio_returns, portfolio_value)
        marginal_var = beta * portfolio_var * position_weight
        
        return marginal_var
    
    def component_var(
        self,
        returns_df: pd.DataFrame,
        weights: Dict[str, float],
        portfolio_value: float = 1000000
    ) -> Dict[str, float]:
        """
        Calculate Component VaR for each position.
        
        Args:
            returns_df: DataFrame of returns for each asset
            weights: Dict of asset weights
            portfolio_value: Total portfolio value
            
        Returns:
            Dict of asset to component VaR
        """
        if returns_df.empty:
            return {}
        
        # Calculate portfolio returns
        aligned_weights = [weights.get(col, 0) for col in returns_df.columns]
        portfolio_returns = (returns_df * aligned_weights).sum(axis=1)
        
        component_vars = {}
        for asset in returns_df.columns:
            if asset in weights and weights[asset] > 0:
                marginal = self.marginal_var(
                    portfolio_returns,
                    returns_df[asset],
                    portfolio_value,
                    weights[asset]
                )
                component_vars[asset] = marginal
        
        return component_vars


class MonteCarloSimulator:
    """
    Monte Carlo simulation for portfolio analysis.
    Supports multiple distribution types and correlation structures.
    """
    
    def __init__(self, num_simulations: int = 10000, random_seed: int = 42):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            num_simulations: Number of simulation paths
            random_seed: Random seed for reproducibility
        """
        self.num_simulations = num_simulations
        np.random.seed(random_seed)
    
    def simulate_gbm(
        self,
        initial_value: float,
        expected_return: float,
        volatility: float,
        time_horizon: int,
        dt: float = 1/252
    ) -> np.ndarray:
        """
        Simulate using Geometric Brownian Motion.
        
        Args:
            initial_value: Starting portfolio value
            expected_return: Annual expected return
            volatility: Annual volatility
            time_horizon: Days to simulate
            dt: Time step (1/252 for daily)
            
        Returns:
            Array of simulated paths (num_simulations x time_horizon)
        """
        # Drift and diffusion
        drift = (expected_return - 0.5 * volatility**2) * dt
        diffusion = volatility * np.sqrt(dt)
        
        # Generate random shocks
        shocks = np.random.standard_normal((self.num_simulations, time_horizon))
        
        # Calculate daily returns
        daily_returns = drift + diffusion * shocks
        
        # Calculate price paths
        paths = np.zeros((self.num_simulations, time_horizon + 1))
        paths[:, 0] = initial_value
        
        for t in range(time_horizon):
            paths[:, t + 1] = paths[:, t] * np.exp(daily_returns[:, t])
        
        return paths
    
    def simulate_correlated_assets(
        self,
        initial_values: Dict[str, float],
        expected_returns: Dict[str, float],
        volatilities: Dict[str, float],
        correlation_matrix: pd.DataFrame,
        time_horizon: int
    ) -> Dict[str, np.ndarray]:
        """
        Simulate correlated assets.
        
        Args:
            initial_values: Dict of asset to initial value
            expected_returns: Dict of asset to expected return
            volatilities: Dict of asset to volatility
            correlation_matrix: Correlation matrix DataFrame
            time_horizon: Days to simulate
            
        Returns:
            Dict of asset to simulated paths
        """
        assets = list(initial_values.keys())
        n_assets = len(assets)
        
        # Cholesky decomposition for correlated random numbers
        try:
            cholesky = np.linalg.cholesky(correlation_matrix.loc[assets, assets].values)
        except np.linalg.LinAlgError:
            # Use identity if not positive definite
            cholesky = np.eye(n_assets)
        
        # Generate correlated random numbers
        uncorrelated = np.random.standard_normal(
            (self.num_simulations, time_horizon, n_assets)
        )
        correlated = np.zeros_like(uncorrelated)
        
        for t in range(time_horizon):
            correlated[:, t, :] = uncorrelated[:, t, :] @ cholesky.T
        
        # Simulate each asset
        dt = 1/252
        paths = {}
        
        for i, asset in enumerate(assets):
            drift = (expected_returns[asset] - 0.5 * volatilities[asset]**2) * dt
            diffusion = volatilities[asset] * np.sqrt(dt)
            
            asset_paths = np.zeros((self.num_simulations, time_horizon + 1))
            asset_paths[:, 0] = initial_values[asset]
            
            for t in range(time_horizon):
                returns = drift + diffusion * correlated[:, t, i]
                asset_paths[:, t + 1] = asset_paths[:, t] * np.exp(returns)
            
            paths[asset] = asset_paths
        
        return paths
    
    def run_portfolio_simulation(
        self,
        portfolio_value: float,
        expected_return: float,
        volatility: float,
        time_horizon: int = 252,
        target_return: float = 0.0
    ) -> MonteCarloResult:
        """
        Run full portfolio simulation with analysis.
        
        Args:
            portfolio_value: Initial portfolio value
            expected_return: Annual expected return
            volatility: Annual volatility
            time_horizon: Days to simulate
            target_return: Target return for probability calculation
            
        Returns:
            MonteCarloResult with full analysis
        """
        # Simulate paths
        paths = self.simulate_gbm(
            portfolio_value,
            expected_return,
            volatility,
            time_horizon
        )
        
        # Final values
        final_values = paths[:, -1]
        
        # Calculate statistics
        mean_final = np.mean(final_values)
        median_final = np.median(final_values)
        std_final = np.std(final_values)
        
        percentiles = np.percentile(final_values, [5, 25, 75, 95])
        
        # Probabilities
        prob_loss = np.mean(final_values < portfolio_value)
        target_value = portfolio_value * (1 + target_return)
        prob_target = np.mean(final_values >= target_value)
        
        return MonteCarloResult(
            simulations=self.num_simulations,
            time_horizon=time_horizon,
            mean_final_value=mean_final,
            median_final_value=median_final,
            std_final_value=std_final,
            percentile_5=percentiles[0],
            percentile_25=percentiles[1],
            percentile_75=percentiles[2],
            percentile_95=percentiles[3],
            probability_of_loss=prob_loss,
            probability_of_target=prob_target,
            target_return=target_return,
            paths=paths
        )


class StressTester:
    """
    Stress testing framework for portfolio analysis.
    Supports historical scenarios and custom stress tests.
    """
    
    # Historical stress scenarios with sector impacts
    HISTORICAL_SCENARIOS = {
        StressScenario.MARKET_CRASH_2008: {
            'name': '2008 Financial Crisis',
            'description': 'Global financial crisis with credit freeze',
            'overall_impact': -0.50,
            'sector_impacts': {
                'Financials': -0.70,
                'Real Estate': -0.55,
                'Consumer Discretionary': -0.45,
                'Industrials': -0.40,
                'Technology': -0.45,
                'Energy': -0.35,
                'Materials': -0.40,
                'Healthcare': -0.25,
                'Consumer Staples': -0.20,
                'Utilities': -0.25,
                'Communication Services': -0.35
            },
            'duration_days': 365,
            'recovery_days': 1000
        },
        StressScenario.COVID_CRASH_2020: {
            'name': 'COVID-19 Crash',
            'description': 'Pandemic-induced market crash',
            'overall_impact': -0.34,
            'sector_impacts': {
                'Energy': -0.60,
                'Financials': -0.40,
                'Industrials': -0.35,
                'Real Estate': -0.30,
                'Consumer Discretionary': -0.25,
                'Materials': -0.25,
                'Communication Services': -0.15,
                'Technology': -0.10,
                'Healthcare': -0.05,
                'Consumer Staples': -0.10,
                'Utilities': -0.20
            },
            'duration_days': 33,
            'recovery_days': 150
        },
        StressScenario.DOT_COM_BUST: {
            'name': 'Dot-Com Bust',
            'description': 'Technology bubble collapse',
            'overall_impact': -0.45,
            'sector_impacts': {
                'Technology': -0.75,
                'Communication Services': -0.65,
                'Consumer Discretionary': -0.35,
                'Financials': -0.30,
                'Industrials': -0.25,
                'Healthcare': -0.15,
                'Energy': -0.10,
                'Consumer Staples': -0.10,
                'Utilities': -0.15,
                'Materials': -0.20,
                'Real Estate': -0.20
            },
            'duration_days': 730,
            'recovery_days': 2000
        },
        StressScenario.BLACK_MONDAY: {
            'name': 'Black Monday 1987',
            'description': 'Largest single-day market crash',
            'overall_impact': -0.22,
            'sector_impacts': {
                'Financials': -0.25,
                'Technology': -0.23,
                'Industrials': -0.22,
                'Consumer Discretionary': -0.22,
                'Materials': -0.20,
                'Energy': -0.18,
                'Healthcare': -0.18,
                'Consumer Staples': -0.15,
                'Utilities': -0.15,
                'Real Estate': -0.20,
                'Communication Services': -0.20
            },
            'duration_days': 1,
            'recovery_days': 500
        },
        StressScenario.INTEREST_RATE_SHOCK: {
            'name': 'Interest Rate Shock',
            'description': 'Sudden 300bp rate increase',
            'overall_impact': -0.20,
            'sector_impacts': {
                'Real Estate': -0.35,
                'Utilities': -0.30,
                'Financials': -0.15,
                'Technology': -0.25,
                'Consumer Discretionary': -0.20,
                'Consumer Staples': -0.10,
                'Healthcare': -0.15,
                'Industrials': -0.18,
                'Materials': -0.15,
                'Energy': -0.10,
                'Communication Services': -0.20
            },
            'duration_days': 90,
            'recovery_days': 365
        },
        StressScenario.INFLATION_SURGE: {
            'name': 'Inflation Surge',
            'description': 'CPI jumps to 10%+',
            'overall_impact': -0.25,
            'sector_impacts': {
                'Technology': -0.35,
                'Consumer Discretionary': -0.30,
                'Financials': -0.20,
                'Real Estate': -0.25,
                'Utilities': -0.20,
                'Consumer Staples': -0.15,
                'Healthcare': -0.15,
                'Industrials': -0.20,
                'Materials': -0.10,
                'Energy': 0.10,  # Energy often benefits
                'Communication Services': -0.25
            },
            'duration_days': 180,
            'recovery_days': 540
        },
        StressScenario.GEOPOLITICAL_CRISIS: {
            'name': 'Geopolitical Crisis',
            'description': 'Major geopolitical conflict',
            'overall_impact': -0.15,
            'sector_impacts': {
                'Energy': 0.20,  # Energy often rises
                'Defense': 0.15,
                'Financials': -0.20,
                'Technology': -0.15,
                'Consumer Discretionary': -0.20,
                'Industrials': -0.15,
                'Materials': -0.10,
                'Healthcare': -0.05,
                'Consumer Staples': -0.05,
                'Utilities': -0.10,
                'Real Estate': -0.15,
                'Communication Services': -0.12
            },
            'duration_days': 60,
            'recovery_days': 180
        }
    }
    
    def __init__(self):
        """Initialize stress tester."""
        self.scenarios = self.HISTORICAL_SCENARIOS.copy()
    
    def add_custom_scenario(
        self,
        name: str,
        description: str,
        overall_impact: float,
        sector_impacts: Dict[str, float],
        duration_days: int = 30,
        recovery_days: int = 90
    ) -> None:
        """Add a custom stress scenario."""
        self.scenarios[name] = {
            'name': name,
            'description': description,
            'overall_impact': overall_impact,
            'sector_impacts': sector_impacts,
            'duration_days': duration_days,
            'recovery_days': recovery_days
        }
    
    def run_stress_test(
        self,
        portfolio: Dict[str, Dict[str, Any]],
        scenario: Union[StressScenario, str],
        portfolio_value: float = 1000000
    ) -> StressTestResult:
        """
        Run stress test on portfolio.
        
        Args:
            portfolio: Dict of symbol to {'value': float, 'sector': str}
            scenario: Stress scenario to test
            portfolio_value: Total portfolio value
            
        Returns:
            StressTestResult
        """
        # Get scenario data
        if isinstance(scenario, StressScenario):
            scenario_data = self.scenarios.get(scenario)
        else:
            scenario_data = self.scenarios.get(scenario)
        
        if not scenario_data:
            scenario_data = self.scenarios.get(StressScenario.MARKET_CRASH_2008)
        
        sector_impacts = scenario_data['sector_impacts']
        
        # Calculate impact for each position
        position_impacts = {}
        worst_impact = 0
        worst_asset = ""
        best_impact = float('-inf')
        best_asset = ""
        
        for symbol, info in portfolio.items():
            position_value = info.get('value', 0)
            sector = info.get('sector', 'Other')
            
            # Get sector impact or use overall
            impact_pct = sector_impacts.get(sector, scenario_data['overall_impact'])
            impact_value = position_value * impact_pct
            
            position_impacts[symbol] = {
                'value': position_value,
                'impact': impact_value,
                'impact_pct': impact_pct
            }
            
            if impact_value < worst_impact:
                worst_impact = impact_value
                worst_asset = symbol
            
            if impact_pct > best_impact:
                best_impact = impact_pct
                best_asset = symbol
        
        # Calculate total impact
        total_impact = sum(p['impact'] for p in position_impacts.values())
        impact_percentage = total_impact / portfolio_value if portfolio_value > 0 else 0
        
        # Risk mitigation suggestions
        mitigations = self._generate_mitigation_suggestions(
            portfolio, position_impacts, scenario_data
        )
        
        return StressTestResult(
            scenario=scenario_data['name'],
            portfolio_impact=total_impact,
            impact_percentage=impact_percentage * 100,
            worst_asset=worst_asset,
            worst_asset_impact=position_impacts.get(worst_asset, {}).get('impact', 0),
            best_asset=best_asset,
            best_asset_impact=position_impacts.get(best_asset, {}).get('impact', 0),
            recovery_time_estimate=scenario_data['recovery_days'],
            risk_mitigation=mitigations
        )
    
    def _generate_mitigation_suggestions(
        self,
        portfolio: Dict[str, Dict[str, Any]],
        impacts: Dict[str, Dict[str, float]],
        scenario: Dict[str, Any]
    ) -> List[str]:
        """Generate risk mitigation suggestions."""
        suggestions = []
        
        # Check concentration
        total_value = sum(p.get('value', 0) for p in portfolio.values())
        for symbol, info in portfolio.items():
            weight = info.get('value', 0) / total_value if total_value > 0 else 0
            if weight > 0.2:
                suggestions.append(
                    f"Consider reducing {symbol} position (currently {weight*100:.1f}% of portfolio)"
                )
        
        # Sector concentration
        sector_values = {}
        for symbol, info in portfolio.items():
            sector = info.get('sector', 'Other')
            sector_values[sector] = sector_values.get(sector, 0) + info.get('value', 0)
        
        for sector, value in sector_values.items():
            weight = value / total_value if total_value > 0 else 0
            if weight > 0.3:
                suggestions.append(
                    f"Sector concentration risk: {sector} is {weight*100:.1f}% of portfolio"
                )
        
        # Hedging suggestions based on scenario
        if 'Interest Rate' in scenario.get('name', ''):
            suggestions.append("Consider interest rate hedges (short treasury futures)")
        
        if 'Inflation' in scenario.get('name', ''):
            suggestions.append("Consider inflation hedges (TIPS, commodities, real assets)")
        
        if scenario.get('overall_impact', 0) < -0.20:
            suggestions.append("Consider portfolio insurance (put options on index)")
            suggestions.append("Maintain adequate cash buffer for opportunities")
        
        return suggestions
    
    def run_all_scenarios(
        self,
        portfolio: Dict[str, Dict[str, Any]],
        portfolio_value: float = 1000000
    ) -> List[StressTestResult]:
        """Run all predefined stress scenarios."""
        results = []
        
        for scenario in StressScenario:
            if scenario != StressScenario.CUSTOM:
                result = self.run_stress_test(portfolio, scenario, portfolio_value)
                results.append(result)
        
        return results


class RiskAnalyzer:
    """
    Comprehensive risk analysis for portfolios.
    Calculates all major risk metrics.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize risk analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.rf_rate = risk_free_rate
        self.var_calculator = ValueAtRisk()
    
    def calculate_all_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        portfolio_value: float = 1000000
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Portfolio returns series
            benchmark_returns: Optional benchmark returns
            portfolio_value: Current portfolio value
            
        Returns:
            RiskMetrics object
        """
        if returns.empty:
            return RiskMetrics()
        
        returns_clean = returns.dropna()
        
        # Basic volatility
        volatility = returns_clean.std()
        annualized_vol = volatility * np.sqrt(252)
        
        # VaR calculations
        var_95 = self.var_calculator.historical_var(returns_clean, portfolio_value)
        var_99 = ValueAtRisk(0.99).historical_var(returns_clean, portfolio_value)
        cvar_95 = self.var_calculator.conditional_var(returns_clean, portfolio_value)
        cvar_99 = ValueAtRisk(0.99).conditional_var(returns_clean, portfolio_value)
        
        # Sharpe Ratio
        excess_returns = returns_clean - self.rf_rate / 252
        sharpe = np.sqrt(252) * excess_returns.mean() / volatility if volatility > 0 else 0
        
        # Sortino Ratio (downside volatility)
        downside_returns = returns_clean[returns_clean < 0]
        downside_vol = downside_returns.std() if len(downside_returns) > 0 else volatility
        sortino = np.sqrt(252) * excess_returns.mean() / downside_vol if downside_vol > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + returns_clean).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        annual_return = returns_clean.mean() * 252
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Beta and Tracking Error (if benchmark provided)
        beta = 0
        tracking_error = 0
        information_ratio = 0
        
        if benchmark_returns is not None:
            bench_clean = benchmark_returns.dropna()
            common_idx = returns_clean.index.intersection(bench_clean.index)
            
            if len(common_idx) > 0:
                port_aligned = returns_clean.loc[common_idx]
                bench_aligned = bench_clean.loc[common_idx]
                
                # Beta
                covariance = np.cov(port_aligned, bench_aligned)[0, 1]
                bench_variance = bench_aligned.var()
                beta = covariance / bench_variance if bench_variance > 0 else 0
                
                # Tracking Error
                active_returns = port_aligned - bench_aligned
                tracking_error = active_returns.std() * np.sqrt(252)
                
                # Information Ratio
                active_return_annual = active_returns.mean() * 252
                information_ratio = active_return_annual / tracking_error if tracking_error > 0 else 0
        
        # Higher moments
        skewness = returns_clean.skew()
        kurtosis = returns_clean.kurtosis()
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            volatility=volatility,
            annualized_volatility=annualized_vol,
            beta=beta,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            skewness=skewness,
            kurtosis=kurtosis
        )
    
    def rolling_risk_metrics(
        self,
        returns: pd.Series,
        window: int = 63  # ~3 months
    ) -> pd.DataFrame:
        """Calculate rolling risk metrics."""
        if returns.empty or len(returns) < window:
            return pd.DataFrame()
        
        metrics = pd.DataFrame(index=returns.index[window-1:])
        
        # Rolling volatility
        metrics['volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling Sharpe
        excess = returns - self.rf_rate / 252
        metrics['sharpe'] = (
            excess.rolling(window).mean() / returns.rolling(window).std()
        ) * np.sqrt(252)
        
        # Rolling max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window).max()
        drawdown = (cumulative - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdown.rolling(window).min()
        
        # Rolling VaR
        metrics['var_95'] = returns.rolling(window).quantile(0.05) * -1
        
        # Rolling skewness
        metrics['skewness'] = returns.rolling(window).skew()
        
        return metrics


class PortfolioOptimizer:
    """
    Portfolio optimization with risk constraints.
    Mean-variance, minimum variance, and risk parity optimization.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize optimizer."""
        self.rf_rate = risk_free_rate
    
    def mean_variance_optimize(
        self,
        returns: pd.DataFrame,
        target_return: Optional[float] = None,
        max_volatility: Optional[float] = None,
        constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, float]:
        """
        Mean-variance portfolio optimization.
        
        Args:
            returns: DataFrame of asset returns
            target_return: Target annual return
            max_volatility: Maximum allowed volatility
            constraints: Dict of asset to (min_weight, max_weight)
            
        Returns:
            Dict of optimal weights
        """
        if not SCIPY_AVAILABLE or returns.empty:
            return {col: 1/len(returns.columns) for col in returns.columns}
        
        n_assets = len(returns.columns)
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Objective: maximize Sharpe ratio or minimize volatility
        def objective(weights):
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (port_return - self.rf_rate) / port_vol if port_vol > 0 else 0
            return -sharpe  # Negative for minimization
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        if target_return is not None:
            cons.append({
                'type': 'eq',
                'fun': lambda x: np.dot(x, mean_returns) - target_return
            })
        
        if max_volatility is not None:
            cons.append({
                'type': 'ineq',
                'fun': lambda x: max_volatility - np.sqrt(np.dot(x.T, np.dot(cov_matrix, x)))
            })
        
        # Bounds
        if constraints:
            bounds = [constraints.get(col, (0, 1)) for col in returns.columns]
        else:
            bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        if result.success:
            weights = result.x
        else:
            weights = x0
        
        return {col: w for col, w in zip(returns.columns, weights)}
    
    def minimum_variance(
        self,
        returns: pd.DataFrame,
        constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, float]:
        """Find minimum variance portfolio."""
        if not SCIPY_AVAILABLE or returns.empty:
            return {col: 1/len(returns.columns) for col in returns.columns}
        
        n_assets = len(returns.columns)
        cov_matrix = returns.cov() * 252
        
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if constraints:
            bounds = [constraints.get(col, (0, 1)) for col in returns.columns]
        else:
            bounds = [(0, 1) for _ in range(n_assets)]
        
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        weights = result.x if result.success else x0
        return {col: w for col, w in zip(returns.columns, weights)}
    
    def risk_parity(
        self,
        returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Risk parity optimization.
        Equal risk contribution from each asset.
        """
        if not SCIPY_AVAILABLE or returns.empty:
            return {col: 1/len(returns.columns) for col in returns.columns}
        
        n_assets = len(returns.columns)
        cov_matrix = returns.cov() * 252
        
        def risk_budget_objective(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Marginal risk contribution
            mrc = np.dot(cov_matrix, weights) / port_vol
            
            # Risk contribution
            rc = weights * mrc
            
            # Target: equal risk contribution
            target_rc = port_vol / n_assets
            
            # Sum of squared differences
            return np.sum((rc - target_rc)**2)
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0.01, 1) for _ in range(n_assets)]
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            risk_budget_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        weights = result.x if result.success else x0
        return {col: w for col, w in zip(returns.columns, weights)}


class RiskDashboard:
    """
    Comprehensive risk management dashboard.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize risk dashboard."""
        self.var_calculator = ValueAtRisk()
        self.monte_carlo = MonteCarloSimulator()
        self.stress_tester = StressTester()
        self.risk_analyzer = RiskAnalyzer(risk_free_rate)
        self.optimizer = PortfolioOptimizer(risk_free_rate)
    
    def full_risk_analysis(
        self,
        portfolio: Dict[str, Dict[str, Any]],
        returns_data: Dict[str, pd.Series],
        benchmark_returns: Optional[pd.Series] = None,
        portfolio_value: float = 1000000
    ) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis.
        
        Args:
            portfolio: Dict of symbol to position info
            returns_data: Dict of symbol to returns series
            benchmark_returns: Optional benchmark returns
            portfolio_value: Total portfolio value
            
        Returns:
            Comprehensive risk analysis
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': portfolio_value
        }
        
        # Calculate portfolio returns
        total_value = sum(p.get('value', 0) for p in portfolio.values())
        weights = {
            sym: info.get('value', 0) / total_value 
            for sym, info in portfolio.items()
        } if total_value > 0 else {}
        
        if returns_data:
            returns_df = pd.DataFrame(returns_data)
            aligned_weights = [weights.get(col, 0) for col in returns_df.columns]
            portfolio_returns = (returns_df * aligned_weights).sum(axis=1)
            
            # Risk metrics
            metrics = self.risk_analyzer.calculate_all_metrics(
                portfolio_returns,
                benchmark_returns,
                portfolio_value
            )
            
            results['risk_metrics'] = {
                'var_95': metrics.var_95,
                'var_99': metrics.var_99,
                'cvar_95': metrics.cvar_95,
                'cvar_99': metrics.cvar_99,
                'volatility': metrics.annualized_volatility,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'max_drawdown': metrics.max_drawdown,
                'beta': metrics.beta,
                'tracking_error': metrics.tracking_error,
                'information_ratio': metrics.information_ratio,
                'skewness': metrics.skewness,
                'kurtosis': metrics.kurtosis
            }
            
            # Monte Carlo simulation
            annual_return = portfolio_returns.mean() * 252
            annual_vol = portfolio_returns.std() * np.sqrt(252)
            
            mc_result = self.monte_carlo.run_portfolio_simulation(
                portfolio_value,
                annual_return,
                annual_vol,
                time_horizon=252,
                target_return=0.10
            )
            
            results['monte_carlo'] = {
                'mean_final_value': mc_result.mean_final_value,
                'median_final_value': mc_result.median_final_value,
                'percentile_5': mc_result.percentile_5,
                'percentile_95': mc_result.percentile_95,
                'probability_of_loss': mc_result.probability_of_loss,
                'probability_of_target': mc_result.probability_of_target
            }
        
        # Stress tests
        stress_results = self.stress_tester.run_all_scenarios(portfolio, portfolio_value)
        results['stress_tests'] = [
            {
                'scenario': r.scenario,
                'impact': r.portfolio_impact,
                'impact_pct': r.impact_percentage,
                'worst_asset': r.worst_asset,
                'recovery_estimate': r.recovery_time_estimate
            }
            for r in stress_results
        ]
        
        # Position concentration
        results['concentration'] = {
            sym: info.get('value', 0) / portfolio_value * 100
            for sym, info in portfolio.items()
        }
        
        return results
    
    def generate_risk_report(
        self,
        analysis: Dict[str, Any]
    ) -> str:
        """Generate formatted risk report."""
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    RISK MANAGEMENT REPORT                        ║
╠══════════════════════════════════════════════════════════════════╣
║ Generated: {analysis['timestamp'][:19]}                          ║
║ Portfolio Value: ${analysis['portfolio_value']:,.2f}                       ║
╚══════════════════════════════════════════════════════════════════╝

📊 RISK METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        if 'risk_metrics' in analysis:
            rm = analysis['risk_metrics']
            report += f"""
Value at Risk (95%):     ${rm['var_95']:,.2f}
Value at Risk (99%):     ${rm['var_99']:,.2f}
Expected Shortfall (95%): ${rm['cvar_95']:,.2f}
Expected Shortfall (99%): ${rm['cvar_99']:,.2f}

Annualized Volatility:   {rm['volatility']*100:.2f}%
Sharpe Ratio:            {rm['sharpe_ratio']:.3f}
Sortino Ratio:           {rm['sortino_ratio']:.3f}
Maximum Drawdown:        {rm['max_drawdown']*100:.2f}%
Beta:                    {rm['beta']:.3f}
Information Ratio:       {rm['information_ratio']:.3f}

Return Distribution:
  Skewness:              {rm['skewness']:.3f}
  Kurtosis:              {rm['kurtosis']:.3f}
"""
        
        if 'monte_carlo' in analysis:
            mc = analysis['monte_carlo']
            report += f"""
🎲 MONTE CARLO SIMULATION (1-Year Horizon)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expected Value:          ${mc['mean_final_value']:,.2f}
Median Value:            ${mc['median_final_value']:,.2f}
5th Percentile:          ${mc['percentile_5']:,.2f}
95th Percentile:         ${mc['percentile_95']:,.2f}
Probability of Loss:     {mc['probability_of_loss']*100:.1f}%
Probability of 10% Gain: {mc['probability_of_target']*100:.1f}%
"""
        
        if 'stress_tests' in analysis:
            report += """
⚡ STRESS TEST RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Scenario                           Impact         Recovery
"""
            for st in analysis['stress_tests']:
                scenario_name = st['scenario'][:32].ljust(32)
                impact = f"{st['impact_pct']:+.1f}%".rjust(10)
                recovery = f"{st['recovery_estimate']}d".rjust(10)
                report += f"{scenario_name}  {impact}  {recovery}\n"
        
        if 'concentration' in analysis:
            report += """
📈 POSITION CONCENTRATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            sorted_conc = sorted(
                analysis['concentration'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for symbol, weight in sorted_conc[:10]:
                bar = '█' * int(weight / 5) + '░' * (20 - int(weight / 5))
                report += f"{symbol.ljust(8)} {bar} {weight:5.1f}%\n"
        
        report += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  This report is for informational purposes only.
    Past performance does not guarantee future results.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return report


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=504, freq='D')
    
    # Sample portfolio
    portfolio = {
        'AAPL': {'value': 250000, 'sector': 'Technology'},
        'GOOGL': {'value': 200000, 'sector': 'Technology'},
        'JPM': {'value': 150000, 'sector': 'Financials'},
        'JNJ': {'value': 150000, 'sector': 'Healthcare'},
        'XOM': {'value': 100000, 'sector': 'Energy'},
        'PG': {'value': 75000, 'sector': 'Consumer Staples'},
        'NEE': {'value': 75000, 'sector': 'Utilities'}
    }
    
    # Generate sample returns
    returns_data = {}
    for symbol in portfolio.keys():
        returns_data[symbol] = pd.Series(
            np.random.normal(0.0005, 0.02, len(dates)),
            index=dates
        )
    
    # Initialize dashboard
    dashboard = RiskDashboard()
    
    # Run full analysis
    analysis = dashboard.full_risk_analysis(
        portfolio,
        returns_data,
        portfolio_value=1000000
    )
    
    # Generate report
    report = dashboard.generate_risk_report(analysis)
    print(report)
    
    # Individual VaR calculations
    var_calc = ValueAtRisk(0.95)
    combined_returns = pd.DataFrame(returns_data).mean(axis=1)
    
    print("\n📊 VaR Analysis")
    print(f"Historical VaR (95%): ${var_calc.historical_var(combined_returns, 1000000):,.2f}")
    print(f"Parametric VaR (95%): ${var_calc.parametric_var(combined_returns, 1000000):,.2f}")
    print(f"Monte Carlo VaR (95%): ${var_calc.monte_carlo_var(combined_returns, 1000000):,.2f}")
    print(f"CVaR (95%): ${var_calc.conditional_var(combined_returns, 1000000):,.2f}")
