"""
Scenario Agent
===============
Simulate macro shocks continuously.

Frequency: Runs in background / on-demand
Consumes: REGIME_DETECTED, RISK_APPROVED
Produces: SCENARIO_RESULT

Responsibilities:
- Rate hike simulation
- Volatility spikes
- Crypto contagion modeling
- Portfolio resilience scoring
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from agents.base_agent import BaseAgent
from core.event_bus import Event, EventType

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# Pre-defined macro shock scenarios
MACRO_SCENARIOS = {
    "rate_hike_100bp": {
        "description": "Fed raises rates by 100bp",
        "equity_shock": -0.05,
        "bond_shock": -0.03,
        "vol_multiplier": 2.0,
        "crypto_shock": -0.15,
        "duration_days": 30,
    },
    "rate_hike_50bp": {
        "description": "Fed raises rates by 50bp",
        "equity_shock": -0.02,
        "bond_shock": -0.015,
        "vol_multiplier": 1.5,
        "crypto_shock": -0.08,
        "duration_days": 15,
    },
    "vol_spike": {
        "description": "VIX doubles (market fear event)",
        "equity_shock": -0.08,
        "bond_shock": 0.01,
        "vol_multiplier": 3.0,
        "crypto_shock": -0.20,
        "duration_days": 10,
    },
    "crypto_contagion": {
        "description": "Major crypto exchange collapse",
        "equity_shock": -0.02,
        "bond_shock": 0.005,
        "vol_multiplier": 1.3,
        "crypto_shock": -0.40,
        "duration_days": 20,
    },
    "black_swan": {
        "description": "Extreme tail event (pandemic, war)",
        "equity_shock": -0.20,
        "bond_shock": 0.03,
        "vol_multiplier": 5.0,
        "crypto_shock": -0.50,
        "duration_days": 60,
    },
    "stagflation": {
        "description": "Persistent inflation + recession",
        "equity_shock": -0.10,
        "bond_shock": -0.05,
        "vol_multiplier": 2.5,
        "crypto_shock": -0.25,
        "duration_days": 90,
    },
    "recovery_rally": {
        "description": "Post-crisis recovery",
        "equity_shock": 0.12,
        "bond_shock": -0.01,
        "vol_multiplier": 0.7,
        "crypto_shock": 0.30,
        "duration_days": 60,
    },
}


class ScenarioAgent(BaseAgent):
    """
    Agent 8: Simulate macro shocks continuously.
    
    Responsibilities:
    - Rate hike simulation
    - Volatility spikes
    - Crypto contagion modeling
    - Portfolio resilience scoring
    """
    
    def __init__(self):
        super().__init__(
            name="ScenarioAgent",
            subscriptions=[
                EventType.REGIME_DETECTED.value,
                EventType.RISK_APPROVED.value,
            ]
        )
        self._scenario_results: Dict[str, List[Dict]] = {}
        self._portfolio_weights: Dict[str, float] = {}
        self._asset_classes: Dict[str, str] = {}  # symbol -> asset_class
    
    def initialize(self):
        """Load scenario definitions."""
        self._log(f"Loaded {len(MACRO_SCENARIOS)} macro scenarios")
    
    def consume(self, event: Event):
        """Track regime changes and approved positions for scenario inputs."""
        payload = event.get_payload()
        
        if event.event_type == EventType.REGIME_DETECTED.value:
            regime = payload.get("regime")
            self._log(f"Regime update: {regime}")
        
        elif event.event_type == EventType.RISK_APPROVED.value:
            symbol = payload.get("symbol")
            size = payload.get("position_size", 0)
            self._portfolio_weights[symbol] = size
    
    def produce(self) -> Optional[Event]:
        return None
    
    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "scenarios_available": len(MACRO_SCENARIOS),
            "scenarios_run": sum(len(v) for v in self._scenario_results.values()),
            "tracked_positions": len(self._portfolio_weights),
        }
    
    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────
    
    def simulate(self, returns: pd.DataFrame,
                 weights: Optional[Dict[str, float]] = None,
                 scenarios: Optional[List[str]] = None,
                 n_simulations: int = 1000) -> Dict[str, Any]:
        """
        Run full scenario simulation suite.
        
        Args:
            returns: Historical returns DataFrame
            weights: Portfolio weights (defaults to tracked positions)
            scenarios: Specific scenarios to run (None = all)
            n_simulations: Monte Carlo paths per scenario
        
        Returns:
            Comprehensive scenario analysis.
        """
        weights = weights or self._portfolio_weights
        scenario_names = scenarios or list(MACRO_SCENARIOS.keys())
        
        results = {}
        for name in scenario_names:
            if name not in MACRO_SCENARIOS:
                continue
            scenario = MACRO_SCENARIOS[name]
            result = self._run_scenario(returns, weights, name, scenario, n_simulations)
            results[name] = result
        
        # Portfolio resilience score
        resilience = self._compute_resilience(results)
        
        output = {
            "scenarios": results,
            "resilience_score": resilience,
            "portfolio_weights": weights,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Publish
        self._publish(
            EventType.SCENARIO_RESULT.value,
            {
                "resilience_score": resilience,
                "scenarios_run": len(results),
                "worst_scenario": min(results.items(),
                                       key=lambda x: x[1].get("expected_loss", 0))[0] if results else None,
            }
        )
        
        return output
    
    def run_scenarios(self, returns, symbol: str = "PORTFOLIO",
                      n_simulations: int = 500) -> Dict[str, Any]:
        """
        Simplified scenario runner for the orchestrator.
        
        Accepts returns as a Series or DataFrame and runs all macro
        scenarios, returning a resilience score.
        """
        # Convert Series to DataFrame if needed
        if isinstance(returns, pd.Series):
            returns = returns.to_frame(name=symbol)
        
        # Use tracked weights or equal weight
        weights = self._portfolio_weights or {c: 1.0 / len(returns.columns) for c in returns.columns}
        
        return self.simulate(returns, weights=weights, n_simulations=n_simulations)
    
    def simulate_rate_hike(self, returns: pd.DataFrame,
                           bp_change: int = 50) -> Dict[str, Any]:
        """Simulate a specific rate hike."""
        shock_factor = bp_change / 10000
        scenario = {
            "equity_shock": -shock_factor * 5,
            "bond_shock": -shock_factor * 3,
            "vol_multiplier": 1.0 + shock_factor * 50,
            "crypto_shock": -shock_factor * 15,
            "duration_days": max(10, bp_change // 5),
        }
        return self._run_scenario(returns, self._portfolio_weights,
                                  f"rate_hike_{bp_change}bp", scenario)
    
    def simulate_crypto_contagion(self, returns: pd.DataFrame,
                                   severity: float = 0.5) -> Dict[str, Any]:
        """Simulate crypto contagion with variable severity."""
        scenario = {
            "equity_shock": -0.02 * severity,
            "bond_shock": 0.005,
            "vol_multiplier": 1.0 + severity * 2,
            "crypto_shock": -0.40 * severity,
            "duration_days": int(20 * severity),
        }
        return self._run_scenario(returns, self._portfolio_weights,
                                  "crypto_contagion_custom", scenario)
    
    # ──────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────
    
    def _run_scenario(self, returns: pd.DataFrame,
                      weights: Dict[str, float],
                      scenario_name: str,
                      scenario: Dict[str, Any],
                      n_simulations: int = 1000) -> Dict[str, Any]:
        """Run a single scenario with Monte Carlo simulation."""
        if returns.empty or not weights:
            return {"error": "Insufficient data", "expected_loss": 0.0}
        
        common = [c for c in returns.columns if c in weights]
        if not common:
            return {"error": "No matching symbols", "expected_loss": 0.0}
        
        w = np.array([weights.get(c, 0) for c in common])
        ret_matrix = returns[common].values
        
        # Apply shocks
        equity_shock = scenario.get("equity_shock", 0)
        vol_mult = scenario.get("vol_multiplier", 1.0)
        
        # Monte Carlo paths
        mean_ret = ret_matrix.mean(axis=0)
        cov = np.cov(ret_matrix.T) if ret_matrix.shape[0] > 1 else np.eye(len(common)) * 0.01
        
        # Ensure cov is 2D
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        elif cov.ndim == 1:
            cov = np.diag(cov)
        
        shocked_mean = mean_ret + equity_shock
        shocked_cov = cov * vol_mult
        
        try:
            simulated = np.random.multivariate_normal(
                shocked_mean, shocked_cov, size=n_simulations
            )
        except Exception:
            simulated = np.random.normal(
                equity_shock, np.sqrt(np.diag(shocked_cov).mean()), 
                size=(n_simulations, len(common))
            )
        
        portfolio_returns = simulated @ w
        
        expected_loss = float(np.mean(portfolio_returns))
        worst_case = float(np.percentile(portfolio_returns, 1))
        var_95 = float(np.percentile(portfolio_returns, 5))
        cvar_95 = float(portfolio_returns[portfolio_returns <= var_95].mean()) if np.any(portfolio_returns <= var_95) else var_95
        
        result = {
            "scenario": scenario_name,
            "description": scenario.get("description", ""),
            "expected_loss": round(expected_loss, 6),
            "worst_case_1pct": round(worst_case, 6),
            "var_95": round(var_95, 6),
            "cvar_95": round(cvar_95, 6),
            "probability_of_loss": round(float((portfolio_returns < 0).mean()), 4),
            "n_simulations": n_simulations,
        }
        
        # Cache result
        if scenario_name not in self._scenario_results:
            self._scenario_results[scenario_name] = []
        self._scenario_results[scenario_name].append(result)
        
        return result
    
    def _compute_resilience(self, results: Dict[str, Dict]) -> float:
        """
        Portfolio resilience score (0-1).
        
        Higher = more resilient to shocks.
        """
        if not results:
            return 0.5
        
        losses = [r.get("expected_loss", 0) for r in results.values()]
        worst_losses = [r.get("worst_case_1pct", 0) for r in results.values()]
        
        avg_loss = np.mean(losses)
        avg_worst = np.mean(worst_losses)
        
        # Normalize: no loss = 1.0, -20% average worst = 0.0
        score = max(0.0, min(1.0, 1.0 + (avg_worst / 0.20)))
        return round(float(score), 4)
