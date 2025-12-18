import numpy as np
import pandas as pd
from scipy.optimize import minimize

class HRPOptimizer:
    """Hierarchical Risk Parity optimizer."""
    
    def __init__(self, returns):
        self.returns = returns
        self.weights = None
    
    def optimize(self):
        """Calculate HRP weights."""
        # Calculate covariance matrix
        cov_matrix = self.returns.cov()
        
        # Simple equal-weighted for now (basic implementation)
        n_assets = len(self.returns.columns)
        self.weights = np.array([1/n_assets] * n_assets)
        
        return self.weights
    
    def get_weights(self):
        """Return the calculated weights."""
        if self.weights is None:
            self.optimize()
        return self.weights

class RiskParity:
    """Risk Parity portfolio optimizer."""
    
    def __init__(self, cov_matrix):
        self.cov_matrix = cov_matrix
        self.weights = None
    
    def optimize(self):
        """Calculate Risk Parity weights."""
        n_assets = len(self.cov_matrix)
        
        def risk_parity_objective(weights):
            # Calculate portfolio volatility
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
            
            # Calculate marginal contribution to risk
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            
            # Risk parity objective: minimize difference from equal risk contribution
            equal_contrib = portfolio_vol / n_assets
            return np.sum((marginal_contrib - equal_contrib) ** 2)
        
        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess: equal weights
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(risk_parity_objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        self.weights = result.x
        return self.weights
    
    def get_weights(self):
        """Return the calculated weights."""
        if self.weights is None:
            self.optimize()
        return self.weights
