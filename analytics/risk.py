import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from portfolio.manager import PortfolioManager

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from data.openbb_integration import OpenBBIntegration
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False

try:
    import core.java_optimizer as java_optimizer
    JAVA_OPTIMIZER_AVAILABLE = java_optimizer.java_available()
except ImportError:
    JAVA_OPTIMIZER_AVAILABLE = False

class RiskManager:
    def __init__(self, portfolio_manager):
        self.portfolio_manager = portfolio_manager
        self.openbb = OpenBBIntegration() if OPENBB_AVAILABLE else None
    
    def calculate_portfolio_risk(self):
        # Get portfolio data
        portfolio = self.portfolio_manager.get_portfolio()
        if portfolio.empty:
            raise ValueError("Portfolio is empty")
        
        # Get asset prices and weights
        tickers = portfolio[portfolio['Ticker'] != 'CASH']['Ticker'].unique().tolist()
        if not tickers:
            raise ValueError("No assets in portfolio")
        
        # Fetch historical data
        try:
            data = yf.download(tickers, period='1y')['Adj Close']
        except Exception as e:
            raise ValueError(f"Error fetching data: {e}")
        
        # Calculate daily returns
        returns = data.pct_change().dropna()
        
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Get portfolio weights
        latest_portfolio = self.portfolio_manager.get_portfolio()
        asset_weights = []
        for ticker in tickers:
            asset = latest_portfolio[latest_portfolio['Ticker'] == ticker]
            if not asset.empty:
                asset_weights.append(asset['Allocation %'].values[0] / 100)
            else:
                asset_weights.append(0)
        
        # Calculate portfolio variance
        portfolio_variance = np.dot(np.array(asset_weights).T, np.dot(cov_matrix, np.array(asset_weights)))
        
        # Calculate portfolio volatility (standard deviation)
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate Value at Risk (VaR)
        portfolio_returns = (self.portfolio_manager.portfolio_value + portfolio['Current Value'].dropna().sum()) - 1000000
        sorted_returns = np.sort(portfolio_returns)
        var_95 = np.percentile(sorted_returns, 5)
        
        # Calculate Conditional Value at Risk (CVaR)
        cvar_95 = sorted_returns[sorted_returns <= var_95].mean()
        
        return {
            'volatility': portfolio_volatility,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def optimize_portfolio(self):
        # Get portfolio data
        portfolio = self.portfolio_manager.get_portfolio()
        if portfolio.empty:
            raise ValueError("Portfolio is empty")
        
        # Get asset prices and weights
        tickers = portfolio[portfolio['Ticker'] != 'CASH']['Ticker'].unique().tolist()
        if not tickers:
            raise ValueError("No assets in portfolio")
        
        # Fetch historical data
        try:
            data = yf.download(tickers, period='1y')['Adj Close']
        except Exception as e:
            raise ValueError(f"Error fetching data: {e}")
        
        # Calculate daily returns
        returns = data.pct_change().dropna()
        
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Calculate expected returns
        expected_returns = returns.mean()
        
        # Define optimization problem
        if CVXPY_AVAILABLE:
            weights = cp.Variable(len(tickers))
            portfolio_variance = cp.quad_form(weights, cov_matrix)
            objective = cp.Minimize(portfolio_variance)
            constraints = [
                cp.sum(weights) == 1,
                weights >= 0
            ]

            # Solve optimization problem
            problem = cp.Problem(objective, constraints)
            problem.solve()
            optimal_weights = weights.value
        elif JAVA_OPTIMIZER_AVAILABLE:
            # Pure-Java mean-variance optimization (closed form)
            try:
                java_result = java_optimizer.optimize_portfolio(
                    tickers,
                    np.asarray(expected_returns, dtype=float),
                    np.asarray(cov_matrix, dtype=float),
                )
                weights_map = java_result["min_variance"]
                optimal_weights = np.array(
                    [weights_map[t] for t in tickers], dtype=float)
            except Exception:
                optimal_weights = None
            if optimal_weights is None or not np.all(np.isfinite(optimal_weights)):
                raise ImportError("Java optimizer failed for portfolio optimization")
        elif SCIPY_AVAILABLE:
            cov = cov_matrix.values

            def portfolio_variance(w):
                return w @ cov @ w

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, 1)] * len(tickers)
            x0 = np.ones(len(tickers)) / len(tickers)

            result = minimize(portfolio_variance, x0, method='SLSQP',
                              bounds=bounds, constraints=constraints)
            optimal_weights = result.x
        else:
            raise ImportError("cvxpy or scipy required for portfolio optimization")
        
        # Map weights to tickers
        optimization_results = {ticker: weight for ticker, weight in zip(tickers, optimal_weights)}
        
        return optimization_results
    
    def black_litterman(self, views, confidences):
        # Get market implied returns
        market_caps = {
            'AAPL': 2.68e12,
            'MSFT': 2.25e12,
            'AMZN': 1.17e12,
            'META': 7.3e11,
            'TSLA': 4.7e11
        }
        
        # Calculate market portfolio weights
        total_market_cap = sum(market_caps.values())
        market_weights = {ticker: market_cap / total_market_cap for ticker, market_cap in market_caps.items()}
        
        # Get historical data
        tickers = list(market_weights.keys())
        data = yf.download(tickers, period='1y')['Adj Close']
        returns = data.pct_change().dropna()
        
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Calculate market returns
        market_returns = (returns * pd.Series(market_weights)).sum(axis=1)
        risk_aversion = 2.5
        market_premium = market_returns.mean()
        
        # Calculate implied market returns
        implied_returns = risk_aversion * (cov_matrix @ np.array(list(market_weights.values())))
        
        # Process views
        p_matrix = []
        q_vector = []
        omega_matrix = []
        
        for view, confidence in zip(views, confidences):
            ticker, direction, threshold = view
            p_matrix.append([1 if t == ticker else 0 for t in tickers])
            q_vector.append(threshold if direction == 'up' else -threshold)
            omega_matrix.append([confidence])
        
        p_matrix = np.array(p_matrix)
        q_vector = np.array(q_vector)
        omega_matrix = np.diag(omega_matrix)
        
        # Calculate Black-Litterman returns
        bl_returns = implied_returns + ((cov_matrix @ p_matrix.T) @ np.linalg.inv(p_matrix @ cov_matrix @ p_matrix.T + omega_matrix)) @ (q_vector - p_matrix @ implied_returns)
        
        # Optimize portfolio with Black-Litterman returns
        if CVXPY_AVAILABLE:
            weights = cp.Variable(len(tickers))
            portfolio_variance = cp.quad_form(weights, cov_matrix)
            objective = cp.Minimize(portfolio_variance)
            constraints = [
                cp.sum(weights) == 1,
                weights >= 0
            ]
            problem = cp.Problem(objective, constraints)
            problem.solve()
            optimal_weights = weights.value
        elif SCIPY_AVAILABLE:
            cov = cov_matrix.values

            def bl_portfolio_variance(w):
                return w @ cov @ w

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, 1)] * len(tickers)
            x0 = np.ones(len(tickers)) / len(tickers)

            result = minimize(bl_portfolio_variance, x0, method='SLSQP',
                              bounds=bounds, constraints=constraints)
            optimal_weights = result.x
        else:
            raise ImportError("cvxpy or scipy required for Black-Litterman optimization")

        return {ticker: weight for ticker, weight in zip(tickers, optimal_weights)}

class HRPOptimizer:
    def __init__(self, returns):
        self.returns = returns
    
    def _get_quasi_diagonal(self, link):
        quasi_diag = pd.Series([link[i] for i in range(len(self.returns.columns))])
        indices = np.where(quasi_diag == 0)[0]
        while len(indices) > 0:
            min_index = indices[0]
            quasi_diag[quasi_diag == quasi_diag[min_index]] = quasi_diag[quasi_diag == quasi_diag[min_index]].min()
            indices = np.where(quasi_diag == quasi_diag.min())[0]
        return quasi_diag.argsort()
    
    def _get_cluster_variance(self, cluster, cov_matrix):
        return cov_matrix.iloc[cluster, cluster].values.mean() * len(cluster)
    
    def _hierarchical_clustering(self):
        # Calculate correlation matrix
        corr = self.returns.corr()
        
        # Calculate distance matrix
        dissimilarity = 1 - corr.abs()
        
        # Perform hierarchical clustering
        from scipy.cluster.hierarchy import linkage
        link = linkage(dissimilarity, 'single')
        
        return link
    
    def optimize(self):
        # Perform hierarchical clustering
        link = self._hierarchical_clustering()
        
        # Get quasi-diagonal indices
        quasi_diag = self._get_quasi_diagonal(link)
        
        # Initialize cluster indices
        cluster_indices = [quasi_diag[i:i+1] for i in range(len(quasi_diag))]
        
        # Initialize covariance matrix
        cov_matrix = self.returns.cov()
        
        # Optimize recursively
        while len(cluster_indices) > 1:
            # Identify two closest clusters
            min_distance = np.inf
            for i in range(len(cluster_indices)):
                for j in range(i+1, len(cluster_indices)):
                    current_distance = np.linalg.norm(cov_matrix.iloc[cluster_indices[i], cluster_indices[j]].values)
                    if current_distance < min_distance:
                        min_distance = current_distance
                        min_i, min_j = i, j
            
            # Merge two closest clusters
            cluster_i = cluster_indices.pop(min_i)
            cluster_j = cluster_indices.pop(min_j - 1 if min_j > min_i else min_j)
            merged_cluster = cluster_i.tolist() + cluster_j.tolist()
            cluster_indices.append(merged_cluster)
            
            # Update covariance matrix
            merged_cov = self._get_cluster_variance(merged_cluster, cov_matrix)
            cov_matrix = cov_matrix.copy()
            cov_matrix.loc[merged_cluster, merged_cluster] = merged_cov
        
        # Calculate final weights
        weights = np.ones(len(quasi_diag)) / len(quasi_diag)
        for cluster in reversed(cluster_indices):
            if len(cluster) > 1:
                cluster_var = self._get_cluster_variance(cluster, cov_matrix)
                sub_weights = np.ones(len(cluster)) / len(cluster)
                for i in range(len(cluster)):
                    weights[cluster[i]] *= sub_weights[i]
        
        return weights

class RiskParity:
    def __init__(self, cov_matrix):
        self.cov_matrix = cov_matrix
    
    def optimize(self):
        # Calculate asset volatilities
        volatilities = np.sqrt(np.diag(self.cov_matrix))
        
        # Calculate initial weights (inverse volatility)
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        
        # Optimize for risk parity
        from scipy.optimize import minimize
        
        def risk_parity_objective(weights):
            # Calculate portfolio variance
            portfolio_var = np.dot(weights.T, np.dot(self.cov_matrix, weights))
            
            # Calculate marginal risk contributions
            mrc = np.dot(self.cov_matrix, weights) / portfolio_var
            
            # Calculate asset risk contributions
            rc = weights * mrc
            
            # Calculate risk parity objective
            objective = np.sum((rc - rc.mean())**2)
            
            return objective
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(self.cov_matrix)))
        result = minimize(risk_parity_objective, weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x

if __name__ == "__main__":
    # Create portfolio manager
    portfolio_manager = PortfolioManager()
    
    # Add some example assets
    try:
        portfolio_manager.add_asset("AAPL", 100)
        portfolio_manager.add_asset("MSFT", 50)
        portfolio_manager.add_asset("AMZN", 20)
    except ValueError as e:
        print(str(e))
        exit()
    
    # Initialize risk manager
    risk_manager = RiskManager(portfolio_manager)
    
    # Calculate portfolio risk
    try:
        risk = risk_manager.calculate_portfolio_risk()
        print("\nPortfolio Risk Metrics:")
        print(f"Volatility: {risk['volatility']:.4f}")
        print(f"VaR (95%): {risk['var_95']:.4f}")
        print(f"CVaR (95%): {risk['cvar_95']:.4f}")
    except ValueError as e:
        print(str(e))
        exit()
    
    # Optimize portfolio
    try:
        optimized_weights = risk_manager.optimize_portfolio()
        print("\nOptimized Portfolio Weights:")
        for ticker, weight in optimized_weights.items():
            print(f"{ticker}: {weight:.4f}")
    except ValueError as e:
        print(str(e))
        exit()
    
    # Black-Litterman optimization
    views = [
        ('AAPL', 'up', 0.1),
        ('MSFT', 'down', 0.05)
    ]
    confidences = [0.3, 0.2]
    
    try:
        bl_weights = risk_manager.black_litterman(views, confidences)
        print("\nBlack-Litterman Portfolio Weights:")
        for ticker, weight in bl_weights.items():
            print(f"{ticker}: {weight:.4f}")
    except ValueError as e:
        print(str(e))
        exit()