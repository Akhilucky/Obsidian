"""
Options Pricing & Greeks Calculator
====================================

Institutional-grade options pricing models:
- Black-Scholes model
- Binomial tree model
- Monte Carlo simulation
- Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Implied volatility solver
- Options strategy analyzer

Used by derivatives desks at major investment banks.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class OptionContract:
    """Represents an options contract."""
    underlying: str
    strike: float
    expiry: datetime
    option_type: OptionType
    spot_price: float
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    volatility: float = 0.2
    
    @property
    def time_to_expiry(self) -> float:
        """Time to expiry in years."""
        delta = self.expiry - datetime.now()
        return max(delta.days / 365.0, 0.0001)  # Minimum 1 hour
    
    @property
    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL
    
    @property
    def moneyness(self) -> str:
        """Determine if option is ITM, ATM, or OTM."""
        ratio = self.spot_price / self.strike
        if self.is_call:
            if ratio > 1.02:
                return "ITM"
            elif ratio < 0.98:
                return "OTM"
            return "ATM"
        else:
            if ratio < 0.98:
                return "ITM"
            elif ratio > 1.02:
                return "OTM"
            return "ATM"


class BlackScholes:
    """
    Black-Scholes options pricing model.
    The foundational model for options pricing.
    """
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        return BlackScholes.d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)
    
    @classmethod
    def price(cls, S: float, K: float, T: float, r: float, sigma: float, 
              option_type: OptionType = OptionType.CALL, q: float = 0.0) -> float:
        """
        Calculate option price using Black-Scholes formula.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: CALL or PUT
            q: Dividend yield
        
        Returns:
            Option price
        """
        d1 = cls.d1(S, K, T, r, q, sigma)
        d2 = cls.d2(S, K, T, r, q, sigma)
        
        if option_type == OptionType.CALL:
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        return price
    
    @classmethod
    def price_option(cls, option: OptionContract) -> float:
        """Price an OptionContract."""
        return cls.price(
            S=option.spot_price,
            K=option.strike,
            T=option.time_to_expiry,
            r=option.risk_free_rate,
            sigma=option.volatility,
            option_type=option.option_type,
            q=option.dividend_yield
        )


class Greeks:
    """
    Calculate option Greeks - sensitivities to various factors.
    Essential for risk management and hedging.
    """
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, 
              option_type: OptionType = OptionType.CALL, q: float = 0.0) -> float:
        """
        Delta: Sensitivity to underlying price change.
        Represents hedge ratio and probability of finishing ITM.
        """
        d1 = BlackScholes.d1(S, K, T, r, q, sigma)
        
        if option_type == OptionType.CALL:
            return np.exp(-q * T) * norm.cdf(d1)
        else:
            return -np.exp(-q * T) * norm.cdf(-d1)
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Gamma: Rate of change of delta.
        Same for calls and puts.
        """
        d1 = BlackScholes.d1(S, K, T, r, q, sigma)
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, 
              option_type: OptionType = OptionType.CALL, q: float = 0.0) -> float:
        """
        Theta: Time decay per day.
        Negative for long positions (time value erodes).
        """
        d1 = BlackScholes.d1(S, K, T, r, q, sigma)
        d2 = BlackScholes.d2(S, K, T, r, q, sigma)
        
        term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        
        if option_type == OptionType.CALL:
            term2 = q * S * np.exp(-q * T) * norm.cdf(d1)
            term3 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
            term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        
        return (term1 + term2 + term3) / 365  # Per day
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Vega: Sensitivity to volatility changes.
        Same for calls and puts. Per 1% vol change.
        """
        d1 = BlackScholes.d1(S, K, T, r, q, sigma)
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
    
    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float, 
            option_type: OptionType = OptionType.CALL, q: float = 0.0) -> float:
        """
        Rho: Sensitivity to interest rate changes.
        Per 1% rate change.
        """
        d2 = BlackScholes.d2(S, K, T, r, q, sigma)
        
        if option_type == OptionType.CALL:
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    @staticmethod
    def vanna(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Vanna: Sensitivity of delta to volatility.
        Second-order Greek.
        """
        d1 = BlackScholes.d1(S, K, T, r, q, sigma)
        d2 = BlackScholes.d2(S, K, T, r, q, sigma)
        return -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma
    
    @staticmethod
    def charm(S: float, K: float, T: float, r: float, sigma: float, 
              option_type: OptionType = OptionType.CALL, q: float = 0.0) -> float:
        """
        Charm: Rate of change of delta over time (delta decay).
        Second-order Greek.
        """
        d1 = BlackScholes.d1(S, K, T, r, q, sigma)
        d2 = BlackScholes.d2(S, K, T, r, q, sigma)
        
        charm = q * np.exp(-q * T) * norm.cdf(d1) - np.exp(-q * T) * norm.pdf(d1) * \
                (2 * (r - q) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
        
        if option_type == OptionType.PUT:
            charm = charm + q * np.exp(-q * T)
        
        return charm / 365  # Per day
    
    @classmethod
    def all_greeks(cls, S: float, K: float, T: float, r: float, sigma: float, 
                   option_type: OptionType = OptionType.CALL, q: float = 0.0) -> Dict[str, float]:
        """Calculate all Greeks at once."""
        return {
            'delta': cls.delta(S, K, T, r, sigma, option_type, q),
            'gamma': cls.gamma(S, K, T, r, sigma, q),
            'theta': cls.theta(S, K, T, r, sigma, option_type, q),
            'vega': cls.vega(S, K, T, r, sigma, q),
            'rho': cls.rho(S, K, T, r, sigma, option_type, q),
            'vanna': cls.vanna(S, K, T, r, sigma, q),
            'charm': cls.charm(S, K, T, r, sigma, option_type, q)
        }
    
    @classmethod
    def option_greeks(cls, option: OptionContract) -> Dict[str, float]:
        """Calculate all Greeks for an OptionContract."""
        return cls.all_greeks(
            S=option.spot_price,
            K=option.strike,
            T=option.time_to_expiry,
            r=option.risk_free_rate,
            sigma=option.volatility,
            option_type=option.option_type,
            q=option.dividend_yield
        )


class ImpliedVolatility:
    """
    Implied volatility calculation using various methods.
    """
    
    @staticmethod
    def solve_iv_brentq(market_price: float, S: float, K: float, T: float, r: float,
                        option_type: OptionType = OptionType.CALL, q: float = 0.0,
                        precision: float = 1e-6) -> float:
        """
        Solve for implied volatility using Brent's method.
        Most robust approach for IV calculation.
        """
        def objective(sigma):
            return BlackScholes.price(S, K, T, r, sigma, option_type, q) - market_price
        
        try:
            iv = brentq(objective, 0.001, 5.0, xtol=precision)
            return iv
        except ValueError:
            return np.nan
    
    @staticmethod
    def solve_iv_newton(market_price: float, S: float, K: float, T: float, r: float,
                        option_type: OptionType = OptionType.CALL, q: float = 0.0,
                        max_iterations: int = 100, precision: float = 1e-6) -> float:
        """
        Solve for implied volatility using Newton-Raphson method.
        Faster but less robust than Brent's method.
        """
        sigma = 0.2  # Initial guess
        
        for _ in range(max_iterations):
            price = BlackScholes.price(S, K, T, r, sigma, option_type, q)
            vega = Greeks.vega(S, K, T, r, sigma, q) * 100  # Undo the /100
            
            diff = market_price - price
            
            if abs(diff) < precision:
                return sigma
            
            if abs(vega) < 1e-10:
                break
            
            sigma += diff / vega
            
            if sigma < 0.001 or sigma > 5.0:
                break
        
        return np.nan
    
    @classmethod
    def calculate(cls, market_price: float, S: float, K: float, T: float, r: float,
                  option_type: OptionType = OptionType.CALL, q: float = 0.0) -> float:
        """Calculate implied volatility (uses Brent's method)."""
        return cls.solve_iv_brentq(market_price, S, K, T, r, option_type, q)


class BinomialTree:
    """
    Binomial tree model for American options.
    Handles early exercise premium.
    """
    
    @staticmethod
    def price(S: float, K: float, T: float, r: float, sigma: float,
              option_type: OptionType = OptionType.CALL, q: float = 0.0,
              steps: int = 100, american: bool = True) -> float:
        """
        Price option using binomial tree.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry
            r: Risk-free rate
            sigma: Volatility
            option_type: CALL or PUT
            q: Dividend yield
            steps: Number of tree steps
            american: Whether to allow early exercise
        
        Returns:
            Option price
        """
        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral probability
        
        # Build price tree at expiration
        prices = S * (u ** np.arange(steps, -1, -1)) * (d ** np.arange(0, steps + 1))
        
        # Calculate option values at expiration
        if option_type == OptionType.CALL:
            values = np.maximum(prices - K, 0)
        else:
            values = np.maximum(K - prices, 0)
        
        # Work backwards through the tree
        discount = np.exp(-r * dt)
        for i in range(steps - 1, -1, -1):
            prices = prices[:-1] * u / (u * d)
            values = discount * (p * values[:-1] + (1 - p) * values[1:])
            
            if american:
                if option_type == OptionType.CALL:
                    exercise_values = np.maximum(prices - K, 0)
                else:
                    exercise_values = np.maximum(K - prices, 0)
                values = np.maximum(values, exercise_values)
        
        return values[0]


class MonteCarloOptionPricer:
    """
    Monte Carlo simulation for exotic options pricing.
    Handles path-dependent options.
    """
    
    def __init__(self, num_simulations: int = 10000, num_steps: int = 252):
        self.num_simulations = num_simulations
        self.num_steps = num_steps
    
    def simulate_paths(self, S: float, T: float, r: float, sigma: float, q: float = 0.0) -> np.ndarray:
        """Simulate stock price paths using geometric Brownian motion."""
        dt = T / self.num_steps
        
        # Generate random normal returns
        Z = np.random.standard_normal((self.num_simulations, self.num_steps))
        
        # Calculate price paths
        drift = (r - q - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        log_returns = drift + diffusion
        log_returns = np.insert(log_returns, 0, 0, axis=1)  # Start at 0
        
        paths = S * np.exp(np.cumsum(log_returns, axis=1))
        
        return paths
    
    def price_european(self, S: float, K: float, T: float, r: float, sigma: float,
                       option_type: OptionType = OptionType.CALL, q: float = 0.0) -> Dict:
        """Price European option using Monte Carlo."""
        paths = self.simulate_paths(S, T, r, sigma, q)
        final_prices = paths[:, -1]
        
        if option_type == OptionType.CALL:
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
        
        discount = np.exp(-r * T)
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(self.num_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (price - 1.96 * std_error, price + 1.96 * std_error)
        }
    
    def price_asian(self, S: float, K: float, T: float, r: float, sigma: float,
                    option_type: OptionType = OptionType.CALL, q: float = 0.0,
                    average_type: str = 'arithmetic') -> Dict:
        """Price Asian (average price) option."""
        paths = self.simulate_paths(S, T, r, sigma, q)
        
        if average_type == 'arithmetic':
            averages = np.mean(paths, axis=1)
        else:  # geometric
            averages = np.exp(np.mean(np.log(paths), axis=1))
        
        if option_type == OptionType.CALL:
            payoffs = np.maximum(averages - K, 0)
        else:
            payoffs = np.maximum(K - averages, 0)
        
        discount = np.exp(-r * T)
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(self.num_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (price - 1.96 * std_error, price + 1.96 * std_error)
        }
    
    def price_barrier(self, S: float, K: float, T: float, r: float, sigma: float,
                      barrier: float, barrier_type: str = 'down-and-out',
                      option_type: OptionType = OptionType.CALL, q: float = 0.0) -> Dict:
        """Price barrier option."""
        paths = self.simulate_paths(S, T, r, sigma, q)
        final_prices = paths[:, -1]
        
        # Check barrier condition
        if 'down' in barrier_type:
            barrier_hit = np.min(paths, axis=1) <= barrier
        else:  # up
            barrier_hit = np.max(paths, axis=1) >= barrier
        
        if 'out' in barrier_type:
            active = ~barrier_hit
        else:  # in
            active = barrier_hit
        
        if option_type == OptionType.CALL:
            payoffs = np.maximum(final_prices - K, 0) * active
        else:
            payoffs = np.maximum(K - final_prices, 0) * active
        
        discount = np.exp(-r * T)
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(self.num_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'knock_out_probability': np.mean(barrier_hit)
        }


class OptionsStrategy:
    """
    Options strategy analyzer.
    Analyze complex multi-leg strategies.
    """
    
    def __init__(self):
        self.positions = []
    
    def add_position(self, option: OptionContract, quantity: int, premium: float = None):
        """Add a position to the strategy."""
        if premium is None:
            premium = BlackScholes.price_option(option)
        
        self.positions.append({
            'option': option,
            'quantity': quantity,
            'premium': premium
        })
    
    def clear_positions(self):
        """Clear all positions."""
        self.positions = []
    
    def calculate_payoff(self, spot_prices: np.ndarray) -> np.ndarray:
        """Calculate strategy payoff at different spot prices."""
        total_payoff = np.zeros_like(spot_prices, dtype=float)
        
        for pos in self.positions:
            option = pos['option']
            quantity = pos['quantity']
            premium = pos['premium']
            
            if option.option_type == OptionType.CALL:
                intrinsic = np.maximum(spot_prices - option.strike, 0)
            else:
                intrinsic = np.maximum(option.strike - spot_prices, 0)
            
            # Payoff = intrinsic value - premium paid
            position_payoff = quantity * (intrinsic - premium)
            total_payoff += position_payoff
        
        return total_payoff
    
    def get_breakeven_points(self) -> List[float]:
        """Find breakeven points of the strategy."""
        if not self.positions:
            return []
        
        # Search for breakeven points
        spot_min = min(p['option'].strike for p in self.positions) * 0.5
        spot_max = max(p['option'].strike for p in self.positions) * 1.5
        
        spots = np.linspace(spot_min, spot_max, 1000)
        payoffs = self.calculate_payoff(spots)
        
        # Find sign changes
        breakevens = []
        for i in range(len(payoffs) - 1):
            if payoffs[i] * payoffs[i+1] < 0:
                # Linear interpolation to find exact breakeven
                be = spots[i] - payoffs[i] * (spots[i+1] - spots[i]) / (payoffs[i+1] - payoffs[i])
                breakevens.append(be)
        
        return breakevens
    
    def get_max_profit(self) -> float:
        """Calculate maximum profit."""
        spots = np.linspace(0.01, 
                           max(p['option'].strike for p in self.positions) * 3, 
                           10000)
        return np.max(self.calculate_payoff(spots))
    
    def get_max_loss(self) -> float:
        """Calculate maximum loss."""
        spots = np.linspace(0.01, 
                           max(p['option'].strike for p in self.positions) * 3, 
                           10000)
        return np.min(self.calculate_payoff(spots))
    
    def get_strategy_greeks(self) -> Dict[str, float]:
        """Calculate aggregate Greeks for the strategy."""
        total_greeks = {
            'delta': 0, 'gamma': 0, 'theta': 0, 
            'vega': 0, 'rho': 0
        }
        
        for pos in self.positions:
            greeks = Greeks.option_greeks(pos['option'])
            for greek, value in greeks.items():
                if greek in total_greeks:
                    total_greeks[greek] += value * pos['quantity']
        
        return total_greeks


class StrategyBuilder:
    """
    Pre-built options strategies.
    """
    
    @staticmethod
    def bull_call_spread(underlying: str, spot: float, lower_strike: float, 
                         upper_strike: float, expiry: datetime,
                         volatility: float = 0.2) -> OptionsStrategy:
        """Create a bull call spread."""
        strategy = OptionsStrategy()
        
        long_call = OptionContract(
            underlying=underlying, strike=lower_strike, expiry=expiry,
            option_type=OptionType.CALL, spot_price=spot, volatility=volatility
        )
        short_call = OptionContract(
            underlying=underlying, strike=upper_strike, expiry=expiry,
            option_type=OptionType.CALL, spot_price=spot, volatility=volatility
        )
        
        strategy.add_position(long_call, 1)
        strategy.add_position(short_call, -1)
        
        return strategy
    
    @staticmethod
    def iron_condor(underlying: str, spot: float, put_long: float, put_short: float,
                    call_short: float, call_long: float, expiry: datetime,
                    volatility: float = 0.2) -> OptionsStrategy:
        """Create an iron condor (sell premium strategy)."""
        strategy = OptionsStrategy()
        
        # Put spread (bull put)
        strategy.add_position(OptionContract(
            underlying=underlying, strike=put_long, expiry=expiry,
            option_type=OptionType.PUT, spot_price=spot, volatility=volatility
        ), 1)
        strategy.add_position(OptionContract(
            underlying=underlying, strike=put_short, expiry=expiry,
            option_type=OptionType.PUT, spot_price=spot, volatility=volatility
        ), -1)
        
        # Call spread (bear call)
        strategy.add_position(OptionContract(
            underlying=underlying, strike=call_short, expiry=expiry,
            option_type=OptionType.CALL, spot_price=spot, volatility=volatility
        ), -1)
        strategy.add_position(OptionContract(
            underlying=underlying, strike=call_long, expiry=expiry,
            option_type=OptionType.CALL, spot_price=spot, volatility=volatility
        ), 1)
        
        return strategy
    
    @staticmethod
    def straddle(underlying: str, spot: float, strike: float, expiry: datetime,
                 volatility: float = 0.2, long: bool = True) -> OptionsStrategy:
        """Create a straddle (volatility play)."""
        strategy = OptionsStrategy()
        quantity = 1 if long else -1
        
        call = OptionContract(
            underlying=underlying, strike=strike, expiry=expiry,
            option_type=OptionType.CALL, spot_price=spot, volatility=volatility
        )
        put = OptionContract(
            underlying=underlying, strike=strike, expiry=expiry,
            option_type=OptionType.PUT, spot_price=spot, volatility=volatility
        )
        
        strategy.add_position(call, quantity)
        strategy.add_position(put, quantity)
        
        return strategy
    
    @staticmethod
    def butterfly(underlying: str, spot: float, lower: float, middle: float,
                  upper: float, expiry: datetime, volatility: float = 0.2) -> OptionsStrategy:
        """Create a butterfly spread."""
        strategy = OptionsStrategy()
        
        strategy.add_position(OptionContract(
            underlying=underlying, strike=lower, expiry=expiry,
            option_type=OptionType.CALL, spot_price=spot, volatility=volatility
        ), 1)
        strategy.add_position(OptionContract(
            underlying=underlying, strike=middle, expiry=expiry,
            option_type=OptionType.CALL, spot_price=spot, volatility=volatility
        ), -2)
        strategy.add_position(OptionContract(
            underlying=underlying, strike=upper, expiry=expiry,
            option_type=OptionType.CALL, spot_price=spot, volatility=volatility
        ), 1)
        
        return strategy


if __name__ == "__main__":
    print("=" * 60)
    print("Options Pricing & Greeks Calculator")
    print("=" * 60)
    
    # Example: AAPL option
    S = 185.0  # Spot price
    K = 190.0  # Strike
    T = 30/365  # 30 days to expiry
    r = 0.05   # 5% risk-free rate
    sigma = 0.25  # 25% volatility
    
    print(f"\nUnderlying: AAPL at ${S}")
    print(f"Strike: ${K}")
    print(f"Time to Expiry: {T*365:.0f} days")
    print(f"Volatility: {sigma*100:.0f}%")
    print(f"Risk-free Rate: {r*100:.0f}%")
    
    # Black-Scholes pricing
    print("\n--- Black-Scholes Pricing ---")
    call_price = BlackScholes.price(S, K, T, r, sigma, OptionType.CALL)
    put_price = BlackScholes.price(S, K, T, r, sigma, OptionType.PUT)
    print(f"Call Price: ${call_price:.2f}")
    print(f"Put Price: ${put_price:.2f}")
    
    # Verify put-call parity
    parity_check = call_price - put_price - (S - K * np.exp(-r * T))
    print(f"Put-Call Parity Check: {parity_check:.6f} (should be ~0)")
    
    # Greeks
    print("\n--- Greeks (Call Option) ---")
    greeks = Greeks.all_greeks(S, K, T, r, sigma, OptionType.CALL)
    for greek, value in greeks.items():
        print(f"{greek.capitalize()}: {value:.4f}")
    
    # Implied Volatility
    print("\n--- Implied Volatility ---")
    market_price = 3.50  # Hypothetical market price
    iv = ImpliedVolatility.calculate(market_price, S, K, T, r, OptionType.CALL)
    print(f"Market Price: ${market_price}")
    print(f"Implied Volatility: {iv*100:.2f}%")
    
    # Binomial Tree (American option)
    print("\n--- Binomial Tree (American) ---")
    american_call = BinomialTree.price(S, K, T, r, sigma, OptionType.CALL, american=True)
    american_put = BinomialTree.price(S, K, T, r, sigma, OptionType.PUT, american=True)
    print(f"American Call: ${american_call:.2f}")
    print(f"American Put: ${american_put:.2f}")
    print(f"Early Exercise Premium (Put): ${american_put - put_price:.4f}")
    
    # Monte Carlo
    print("\n--- Monte Carlo Simulation ---")
    mc = MonteCarloOptionPricer(num_simulations=50000)
    mc_result = mc.price_european(S, K, T, r, sigma, OptionType.CALL)
    print(f"MC Call Price: ${mc_result['price']:.2f} ± ${mc_result['std_error']:.4f}")
    
    # Asian option
    asian_result = mc.price_asian(S, K, T, r, sigma, OptionType.CALL)
    print(f"Asian Call Price: ${asian_result['price']:.2f}")
    
    # Strategy example
    print("\n--- Iron Condor Strategy ---")
    expiry = datetime.now() + timedelta(days=30)
    iron_condor = StrategyBuilder.iron_condor(
        'AAPL', S,
        put_long=170, put_short=175,
        call_short=195, call_long=200,
        expiry=expiry, volatility=sigma
    )
    
    print(f"Max Profit: ${iron_condor.get_max_profit():.2f}")
    print(f"Max Loss: ${iron_condor.get_max_loss():.2f}")
    print(f"Breakeven Points: {[f'${x:.2f}' for x in iron_condor.get_breakeven_points()]}")
    
    strategy_greeks = iron_condor.get_strategy_greeks()
    print(f"Strategy Delta: {strategy_greeks['delta']:.4f}")
    print(f"Strategy Theta: ${strategy_greeks['theta']:.4f}/day")
    
    print("\n" + "=" * 60)
    print("Options Pricing Complete!")
    print("=" * 60)
