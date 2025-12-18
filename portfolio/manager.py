import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class PortfolioManager:
    def __init__(self):
        # Initialize empty portfolio
        self.portfolio = pd.DataFrame(columns=['Ticker', 'Quantity', 'Purchase Price', 'Purchase Date', 'Allocation'])
        self.portfolio_value = 1000000  # Starting capital
    
    def add_asset(self, ticker, quantity):
        # Fetch current price
        try:
            data = yf.download(ticker, period='1d', interval='1m')
            current_price = data['Close'].iloc[-1]
        except:
            raise ValueError(f"Could not fetch data for {ticker}")
        
        # Calculate total cost
        total_cost = quantity * current_price
        
        # Check if we have enough funds
        if total_cost > self.portfolio_value:
            raise ValueError("Insufficient funds")
        
        # Deduct from available funds
        self.portfolio_value -= total_cost
        
        # Add to portfolio
        self.portfolio = pd.concat([
            self.portfolio,
            pd.DataFrame([{
                'Ticker': ticker,
                'Quantity': quantity,
                'Purchase Price': current_price,
                'Purchase Date': datetime.now(),
                'Allocation': quantity * current_price
            }])
        ], ignore_index=True)
    
    def remove_asset(self, ticker):
        # Find asset in portfolio
        asset = self.portfolio[self.portfolio['Ticker'] == ticker]
        if asset.empty:
            raise ValueError(f"{ticker} not found in portfolio")
        
        # Return funds
        self.portfolio_value += asset['Allocation'].values[0]
        
        # Remove from portfolio
        self.portfolio = self.portfolio[self.portfolio['Ticker'] != ticker].reset_index(drop=True)
    
    def get_portfolio(self):
        # Update current values
        if not self.portfolio.empty:
            for i, row in self.portfolio.iterrows():
                try:
                    data = yf.download(row['Ticker'], period='1d', interval='1m')
                    current_price = data['Close'].iloc[-1]
                    self.portfolio.at[i, 'Current Price'] = current_price
                    self.portfolio.at[i, 'Current Value'] = row['Quantity'] * current_price
                    self.portfolio.at[i, 'Profit/Loss'] = (current_price - row['Purchase Price']) * row['Quantity']
                except:
                    self.portfolio.at[i, 'Current Price'] = np.nan
                    self.portfolio.at[i, 'Current Value'] = np.nan
                    self.portfolio.at[i, 'Profit/Loss'] = np.nan
        
        # Calculate total portfolio value
        total_value = self.portfolio_value + self.portfolio['Current Value'].dropna().sum()
        
        # Calculate allocation percentages
        if not self.portfolio.empty and not self.portfolio['Current Value'].dropna().empty():
            self.portfolio['Allocation %'] = (self.portfolio['Current Value'] / total_value) * 100
        
        # Add cash position
        cash_position = pd.DataFrame([{
            'Ticker': 'CASH',
            'Quantity': 1,
            'Purchase Price': self.portfolio_value,
            'Purchase Date': datetime.now(),
            'Allocation': self.portfolio_value,
            'Current Price': self.portfolio_value,
            'Current Value': self.portfolio_value,
            'Profit/Loss': 0,
            'Allocation %': (self.portfolio_value / total_value) * 100
        }])
        
        # Combine and sort
        full_portfolio = pd.concat([self.portfolio, cash_position], ignore_index=True)
        full_portfolio.sort_values('Allocation %', ascending=False, inplace=True)
        
        return full_portfolio
    
    def stress_test(self, severity=5):
        # Simulate market drop based on severity
        market_drop = severity * 2  # Assume severity 1 = 2% drop
        
        # Copy portfolio for simulation
        simulated_portfolio = self.portfolio.copy()
        
        # Apply stress to each asset
        for i, row in simulated_portfolio.iterrows():
            try:
                data = yf.download(row['Ticker'], period='1d', interval='1m')
                current_price = data['Close'].iloc[-1]
                stressed_price = current_price * (1 - market_drop / 100)
                simulated_portfolio.at[i, 'Stressed Value'] = row['Quantity'] * stressed_price
            except:
                simulated_portfolio.at[i, 'Stressed Value'] = np.nan
        
        # Calculate total value before and after stress
        original_value = self.portfolio_value + self.portfolio['Current Value'].dropna().sum()
        stressed_value = self.portfolio_value + simulated_portfolio['Stressed Value'].dropna().sum()
        
        # Calculate impact
        value_change = stressed_value - original_value
        percent_change = (value_change / original_value) * 100
        
        return {
            'value_change': value_change,
            'percent_change': percent_change
        }
    
    def scenario_analysis(self, scenario):
        scenario_impact = {
            'Market Crash': -15,
            'Interest Rate Hike': -8,
            'Geopolitical Crisis': -12
        }
        
        if scenario not in scenario_impact:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        # Simulate impact based on scenario
        impact = scenario_impact[scenario]
        
        # Copy portfolio for simulation
        simulated_portfolio = self.portfolio.copy()
        
        # Apply scenario to each asset
        for i, row in simulated_portfolio.iterrows():
            try:
                data = yf.download(row['Ticker'], period='1d', interval='1m')
                current_price = data['Close'].iloc[-1]
                scenario_price = current_price * (1 + impact / 100)
                simulated_portfolio.at[i, 'Scenario Value'] = row['Quantity'] * scenario_price
            except:
                simulated_portfolio.at[i, 'Scenario Value'] = np.nan
        
        # Calculate total value before and after scenario
        original_value = self.portfolio_value + self.portfolio['Current Value'].dropna().sum()
        scenario_value = self.portfolio_value + simulated_portfolio['Scenario Value'].dropna().sum()
        
        # Calculate impact
        value_change = scenario_value - original_value
        percent_change = (value_change / original_value) * 100
        
        return {
            'value_change': value_change,
            'percent_change': percent_change
        }
    
    def monte_carlo_simulation(self, simulations=1000):
        # Prepare simulation results
        results = []
        
        # Simulate portfolio performance
        for _ in range(simulations):
            simulated_portfolio = self.get_portfolio()
            simulated_value = self.portfolio_value + simulated_portfolio['Current Value'].dropna().sum()
            
            # Apply random market movement
            market_change = np.random.normal(0, 0.05)
            final_value = simulated_value * (1 + market_change)
            results.append(final_value)
        
        return np.array(results)