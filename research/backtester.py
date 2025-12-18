import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import ta
import streamlit as st
from datetime import datetime

class Backtester:
    def __init__(self):
        pass
    
    def prepare_data(self, ticker, start_date, end_date):
        try:
            # Fetch data
            data = yf.download(ticker, start=start_date, end=end_date)
            
            # Calculate features
            data['MA20'] = data['Close'].rolling(20).mean()
            data['MA50'] = data['Close'].rolling(50).mean()
            data['RSI14'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
            data['BB Upper'] = data['MA20'] + 2 * data['Close'].rolling(20).std()
            data['BB Lower'] = data['MA20'] - 2 * data['Close'].rolling(20).std()
            
            # Create target variable
            data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
            
            # Drop NaN values
            data.dropna(inplace=True)
            
            return data
        except Exception as e:
            print(f"Error preparing data: {e}")
            return None
    
    def run_backtest(self, data):
        if data is None or data.empty:
            print("No data to backtest")
            return
        
        # Split data into features and target
        X = data[['MA20', 'MA50', 'RSI14', 'BB Upper', 'BB Lower']]
        y = data['Target']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Initialize and train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred,
            'actual': y_test,
            'model': model,
            'features': X_test
        }

class GridSearch:
    def __init__(self):
        pass
    
    def run_grid_search(self, data):
        if data is None or data.empty:
            print("No data for grid search")
            return
        
        # Split data into features and target
        X = data[['MA20', 'MA50', 'RSI14', 'BB Upper', 'BB Lower']]
        y = data['Target']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Initialize model
        model = RandomForestClassifier(random_state=42)
        
        # Run grid search
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get best parameters and model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        # Evaluate best model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return {
            'best_params': best_params,
            'best_model': best_model,
            'accuracy': accuracy,
            'report': report
        }

class FactorLibrary:
    def __init__(self):
        pass
    
    def calculate_momentum(self, data, window=20):
        return data['Close'].pct_change(window)
    
    def calculate_mean_reversion(self, data, window=20):
        ma = data['Close'].rolling(window).mean()
        return data['Close'] - ma
    
    def calculate_trend_following(self, data, short_window=20, long_window=50):
        ma_short = data['Close'].rolling(short_window).mean()
        ma_long = data['Close'].rolling(long_window).mean()
        return ma_short - ma_long
    
    def calculate_volatility(self, data, window=20):
        return data['Close'].pct_change().rolling(window).std() * np.sqrt(252)
    
    def calculate_volume_trend(self, data, window=20):
        return data['Volume'].rolling(window).mean()

if __name__ == "__main__":
    # Initialize backtester
    backtester = Backtester()
    
    # Prepare data
    data = backtester.prepare_data('AAPL', '2023-01-01', datetime.now().strftime('%Y-%m-%d'))
    
    # Run backtest
    if data is not None:
        results = backtester.run_backtest(data)
        
        if results is not None:
            # Display results
            print(f"Model Accuracy: {results['accuracy']:.4f}")
            print("Classification Report:")
            print(results['report'])
        else:
            print("Backtest returned no results")
        
        # Initialize grid search
        grid_search = GridSearch()
        grid_results = grid_search.run_grid_search(data)
        
        if grid_results is not None:
            print("\nGrid Search Results:")
            print(f"Best Parameters: {grid_results['best_params']}")
            print(f"Best Model Accuracy: {grid_results['accuracy']:.4f}")
            print("Best Model Report:")
            print(grid_results['report'])
        else:
            print("No data for grid search")
        
        # Initialize factor library
        factor_lib = FactorLibrary()
        data['Momentum'] = factor_lib.calculate_momentum(data)
        data['Mean Reversion'] = factor_lib.calculate_mean_reversion(data)
        data['Trend Following'] = factor_lib.calculate_trend_following(data)
        data['Volatility'] = factor_lib.calculate_volatility(data)
        data['Volume Trend'] = factor_lib.calculate_volume_trend(data)
        
        # Display factors
        print("\nSample Factors:")
        print(data[['Momentum', 'Mean Reversion', 'Trend Following', 'Volatility', 'Volume Trend']].tail())