import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
from data.openbb_integration import OpenBBIntegration

class DataIngestor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.openbb = OpenBBIntegration()
    
    def fetch_stock_data(self, ticker, start_date, end_date):
        """Fetch stock data using OpenBB."""
        try:
            data = self.openbb.fetch_stock_data(ticker, start_date, end_date)
            return data
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None
    
    def save_data(self, data, ticker, source='openbb'):
        if data is not None and not data.empty:
            # Clean ticker name for file system
            clean_ticker = ticker.replace(' ', '_').split()[0]
            file_path = os.path.join(self.data_dir, f"{source}_{clean_ticker}.parquet")
            data.to_parquet(file_path)
            print(f"Data saved to {file_path}")
    
    def run(self, tickers, start_date, end_date):
        """Fetch and save data for multiple tickers."""
        for ticker in tickers:
            print(f"\nFetching data for {ticker}...")
            data = self.fetch_stock_data(ticker, start_date, end_date)
            self.save_data(data, ticker, 'openbb')

if __name__ == "__main__":
    ingestor = DataIngestor()
    tickers = ['AAPL US Equity', 'MSFT US Equity', 'AMZN US Equity', 'META US Equity', 'TSLA US Equity']
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    ingestor.run(tickers, start_date, end_date)