import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

try:
    from openbb import obb
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False
    print("OpenBB not available. Install with: pip install openbb")

class OpenBBIntegration:
    """Integration with OpenBB for financial data and analytics."""
    
    def __init__(self):
        self.obb = obb if OPENBB_AVAILABLE else None
    
    def fetch_stock_data(self, ticker, start_date=None, end_date=None, interval='1d'):
        """Fetch stock data using OpenBB or fallback to yfinance."""
        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Extract base ticker if Bloomberg format
            base_ticker = ticker.split()[0] if ' ' in ticker else ticker
            
            # Try OpenBB first
            if OPENBB_AVAILABLE and self.obb is not None:
                try:
                    data = self.obb.equity.price.historical(
                        symbol=base_ticker,
                        start_date=start_date,
                        end_date=end_date,
                        interval=interval
                    )
                    if data is not None:
                        result = data.to_pandas() if hasattr(data, 'to_pandas') else data
                        if result is not None and not result.empty:
                            return result
                except Exception as e:
                    print(f"OpenBB fetch failed: {e}, trying yfinance...")
            
            # Fallback to yfinance
            return yf.download(base_ticker, start=start_date, end=end_date)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def get_quote(self, ticker):
        """Get real-time quote for a ticker."""
        try:
            base_ticker = ticker.split()[0] if ' ' in ticker else ticker
            
            if OPENBB_AVAILABLE and self.obb is not None:
                try:
                    quote = self.obb.equity.quote(symbol=base_ticker)
                    if quote is not None:
                        return quote.to_pandas() if hasattr(quote, 'to_pandas') else quote
                except:
                    pass
            
            # Fallback using yfinance
            ticker_obj = yf.Ticker(base_ticker)
            return ticker_obj.info
        except Exception as e:
            print(f"Error fetching quote: {e}")
            return None
    
    def get_fundamentals(self, ticker):
        """Get fundamental data for a ticker."""
        try:
            base_ticker = ticker.split()[0] if ' ' in ticker else ticker
            data = {}
            
            ticker_obj = yf.Ticker(base_ticker)
            
            # Fetch various fundamental metrics from yfinance
            try:
                data['info'] = ticker_obj.info
            except:
                pass
            
            try:
                data['income'] = ticker_obj.quarterly_income_stmt
            except:
                pass
            
            try:
                data['balance'] = ticker_obj.quarterly_balance_sheet
            except:
                pass
            
            return data
        except Exception as e:
            print(f"Error fetching fundamentals: {e}")
            return None
    
    def get_technical_indicators(self, ticker, indicator='sma', window=20):
        """Get technical indicators for a ticker using ta library."""
        try:
            import ta
            
            base_ticker = ticker.split()[0] if ' ' in ticker else ticker
            
            # Fetch price data
            data = yf.download(base_ticker, period='1y')
            
            if indicator.lower() == 'sma':
                return ta.trend.sma_indicator(data['Close'], window=window)
            elif indicator.lower() == 'ema':
                return ta.trend.ema_indicator(data['Close'], window=window)
            elif indicator.lower() == 'rsi':
                return ta.momentum.rsi(data['Close'], window=window)
            elif indicator.lower() == 'macd':
                return ta.trend.MACD(data['Close']).macd()
            else:
                return None
        except Exception as e:
            print(f"Error fetching technical indicator: {e}")
            return None
    
    def get_news(self, ticker, limit=10):
        """Get latest news for a ticker."""
        try:
            base_ticker = ticker.split()[0] if ' ' in ticker else ticker
            
            # Try OpenBB first
            if OPENBB_AVAILABLE and self.obb is not None:
                try:
                    news = self.obb.news(query=base_ticker, limit=limit)
                    if news is not None:
                        return news.to_pandas() if hasattr(news, 'to_pandas') else news
                except:
                    pass
            
            # Fallback using yfinance
            ticker_obj = yf.Ticker(base_ticker)
            return pd.DataFrame(ticker_obj.news[:limit]) if ticker_obj.news else None
        except Exception as e:
            print(f"Error fetching news: {e}")
            return None

if __name__ == "__main__":
    print(f"OpenBB available: {OPENBB_AVAILABLE}")
    print("OpenBB integration module loaded successfully")
    integrator = OpenBBIntegration()
    
    # Test: Fetch Apple stock data
    print("\nFetching Apple stock data...")
    data = integrator.fetch_stock_data('AAPL', 
                                      start_date='2024-01-01', 
                                      end_date='2024-12-31')
    if data is not None:
        print(f"Data shape: {data.shape}")
        print(data.head())
    else:
        print("Failed to fetch data")

