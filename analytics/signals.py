import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
from features.store import FeatureStore

class SignalGenerator:
    def __init__(self):
        self.feature_store = FeatureStore()
    
    def fetch_features(self, ticker, start_date=None, end_date=None):
        return self.feature_store.get_features(ticker, start_date, end_date)
    
    def generate_signal(self, data):
        if data is None or data.empty:
            print("No data to generate signal")
            return None
        
        # Prepare features and target
        X = data[['ma20', 'ma50', 'rsi14', 'bb_upper', 'bb_lower']]
        y = np.where(data['close'].shift(-1) > data['close'], 1, 0)
        
        # Split data into training and testing sets
        train_size = int(0.8 * len(X))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Initialize and train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate signal for latest data
        latest_features = X.iloc[-1].values.reshape(1, -1)
        signal = model.predict(latest_features)[0]
        
        return {
            'signal': signal,
            'accuracy': accuracy,
            'predictions': y_pred,
            'actual': y_test
        }
    
    def run(self, ticker):
        # Fetch features
        data = self.fetch_features(ticker)
        
        if data is not None and not data.empty:
            # Generate signal
            signal = self.generate_signal(data)
            
            # Save signal to feature store
            signal_df = pd.DataFrame([{
                'ticker': ticker,
                'date': datetime.now().date(),
                'signal': signal['signal'],
                'model_accuracy': signal['accuracy']
            }])
            signal_df.set_index('date', inplace=True)
            self.feature_store.ingest_dataframe(signal_df, 'signals')
            
            return signal
        else:
            return None

if __name__ == "__main__":
    generator = SignalGenerator()
    signal = generator.run('AAPL US Equity')
    
    if signal:
        print(f"Signal for AAPL US Equity: {signal['signal']}")
        print(f"Model Accuracy: {signal['accuracy']:.4f}")