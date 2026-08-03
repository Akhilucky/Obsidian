import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
from pathlib import Path

SIGNALS_DIR = Path(__file__).parent.parent / "signals"


class SignalGenerator:
    def __init__(self):
        self.signals_dir = SIGNALS_DIR
        self.signals_dir.mkdir(exist_ok=True)

    def fetch_features(self, ticker, start_date=None, end_date=None):
        data = yf.download(ticker, start=start_date, end=end_date,
                           progress=False, auto_adjust=False)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [c.lower() for c in data.columns]

        close = data["close"]
        data["ma20"] = close.rolling(20).mean()
        data["ma50"] = close.rolling(50).mean()
        data["rsi14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        data["bb_upper"] = bb.bollinger_hband()
        data["bb_lower"] = bb.bollinger_lband()
        return data.dropna()

    def generate_signal(self, data):
        if data is None or data.empty or len(data) < 50:
            print("No data to generate signal")
            return None

        feature_cols = ['ma20', 'ma50', 'rsi14', 'bb_upper', 'bb_lower']
        X = data[feature_cols]
        y = np.where(data['close'].shift(-1) > data['close'], 1, 0)

        train_size = int(0.8 * len(X))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        latest_features = X.iloc[-1].values.reshape(1, -1)
        signal = model.predict(latest_features)[0]
        proba = model.predict_proba(latest_features)[0]

        return {
            'signal': int(signal),
            'confidence': float(max(proba)),
            'accuracy': float(accuracy),
            'predictions': y_pred,
            'actual': y_test
        }

    def run(self, ticker):
        data = self.fetch_features(ticker)
        if data is not None and not data.empty:
            signal = self.generate_signal(data)
            if signal is None:
                return None

            signal_df = pd.DataFrame([{
                'ticker': ticker,
                'date': datetime.now().date(),
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'model_accuracy': signal['accuracy']
            }])
            signal_df.set_index('date', inplace=True)

            path = self.signals_dir / f"signals_{ticker.replace('.', '_').replace('^', 'IDX_')}.parquet"
            signal_df.to_parquet(path)
            return signal
        return None


if __name__ == "__main__":
    generator = SignalGenerator()
    signal = generator.run('AAPL')

    if signal:
        print(f"Signal for AAPL: {signal['signal']}")
        print(f"Model Accuracy: {signal['accuracy']:.4f}")
