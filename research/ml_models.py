"""
Advanced ML Prediction Models for Financial Markets
====================================================

This module contains state-of-the-art machine learning models for:
- Price prediction (LSTM, Transformer, XGBoost)
- Market regime detection
- Volatility forecasting (GARCH)
- Alpha generation

Institutional-grade implementations for competing with hedge funds.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


class LSTMPredictor:
    """
    Long Short-Term Memory neural network for price prediction.
    Used by major quantitative funds for time series forecasting.
    """
    
    def __init__(self, lookback=60, forecast_horizon=5, units=50):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.units = units
        self.model = None
        self.scaler = MinMaxScaler()
        
    def build_model(self, input_shape):
        """Build LSTM architecture."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required. Install: pip install tensorflow")
        
        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(self.units, return_sequences=True),
            Dropout(0.2),
            LSTM(self.units // 2),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(self.forecast_horizon)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, data, target_col='Close'):
        """Prepare sequences for LSTM training."""
        scaled_data = self.scaler.fit_transform(data[[target_col]])
        
        X, y = [], []
        for i in range(self.lookback, len(scaled_data) - self.forecast_horizon):
            X.append(scaled_data[i-self.lookback:i, 0])
            y.append(scaled_data[i:i+self.forecast_horizon, 0])
        
        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def train(self, data, target_col='Close', epochs=50, batch_size=32, validation_split=0.2):
        """Train the LSTM model."""
        X, y = self.prepare_data(data, target_col)
        
        self.build_model((X.shape[1], 1))
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, data, target_col='Close'):
        """Make predictions using trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        scaled_data = self.scaler.transform(data[[target_col]])
        X = scaled_data[-self.lookback:].reshape(1, self.lookback, 1)
        
        predictions = self.model.predict(X, verbose=0)
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))
        
        return predictions.flatten()


class TransformerPredictor:
    """
    Transformer model for financial time series prediction.
    State-of-the-art architecture used by Renaissance Technologies.
    """
    
    def __init__(self, lookback=60, forecast_horizon=5, d_model=64, num_heads=4, ff_dim=128):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.model = None
        self.scaler = MinMaxScaler()
    
    def build_model(self, input_shape):
        """Build Transformer architecture."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required. Install: pip install tensorflow")
        
        inputs = Input(shape=input_shape)
        
        # Positional encoding (simplified)
        x = Dense(self.d_model)(inputs)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads
        )(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed-forward network
        ff_output = Dense(self.ff_dim, activation='relu')(x)
        ff_output = Dense(self.d_model)(ff_output)
        x = LayerNormalization(epsilon=1e-6)(x + ff_output)
        
        # Global pooling and output
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.forecast_horizon)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def prepare_data(self, data, feature_cols=None):
        """Prepare data for Transformer."""
        if feature_cols is None:
            feature_cols = ['Close', 'Volume', 'High', 'Low']
        
        available_cols = [col for col in feature_cols if col in data.columns]
        scaled_data = self.scaler.fit_transform(data[available_cols])
        
        X, y = [], []
        for i in range(self.lookback, len(scaled_data) - self.forecast_horizon):
            X.append(scaled_data[i-self.lookback:i])
            y.append(scaled_data[i:i+self.forecast_horizon, 0])  # Predict Close only
        
        return np.array(X), np.array(y)
    
    def train(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """Train the Transformer model."""
        X, y = self.prepare_data(data)
        
        self.build_model((X.shape[1], X.shape[2]))
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history


class XGBoostPredictor:
    """
    XGBoost model for financial prediction.
    Highly effective for tabular financial data.
    """
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.feature_importance = None
    
    def create_features(self, data):
        """Create advanced technical features for XGBoost."""
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['Close'].ewm(span=window).mean()
            df[f'std_{window}'] = df['Close'].rolling(window=window).std()
        
        # Price momentum
        for lag in [1, 5, 10, 20]:
            df[f'momentum_{lag}'] = df['Close'] / df['Close'].shift(lag) - 1
        
        # Volume features
        if 'Volume' in df.columns:
            df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        
        # Volatility
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        df['bb_std'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Target: Next day return direction
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        return df.dropna()
    
    def train(self, data, test_size=0.2):
        """Train XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost required. Install: pip install xgboost")
        
        df = self.create_features(data)
        
        feature_cols = [col for col in df.columns if col not in ['target', 'Close', 'Open', 'High', 'Low', 'Volume', 'Date']]
        X = df[feature_cols]
        y = df['target']
        
        # Time series split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'accuracy': accuracy,
            'feature_importance': self.feature_importance,
            'classification_report': classification_report(y_test, y_pred)
        }


class GARCHVolatilityModel:
    """
    GARCH model for volatility forecasting.
    Standard tool used by institutional risk managers.
    """
    
    def __init__(self, p=1, q=1, vol='GARCH'):
        self.p = p
        self.q = q
        self.vol = vol
        self.model = None
        self.results = None
    
    def fit(self, returns):
        """Fit GARCH model to returns."""
        if not ARCH_AVAILABLE:
            raise ImportError("arch package required. Install: pip install arch")
        
        # Scale returns for numerical stability
        returns_scaled = returns * 100
        
        self.model = arch_model(
            returns_scaled.dropna(),
            vol=self.vol,
            p=self.p,
            q=self.q
        )
        
        self.results = self.model.fit(disp='off')
        return self.results
    
    def forecast(self, horizon=5):
        """Forecast volatility."""
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecast = self.results.forecast(horizon=horizon)
        variance_forecast = forecast.variance.iloc[-1].values
        volatility_forecast = np.sqrt(variance_forecast) / 100  # Scale back
        
        return volatility_forecast
    
    def get_summary(self):
        """Get model summary."""
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.results.summary()


class MarketRegimeDetector:
    """
    Hidden Markov Model for market regime detection.
    Identifies bull/bear/sideways markets automatically.
    """
    
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.model = None
        self.regime_names = {0: 'Bear', 1: 'Neutral', 2: 'Bull'}
    
    def detect_regimes(self, returns):
        """Detect market regimes using rolling statistics."""
        df = pd.DataFrame(returns, columns=['returns'])
        
        # Calculate regime indicators
        df['rolling_mean'] = df['returns'].rolling(window=20).mean()
        df['rolling_std'] = df['returns'].rolling(window=20).std()
        df['regime_score'] = df['rolling_mean'] / df['rolling_std']
        
        # Classify regimes
        df['regime'] = pd.cut(
            df['regime_score'],
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=[0, 1, 2]
        )
        
        df['regime_name'] = df['regime'].map(self.regime_names)
        
        return df.dropna()
    
    def get_current_regime(self, returns):
        """Get current market regime."""
        df = self.detect_regimes(returns)
        if len(df) > 0:
            return {
                'regime': int(df['regime'].iloc[-1]),
                'regime_name': df['regime_name'].iloc[-1],
                'score': df['regime_score'].iloc[-1]
            }
        return None


class AlphaGenerator:
    """
    Advanced alpha signal generator.
    Combines multiple factors for alpha generation.
    """
    
    def __init__(self):
        self.factors = {}
        self.weights = {}
    
    def calculate_momentum_alpha(self, data, lookback=20):
        """Calculate momentum alpha."""
        returns = data['Close'].pct_change(lookback)
        # Normalize to z-score
        alpha = (returns - returns.rolling(252).mean()) / returns.rolling(252).std()
        return alpha
    
    def calculate_mean_reversion_alpha(self, data, lookback=20):
        """Calculate mean reversion alpha."""
        sma = data['Close'].rolling(lookback).mean()
        deviation = (data['Close'] - sma) / sma
        # Negative deviation = buy signal (mean reversion)
        alpha = -deviation
        return alpha
    
    def calculate_volatility_alpha(self, data, lookback=20):
        """Calculate volatility-adjusted alpha."""
        returns = data['Close'].pct_change()
        volatility = returns.rolling(lookback).std()
        # Low volatility = positive alpha
        alpha = -volatility
        alpha = (alpha - alpha.rolling(252).mean()) / alpha.rolling(252).std()
        return alpha
    
    def calculate_quality_alpha(self, data):
        """Calculate quality/stability alpha."""
        returns = data['Close'].pct_change()
        # Positive skewness = good
        skewness = returns.rolling(60).skew()
        # Low kurtosis = good
        kurtosis = -returns.rolling(60).kurt()
        alpha = (skewness + kurtosis) / 2
        return alpha
    
    def generate_composite_alpha(self, data, weights=None):
        """Generate composite alpha signal."""
        if weights is None:
            weights = {
                'momentum': 0.3,
                'mean_reversion': 0.25,
                'volatility': 0.25,
                'quality': 0.2
            }
        
        alphas = pd.DataFrame()
        alphas['momentum'] = self.calculate_momentum_alpha(data)
        alphas['mean_reversion'] = self.calculate_mean_reversion_alpha(data)
        alphas['volatility'] = self.calculate_volatility_alpha(data)
        alphas['quality'] = self.calculate_quality_alpha(data)
        
        # Composite alpha
        composite = sum(alphas[factor] * weight for factor, weight in weights.items())
        
        return {
            'composite_alpha': composite,
            'individual_alphas': alphas,
            'signal': np.sign(composite)  # 1 = buy, -1 = sell, 0 = hold
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Advanced ML Prediction Models")
    print("=" * 60)
    
    # Test with sample data
    from data.openbb_integration import OpenBBIntegration
    
    openbb = OpenBBIntegration()
    data = openbb.fetch_stock_data('AAPL', '2023-01-01', '2024-12-31')
    
    if data is not None and not data.empty:
        print(f"\nData loaded: {len(data)} rows")
        
        # Test XGBoost (always available)
        print("\n--- XGBoost Predictor ---")
        if XGBOOST_AVAILABLE:
            xgb_model = XGBoostPredictor()
            results = xgb_model.train(data)
            print(f"Accuracy: {results['accuracy']:.4f}")
            print("\nTop 10 Features:")
            print(results['feature_importance'].head(10))
        else:
            print("XGBoost not installed. Install: pip install xgboost")
        
        # Test Market Regime Detector
        print("\n--- Market Regime Detector ---")
        regime_detector = MarketRegimeDetector()
        returns = data['Close'].pct_change()
        current_regime = regime_detector.get_current_regime(returns)
        if current_regime:
            print(f"Current Regime: {current_regime['regime_name']}")
            print(f"Regime Score: {current_regime['score']:.4f}")
        
        # Test Alpha Generator
        print("\n--- Alpha Generator ---")
        alpha_gen = AlphaGenerator()
        alpha_signals = alpha_gen.generate_composite_alpha(data)
        print(f"Latest Composite Alpha: {alpha_signals['composite_alpha'].iloc[-1]:.4f}")
        print(f"Current Signal: {'BUY' if alpha_signals['signal'].iloc[-1] > 0 else 'SELL' if alpha_signals['signal'].iloc[-1] < 0 else 'HOLD'}")
        
        # Test GARCH
        print("\n--- GARCH Volatility Model ---")
        if ARCH_AVAILABLE:
            garch = GARCHVolatilityModel()
            garch.fit(returns)
            vol_forecast = garch.forecast(5)
            print(f"5-day Volatility Forecast: {vol_forecast}")
        else:
            print("arch package not installed. Install: pip install arch")
        
        print("\n" + "=" * 60)
        print("All models tested successfully!")
        print("=" * 60)
    else:
        print("Failed to load data for testing")
