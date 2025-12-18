"""
Machine Learning Trading Strategies
=====================================
ML-based trading strategies including LSTM for price prediction,
Random Forest for classification, and Reinforcement Learning (DQN, PPO).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
from collections import deque
from enum import Enum
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# Check available ML libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Deep learning models will be limited.")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. ML models will be limited.")


class PredictionType(Enum):
    """Types of predictions."""
    PRICE = "price"
    RETURN = "return"
    DIRECTION = "direction"
    VOLATILITY = "volatility"


class SignalType(Enum):
    """Trading signal types."""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class MLConfig:
    """Configuration for ML strategies."""
    # Data parameters
    lookback_window: int = 60
    prediction_horizon: int = 5
    train_test_split: float = 0.8
    
    # Feature parameters
    use_technical_features: bool = True
    use_volume_features: bool = True
    use_returns_features: bool = True
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    
    # Model parameters
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    
    # Trading parameters
    confidence_threshold: float = 0.6
    position_size: float = 1.0


class FeatureEngineer:
    """
    Feature engineering for ML models.
    Creates technical indicators and derived features.
    """
    
    def __init__(self, config: MLConfig = None):
        self.config = config or MLConfig()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_names = []
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set from OHLCV data.
        """
        df = data.copy()
        
        # Ensure column names are lowercase
        df.columns = [c.lower() for c in df.columns]
        
        features = pd.DataFrame(index=df.index)
        
        # Returns features
        if self.config.use_returns_features:
            features['return_1d'] = df['close'].pct_change(1)
            features['return_5d'] = df['close'].pct_change(5)
            features['return_10d'] = df['close'].pct_change(10)
            features['return_20d'] = df['close'].pct_change(20)
            
            # Log returns
            features['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            # Return momentum
            features['return_momentum'] = features['return_5d'] - features['return_20d']
        
        # Technical features
        if self.config.use_technical_features:
            # Moving averages
            for period in [5, 10, 20, 50]:
                features[f'sma_{period}'] = df['close'].rolling(period).mean()
                features[f'sma_ratio_{period}'] = df['close'] / features[f'sma_{period}']
            
            # EMA
            for period in [12, 26]:
                features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # MACD
            features['macd'] = features['ema_12'] - features['ema_26']
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_hist'] = features['macd'] - features['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            features['rsi_normalized'] = (features['rsi'] - 50) / 50
            
            # Bollinger Bands
            bb_mean = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            features['bb_upper'] = bb_mean + 2 * bb_std
            features['bb_lower'] = bb_mean - 2 * bb_std
            features['bb_position'] = (df['close'] - bb_mean) / (2 * bb_std)
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features['atr'] = tr.rolling(14).mean()
            features['atr_pct'] = features['atr'] / df['close']
            
            # Volatility
            features['volatility_10d'] = df['close'].pct_change().rolling(10).std()
            features['volatility_20d'] = df['close'].pct_change().rolling(20).std()
            features['vol_ratio'] = features['volatility_10d'] / features['volatility_20d']
        
        # Volume features
        if self.config.use_volume_features and 'volume' in df.columns:
            features['volume_sma_10'] = df['volume'].rolling(10).mean()
            features['volume_sma_20'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma_20']
            features['volume_momentum'] = (df['volume'] - features['volume_sma_10']) / features['volume_sma_10']
            
            # Price-Volume features
            features['pv_trend'] = (df['close'] * df['volume']).rolling(10).mean()
            
            # OBV
            obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
            features['obv_normalized'] = (obv - obv.rolling(20).mean()) / obv.rolling(20).std()
        
        # Candlestick features
        features['body'] = (df['close'] - df['open']) / df['open']
        features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        features['body_to_range'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 0.0001)
        
        # Day of week (cyclical encoding)
        if isinstance(df.index, pd.DatetimeIndex):
            features['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 5)
            features['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 5)
        
        # Store feature names
        self.feature_names = list(features.columns)
        
        # Drop NaN rows
        features = features.dropna()
        
        return features
    
    def create_sequences(self, 
                         features: pd.DataFrame,
                         target: pd.Series,
                         lookback: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.
        
        Returns:
            X: (n_samples, lookback, n_features)
            y: (n_samples,)
        """
        lookback = lookback or self.config.lookback_window
        
        # Align indices
        common_idx = features.index.intersection(target.index)
        features = features.loc[common_idx]
        target = target.loc[common_idx]
        
        X, y = [], []
        
        for i in range(lookback, len(features)):
            X.append(features.iloc[i-lookback:i].values)
            y.append(target.iloc[i])
        
        return np.array(X), np.array(y)
    
    def scale_features(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features using StandardScaler."""
        if self.scaler is None:
            return features
        
        if fit:
            scaled = self.scaler.fit_transform(features)
        else:
            scaled = self.scaler.transform(features)
        
        return pd.DataFrame(scaled, index=features.index, columns=features.columns)


# LSTM Model (PyTorch)
if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """LSTM model for price/return prediction."""
        
        def __init__(self, 
                     input_size: int,
                     hidden_size: int = 64,
                     num_layers: int = 2,
                     output_size: int = 1,
                     dropout: float = 0.2):
            super().__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        def forward(self, x):
            # x: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)
            
            # Take last output
            last_out = lstm_out[:, -1, :]
            
            out = self.dropout(last_out)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            
            return out
    
    
    class TransformerModel(nn.Module):
        """Transformer model for sequence prediction."""
        
        def __init__(self,
                     input_size: int,
                     d_model: int = 64,
                     nhead: int = 4,
                     num_layers: int = 2,
                     output_size: int = 1,
                     dropout: float = 0.1):
            super().__init__()
            
            self.input_projection = nn.Linear(input_size, d_model)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            self.output_layer = nn.Linear(d_model, output_size)
        
        def forward(self, x):
            # x: (batch, seq_len, features)
            x = self.input_projection(x)
            x = self.transformer(x)
            
            # Take last position
            x = x[:, -1, :]
            return self.output_layer(x)


class LSTMStrategy:
    """
    LSTM-based price/return prediction strategy.
    """
    
    def __init__(self, config: MLConfig = None):
        self.config = config or MLConfig()
        self.feature_engineer = FeatureEngineer(self.config)
        self.model = None
        self.is_trained = False
        self.training_history = {'train_loss': [], 'val_loss': []}
    
    def _create_target(self, 
                       data: pd.DataFrame,
                       prediction_type: PredictionType = PredictionType.DIRECTION) -> pd.Series:
        """Create target variable."""
        horizon = self.config.prediction_horizon
        
        if prediction_type == PredictionType.PRICE:
            target = data['close'].shift(-horizon)
        elif prediction_type == PredictionType.RETURN:
            target = data['close'].pct_change(horizon).shift(-horizon)
        elif prediction_type == PredictionType.DIRECTION:
            future_return = data['close'].pct_change(horizon).shift(-horizon)
            target = (future_return > 0).astype(int)
        elif prediction_type == PredictionType.VOLATILITY:
            target = data['close'].pct_change().rolling(horizon).std().shift(-horizon)
        
        return target.dropna()
    
    def train(self,
              data: pd.DataFrame,
              prediction_type: PredictionType = PredictionType.DIRECTION,
              validation_split: float = 0.2) -> Dict:
        """
        Train LSTM model.
        
        Args:
            data: OHLCV DataFrame
            prediction_type: Type of prediction to make
            validation_split: Fraction for validation
        
        Returns:
            Training metrics dictionary
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for LSTM training")
        
        # Create features and target
        features = self.feature_engineer.create_features(data)
        features = self.feature_engineer.scale_features(features)
        target = self._create_target(data, prediction_type)
        
        # Create sequences
        X, y = self.feature_engineer.create_sequences(features, target)
        
        if len(X) == 0:
            raise ValueError("Insufficient data for training")
        
        # Train/validation split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create model
        input_size = X_train.shape[2]
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            output_size=1,
            dropout=self.config.dropout
        )
        
        # Loss and optimizer
        if prediction_type == PredictionType.DIRECTION:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            
            # Mini-batch training
            train_losses = []
            indices = np.random.permutation(len(X_train))
            
            for i in range(0, len(X_train), self.config.batch_size):
                batch_idx = indices[i:i+self.config.batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            avg_train_loss = np.mean(train_losses)
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        self.is_trained = True
        self.prediction_type = prediction_type
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            val_preds = self.model(X_val)
            if prediction_type == PredictionType.DIRECTION:
                val_preds = torch.sigmoid(val_preds)
                accuracy = ((val_preds > 0.5).float() == y_val).float().mean().item()
                return {'accuracy': accuracy, 'best_val_loss': best_val_loss, 'epochs_trained': epoch + 1}
        
        return {'best_val_loss': best_val_loss, 'epochs_trained': epoch + 1}
    
    def predict(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Make prediction on new data.
        
        Returns:
            (prediction, confidence)
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained")
        
        # Create features
        features = self.feature_engineer.create_features(data)
        features = self.feature_engineer.scale_features(features, fit=False)
        
        # Get last lookback window
        X = features.iloc[-self.config.lookback_window:].values
        X = torch.FloatTensor(X).unsqueeze(0)  # Add batch dimension
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            
            if self.prediction_type == PredictionType.DIRECTION:
                prob = torch.sigmoid(output).item()
                prediction = 1 if prob > 0.5 else 0
                confidence = abs(prob - 0.5) * 2  # Scale to 0-1
            else:
                prediction = output.item()
                confidence = 0.5  # Regression confidence is harder to estimate
        
        return prediction, confidence
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate trading signal."""
        prediction, confidence = self.predict(data)
        
        if confidence < self.config.confidence_threshold:
            signal = SignalType.HOLD
        elif self.prediction_type == PredictionType.DIRECTION:
            if prediction == 1:
                signal = SignalType.STRONG_BUY if confidence > 0.8 else SignalType.BUY
            else:
                signal = SignalType.STRONG_SELL if confidence > 0.8 else SignalType.SELL
        else:
            # For price/return predictions, compare to current
            current_price = data['close'].iloc[-1]
            if prediction > current_price * 1.02:
                signal = SignalType.BUY
            elif prediction < current_price * 0.98:
                signal = SignalType.SELL
            else:
                signal = SignalType.HOLD
        
        return {
            'signal': signal,
            'prediction': prediction,
            'confidence': confidence,
            'model_type': 'LSTM'
        }
    
    def save_model(self, path: str):
        """Save trained model."""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'scaler': self.feature_engineer.scaler,
            'feature_names': self.feature_engineer.feature_names,
            'prediction_type': self.prediction_type
        }, path)
    
    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path)
        
        self.config = checkpoint['config']
        self.feature_engineer.scaler = checkpoint['scaler']
        self.feature_engineer.feature_names = checkpoint['feature_names']
        self.prediction_type = checkpoint['prediction_type']
        
        # Recreate model architecture
        input_size = len(self.feature_engineer.feature_names)
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True


class RandomForestStrategy:
    """
    Random Forest classification strategy for direction prediction.
    """
    
    def __init__(self, config: MLConfig = None):
        self.config = config or MLConfig()
        self.feature_engineer = FeatureEngineer(self.config)
        self.model = None
        self.is_trained = False
        self.feature_importance = {}
    
    def train(self,
              data: pd.DataFrame,
              n_estimators: int = 100,
              max_depth: int = 10) -> Dict:
        """
        Train Random Forest classifier.
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required for Random Forest")
        
        # Create features and target
        features = self.feature_engineer.create_features(data)
        
        # Target: direction of next n-day return
        horizon = self.config.prediction_horizon
        future_return = data['close'].pct_change(horizon).shift(-horizon)
        target = (future_return > 0).astype(int)
        
        # Align
        common_idx = features.index.intersection(target.dropna().index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]
        
        # Scale features
        X_scaled = self.feature_engineer.scale_features(X)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=1-self.config.train_test_split, shuffle=False
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Feature importance
        self.feature_importance = dict(zip(
            self.feature_engineer.feature_names,
            self.model.feature_importances_
        ))
        
        self.is_trained = True
        
        return {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred),
            'recall': recall_score(y_test, test_pred),
            'f1': f1_score(y_test, test_pred),
            'top_features': sorted(self.feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)[:10]
        }
    
    def predict(self, data: pd.DataFrame) -> Tuple[int, float]:
        """Make prediction with probability."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        features = self.feature_engineer.create_features(data)
        features = self.feature_engineer.scale_features(features, fit=False)
        
        # Get latest row
        X = features.iloc[-1:].values
        
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return prediction, confidence
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate trading signal."""
        prediction, confidence = self.predict(data)
        
        if confidence < self.config.confidence_threshold:
            signal = SignalType.HOLD
        elif prediction == 1:
            signal = SignalType.STRONG_BUY if confidence > 0.7 else SignalType.BUY
        else:
            signal = SignalType.STRONG_SELL if confidence > 0.7 else SignalType.SELL
        
        return {
            'signal': signal,
            'prediction': prediction,
            'confidence': confidence,
            'model_type': 'RandomForest',
            'top_features': list(self.feature_importance.items())[:5]
        }


class GradientBoostingStrategy:
    """
    Gradient Boosting classification strategy.
    """
    
    def __init__(self, config: MLConfig = None):
        self.config = config or MLConfig()
        self.feature_engineer = FeatureEngineer(self.config)
        self.model = None
        self.is_trained = False
    
    def train(self,
              data: pd.DataFrame,
              n_estimators: int = 100,
              learning_rate: float = 0.1,
              max_depth: int = 5) -> Dict:
        """Train Gradient Boosting classifier."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required")
        
        features = self.feature_engineer.create_features(data)
        
        horizon = self.config.prediction_horizon
        future_return = data['close'].pct_change(horizon).shift(-horizon)
        target = (future_return > 0).astype(int)
        
        common_idx = features.index.intersection(target.dropna().index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]
        
        X_scaled = self.feature_engineer.scale_features(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=1-self.config.train_test_split, shuffle=False
        )
        
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        test_pred = self.model.predict(X_test)
        
        self.is_trained = True
        
        return {
            'test_accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred),
            'recall': recall_score(y_test, test_pred),
            'f1': f1_score(y_test, test_pred)
        }
    
    def predict(self, data: pd.DataFrame) -> Tuple[int, float]:
        """Make prediction."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        features = self.feature_engineer.create_features(data)
        features = self.feature_engineer.scale_features(features, fit=False)
        
        X = features.iloc[-1:].values
        prediction = self.model.predict(X)[0]
        confidence = max(self.model.predict_proba(X)[0])
        
        return prediction, confidence


# Reinforcement Learning (if PyTorch available)
if TORCH_AVAILABLE:
    class DQNNetwork(nn.Module):
        """Deep Q-Network for trading."""
        
        def __init__(self, state_size: int, action_size: int = 3, hidden_size: int = 64):
            super().__init__()
            
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, action_size)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)
    
    
    @dataclass
    class Experience:
        """Experience tuple for replay buffer."""
        state: np.ndarray
        action: int
        reward: float
        next_state: np.ndarray
        done: bool


class DQNStrategy:
    """
    Deep Q-Network Reinforcement Learning Strategy.
    
    Actions:
        0: Hold
        1: Buy
        2: Sell
    """
    
    def __init__(self, config: MLConfig = None):
        self.config = config or MLConfig()
        self.feature_engineer = FeatureEngineer(self.config)
        
        # RL parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.replay_buffer_size = 10000
        self.batch_size = 64
        
        self.model = None
        self.target_model = None
        self.replay_buffer: deque = deque(maxlen=self.replay_buffer_size)
        self.is_trained = False
        
        self.action_space = 3  # Hold, Buy, Sell
    
    def _create_state(self, features: pd.DataFrame, idx: int) -> np.ndarray:
        """Create state vector from features."""
        if idx < self.config.lookback_window:
            return None
        
        state = features.iloc[idx-self.config.lookback_window:idx].values.flatten()
        return state
    
    def _calculate_reward(self,
                          action: int,
                          position: int,
                          current_price: float,
                          next_price: float,
                          transaction_cost: float = 0.001) -> float:
        """
        Calculate reward for action.
        
        Position: -1 (short), 0 (flat), 1 (long)
        """
        price_return = (next_price - current_price) / current_price
        
        # Position change cost
        if action == 1 and position != 1:  # Buy
            cost = transaction_cost
        elif action == 2 and position != -1:  # Sell
            cost = transaction_cost
        else:
            cost = 0
        
        # Calculate P&L based on position
        if position == 1:
            pnl = price_return
        elif position == -1:
            pnl = -price_return
        else:
            pnl = 0
        
        return pnl - cost
    
    def train(self, 
              data: pd.DataFrame,
              episodes: int = 100,
              update_target_every: int = 10) -> Dict:
        """
        Train DQN agent.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for DQN")
        
        # Create features
        features = self.feature_engineer.create_features(data)
        features = self.feature_engineer.scale_features(features)
        prices = data['close'].loc[features.index]
        
        # Initialize networks
        state_size = features.shape[1] * self.config.lookback_window
        self.model = DQNNetwork(state_size, self.action_space)
        self.target_model = DQNNetwork(state_size, self.action_space)
        self.target_model.load_state_dict(self.model.state_dict())
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        episode_rewards = []
        
        for episode in range(episodes):
            total_reward = 0
            position = 0  # Start flat
            
            for t in range(self.config.lookback_window, len(features) - 1):
                state = self._create_state(features, t)
                if state is None:
                    continue
                
                # Epsilon-greedy action selection
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.action_space)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        q_values = self.model(state_tensor)
                    action = q_values.argmax().item()
                
                # Update position based on action
                if action == 1:  # Buy
                    new_position = 1
                elif action == 2:  # Sell
                    new_position = -1
                else:  # Hold
                    new_position = position
                
                # Calculate reward
                current_price = prices.iloc[t]
                next_price = prices.iloc[t + 1]
                reward = self._calculate_reward(action, position, current_price, next_price)
                total_reward += reward
                
                # Get next state
                next_state = self._create_state(features, t + 1)
                done = (t == len(features) - 2)
                
                # Store experience
                if next_state is not None:
                    self.replay_buffer.append(Experience(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=done
                    ))
                
                position = new_position
                
                # Train on batch
                if len(self.replay_buffer) >= self.batch_size:
                    self._train_step(optimizer, criterion)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Update target network
            if (episode + 1) % update_target_every == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            episode_rewards.append(total_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                logger.info(f"Episode {episode+1}: Avg Reward = {avg_reward:.4f}, Epsilon = {self.epsilon:.3f}")
        
        self.is_trained = True
        
        return {
            'final_avg_reward': np.mean(episode_rewards[-10:]),
            'total_episodes': episodes,
            'final_epsilon': self.epsilon
        }
    
    def _train_step(self, optimizer, criterion):
        """Single training step on batch."""
        # Sample batch
        batch_size = min(self.batch_size, len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.FloatTensor([e.done for e in batch])
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss and update
        loss = criterion(current_q.squeeze(), target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def predict(self, data: pd.DataFrame) -> int:
        """Predict best action."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        features = self.feature_engineer.create_features(data)
        features = self.feature_engineer.scale_features(features, fit=False)
        
        state = features.iloc[-self.config.lookback_window:].values.flatten()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        return q_values.argmax().item()
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate trading signal."""
        action = self.predict(data)
        
        action_map = {
            0: SignalType.HOLD,
            1: SignalType.BUY,
            2: SignalType.SELL
        }
        
        return {
            'signal': action_map[action],
            'action': action,
            'model_type': 'DQN'
        }


class MLStrategySuite:
    """
    Unified interface for all ML strategies.
    """
    
    def __init__(self, config: MLConfig = None):
        self.config = config or MLConfig()
        self.strategies = {}
        self.ensemble_weights = {}
    
    def add_strategy(self, name: str, strategy: Any, weight: float = 1.0):
        """Add a strategy to the suite."""
        self.strategies[name] = strategy
        self.ensemble_weights[name] = weight
    
    def create_default_strategies(self):
        """Create default set of strategies."""
        self.strategies = {
            'random_forest': RandomForestStrategy(self.config),
            'gradient_boosting': GradientBoostingStrategy(self.config),
        }
        
        if TORCH_AVAILABLE:
            self.strategies['lstm'] = LSTMStrategy(self.config)
            self.strategies['dqn'] = DQNStrategy(self.config)
        
        for name in self.strategies:
            self.ensemble_weights[name] = 1.0
    
    def train_all(self, data: pd.DataFrame) -> Dict:
        """Train all strategies."""
        results = {}
        
        for name, strategy in self.strategies.items():
            logger.info(f"Training {name}...")
            try:
                result = strategy.train(data)
                results[name] = {'status': 'success', 'metrics': result}
            except Exception as e:
                results[name] = {'status': 'error', 'error': str(e)}
                logger.error(f"Error training {name}: {e}")
        
        return results
    
    def get_ensemble_signal(self, data: pd.DataFrame) -> Dict:
        """
        Get ensemble signal from all trained strategies.
        """
        signals = {}
        weighted_score = 0
        total_weight = 0
        
        for name, strategy in self.strategies.items():
            if hasattr(strategy, 'is_trained') and strategy.is_trained:
                try:
                    signal = strategy.generate_signal(data)
                    signals[name] = signal
                    
                    # Convert signal to numeric score
                    signal_score = signal['signal'].value if hasattr(signal['signal'], 'value') else 0
                    weight = self.ensemble_weights.get(name, 1.0)
                    
                    weighted_score += signal_score * weight
                    total_weight += weight
                except Exception as e:
                    logger.warning(f"Error getting signal from {name}: {e}")
        
        if total_weight == 0:
            ensemble_signal = SignalType.HOLD
        else:
            avg_score = weighted_score / total_weight
            if avg_score >= 1.5:
                ensemble_signal = SignalType.STRONG_BUY
            elif avg_score >= 0.5:
                ensemble_signal = SignalType.BUY
            elif avg_score <= -1.5:
                ensemble_signal = SignalType.STRONG_SELL
            elif avg_score <= -0.5:
                ensemble_signal = SignalType.SELL
            else:
                ensemble_signal = SignalType.HOLD
        
        return {
            'ensemble_signal': ensemble_signal,
            'individual_signals': signals,
            'weighted_score': weighted_score / total_weight if total_weight > 0 else 0,
            'agreement': sum(1 for s in signals.values() 
                           if s['signal'] == ensemble_signal) / len(signals) if signals else 0
        }
    
    def backtest(self, data: pd.DataFrame, strategy_name: str = None) -> Dict:
        """
        Backtest a strategy or ensemble.
        """
        if strategy_name:
            strategy = self.strategies.get(strategy_name)
            if not strategy:
                raise ValueError(f"Strategy {strategy_name} not found")
        
        # Simple backtest
        features = FeatureEngineer(self.config).create_features(data)
        
        signals = []
        returns = []
        
        for i in range(self.config.lookback_window + 10, len(features)):
            subset = data.iloc[:i]
            
            if strategy_name:
                signal = self.strategies[strategy_name].generate_signal(subset)
                position = signal['signal'].value if hasattr(signal['signal'], 'value') else 0
            else:
                signal = self.get_ensemble_signal(subset)
                position = signal['ensemble_signal'].value if hasattr(signal['ensemble_signal'], 'value') else 0
            
            # Calculate return
            if i < len(data) - 1:
                daily_return = (data['close'].iloc[i+1] - data['close'].iloc[i]) / data['close'].iloc[i]
                strategy_return = daily_return * np.sign(position)
                returns.append(strategy_return)
                signals.append(position)
        
        if not returns:
            return {'error': 'No returns calculated'}
        
        returns = np.array(returns)
        cumulative = (1 + returns).cumprod()
        
        return {
            'total_return': cumulative[-1] - 1,
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'max_drawdown': np.min(cumulative / np.maximum.accumulate(cumulative) - 1),
            'win_rate': np.sum(returns > 0) / len(returns),
            'num_trades': np.sum(np.abs(np.diff(signals)) > 0)
        }


# Factory function
def create_ml_strategy(
    strategy_type: str,
    config: MLConfig = None
) -> Any:
    """
    Factory function to create ML strategies.
    
    Args:
        strategy_type: 'lstm', 'random_forest', 'gradient_boosting', 'dqn', or 'ensemble'
        config: Strategy configuration
    """
    strategies = {
        'random_forest': RandomForestStrategy,
        'gradient_boosting': GradientBoostingStrategy,
    }
    
    if TORCH_AVAILABLE:
        strategies['lstm'] = LSTMStrategy
        strategies['dqn'] = DQNStrategy
    
    if strategy_type == 'ensemble':
        return MLStrategySuite(config)
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_type}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_type](config)


if __name__ == "__main__":
    print("=== ML Strategy Demo ===\n")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Simulated price data with trend and noise
    trend = np.linspace(100, 150, 500)
    noise = np.random.randn(500) * 5
    seasonal = 10 * np.sin(np.linspace(0, 8*np.pi, 500))
    prices = trend + noise + seasonal
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(500) * 0.01),
        'high': prices * (1 + np.abs(np.random.randn(500)) * 0.02),
        'low': prices * (1 - np.abs(np.random.randn(500)) * 0.02),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 500)
    }, index=dates)
    
    # Test Random Forest
    if SKLEARN_AVAILABLE:
        print("--- Random Forest Strategy ---")
        rf = RandomForestStrategy()
        metrics = rf.train(data)
        print(f"Test Accuracy: {metrics['test_accuracy']:.2%}")
        print(f"F1 Score: {metrics['f1']:.3f}")
        
        signal = rf.generate_signal(data)
        print(f"Current Signal: {signal['signal'].name}")
        print(f"Confidence: {signal['confidence']:.2%}")
        print(f"Top Features: {[f[0] for f in signal['top_features'][:3]]}")
    
    # Test LSTM
    if TORCH_AVAILABLE:
        print("\n--- LSTM Strategy ---")
        lstm = LSTMStrategy()
        lstm.config.epochs = 20  # Reduced for demo
        metrics = lstm.train(data)
        print(f"Accuracy: {metrics.get('accuracy', 'N/A')}")
        
        signal = lstm.generate_signal(data)
        print(f"Current Signal: {signal['signal'].name}")
        print(f"Confidence: {signal['confidence']:.2%}")
    
    print("\n--- Feature Engineering ---")
    fe = FeatureEngineer()
    features = fe.create_features(data)
    print(f"Created {len(fe.feature_names)} features")
    print(f"Sample features: {fe.feature_names[:5]}")
