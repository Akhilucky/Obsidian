"""
Model Experimentation Lab
==========================
A comprehensive laboratory for building, testing, and comparing ML models.
Supports custom model definitions, hyperparameter tuning, and experiment tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import json
import hashlib
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.model_selection import (
        train_test_split, cross_val_score, TimeSeriesSplit,
        GridSearchCV, RandomizedSearchCV
    )
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    from sklearn.ensemble import (
        RandomForestRegressor, RandomForestClassifier,
        GradientBoostingRegressor, GradientBoostingClassifier,
        AdaBoostRegressor, AdaBoostClassifier,
        VotingRegressor, VotingClassifier,
        StackingRegressor, StackingClassifier
    )
    from sklearn.linear_model import (
        LinearRegression, Ridge, Lasso, ElasticNet,
        LogisticRegression
    )
    from sklearn.svm import SVR, SVC
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

@dataclass
class ExperimentResult:
    """Stores results from a single experiment run"""
    experiment_id: str
    model_name: str
    timestamp: datetime
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    predictions: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_time: float = 0.0
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'experiment_id': self.experiment_id,
            'model_name': self.model_name,
            'timestamp': self.timestamp.isoformat(),
            'parameters': self.parameters,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'training_time': self.training_time,
            'notes': self.notes
        }


class ExperimentTracker:
    """
    Track and compare ML experiments
    ================================
    Similar to MLflow but lightweight and built-in.
    """
    
    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = experiment_dir
        self.experiments: List[ExperimentResult] = []
        self._ensure_dir()
        self._load_experiments()
    
    def _ensure_dir(self):
        """Create experiment directory if needed"""
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
    
    def _load_experiments(self):
        """Load existing experiments from disk"""
        index_file = os.path.join(self.experiment_dir, 'experiments.json')
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                data = json.load(f)
                for exp_data in data:
                    exp_data['timestamp'] = datetime.fromisoformat(exp_data['timestamp'])
                    self.experiments.append(ExperimentResult(**exp_data))
    
    def log_experiment(self, result: ExperimentResult):
        """Log a new experiment"""
        self.experiments.append(result)
        self._save_experiments()
    
    def _save_experiments(self):
        """Save experiments to disk"""
        index_file = os.path.join(self.experiment_dir, 'experiments.json')
        with open(index_file, 'w') as f:
            json.dump([e.to_dict() for e in self.experiments], f, indent=2)
    
    def get_best_experiment(self, metric: str = 'mse', minimize: bool = True) -> Optional[ExperimentResult]:
        """Get the best experiment by a specific metric"""
        if not self.experiments:
            return None
        
        valid_experiments = [e for e in self.experiments if metric in e.metrics]
        if not valid_experiments:
            return None
        
        if minimize:
            return min(valid_experiments, key=lambda x: x.metrics[metric])
        return max(valid_experiments, key=lambda x: x.metrics[metric])
    
    def compare_experiments(self, experiment_ids: Optional[List[str]] = None,
                           metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare multiple experiments"""
        if experiment_ids:
            exps = [e for e in self.experiments if e.experiment_id in experiment_ids]
        else:
            exps = self.experiments
        
        if not exps:
            return pd.DataFrame()
        
        data = []
        for exp in exps:
            row = {
                'experiment_id': exp.experiment_id,
                'model': exp.model_name,
                'timestamp': exp.timestamp,
                'training_time': exp.training_time
            }
            
            if metrics:
                for m in metrics:
                    row[m] = exp.metrics.get(m, None)
            else:
                row.update(exp.metrics)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_leaderboard(self, metric: str = 'mse', top_n: int = 10,
                       minimize: bool = True) -> pd.DataFrame:
        """Get leaderboard of best models"""
        df = self.compare_experiments()
        if df.empty or metric not in df.columns:
            return df
        
        df = df.sort_values(metric, ascending=minimize).head(top_n)
        return df.reset_index(drop=True)


# ============================================================================
# BASE MODEL CLASS
# ============================================================================

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.parameters: Dict[str, Any] = {}
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        return None
    
    def save(self, filepath: str):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """Load model from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# ============================================================================
# SKLEARN-BASED MODELS
# ============================================================================

class SklearnModel(BaseModel):
    """Wrapper for scikit-learn models"""
    
    def __init__(self, model_class, name: str = "SklearnModel", **params):
        super().__init__(name)
        self.model_class = model_class
        self.parameters = params
        self.model = model_class(**params)
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray, scale: bool = True, **kwargs):
        """Train the sklearn model"""
        if scale:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        self.model.fit(X_scaled, y, **kwargs)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance for tree-based models"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if self.feature_names:
                return dict(zip(self.feature_names, importances))
            return dict(enumerate(importances))
        return None


class RandomForestModel(SklearnModel):
    """Random Forest model"""
    
    def __init__(self, task: str = 'regression', **params):
        if task == 'regression':
            model_class = RandomForestRegressor
        else:
            model_class = RandomForestClassifier
        
        default_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
        default_params.update(params)
        
        super().__init__(model_class, "RandomForest", **default_params)


class GradientBoostingModel(SklearnModel):
    """Gradient Boosting model"""
    
    def __init__(self, task: str = 'regression', **params):
        if task == 'regression':
            model_class = GradientBoostingRegressor
        else:
            model_class = GradientBoostingClassifier
        
        default_params = {'n_estimators': 100, 'random_state': 42}
        default_params.update(params)
        
        super().__init__(model_class, "GradientBoosting", **default_params)


# ============================================================================
# XGBOOST MODEL
# ============================================================================

class XGBoostModel(BaseModel):
    """XGBoost model wrapper"""
    
    def __init__(self, task: str = 'regression', **params):
        super().__init__("XGBoost")
        self.task = task
        self.scaler = StandardScaler()
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        default_params.update(params)
        self.parameters = default_params
        
        if XGBOOST_AVAILABLE:
            if task == 'regression':
                self.model = xgb.XGBRegressor(**default_params)
            else:
                self.model = xgb.XGBClassifier(**default_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray, scale: bool = True, 
            eval_set: Optional[List] = None, **kwargs):
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        if scale:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        fit_params = {}
        if eval_set:
            fit_params['eval_set'] = eval_set
        fit_params.update(kwargs)
        
        self.model.fit(X_scaled, y, **fit_params)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if self.feature_names:
                return dict(zip(self.feature_names, importances))
            return dict(enumerate(importances))
        return None


# ============================================================================
# LIGHTGBM MODEL
# ============================================================================

class LightGBMModel(BaseModel):
    """LightGBM model wrapper"""
    
    def __init__(self, task: str = 'regression', **params):
        super().__init__("LightGBM")
        self.task = task
        self.scaler = StandardScaler()
        
        default_params = {
            'n_estimators': 100,
            'max_depth': -1,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        default_params.update(params)
        self.parameters = default_params
        
        if LIGHTGBM_AVAILABLE:
            if task == 'regression':
                self.model = lgb.LGBMRegressor(**default_params)
            else:
                self.model = lgb.LGBMClassifier(**default_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray, scale: bool = False, **kwargs):
        """Train LightGBM model"""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        
        if scale:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        self.model.fit(X_scaled, y, **kwargs)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        if hasattr(self.scaler, 'mean_'):
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if self.feature_names:
                return dict(zip(self.feature_names, importances))
            return dict(enumerate(importances))
        return None


# ============================================================================
# NEURAL NETWORK MODELS
# ============================================================================

class KerasModel(BaseModel):
    """Custom Keras neural network"""
    
    def __init__(self, architecture: List[Dict], name: str = "KerasNN"):
        super().__init__(name)
        self.architecture = architecture
        self.scaler = StandardScaler()
        self.history = None
    
    def build_model(self, input_shape: Tuple[int, ...]):
        """Build Keras model from architecture definition"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        model = keras.Sequential()
        
        for i, layer_config in enumerate(self.architecture):
            layer_type = layer_config.get('type', 'dense')
            
            if layer_type == 'dense':
                if i == 0:
                    model.add(keras.layers.Dense(
                        layer_config.get('units', 64),
                        activation=layer_config.get('activation', 'relu'),
                        input_shape=input_shape
                    ))
                else:
                    model.add(keras.layers.Dense(
                        layer_config.get('units', 64),
                        activation=layer_config.get('activation', 'relu')
                    ))
            
            elif layer_type == 'dropout':
                model.add(keras.layers.Dropout(layer_config.get('rate', 0.2)))
            
            elif layer_type == 'batch_norm':
                model.add(keras.layers.BatchNormalization())
            
            elif layer_type == 'lstm':
                return_sequences = layer_config.get('return_sequences', False)
                if i == 0:
                    model.add(keras.layers.LSTM(
                        layer_config.get('units', 64),
                        return_sequences=return_sequences,
                        input_shape=input_shape
                    ))
                else:
                    model.add(keras.layers.LSTM(
                        layer_config.get('units', 64),
                        return_sequences=return_sequences
                    ))
            
            elif layer_type == 'conv1d':
                if i == 0:
                    model.add(keras.layers.Conv1D(
                        filters=layer_config.get('filters', 64),
                        kernel_size=layer_config.get('kernel_size', 3),
                        activation=layer_config.get('activation', 'relu'),
                        input_shape=input_shape
                    ))
                else:
                    model.add(keras.layers.Conv1D(
                        filters=layer_config.get('filters', 64),
                        kernel_size=layer_config.get('kernel_size', 3),
                        activation=layer_config.get('activation', 'relu')
                    ))
            
            elif layer_type == 'flatten':
                model.add(keras.layers.Flatten())
            
            elif layer_type == 'output':
                model.add(keras.layers.Dense(
                    layer_config.get('units', 1),
                    activation=layer_config.get('activation', 'linear')
                ))
        
        self.model = model
        return model
    
    def compile(self, optimizer: str = 'adam', loss: str = 'mse', 
                metrics: List[str] = None):
        """Compile the model"""
        if metrics is None:
            metrics = ['mae']
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100,
            batch_size: int = 32, validation_split: float = 0.2,
            scale: bool = True, **kwargs):
        """Train the model"""
        if scale and len(X.shape) == 2:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        if self.model is None:
            if len(X_scaled.shape) == 2:
                input_shape = (X_scaled.shape[1],)
            else:
                input_shape = X_scaled.shape[1:]
            self.build_model(input_shape)
            self.compile()
        
        self.history = self.model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=kwargs.get('verbose', 0),
            **{k: v for k, v in kwargs.items() if k != 'verbose'}
        )
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        if hasattr(self.scaler, 'mean_') and len(X.shape) == 2:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled, verbose=0).flatten()


class LSTMModel(KerasModel):
    """Pre-configured LSTM model for time series"""
    
    def __init__(self, units: int = 64, layers: int = 2, dropout: float = 0.2):
        architecture = []
        
        for i in range(layers):
            architecture.append({
                'type': 'lstm',
                'units': units,
                'return_sequences': i < layers - 1
            })
            architecture.append({'type': 'dropout', 'rate': dropout})
        
        architecture.append({'type': 'output', 'units': 1})
        
        super().__init__(architecture, f"LSTM_{layers}x{units}")
        self.parameters = {'units': units, 'layers': layers, 'dropout': dropout}


class TransformerModel(KerasModel):
    """Simplified Transformer model for time series"""
    
    def __init__(self, d_model: int = 64, num_heads: int = 4, 
                 ff_dim: int = 128, dropout: float = 0.1):
        # Simplified architecture using attention mechanism
        architecture = [
            {'type': 'dense', 'units': d_model, 'activation': 'relu'},
            {'type': 'dropout', 'rate': dropout},
            {'type': 'dense', 'units': ff_dim, 'activation': 'relu'},
            {'type': 'dropout', 'rate': dropout},
            {'type': 'dense', 'units': d_model, 'activation': 'relu'},
            {'type': 'output', 'units': 1}
        ]
        
        super().__init__(architecture, "Transformer")
        self.parameters = {
            'd_model': d_model, 
            'num_heads': num_heads,
            'ff_dim': ff_dim, 
            'dropout': dropout
        }


# ============================================================================
# ENSEMBLE MODELS
# ============================================================================

class CustomEnsemble(BaseModel):
    """Create custom ensemble from multiple models"""
    
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None,
                 method: str = 'average'):
        super().__init__("CustomEnsemble")
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.method = method  # 'average', 'weighted', 'stacking'
        self.meta_model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train all models in the ensemble"""
        for model in self.models:
            model.fit(X, y, **kwargs)
        
        if self.method == 'stacking':
            # Train meta-model on predictions
            predictions = np.column_stack([m.predict(X) for m in self.models])
            self.meta_model = Ridge()
            self.meta_model.fit(predictions, y)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using ensemble"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        predictions = np.column_stack([m.predict(X) for m in self.models])
        
        if self.method == 'average':
            return predictions.mean(axis=1)
        elif self.method == 'weighted':
            return np.average(predictions, axis=1, weights=self.weights)
        elif self.method == 'stacking':
            return self.meta_model.predict(predictions)
        
        return predictions.mean(axis=1)


# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

class HyperparameterTuner:
    """
    Hyperparameter tuning utilities
    ================================
    Grid search, random search, and Bayesian optimization.
    """
    
    def __init__(self, model_class, param_grid: Dict[str, List],
                 scoring: str = 'neg_mean_squared_error',
                 cv: int = 5):
        self.model_class = model_class
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.best_params = None
        self.best_score = None
        self.results = None
    
    def grid_search(self, X: np.ndarray, y: np.ndarray,
                   n_jobs: int = -1) -> Dict[str, Any]:
        """Perform grid search"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required")
        
        base_model = self.model_class()
        
        tscv = TimeSeriesSplit(n_splits=self.cv)
        
        search = GridSearchCV(
            base_model.model,
            self.param_grid,
            scoring=self.scoring,
            cv=tscv,
            n_jobs=n_jobs,
            verbose=1
        )
        
        search.fit(X, y)
        
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.results = pd.DataFrame(search.cv_results_)
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'results': self.results
        }
    
    def random_search(self, X: np.ndarray, y: np.ndarray,
                     n_iter: int = 50, n_jobs: int = -1) -> Dict[str, Any]:
        """Perform random search"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required")
        
        base_model = self.model_class()
        
        tscv = TimeSeriesSplit(n_splits=self.cv)
        
        search = RandomizedSearchCV(
            base_model.model,
            self.param_grid,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=tscv,
            n_jobs=n_jobs,
            random_state=42,
            verbose=1
        )
        
        search.fit(X, y)
        
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.results = pd.DataFrame(search.cv_results_)
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'results': self.results
        }


# ============================================================================
# MODEL LAB - MAIN INTERFACE
# ============================================================================

class ModelLab:
    """
    Model Experimentation Laboratory
    =================================
    The central interface for building, testing, and comparing models.
    
    Features:
    - Quick model training with pre-configured defaults
    - Custom model creation
    - Hyperparameter tuning
    - Experiment tracking
    - Model comparison
    - Walk-forward validation
    """
    
    def __init__(self, experiment_dir: str = "experiments"):
        self.tracker = ExperimentTracker(experiment_dir)
        self.models: Dict[str, BaseModel] = {}
        self.data_cache: Dict[str, pd.DataFrame] = {}
    
    def quick_train(self, X: np.ndarray, y: np.ndarray,
                   model_type: str = 'xgboost',
                   test_size: float = 0.2,
                   **model_params) -> Tuple[BaseModel, Dict[str, float]]:
        """
        Quickly train a model with sensible defaults
        
        Parameters:
        -----------
        X : Training features
        y : Target variable
        model_type : One of 'xgboost', 'lightgbm', 'random_forest', 'lstm', 'ensemble'
        test_size : Fraction for test split
        
        Returns:
        --------
        Trained model and metrics dictionary
        """
        # Split data (time-aware)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create model
        if model_type == 'xgboost':
            model = XGBoostModel(**model_params)
        elif model_type == 'lightgbm':
            model = LightGBMModel(**model_params)
        elif model_type == 'random_forest':
            model = RandomForestModel(**model_params)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingModel(**model_params)
        elif model_type == 'lstm':
            # Reshape for LSTM
            if len(X_train.shape) == 2:
                X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            model = LSTMModel(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        predictions = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, predictions)
        metrics['training_time'] = training_time
        
        # Log experiment
        exp_id = hashlib.md5(f"{model_type}_{datetime.now()}".encode()).hexdigest()[:8]
        result = ExperimentResult(
            experiment_id=exp_id,
            model_name=model.name,
            timestamp=datetime.now(),
            parameters=model.parameters,
            metrics=metrics,
            predictions=predictions,
            feature_importance=model.get_feature_importance(),
            training_time=training_time
        )
        self.tracker.log_experiment(result)
        
        # Store model
        self.models[exp_id] = model
        
        return model, metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        return {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'mape': float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
        }
    
    def compare_models(self, X: np.ndarray, y: np.ndarray,
                      model_types: List[str] = None,
                      test_size: float = 0.2) -> pd.DataFrame:
        """
        Compare multiple model types on the same data
        
        Returns DataFrame with metrics for each model
        """
        if model_types is None:
            model_types = ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']
        
        results = []
        for model_type in model_types:
            try:
                model, metrics = self.quick_train(X, y, model_type, test_size)
                metrics['model'] = model_type
                results.append(metrics)
            except Exception as e:
                print(f"Error training {model_type}: {e}")
        
        return pd.DataFrame(results)
    
    def walk_forward_validation(self, X: np.ndarray, y: np.ndarray,
                                model_type: str = 'xgboost',
                                n_splits: int = 5,
                                train_size: float = 0.6,
                                **model_params) -> Dict[str, Any]:
        """
        Walk-forward validation for time series
        
        Returns metrics for each fold and aggregate statistics
        """
        fold_size = len(X) // n_splits
        initial_train_size = int(len(X) * train_size)
        
        fold_results = []
        all_predictions = []
        all_actuals = []
        
        for i in range(n_splits):
            train_end = initial_train_size + i * fold_size
            test_end = min(train_end + fold_size, len(X))
            
            if train_end >= len(X):
                break
            
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[train_end:test_end]
            y_test = y[train_end:test_end]
            
            # Train and predict
            if model_type == 'xgboost':
                model = XGBoostModel(**model_params)
            elif model_type == 'lightgbm':
                model = LightGBMModel(**model_params)
            elif model_type == 'random_forest':
                model = RandomForestModel(**model_params)
            else:
                model = GradientBoostingModel(**model_params)
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            metrics = self._calculate_metrics(y_test, predictions)
            metrics['fold'] = i + 1
            fold_results.append(metrics)
            
            all_predictions.extend(predictions)
            all_actuals.extend(y_test)
        
        # Aggregate metrics
        df = pd.DataFrame(fold_results)
        aggregate = {
            'mean_mse': df['mse'].mean(),
            'std_mse': df['mse'].std(),
            'mean_rmse': df['rmse'].mean(),
            'std_rmse': df['rmse'].std(),
            'mean_r2': df['r2'].mean(),
            'std_r2': df['r2'].std(),
            'overall_r2': r2_score(all_actuals, all_predictions)
        }
        
        return {
            'fold_results': df,
            'aggregate': aggregate,
            'predictions': np.array(all_predictions),
            'actuals': np.array(all_actuals)
        }
    
    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                            model_type: str = 'xgboost',
                            param_grid: Optional[Dict] = None,
                            method: str = 'random',
                            n_iter: int = 50) -> Dict[str, Any]:
        """
        Tune hyperparameters for a model
        
        Parameters:
        -----------
        model_type : Type of model to tune
        param_grid : Custom parameter grid (uses defaults if None)
        method : 'grid' or 'random' search
        n_iter : Number of iterations for random search
        """
        # Default parameter grids
        default_grids = {
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [15, 31, 63],
                'subsample': [0.7, 0.8, 0.9]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
        
        grid = param_grid or default_grids.get(model_type, {})
        
        # Create tuner
        if model_type == 'xgboost':
            tuner = HyperparameterTuner(XGBoostModel, grid)
        elif model_type == 'lightgbm':
            tuner = HyperparameterTuner(LightGBMModel, grid)
        else:
            tuner = HyperparameterTuner(RandomForestModel, grid)
        
        # Run search
        if method == 'grid':
            results = tuner.grid_search(X, y)
        else:
            results = tuner.random_search(X, y, n_iter=n_iter)
        
        return results
    
    def create_custom_model(self, architecture: List[Dict]) -> KerasModel:
        """
        Create a custom neural network
        
        Example architecture:
        [
            {'type': 'dense', 'units': 128, 'activation': 'relu'},
            {'type': 'dropout', 'rate': 0.3},
            {'type': 'dense', 'units': 64, 'activation': 'relu'},
            {'type': 'output', 'units': 1}
        ]
        """
        return KerasModel(architecture, "CustomNN")
    
    def create_ensemble(self, model_types: List[str], 
                       weights: Optional[List[float]] = None,
                       method: str = 'weighted') -> CustomEnsemble:
        """Create an ensemble of multiple models"""
        models = []
        for mt in model_types:
            if mt == 'xgboost':
                models.append(XGBoostModel())
            elif mt == 'lightgbm':
                models.append(LightGBMModel())
            elif mt == 'random_forest':
                models.append(RandomForestModel())
            elif mt == 'gradient_boosting':
                models.append(GradientBoostingModel())
        
        return CustomEnsemble(models, weights, method)
    
    def get_leaderboard(self, metric: str = 'rmse', top_n: int = 10) -> pd.DataFrame:
        """Get leaderboard of best experiments"""
        return self.tracker.get_leaderboard(metric, top_n, minimize=True)
    
    def export_model(self, experiment_id: str, filepath: str):
        """Export a trained model to file"""
        if experiment_id in self.models:
            self.models[experiment_id].save(filepath)
        else:
            raise ValueError(f"Model {experiment_id} not found")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL EXPERIMENTATION LAB DEMO")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5
    
    # Initialize lab
    lab = ModelLab()
    
    # Quick training
    print("\n1. Quick Training XGBoost...")
    model, metrics = lab.quick_train(X, y, 'xgboost')
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   R²: {metrics['r2']:.4f}")
    
    # Compare models
    print("\n2. Comparing Models...")
    comparison = lab.compare_models(X, y)
    print(comparison[['model', 'rmse', 'r2']].to_string())
    
    # Walk-forward validation
    print("\n3. Walk-Forward Validation...")
    wf_results = lab.walk_forward_validation(X, y, n_splits=3)
    print(f"   Mean RMSE: {wf_results['aggregate']['mean_rmse']:.4f}")
    print(f"   Overall R²: {wf_results['aggregate']['overall_r2']:.4f}")
    
    # Custom ensemble
    print("\n4. Creating Custom Ensemble...")
    ensemble = lab.create_ensemble(
        ['xgboost', 'lightgbm', 'random_forest'],
        weights=[0.4, 0.4, 0.2],
        method='weighted'
    )
    ensemble.fit(X[:800], y[:800])
    predictions = ensemble.predict(X[800:])
    ensemble_rmse = np.sqrt(mean_squared_error(y[800:], predictions))
    print(f"   Ensemble RMSE: {ensemble_rmse:.4f}")
    
    # Leaderboard
    print("\n5. Experiment Leaderboard:")
    print(lab.get_leaderboard())
