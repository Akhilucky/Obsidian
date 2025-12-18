import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

class GridSearch:
    """Grid search optimizer for hyperparameter tuning."""
    
    def __init__(self, model=None, param_grid=None, cv=3):
        self.model = model or RandomForestClassifier()
        self.param_grid = param_grid or {}
        self.cv = cv
        self.grid_search = None
    
    def fit(self, X, y):
        """Fit the grid search model."""
        self.grid_search = GridSearchCV(self.model, self.param_grid, cv=self.cv)
        self.grid_search.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions using the best model."""
        if self.grid_search is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.grid_search.predict(X)
    
    def get_best_params(self):
        """Return the best parameters found."""
        if self.grid_search is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.grid_search.best_params_
    
    def get_best_score(self):
        """Return the best cross-validation score."""
        if self.grid_search is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.grid_search.best_score_
