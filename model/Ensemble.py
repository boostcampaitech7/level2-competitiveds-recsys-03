import pandas as pd
from typing import Any
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor

class Voting:
    def __init__(self, 
                 models: list[tuple[str, BaseEstimator]],
                 weights: list[float],
                 random_seed: int = 42):
        self.random_seed = random_seed
        self.models = models
        self.weights = weights
    
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.voting_model = VotingRegressor(estimators=self.models, weights=self.weights)
        self.voting_model.fit(X_train, y_train)
        return self.voting_model
    
    def predict(self, X_valid: pd.DataFrame):
        if self.voting_model == None:
            raise ValueError("Model is not trained.")
        return self.voting_model.predict(X_valid)

class Stacking:
    def __init__(self,
                 models: list[tuple[str, BaseEstimator]],
                 meta_model,
                 random_seed=42):
        self.random_seed = random_seed
        self.models = models
        self.meta_model = meta_model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.stacking_model = StackingRegressor(self.models, self.meta_model, cv=5, n_jobs=-1)
        self.stacking_model.fit(X_train, y_train)
        return self.stacking_model
    
    def predict(self, X_valid: pd.DataFrame):
        if self.stacking_model == None:
            raise ValueError("Model is not trained.")
        return self.stacking_model.predict(X_valid)