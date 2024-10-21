import pandas as pd
from typing import Any
from sklearn.ensemble import VotingRegressor

class Voting:
    def __init__(self, 
                 models: list[tuple[str, Any]],
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