import pandas as pd
import numpy as np
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
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> object:
        """
        모델 객체를 정의하고 fit하는 함수입니다.

        Args:
            X_train (pd.DataFrame): 독립 변수 데이터
            y_train (pd.Series): 예측 변수 데이터

        Returns:
            object: fit까지 완료된 모델 객체
        """
        self.voting_model = VotingRegressor(estimators=self.models, weights=self.weights)
        self.voting_model.fit(X_train, y_train)
        return self.voting_model
    
    def predict(self, X_valid: pd.DataFrame) -> np.ndarray:
        """
        fit된 모델을 기반으로 예측값을 출력하는 함수입니다.

        Args:
            X_valid (pd.DataFrame): 검증 데이터셋

        Raises:
            ValueError: fit을 하지 않고 해당 함수를 실행했을 때 발생

        Returns:
            np.ndarray: 예측 결과
        """
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
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> object:
        """
        모델 객체를 정의하고 fit하는 함수입니다.

        Args:
            X_train (pd.DataFrame): 독립 변수 데이터
            y_train (pd.Series): 예측 변수 데이터

        Returns:
            object: fit까지 완료된 모델 객체
        """
        self.stacking_model = StackingRegressor(self.models, self.meta_model, cv=5, n_jobs=-1)
        self.stacking_model.fit(X_train, y_train)
        return self.stacking_model
    
    def predict(self, X_valid: pd.DataFrame) -> np.ndarray:
        """
        fit된 모델을 기반으로 예측값을 출력하는 함수입니다.

        Args:
            X_valid (pd.DataFrame): 검증 데이터셋

        Raises:
            ValueError: fit을 하지 않고 해당 함수를 실행했을 때 발생

        Returns:
            np.ndarray: 예측 결과
        """
        if self.stacking_model == None:
            raise ValueError("Model is not trained.")
        return self.stacking_model.predict(X_valid)