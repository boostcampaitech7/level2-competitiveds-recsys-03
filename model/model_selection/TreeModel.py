from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from wandb.integration.xgboost import WandbCallback
from wandb.integration.lightgbm import wandb_callback, log_summary
import pandas as pd
import numpy as np

class XGBoost:
    def __init__(self, random_seed: int = 42, **params):
        self.params = params
        self.random_seed = random_seed
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> object:
        """모델 객체를 정의하고 fit하는 함수입니다.

        Args:
            X_train (pd.DataFrame): 독립 변수 데이터
            y_train (pd.Series): 예측 변수 데이터

        Returns:
            object: fit까지 완료된 모델 객체
        """
        self.model = XGBRegressor(**self.params, tree_method="hist", device="cuda", random_state=self.random_seed, n_jobs=-1)
        # Feature Importance
        # self.model = XGBRegressor(**self.params, tree_method="hist", device="cuda", random_state=self.random_seed, callbacks=[WandbCallback(log_model=True)], n_jobs=-1) 
        self.model.fit(X_train, y_train)
        return self.model
    
    def train_cls(self, X_train: pd.DataFrame, y_train: pd.Series) -> object:
        """
        모델 객체를 정의하고 fit하는 함수입니다.

        Args:
            X_train (pd.DataFrame): 독립 변수 데이터
            y_train (pd.Series): 예측 변수 데이터

        Returns:
            object: fit까지 완료된 모델 객체
        """
        self.model = XGBClassifier(**self.params, device="cuda", random_state=self.random_seed, use_label_encoder=False, n_jobs=-1)
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X_valid: pd.DataFrame) -> np.ndarray:
        """fit된 모델을 기반으로 예측값을 출력하는 함수입니다.

        Args:
            X_valid (pd.DataFrame): 검증 데이터셋

        Raises:
            ValueError: fit을 하지 않고 해당 함수를 실행했을 때 발생

        Returns:
            np.ndarray: 예측 결과
        """
        if self.model == None:
            raise ValueError("Model is not trained.")
        return self.model.predict(X_valid)

        
    def predict_proba(self, X_valid: pd.DataFrame) -> np.ndarray:
        """
        fit된 모델을 기반으로 예측값을 출력하는 함수입니다.

        Args:
            X_valid (pd.DataFrame): 검증 데이터셋

        Raises:
            ValueError: fit을 하지 않고 해당 함수를 실행했을 때 발생

        Returns:
            np.ndarray: 예측 결과
        """
        if self.model == None:
            raise ValueError("Model is not trained.")
        return self.model.predict_proba(X_valid)

class LightGBM:
    def __init__(self, random_seed: int = 42, **params):
        self.params = params
        self.random_seed = random_seed
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> object:
        """모델 객체를 정의하고 fit하는 함수입니다.

        Args:
            X_train (pd.DataFrame): 독립 변수 데이터
            y_train (pd.Series): 예측 변수 데이터

        Returns:
            object: fit까지 완료된 모델 객체
        """
        self.model = LGBMRegressor(**self.params, method="hist", device="cpu", random_state=self.random_seed, n_jobs=-1)
        self.model.fit(X_train, y_train)

        # Feature Importance
        # self.model.fit(X_train, y_train, callbacks=[wandb_callback()])
        # booster = self.model.booster_
        # log_summary(booster, save_model_checkpoint=True)

        return self.model
    
    def predict(self, X_valid: pd.DataFrame) -> np.ndarray:
        """fit된 모델을 기반으로 예측값을 출력하는 함수입니다.

        Args:
            X_valid (pd.DataFrame): 검증 데이터셋

        Raises:
            ValueError: fit을 하지 않고 해당 함수를 실행했을 때 발생

        Returns:
            np.ndarray: 예측 결과
        """
        if self.model == None:
            raise ValueError("Model is not trained.")
        return self.model.predict(X_valid)


class CatBoost:
    def __init__(self, random_seed: int = 42, **params):
        self.params = params
        self.random_seed = random_seed
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> object:
        """모델 객체를 정의하고 fit하는 함수입니다.

        Args:
            X_train (pd.DataFrame): 독립 변수 데이터
            y_train (pd.Series): 예측 변수 데이터

        Returns:
            object: fit까지 완료된 모델 객체
        """
        self.model = CatBoostRegressor(**self.params, random_state=self.random_seed, n_jobs=-1)
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X_valid: pd.DataFrame) -> np.ndarray:
        """fit된 모델을 기반으로 예측값을 출력하는 함수입니다.

        Args:
            X_valid (pd.DataFrame): 검증 데이터셋

        Raises:
            ValueError: fit을 하지 않고 해당 함수를 실행했을 때 발생

        Returns:
            np.ndarray: 예측 결과
        """
        if self.model == None:
            raise ValueError("Model is not trained.")
        return self.model.predict(X_valid)

class RandomForest:
    def __init__(self, random_seed: int = 42, **params):
        self.params = params
        self.random_seed = random_seed
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> object:
        """모델 객체를 정의하고 fit하는 함수입니다.

        Args:
            X_train (pd.DataFrame): 독립 변수 데이터
            y_train (pd.Series): 예측 변수 데이터

        Returns:
            Any: fit까지 완료된 모델 객체
        """
        self.model = RandomForestRegressor(**self.params, random_state=self.random_seed, n_jobs=-1)
        self.model.fit(X_train, y_train)
        return self.model
    
    def train_cls(self, X_train: pd.DataFrame, y_train: pd.Series) -> object:
        """
        모델 객체를 정의하고 fit하는 함수입니다.

        Args:
            X_train (pd.DataFrame): 독립 변수 데이터
            y_train (pd.Series): 예측 변수 데이터

        Returns:
            object: fit까지 완료된 모델 객체
        """
        self.model = RandomForestClassifier(**self.params, random_state=self.random_seed, n_jobs=-1)
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X_valid: pd.DataFrame) -> np.ndarray:
        """fit된 모델을 기반으로 예측값을 출력하는 함수입니다.

        Args:
            X_valid (pd.DataFrame): 검증 데이터셋

        Raises:
            ValueError: fit을 하지 않고 해당 함수를 실행했을 때 발생

        Returns:
            np.ndarray: 예측 결과
        """
        if self.model == None:
            raise ValueError("Model is not trained.")
        return self.model.predict(X_valid)
    
    def predict_proba(self, X_valid: pd.DataFrame) -> np.ndarray:
        """
        fit된 모델을 기반으로 예측값을 출력하는 함수입니다.

        Args:
            X_valid (pd.DataFrame): 검증 데이터셋

        Raises:
            ValueError: fit을 하지 않고 해당 함수를 실행했을 때 발생

        Returns:
            np.ndarray: 예측 결과
        """
        if self.model == None:
            raise ValueError("Model is not trained.")
        return self.model.predict_proba(X_valid)