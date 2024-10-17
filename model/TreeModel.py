from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import pandas as pd

class XGBoost:
    def __init__(self, random_seed: int = 42, **params):
        self.params = params
        self.random_seed = random_seed
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """모델 객체를 정의하고 fit하는 함수입니다.

        Args:
            X_train (pd.DataFrame): 독립 변수 데이터
            y_train (pd.Series): 예측 변수 데이터

        Returns:
            Any: fit까지 완료된 모델 객체
        """
        self.model = XGBRegressor(**self.params, tree_method="hist", device="cuda", random_state=self.random_seed)
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X_valid: pd.DataFrame):
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


class LightGBM:
    def __init__(self, random_seed: int = 42, **params):
        self.params = params
        self.random_seed = random_seed
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """모델 객체를 정의하고 fit하는 함수입니다.

        Args:
            X_train (pd.DataFrame): 독립 변수 데이터
            y_train (pd.Series): 예측 변수 데이터

        Returns:
            Any: fit까지 완료된 모델 객체
        """
        self.model = LGBMRegressor(**self.params, method="hist", device="cpu", random_state=self.random_seed)
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X_valid: pd.DataFrame):
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
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """모델 객체를 정의하고 fit하는 함수입니다.

        Args:
            X_train (pd.DataFrame): 독립 변수 데이터
            y_train (pd.Series): 예측 변수 데이터

        Returns:
            Any: fit까지 완료된 모델 객체
        """
        self.model = CatBoostRegressor(**self.params, random_state=self.random_seed)
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X_valid: pd.DataFrame):
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