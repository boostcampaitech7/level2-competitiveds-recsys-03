from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import h2o
from h2o.estimators import H2ORandomForestEstimator
from wandb.integration.xgboost import WandbCallback
from wandb.integration.lightgbm import wandb_callback, log_summary
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
        self.model = XGBRegressor(**self.params, tree_method="hist", device="cuda", random_state=self.random_seed, n_jobs=-1)
        # Feature Importance
        # self.model = XGBRegressor(**self.params, tree_method="hist", device="cuda", random_state=self.random_seed, callbacks=[WandbCallback(log_model=True)], n_jobs=-1) 
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
        self.model = LGBMRegressor(**self.params, method="hist", device="cpu", random_state=self.random_seed, n_jobs=-1)
        self.model.fit(X_train, y_train)

        # Feature Importance
        # self.model.fit(X_train, y_train, callbacks=[wandb_callback()])
        # booster = self.model.booster_
        # log_summary(booster, save_model_checkpoint=True)

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
        self.model = CatBoostRegressor(**self.params, random_state=self.random_seed, n_jobs=-1)
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

class RandomForest:
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
        self.model = RandomForestRegressor(**self.params, random_state=self.random_seed, n_jobs=-1)
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

class H2ORandomForest:
    def __init__(self, random_seed: int = 42, **params):
        self.params = params
        self.random_seed = random_seed
        if not h2o.is_running():
            h2o.init() # H2O 클러스터 초기화
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """모델 객체를 정의하고 fit하는 함수입니다.

        Args:
            X_train (pd.DataFrame): 독립 변수 데이터
            y_train (pd.Series): 예측 변수 데이터

        Returns:
            Any: fit까지 완료된 모델 객체
        """
        # H2O 데이터프레임으로 변환
        train_h2o = h2o.H2OFrame(X_train)
        train_h2o['target'] = h2o.H2OFrame(y_train)

        # H2O 랜덤 포레스트 모델 학습
        self.model = H2ORandomForestEstimator(**self.params)
        self.model.train(x=X_train.columns.tolist(), y='target', training_frame=train_h2o)
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
        valid_h2o = h2o.H2OFrame(X_valid) # H2O 데이터프레임으로 변환
        predictions_h2o = self.model.predict(valid_h2o) # 예측 수행
        return predictions_h2o.as_data_frame()["predict"].values # 예측 결과 반환