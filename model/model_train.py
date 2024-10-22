import pandas as pd
import numpy as np
from typing import Any
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from model.TreeModel import XGBoost, LightGBM, CatBoost
from model.Ensemble import Voting, Stacking
import optuna
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_SEED = 42

def set_model(
        model_name: str,
        params: Any = None,
        models: list[tuple[str, BaseEstimator]] = None,
        weights: list[float] = None,
        meta_model: BaseEstimator = None
    ) -> BaseEstimator:
    """
    주어진 모델 이름에 따라 모델을 생성하고 반환하는 함수입니다.

    Args:
        model_name (str): 생성하려는 모델 이름
        params: 모델 생성 시 사용할 하이퍼파라미터
        models (list[tuple[str, BaseEstimator]]): (앙상블) 앙상블을 수행할 모델
        weights (list[float]): (보팅) 보팅 가중치 배열
        meta model (BaseEstimator): (스태킹) 메타 모델

    Returns:
        BaseEstimator: 생성된 모델 객체
    """
    match model_name:
        case "xgboost":
            model = XGBoost(**params)
        case "lightgbm":
            model = LightGBM(**params)
        case "catboost":
            model = CatBoost(**params)
        case "voting":
            model = Voting(models=models, weights=weights)
        case "stacking":
            model = Stacking(models=models, meta_model=meta_model)
    return model

def cv_train(model, X: pd.DataFrame, y: pd.DataFrame, verbose: bool = True) -> float:
    """
    K-Fold를 이용하여 Cross Validation을 수행하는 함수입니다.

    Args:
        model: 수행하려는 모델
        X (pd.DataFrame): 독립 변수
        y (pd.DataFrame): 예측 변수. deposit과 log_deposit 열로 나뉨.
        verbose (bool, optional): Fold별 진행상황을 출력할지 여부. Defaults to True.

    Returns:
        float: 평균 MAE
    """
    cv = 5
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

    mae_list = []
    for i, (train_idx, valid_idx) in enumerate(kfold.split(X, y), start=1):
        if verbose: print(f"training...[{i}/{cv}]")

        X_train, y_train = X.loc[train_idx, :], y.loc[train_idx, "log_deposit"]
        X_valid, y_valid = X.loc[valid_idx, :], y.loc[valid_idx, "deposit"]

        model.train(X_train, y_train)

        y_pred = model.predict(X_valid)
        y_pred = np.expm1(y_pred)
        fold_mae = mean_absolute_error(y_valid, y_pred)
        if verbose: print(f"Valid MAE: {fold_mae:.4f}")
        mae_list.append(fold_mae)

    mae = np.mean(mae_list)
    if verbose:
        print("### K-fold Result ###")
        print(f"Valid MAE: {mae:.4f}")
    
    return mae

def optuna_train(
        model_name: str,
        X: pd.DataFrame,
        y: pd.DataFrame,
        n_trials: int = 50
) -> tuple[dict, float]:
    """
    Optuna를 사용하여 주어진 모델의 하이퍼파라미터를 최적화하는 함수입니다.

    Args:
        model_name (str): 최적화할 모델의 이름
        X (pd.DataFrame): 독립 변수
        y (pd.DataFrame): 예측 변수
        n_trials (int): optuna trial 수

    Returns:
        tuple[dict, float]:
            - dict: 최적의 하이퍼파라미터
            - float: 최적의 하이퍼파라미터에 대한 성능 지표(MAE)
    """
    def objective(trial):
        match model_name:
            case "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                    "max_depth": trial.suggest_int("max_depth", 5, 12),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                }
            case "lightgbm":
                params = {
                    "verbose": -1,
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 5, 12),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                    "objective": "regression_l1"
                }
            case "catboost":
                params = {
                    "verbose": 0,
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "iterations": trial.suggest_int("iterations", 50, 500),
                    "depth": trial.suggest_int("depth", 3, 10),
                    "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
                    # "bagging_temperature": trial.suggest_loguniform("bagging_temperature", 0.01, 1),
                    # "border_count": trial.suggest_int("border_count", 32, 255),
                    "cat_features": ["contract_day"],
                    "task_type": "GPU",
                    "devices": "cuda",
                }
        model = set_model(model_name=model_name, params=params)
        return cv_train(model, X, y, verbose=False)
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

def voting_train(
        models: list[str],
        X: pd.DataFrame,
        y: pd.DataFrame,
        n_trials: int = 50
) -> tuple[list[float], dict, float]:
    """
    optuna를 이용한 Voting Regressor 최적화 함수입니다.

    Args:
        models (list[str]): 보팅을 수행할 모델의 리스트.
        X (pd.DataFrame): 독립 변수
        y (pd.DataFrame): 예측 변수
        n_trials (int, optional): optuna 시행 횟수. Defaults to 50.

    Returns:
        tuple[list[float], dict, float]:
            - list[float]: 최적의 보팅 가중치
            - dict: 각 모델별 최적의 하이퍼파라미터를 담은 딕셔너리
            - float: 최적의 하이퍼파라미터에 대한 성능 지표(MAE)
    """
    def objective(trial):
        model_params = []
        for model_name in models:
            # 개별 모델 및 하이퍼파라미터 정의
            match model_name:
                case "xgboost":
                    params = {
                        "n_estimators": trial.suggest_int("XGB_n_estimators", 50, 300),
                        "learning_rate": trial.suggest_float("XGB_learning_rate", 0.01, 0.2),
                        "max_depth": trial.suggest_int("XGB_max_depth", 5, 12),
                        "subsample": trial.suggest_float("XGB_subsample", 0.5, 1.0),
                    }
                    model = XGBRegressor(**params, random_state=42, device="cuda")
                case "lightgbm":
                    params = {
                        "verbose": -1,
                        "n_estimators": trial.suggest_int("LGBM_n_estimators", 50, 300),
                        "learning_rate": trial.suggest_float("LGBM_learning_rate", 0.01, 0.3, log=True),
                        "max_depth": trial.suggest_int("LGBM_max_depth", 5, 12),
                        "subsample": trial.suggest_float("LGBM_subsample", 0.5, 1.0),
                        "num_leaves": trial.suggest_int("LGBM_num_leaves", 20, 150),
                        "objective": "regression_l1"
                    }
                    model = LGBMRegressor(**params, random_state=42, device="cuda")
                case "catboost":
                    params = {
                        "verbose": 0,
                        "learning_rate": trial.suggest_float("Cat_learning_rate", 0.01, 0.3),
                        "iterations": trial.suggest_int("Cat_iterations", 50, 500),
                        "depth": trial.suggest_int("Cat_depth", 3, 10),
                        "l2_leaf_reg": trial.suggest_int("Cat_l2_leaf_reg", 1, 10),
                        # "bagging_temperature": trial.suggest_loguniform("bagging_temperature", 0.01, 1),
                        # "border_count": trial.suggest_int("border_count", 32, 255),
                        "cat_features": ["contract_day"],
                        "devices": "cuda",
                    }
                    model = CatBoostRegressor(**params, random_state=42)
            # 통합 모델 정의
            model_params.append((model_name, model))

        # 가중치 설정
        weights = []
        for model_name in models:
            weight = trial.suggest_float(f"{model_name} weight", 0.0, 1.0)
            weights.append(weight)

        # 가중치의 합이 1이 되도록 정규화
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)  # 모든 가중치를 동일하게 설정
        voting_model = set_model(model_name="voting", models=model_params, weights=weights)

        trial.set_user_attr("models", model_params)
        trial.set_user_attr("weights", weights)
        return cv_train(voting_model, X, y, verbose=False)
    
    # 최적화 수행
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    best_models = study.best_trial.user_attrs["models"]
    best_weights = study.best_trial.user_attrs["weights"]

    return best_weights, best_models, study.best_value

def stacking_train(
        models: list[str],
        X: pd.DataFrame,
        y: pd.DataFrame,
        meta_model: BaseEstimator = LinearRegression(),
        n_trials: int = 50
) -> tuple[dict, float]:
    """
    optuna를 이용한 Stacking Regressor 최적화 함수입니다.

    Args:
        models (list[str]): 앙상블을 수행할 모델의 리스트
        X (pd.DataFrame): 독립 변수
        y (pd.DataFrame): 예측 변수
        meta_model (BaseEstimator, optional): 메타 모델. Defaults to LinearRegression().
        n_trials (int, optional): optuna의 trial 횟수. Defaults to 50.

    Returns:
        tuple[dict, float]:
            - dict: 최적의 하이퍼파라미터
            - float: 최적의 하이퍼파라미터에 대한 성능 지표(MAE)
    """
    def objective(trial):
        model_params = []
        for model_name in models:
            # 개별 모델 및 하이퍼파라미터 정의
            match model_name:
                case "xgboost":
                    params = {
                        "n_estimators": trial.suggest_int("XGB_n_estimators", 50, 300),
                        "learning_rate": trial.suggest_float("XGB_learning_rate", 0.01, 0.2),
                        "max_depth": trial.suggest_int("XGB_max_depth", 5, 12),
                        "subsample": trial.suggest_float("XGB_subsample", 0.5, 1.0),
                    }
                    model = XGBRegressor(**params, random_state=42, device="cuda")
                case "lightgbm":
                    params = {
                        "verbose": -1,
                        "n_estimators": trial.suggest_int("LGBM_n_estimators", 50, 300),
                        "learning_rate": trial.suggest_float("LGBM_learning_rate", 0.01, 0.3, log=True),
                        "max_depth": trial.suggest_int("LGBM_max_depth", 5, 12),
                        "subsample": trial.suggest_float("LGBM_subsample", 0.5, 1.0),
                        "num_leaves": trial.suggest_int("LGBM_num_leaves", 20, 150),
                        "objective": "regression_l1"
                    }
                    model = LGBMRegressor(**params, random_state=42, device="cuda")
                case "catboost":
                    params = {
                        "verbose": 0,
                        "learning_rate": trial.suggest_float("Cat_learning_rate", 0.01, 0.3),
                        "iterations": trial.suggest_int("Cat_iterations", 50, 500),
                        "depth": trial.suggest_int("Cat_depth", 3, 10),
                        "l2_leaf_reg": trial.suggest_int("Cat_l2_leaf_reg", 1, 10),
                        # "bagging_temperature": trial.suggest_loguniform("bagging_temperature", 0.01, 1),
                        # "border_count": trial.suggest_int("border_count", 32, 255),
                        "cat_features": ["contract_day"],
                        "devices": "cuda",
                    }
                    model = CatBoostRegressor(**params, random_state=42)
            # 통합 모델 정의
            model_params.append((model_name, model))

        # 스태킹 모델 정의
        stacking_model = set_model(model_name="stacking", models=model_params, meta_model=meta_model)

        trial.set_user_attr("models", model_params)
        return cv_train(stacking_model, X, y, verbose=False)
    
    # 최적화 수행
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    best_models = study.best_trial.user_attrs["models"]

    return best_models, study.best_value