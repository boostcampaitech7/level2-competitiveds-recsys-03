import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from typing import Any
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from model.TreeModel import XGBoost, LightGBM, CatBoost
from model.Ensemble import Voting
import optuna
import optuna.logging
# Optuna의 기본 로그 출력을 비활성화
optuna.logging.set_verbosity(optuna.logging.WARNING)
RANDOM_SEED = 42

def set_model(model_name: str, params, models: dict[str, Any] = None):
    """
    주어진 모델 이름에 따라 모델을 생성하고 반환하는 함수입니다.

    Args:
        model_name (str): 생성하려는 모델 이름
        params: 모델 생성 시 사용할 하이퍼파라미터

    Returns:
        model (object): 생성된 모델 객체
    """
    match model_name:
        case "xgboost":
            model = XGBoost(**params)
        case "lightgbm":
            model = LightGBM(**params)
        case "catboost":
            model = CatBoost(**params)
        case "Voting":
            model = Voting(models=models, weights=params)
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

    mae_list, mae_list_train = [], []
    for i, (train_idx, valid_idx) in enumerate(kfold.split(X, y), start=1):
        if verbose: print(f"training...[{i}/{cv}]")

        X_train, y_train = X.loc[train_idx, :], y.loc[train_idx, "log_deposit"]
        X_valid, y_valid = X.loc[valid_idx, :], y.loc[valid_idx, "deposit"]

        model.train(X_train, y_train)

        y_pred = model.predict(X_valid)
        y_pred = np.expm1(y_pred)

        y_pred_train = model.predict(X_train)
        y_pred_train = np.expm1(y_pred_train)
        fold_mae = mean_absolute_error(y_valid, y_pred)
        fold_mae_train = mean_absolute_error(np.expm1(y_train), y_pred_train)
        if verbose: print(f"Valid MAE: {fold_mae:.4f}")
        mae_list.append(fold_mae)
        mae_list_train.append(fold_mae_train)

    mae = np.mean(mae_list)
    mae_train = np.mean(mae_list_train)
    if verbose:
        print("### K-fold Result ###")
        print(f"Valid MAE: {mae:.4f}")
    
    return mae, mae_train

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
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "max_depth": trial.suggest_int("max_depth", 5, 12),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "n_jobs": -1
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
        model = set_model(model_name, **params)
        val_mae, train_mae = cv_train(model, X, y, verbose=False)
        print(datetime.now().strftime(f"[%Y-%m-%d %H:%M:%S]"), end=" ")
        print(f"Trial {trial.number}", end=" ===> ")
        print(f"Train Value: {train_mae:.4f}", end=" ")
        return val_mae
    
    # 콜백 함수 정의 (현재 trial 값과 최적 trial 값 출력)
    def print_formatted_params(study, trial):
        print(f"Valid Value: {trial.value:.4f}", end=", ")
        
        # 현재까지 최적의 trial 값 출력
        if study.best_value is not None:
            print(f"Best Value: {study.best_value:.4f}", end=" | ")
        
        formatted_params = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in trial.params.items()}
        print(formatted_params)
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, callbacks=[print_formatted_params])
    return study.best_params, study.best_value

def voting_train(
        models: list[str],
        X: pd.DataFrame,
        y: pd.DataFrame,
        n_trials: int = 50
    ) -> tuple[dict, float]:
    """
    optuna를 이용한 Voting Regressor 최적화 함수입니다.

    Args:
        models (list[str]): 보팅을 수행할 모델의 리스트.
        X (pd.DataFrame): 독립 변수
        y (pd.DataFrame): 예측 변수
        n_trials (int, optional): optuna 시행 횟수. Defaults to 50.

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
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                        "max_depth": trial.suggest_int("max_depth", 5, 12),
                        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    }
                    model = XGBRegressor(**params, random_state=42, device="cuda")
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
                    model = LGBMRegressor(**params, random_state=42, device="cuda")
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
        
        voting_model = set_model(model_name="Voting", models=model_params, params=weights)
        return cv_train(voting_model, X, y, verbose=False)
    
    # 최적화 수행
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value