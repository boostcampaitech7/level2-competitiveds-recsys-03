import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from model.TreeModel import XGBoost, LightGBM, CatBoost
import optuna
RANDOM_SEED = 42

def set_model(model_name: str, **params):
    match model_name:
        case "xgboost":
            model = XGBoost(**params)
        case "lightgbm":
            model = LightGBM(**params)
        case "catboost":
            model = CatBoost(**params)
    return model

def cv_train(model, X: pd.DataFrame, y: pd.DataFrame, verbose: bool = True) -> float:
    """K-Fold를 이용하여 Cross Validation을 수행하는 함수입니다.

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

def optuna_train(model_name: str, X: pd.DataFrame, y: pd.DataFrame) -> tuple[dict, float]:
    def objective(trial):
        match model_name:
            case "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                    "max_depth": trial.suggest_int("max_depth", 5, 12),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "gamma": trial.suggest_float("gamma", 0, 5)
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
        return cv_train(model, X, y, verbose=False)
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=50)
    return study.best_params, study.best_value