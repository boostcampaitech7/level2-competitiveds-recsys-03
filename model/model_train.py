import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from model.TreeModel import XGBoost, LightGBM, CatBoost
import optuna
RANDOM_SEED = 42

def set_model(model_name: str, train_type: str, **params):
    match model_name:
        case "xgboost":
            model = XGBoost(**params)
        case "lightgbm":
            model = LightGBM(**params)
        case "catboost":
            model = CatBoost(**params)
    return model

def cv_train(model, X, y, verbose: bool = True):
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

def optuna_train(model_name, X, y):
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
                "verbose" : -1,
                "num_leaves" : trial.suggest_int("num_leaves", 20, 150),
                "learning_rate" : trial.suggest_loguniform("learning_rate", 0.01, 0.3),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth" : trial.suggest_int("max_depth", 5, 12), # 과적합 방지
                "subsample" : trial.suggest_uniform("subsample", 0.5, 1.0), # 데이터 샘플링 비율, 과적합 방지
                # # 정규화 파라미터 범위 조정
                # "reg_alpha" : trial.suggest_loguniform("reg_alpha", 1e-3, 1.0), 
                # "reg_lambda" : trial.suggest_loguniform("reg_lambda", 1e-3, 1.0),
                "random_state" : 42,
                #"force_col_wise": True,
                #"device" : "gpu",
                "objective": "regression_l1" 
            }
            case "catboost":
                params = {
                    "iterations": trial.suggest_int("iterations", 100, 1000),
                    "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
                    "depth": trial.suggest_int("depth", 5, 12),
                    "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
                    "bagging_temperature": trial.suggest_loguniform("bagging_temperature", 0.01, 1),
                    "border_count": trial.suggest_int("border_count", 32, 255),
                    "cat_features": ["contract_year",
                                    #  "contract_month",
                                    #  "contract_day",
                                    "contract_type"],
                    "task_type": "GPU",
                    "devices": "0",
                    "verbose": 0
                }
        model = set_model(model_name, train_type="regression", **params)
        return cv_train(model, X, y, verbose=False)
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=50)
    return study.best_params, study.best_value