import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from TreeModel import XGBoost, LightGBM, CatBoost
RANDOM_SEED = 42

def set_model(model_name: str, train_type: str, **params) -> Any:
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