from data.feature_engineering import find_nearest_haversine_distance
from data.load_dataset import load_dataset
from model.inference import save_csv
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import optuna
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

# 메인 실행 코드
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "수도권 아파트 전세 예측", description="사용법 python train.py --model 모델명(소문자)")
    parser.add_argument("--model", type=str, choices=["xgboost", "lightgbm", "catboost", "ensemble"], default="xgboost", help="Select the model to train")
    args = parser.parse_args()
    # 1. 데이터 로드
    train_data, test_data, sample_submission = load_dataset()
    
    # 2. 데이터 전처리
    # 위치 중복도 낮은 행 삭제
    groups = train_data.groupby(["latitude", "longitude"])["index"].count()
    conditioned_groups_index = groups[(groups >= 2) & (groups <= 5)].index # 이 범위를 파라미터로 조정하는걸로
    small_groups = train_data[
        train_data["latitude"].isin(conditioned_groups_index.get_level_values(0)) &
        train_data["longitude"].isin(conditioned_groups_index.get_level_values(1))
    ]
    train_data.drop(small_groups.index, axis=0, inplace=True)
    # built_year > 2024 행 삭제
    train_data = train_data[train_data["built_year"] < 2024]
    train_data.reset_index(drop=True, inplace=True)
  
    # log 변환
    train_data["log_deposit"] = np.log1p(train_data["deposit"])
    train_data["log_area_m2"] = np.log1p(train_data["area_m2"])
    train_data["log_school_distance"] = np.log1p(train_data["nearest_school_distance"])
    train_data["log_park_distance"] = np.log1p(train_data["nearest_park_distance"])
    train_data["log_subway_distance"] = np.log1p(train_data["nearest_subway_distance"])
    test_data["log_area_m2"] = np.log1p(test_data["area_m2"])
    test_data["log_school_distance"] = np.log1p(test_data["nearest_school_distance"])
    test_data["log_park_distance"] = np.log1p(test_data["nearest_park_distance"])
    test_data["log_subway_distance"] = np.log1p(test_data["nearest_subway_distance"])
    
    # 3. Feature Select
    # train 피처 및 타겟 설정
    train_cols = [
        "deposit",
        "log_deposit",
        "log_area_m2",
        "built_year",
        "latitude",
        "longitude",
        "log_subway_distance",
        "log_school_distance",
        "log_park_distance",
        "contract_year_month",
        "contract_day",
    ]
    train_data = train_data[train_cols]
    
    # test 피처 및 타겟 설정
    test_cols = [
        "log_area_m2",
        "built_year",
        "latitude",
        "longitude",
        "log_subway_distance",
        "log_school_distance",
        "log_park_distance",
        "contract_year_month",
        "contract_day",
    ]
    test_data = test_data[test_cols]
    
    # 데이터 분리
    X = train_data.drop(columns=["deposit", "log_deposit"], inplace=False)
    y = train_data[["deposit", "log_deposit"]]
    
    # 4. 모델 학습 및 평가(모듈화 필요)
    best_params = {'n_estimators': 249, 'learning_rate': 0.1647758714498898, 'max_depth': 12, 'subsample': 0.9996749158433582}
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    mae = []
    for train_idx, valid_idx in kfold.split(X, train_data["deposit"]):
        X_train, y_train = X.loc[train_idx, :], y.loc[train_idx, "log_deposit"]
        X_valid, y_valid = X.loc[valid_idx, :], y.loc[valid_idx, "deposit"]
        best_model = xgb.XGBRegressor(**best_params, tree_method="hist", device = "cuda", random_state=42)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_valid)
        y_pred = np.expm1(y_pred)
        mae.append(mean_absolute_error(y_valid, y_pred))

    print(f"{np.mean(mae):.4f}")
    
    # 5. 테스트 데이터에 대한 예측 및 제출 파일 생성
    save_csv(best_model, test_data, sample_submission)
    