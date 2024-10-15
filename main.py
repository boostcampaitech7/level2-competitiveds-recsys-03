from data.feature_engineering import find_nearest_haversine_distance
from data.load_dataset import load_dataset
from data.merge_dataset import merge_dataset
from model.inference import save_csv
from model.feature_select import select_features
from model.data_split import split_features_and_target
from model.log_transformation import apply_log_transformation
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import optuna
import wandb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from model.model_train import cv_train, set_model

# 메인 실행 코드
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "수도권 아파트 전세 예측", description="사용법 python train.py --model 모델명(소문자)")
    parser.add_argument("--model", type=str, choices=["xgboost", "lightgbm", "catboost", "ensemble"], default="xgboost", help="Select the model to train")
    parser.add_argument("--project", type=str, default="no_name", help="Input the project name")
    parser.add_argument("--run", type=str, default="no_name", help="Input the run name")
    args = parser.parse_args()

    # 0. WandB 초기화
    wandb.init(
        settings=wandb.Settings(start_method="thread"),
        dir=None,  # 로컬에 로그 저장하지 않음
        entity="remember-us", # team name,
        project=args.project, # project name
        name=args.run, # run name
        config={
            "random_state": 42,
            "device": "cuda"
        } # common setting
    )

    # 1. 데이터 로드
    # 기존 데이터 불러오기
    train_data, test_data, sample_submission, interest_data, subway_data, school_data, park_data = load_dataset()
    # 기존 데이터에 새로운 feature들을 병합한 데이터프레임 불러오기
    train_data, test_data = merge_dataset(train_data, test_data, interest_data, subway_data, school_data, park_data)
    
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
    train_data, test_data = apply_log_transformation(train_data, test_data)
    
    # Feature Select
    train_data, test_data, train_cols = select_features(train_data, test_data)
    
    # train_data split
    X, y = split_features_and_target(train_data)
    
    # Model train and evaulate
    best_params = {'n_estimators': 249, 'learning_rate': 0.1647758714498898, 'max_depth': 12, 'subsample': 0.9996749158433582}
    best_model = set_model(args.model, train_type="regression", **best_params)
    mae = cv_train(best_model, X, y)

    best_model = best_model.train(X, y["log_deposit"])
    
    # WandB log and finish
    wandb.log({
        "features": train_cols,
        "model": args.model,
        "params": best_params,
        "valid MAE": mae
    })
    wandb.finish()
    
    # 테스트 데이터에 대한 예측 및 제출 파일 생성
    save_csv(best_model, test_data, sample_submission)

    print(f"🧼 [{args.project} - {args.run}] Completed 🧼")
    