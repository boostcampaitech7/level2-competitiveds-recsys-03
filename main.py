from data.load_dataset import load_dataset
from data.merge_dataset import merge_dataset
from data.feature_engineering import *
from data.data_preprocessing import *
from model.inference import save_csv
from model.feature_select import select_features
from model.data_split import split_features_and_target
from model.log_transformation import apply_log_transformation
from model.model_train import cv_train, set_model, optuna_train
import argparse
import os
import wandb

# 메인 실행 코드
if __name__ == "__main__":

    ### 0. Argument Parsing

    parser = argparse.ArgumentParser(prog = "수도권 아파트 전세 예측", description="사용법 python train.py --model 모델명(소문자)")
    parser.add_argument("--model", type=str, choices=["xgboost", "lightgbm", "catboost", "ensemble"], default="xgboost", help="Select the model to train")
    parser.add_argument("--optuna", type=str, choices=["on", "off"], default="off", help="Select Optuna option")
    parser.add_argument("--project", type=str, default="no_name", help="Input the project name")
    parser.add_argument("--run", type=str, default="no_name", help="Input the run name")
    args = parser.parse_args()

    ### 1. WandB Initialization

    wandb.init(
        settings=wandb.Settings(start_method="thread"),
        dir=None,  # 로컬에 로그 저장하지 않음
        entity="remember-us", # team name,
        project=args.project, # project name
        name=args.run, # run name
        config={
            "User": os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        } # common setting
    )

    ### 2. Data Load

    # 기존 데이터 불러오기
    train_data, test_data, sample_submission, interest_data, subway_data, school_data, park_data = load_dataset()
    # 기존 데이터에 새로운 feature들을 병합한 데이터프레임 불러오기
    train_data, test_data = merge_dataset(train_data, test_data, interest_data, subway_data, school_data, park_data)
    
    ### 3. Data Preprocessing

    # 위치 중복도 낮은 행 삭제
    train_data = delete_low_density(train_data, 2, 6)
    
    # built_year > 2024 행 삭제
    train_data = train_data[train_data["built_year"] < 2024]
    train_data.reset_index(drop=True, inplace=True)
    
    ### 4. Feature Engineering

    # log 변환
    train_data, test_data = apply_log_transformation(train_data, test_data)

    # # clustering
    # feature_columns = [
    #     "latitude",
    #     "longitude",
    #     "log_subway_distance",
    #     "log_school_distance",
    #     "log_park_distance",
    #     "num_of_subways_within_radius",
    #     "num_of_schools_within_radius",
    #     "num_of_parks_within_radius",
    # ]
    # coords = train_data[feature_columns]
    # cm = ClusteringModel(data=coords)
    # n_clusters = 20
    # print("n_clusters:", n_clusters)
    # kmeans_model = cm.kmeans_clustering(n_clusters, train_data, test_data, feature_columns, "cluster")
    # train_data["cluster"] = kmeans_model.predict(train_data[feature_columns])
    # test_data["cluster"] = kmeans_model.predict(test_data[feature_columns])

    # train_data split
    X, y = split_features_and_target(train_data)
    
    # Feature Select
    X, test_data = select_features(X, y, test_data)
    
    ### 5. Model Train and Evaulate
    
    if args.optuna == "on":
        best_params, mae = optuna_train(args.model, X, y)
    else:
        best_params = {
            'n_estimators': 249,
            'learning_rate': 0.1647758714498898,
            'max_depth': 12,
            'subsample': 0.9996749158433582
        }

        best_model = set_model(args.model, **best_params)
        mae = cv_train(best_model, X, y)

    best_model = set_model(args.model, **best_params)
    best_model = best_model.train(X, y["log_deposit"])

    ### 6. WandB Log and Finish

    wandb.log({
        "features": list(X.columns),
        "model": args.model,
        "params": best_params,
        "valid MAE": mae
    })
    wandb.finish()
    
    ### 7. Inference

    save_csv(best_model, test_data, sample_submission)

    print(f"🧼 [{args.project} - {args.run}] Completed 🧼")
    