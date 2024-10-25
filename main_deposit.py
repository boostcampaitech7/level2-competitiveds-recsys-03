from data.load_dataset import load_dataset
from data.merge_dataset import merge_dataset
from data.data_preprocessing import *
from data.feature_engineering import *
from model.feature_select import select_features
from model.data_split import split_features_and_target
from model.model_train import *
from model.deposit_group import *
import argparse
import os
import wandb
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 메인 실행 코드
if __name__ == "__main__":

    ### 0. Argument Parsing

    parser = argparse.ArgumentParser(prog = "수도권 아파트 전세 예측", description="사용법 python train.py --model 모델명(소문자)")
    parser.add_argument("--model", type=str, choices=["xgboost", "randomforest"], default="xgboost", help="Select the model to train")
    parser.add_argument("--optuna", type=str, choices=["on", "off"], default="off", help="Select Optuna option")
    parser.add_argument("--project", type=str, default="deposit", help="Input the project name")
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

    # urgent_sale_cut 사용시 필요한 예시 코드
    # cut_index = urgent_sale_cut(train_data, 2.5)

    # 위치 중복도 낮은 행 삭제
    train_data = delete_low_density(train_data, 2, 5)
    
    # built_year > 2024 행 삭제
    train_data = train_data[train_data["built_year"] < 2024]
    train_data.reset_index(drop=True, inplace=True)
    
    ### 4. Feature Engineering

    # log 변환
    train_data, test_data = apply_log_transformation(train_data, test_data)

    # 가격 범주화
    sorted_train_data = train_data.sort_values(by="deposit").reset_index(drop=True) # 데이터 정렬 및 인덱스 리셋
    sorted_train_data["deposit_group"] = sorted_train_data["deposit"].apply(categorize_deposit) # 그룹화 적용
    train_data = sorted_train_data
    print(train_data.groupby("deposit_group")["deposit"].agg(["min", "max", "mean", "count"])) # 그룹별 통계 출력

    # train_data split
    X, y = split_features_and_target(train_data, target=["deposit_group"])
    
    # Feature Select
    X, test_data = select_features(X, y, test_data)
    
    ### 5-1. [deposit_group] Model Train and Evaulate

    if args.optuna == "on":
        best_params, _ = deposit_optuna_train(args.model, "cls", X, y)
    else:
        best_params = {
            "n_estimators": 277,
            "learning_rate": 0.16274957919272004,
            "max_depth": 12,
            "subsample": 0.6367009952175001,
            "colsample_bytree": 0.7892810174117046,
            "gamma": 0.5556970036799329
        }
    
    best_model = set_model(args.model, best_params)
    best_model.train_cls(X, y)
    predicted_groups = best_model.predict(test_data)
    test_data["predicted_group"] = predicted_groups
    
    print(f"✨ Classification Completed 💫")

    ### 5-2. [deposit] Model Train and Evaulate
    
    # deposit_group별로 회귀 모델 훈련 및 예측
    group_models, group_params, mae = train_regressors_per_group(args.model, train_data, list(X.columns), args.optuna)
    y_test_pred = predict_per_group(test_data, group_models, list(X.columns))

    ### 6. WandB Log and Finish

    group_params_str = json.dumps(group_params, indent=4)
    wandb.log({
        "features": list(X.columns),
        "model": args.model,
        "params": best_params,
        "group_params": group_params_str,
        "valid MAE": mae,
        "optuna": args.optuna
    })
    wandb.finish()
    
    ### 7. Inference

    sample_submission["deposit"] = y_test_pred
    file_name = args.run + ".csv"
    sample_submission.to_csv(file_name, index=False)

    print(f"🧼 [{args.project} - {args.run}] Completed 🧼")
    