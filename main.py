from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from data.load_dataset import load_dataset
from data.merge_dataset import merge_dataset
from data.feature_engineering import *
from model.inference import save_csv
from model.feature_select import select_features
from model.data_split import split_features_and_target
from model.model_train import *
import argparse
import os
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 메인 실행 코드
if __name__ == "__main__":

    ### 0. Argument Parsing

    parser = argparse.ArgumentParser(prog = "수도권 아파트 전세 예측", description="사용법 python train.py --model 모델명(소문자)")
    parser.add_argument("--model", type=str, choices=["xgboost", "lightgbm", "catboost", "voting", "stacking"], default="xgboost", help="Select the model to train")
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
    
    ### 4. Feature Engineering

    # log 변환
    train_data, test_data = apply_log_transformation(train_data, test_data)

    # train_data split
    X, y = split_features_and_target(train_data)
    
    # Feature Select
    X, test_data = select_features(X, y, test_data)
    
    ### 5. Model Train and Evaulate
    match args.model:
        case "voting":
            models = ["xgboost", "catboost"]
            best_weights, best_models, mae = voting_train(models, X, y)
            best_model = set_model(model_name="voting", weights=best_weights, models=best_models)
            best_model = best_model.train(X, y["log_deposit"])
            best_params = str(best_models)
        case "stacking":
            meta_model = LinearRegression()
            models = ["xgboost", "catboost"]
            best_models, mae = stacking_train(models, X, y, n_trials=2)
            best_model = set_model(model_name="stacking", models=best_models, meta_model=meta_model)
            best_model = best_model.train(X, y["log_deposit"])
            best_params = str(best_models)
        case _:
            if args.optuna == "on":
                best_params, mae = optuna_train(args.model, X, y)
                best_model = set_model(args.model, **best_params)
                best_model = best_model.train(X, y["log_deposit"])
            else:
                best_params = {
                    'n_estimators': 249,
                    'learning_rate': 0.1647758714498898,
                    'max_depth': 12,
                    'subsample': 0.9996749158433582
                }

                best_model = set_model(args.model, **best_params)
                mae = cv_train(best_model, X, y)

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
    