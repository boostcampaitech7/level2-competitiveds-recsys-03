from tqdm import tqdm
from model.model_train import *

def categorize_deposit(deposit: float) -> int:
    """
    deposit을 기준으로 그룹을 분류하는 함수입니다.

    Args:
        deposit (float)

    Returns:
        int: deposit에 따라 분류된 그룹 번호
    """
    if deposit < 10000:
        return 0
    elif deposit <= 100000:
        return 1
    elif deposit <= 200000:
        return 2
    elif deposit <= 300000:
        return 3
    elif deposit <= 400000:
        return 4
    elif deposit <= 500000:
        return 5
    elif deposit <= 600000:
        return 6
    else:
        return 7 

def train_regressors_per_group(
        model_name: str,
        train_data: pd.DataFrame,
        selected_cols: list,
        optuna: str
    ) -> tuple[dict, dict, float]:
    """
    각 deposit_group별로 모델을 훈련하고 성능을 평가하는 함수입니다.

    Args:
        model_name (str): 사용할 모델 이름
        train_data (pd.DataFrame): 훈련 데이터
        selected_cols (list): 모델 훈련에 사용될 feature의 열 이름 리스트
        optuna (str): Optuna 최적화 사용 여부. "on" 또는 "off".

    Returns:
        tuple[dict, dict, float]:
            - group_models (dict): 각 그룹에 대한 훈련된 모델을 포함하는 딕셔너리.
            - group_params (dict): 각 그룹에 대한 최적의 하이퍼파라미터를 포함하는 딕셔너리.
            - mean_score (float): 모든 그룹에 대한 평균 절대 오차 (Mean Absolute Error).
    """
    group_models = {}
    group_params = {}
    group_scores = {}

    for group in tqdm(train_data["deposit_group"].unique(), desc="Training models per group"):
        group_data = train_data[train_data["deposit_group"] == group]
        X_group = group_data[selected_cols]
        y_group = group_data["deposit"]

        # 모델 훈련
        match optuna:
            case "on":
                best_params, best_value = deposit_optuna_train(model_name, "reg", X_group, y_group)
                group_params[group] = best_params
                group_scores[group] = best_value
            case "off":
                group_params = {
                    0: {
                        "n_estimators": 269,
                        "learning_rate": 0.0594699816408674,
                        "max_depth": 11,
                        "subsample": 0.7547912219027157,
                        "colsample_bytree": 0.7020843771180812,
                        "gamma": 3.037806599477243},
                    1: {
                        "n_estimators": 276,
                        "learning_rate": 0.15579191199373718,
                        "max_depth": 12,
                        "subsample": 0.909150931054429,
                        "colsample_bytree": 0.8709809907337003,
                        "gamma": 3.936332525239126},
                    2: {
                        "n_estimators": 187,
                        "learning_rate": 0.04512234654985014,
                        "max_depth": 12,
                        "subsample": 0.8875664116805573,
                        "colsample_bytree": 0.9697494707820946,
                        "gamma": 4.474136752138244},
                    3: {
                        "n_estimators": 279,
                        "learning_rate": 0.11548075621633985,
                        "max_depth": 5,
                        "subsample": 0.6857659688575958,
                        "colsample_bytree": 0.86707596884712,
                        "gamma": 0.2970741820173067},
                    4: {
                        "n_estimators": 262,
                        "learning_rate": 0.10181884312738954,
                        "max_depth": 12,
                        "subsample": 0.9636784876731649,
                        "colsample_bytree": 0.9301563662590965,
                        "gamma": 3.9023500438592036},
                    5: {
                        "n_estimators": 144,
                        "learning_rate": 0.19063571821788408,
                        "max_depth": 10,
                        "subsample": 0.7993292420985183,
                        "colsample_bytree": 0.5780093202212182,
                        "gamma": 0.7799726016810132},
                    6: {
                        "n_estimators": 98,
                        "learning_rate": 0.13418531015780658,
                        "max_depth": 7,
                        "subsample": 0.8210566991625188,
                        "colsample_bytree": 0.91306660229789,
                        "gamma": 1.1997602717553963},
                    7: {
                        "n_estimators": 237,
                        "learning_rate": 0.1903026381932035,
                        "max_depth": 8,
                        "subsample": 0.6737126835787389,
                        "colsample_bytree": 0.7374821279913889,
                        "gamma": 1.1574290155684595}
                }
                best_params = group_params[group]
                score = deposit_train(model_name, "reg", X_group, y_group, best_params)
                group_scores[group] = score

        model = set_model(model_name, best_params)
        model.train(X_group, y_group)
        
        # 각 그룹에 해당하는 모델 저장
        group_models[group] = model
    
    # mean MAE 계산
    counts = train_data.groupby("deposit_group")["deposit"].count()
    scores = sum(score * counts[group] for group, score in group_scores.items())
    total_count = counts.sum()
    mean_score = scores / total_count

    # 점수 출력
    print(f"⭐ group MAE: {group_scores}")
    print(f"⭐ Mean MAE: {mean_score:.4f}")
        
    return group_models, group_params, mean_score

def predict_per_group(
        test_data: pd.DataFrame, 
        group_models: dict,
        selected_cols: list
    ) -> np.ndarray:
    """
    각 deposit_group별로 예측을 수행하는 함수입니다.

    Args:
        test_data (pd.DataFrame): 테스트 데이터
        group_models (dict): 각 그룹에 대한 훈련된 모델을 포함하는 딕셔너리
        selected_cols (list): 예측에 사용될 feature의 열 이름 리스트

    Returns:
        np.ndarray: 각 테스트 샘플에 대한 예측값 배열
    """
    # 예측값을 저장할 배열 초기화
    y_pred = np.zeros(len(test_data))
    
    # 그룹별로 데이터 분리 후 예측
    for group, model in group_models.items():
        group_data = test_data[test_data["predicted_group"] == group]
        X_group = group_data[selected_cols]
        
        # 각 그룹에 대해 예측
        if len(X_group) > 0:  # 해당 그룹에 데이터가 있는 경우만 예측
            y_pred_group = model.predict(X_group)
            y_pred[test_data["predicted_group"] == group] = y_pred_group

    return y_pred