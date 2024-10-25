from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import Lasso
import pandas as pd
import xgboost as xgb

def select_features(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    학습 데이터와 테스트 데이터에서 사용할 피처(컬럼)를 선택하는 함수.

    Args:
        train_data (pd.DataFrame): 학습에 사용할 데이터프레임.
        test_data (pd.DataFrame): 테스트에 사용할 데이터프레임.

    Returns:
        tuple:
            - train_data_selected (pd.DataFrame): 선택된 피처들로 구성된 학습 데이터.
            - test_data_selected (pd.DataFrame): 선택된 피처들로 구성된 테스트 데이터.
    """

    selected_cols = [
        "log_area_m2",
        "built_year",
        "latitude",
        "longitude",
        "log_leader_distance",
        "log_subway_distance",
        "log_school_distance",
        "log_park_distance",
        "contract_year_month",
        # "contract_day",
        "num_of_subways_within_radius",
        "park",
        "region"
    ]

    # selected_cols = select_kbest(X[selected_cols], y, "log_deposit")
    # selected_cols = select_rfe(X[selected_cols], y, "log_deposit", model_type="lasso")

    # 피처 선택
    X_selected = train_data[selected_cols]
    test_data_selected = test_data[selected_cols]

    return X_selected, test_data_selected

def select_kbest(X, y, target, k=10):
    """
    SelectKBest 방법을 사용하여 상위 k개의 특성 선택

    Args:
        X (DataFrame): 독립변수
        y (DataFrame): 종속변수
        target (str): 종속변수 열 중 실제 사용할 target 열 이름
        k (int, optional): 선택할 상위 k개 특성의 수 (Defaults to 10)

    Returns:
        List[str]: 선택된 상위 k개의 특성의 열 이름 리스트
    """
    # SelectKBest 적용
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y[target])

    # 선택된 특성의 열 이름 리스트 반환
    selected_cols = X.columns[selector.get_support()].tolist()

    return selected_cols

def select_rfe(X, y, target, n_features_to_select=10, model_type="xgboost"):
    """
    RFE(Recursive Feature Elimination)을 사용하여 피처 선택

    Args:
        X (DataFrame): 독립변수
        y (DataFrame): 종속변수
        target (str): 종속변수 열 중 실제 사용할 target 열 이름
        n_features_to_select (int, optional): 선택할 피처의 수 (Defaults to 10)
        model_type (str): 사용할 모델 유형 (xgboost 또는 lasso)
    
    Returns:
        List[str]: 선택된 상위 k개의 특성의 열 이름 리스트
    """
    if model_type == "xgboost":
        model = xgb.XGBRegressor()  # XGBoost 회귀 모델
    elif model_type == "lasso":
        model = Lasso(alpha=1.0)  # Lasso 회귀 모델

    selector = RFE(estimator=model, n_features_to_select=n_features_to_select)
    selector.fit(X, y[target])

    # 선택된 특성의 열 이름 리스트 반환
    selected_cols = X.columns[selector.get_support()].tolist()

    return selected_cols