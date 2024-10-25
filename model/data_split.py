import pandas as pd

def split_features_and_target(train_data: pd.DataFrame):
    """
    학습 데이터에서 피처와 타겟 변수를 분리하는 함수.

    Args:
        train_data (pd.DataFrame): 학습에 사용할 데이터프레임. 
                                "deposit"과 "log_deposit" 열이 포함되어 있어야 함.

    Returns:
        tuple: 
            - X (pd.DataFrame): "deposit"과 "log_deposit" 열을 제외한 피처 데이터.
            - y (pd.DataFrame): 타겟 데이터로 "deposit"과 "log_deposit" 열로 구성됨.
    """
    # X: 피처 데이터 (deposit과 log_deposit 열을 제외한 나머지)
    X = train_data.drop(columns=["deposit", "log_deposit"], inplace=False)
    
    # y: 타겟 데이터 (deposit과 log_deposit 열)
    y = train_data[["deposit", "log_deposit"]]
    return X, y