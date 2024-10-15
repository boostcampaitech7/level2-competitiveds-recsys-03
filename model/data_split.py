def split_features_and_target(train_data):
    # X: 피처 데이터 (deposit과 log_deposit 열을 제외한 나머지)
    X = train_data.drop(columns=["deposit", "log_deposit"], inplace=False)
    
    # y: 타겟 데이터 (deposit과 log_deposit 열)
    y = train_data[["deposit", "log_deposit"]]
    return X, y