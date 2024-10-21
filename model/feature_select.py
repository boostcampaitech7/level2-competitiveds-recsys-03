def select_features(train_data, test_data):
    """
    학습 데이터와 테스트 데이터에서 사용할 피처(컬럼)를 선택하는 함수.

    Args:
        train_data (DataFrame): 학습에 사용할 데이터프레임.
        test_data (DataFrame): 테스트에 사용할 데이터프레임.

    Returns:
        tuple:
            - train_data_selected (DataFrame): 선택된 피처들로 구성된 학습 데이터.
            - test_data_selected (DataFrame): 선택된 피처들로 구성된 테스트 데이터.
    """
    # 학습 데이터에서 사용할 피처들
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
        #"region",
        #"region_mean",
    ]
    # 테스트 데이터에서 사용할 피처들
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
        #"region",
        #"region_mean",
    ]
    # 피처 선택
    train_data_selected = train_data[train_cols]
    test_data_selected = test_data[test_cols]

    return train_data_selected, test_data_selected