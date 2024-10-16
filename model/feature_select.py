def select_features(train_data, test_data):
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
        "nearest_subway_num",
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
        "nearest_subway_num",
    ]
    # 피처 선택
    train_data_selected = train_data[train_cols]
    test_data_selected = test_data[test_cols]

    return train_data_selected, test_data_selected