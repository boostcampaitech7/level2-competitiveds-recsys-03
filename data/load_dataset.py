import os
import pandas as pd
import numpy as np
from data.feature_engineering import find_nearest_haversine_distance

def load_dataset():
    # 파일 경로 지정
    data_path: str = "~/house/data"

    # train, test data 불러오기
    train_data: pd.DataFrame = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_data: pd.DataFrame = pd.read_csv(os.path.join(data_path, "test.csv"))
    sample_submission: pd.DataFrame = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))

    # 금리, 지하철, 학교, 공원 정보 불러오기
    interest_data: pd.DataFrame = pd.read_csv(os.path.join(data_path, "interestRate.csv"))
    subway_data: pd.DataFrame = pd.read_csv(os.path.join(data_path, "subwayInfo.csv"))
    school_data: pd.DataFrame = pd.read_csv(os.path.join(data_path, "schoolinfo.csv"))
    park_data: pd.DataFrame = pd.read_csv(os.path.join(data_path, "parkInfo.csv"))

    
    ### 금리: 계약 연월 기준으로 interest_data를 train_data로 병합
    train_data: pd.DataFrame = pd.merge(train_data, interest_data, left_on="contract_year_month", right_on="year_month", how="left")
    train_data: pd.DataFrame = train_data.drop(columns=["year_month"])
    
    test_data: pd.DataFrame = pd.merge(test_data, interest_data, left_on="contract_year_month", right_on="year_month", how="left")
    test_data: pd.DataFrame = test_data.drop(columns=["year_month"])

    
    ### 최단거리
    # train_data에서 위도, 경도 중복 행을 제외하고 추출
    unique_loc_train: pd.DataFrame = train_data[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    unique_loc_test: pd.DataFrame = test_data[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    
    # train_data에 최단거리 지하철 정보 추가
    merged_df: pd.DataFrame = find_nearest_haversine_distance(unique_loc_train, subway_data)
    merged_df: pd.DataFrame = pd.concat([merged_df, unique_loc_train], axis=1)
    train_data: pd.DataFrame = pd.merge(train_data, merged_df, on=["latitude", "longitude"], how="left")
    train_data.rename(columns={
        "nearest_distance": "nearest_subway_distance",
        "nearest_latitude": "nearest_subway_latitude",
        "nearest_longitude": "nearest_subway_longitude"
        }, 
        inplace=True
    )

    # test_data에 최단거리 지하철 정보 추가
    merged_df: pd.DataFrame = find_nearest_haversine_distance(unique_loc_train, subway_data)
    merged_df: pd.DataFrame = pd.concat([merged_df, unique_loc_train], axis=1)
    test_data: pd.DataFrame = pd.merge(test_data, merged_df, on=["latitude", "longitude"], how="left")
    test_data.rename(columns={
        "nearest_distance": "nearest_subway_distance",
        "nearest_latitude": "nearest_subway_latitude",
        "nearest_longitude": "nearest_subway_longitude"
        }, 
        inplace=True
    )

    # train_data에 최단거리 학교 정보 추가
    merged_df: pd.DataFrame = find_nearest_haversine_distance(unique_loc_train, school_data)
    merged_df: pd.DataFrame = pd.concat([merged_df, unique_loc_train], axis=1)
    train_data: pd.DataFrame = pd.merge(train_data, merged_df, on=["latitude", "longitude"], how="left")
    train_data.rename(columns={
        "nearest_distance": "nearest_school_distance",
        "nearest_latitude": "nearest_school_latitude",
        "nearest_longitude": "nearest_school_longitude"
        },
        inplace=True
    )

    # test_data에 최단거리 학교 정보 추가
    merged_df: pd.DataFrame = find_nearest_haversine_distance(unique_loc_train, school_data)
    merged_df: pd.DataFrame = pd.concat([merged_df, unique_loc_train], axis=1)
    test_data: pd.DataFrame = pd.merge(test_data, merged_df, on=["latitude", "longitude"], how="left")
    test_data.rename(columns={
        "nearest_distance": "nearest_school_distance",
        "nearest_latitude": "nearest_school_latitude",
        "nearest_longitude": "nearest_school_longitude"
        },
        inplace=True
    )

    # train_data에 최단거리 공원 정보 추가
    merged_df: pd.DataFrame = find_nearest_haversine_distance(unique_loc_train, park_data)
    merged_df: pd.DataFrame = pd.concat([merged_df, unique_loc_train], axis=1)
    train_data: pd.DataFrame = pd.merge(train_data, merged_df, on=["latitude", "longitude"], how="left")
    train_data.rename(columns={
        "nearest_distance": "nearest_park_distance",
        "nearest_latitude": "nearest_park_latitude",
        "nearest_longitude": "nearest_park_longitude"
        },
        inplace=True
    )

    # test_data에 최단거리 공원 정보 추가
    merged_df: pd.DataFrame = find_nearest_haversine_distance(unique_loc_train, park_data)
    merged_df: pd.DataFrame = pd.concat([merged_df, unique_loc_train], axis=1)
    test_data: pd.DataFrame = pd.merge(test_data, merged_df, on=["latitude", "longitude"], how="left")
    test_data.rename(columns={
        "nearest_distance": "nearest_park_distance",
        "nearest_latitude": "nearest_park_latitude",
        "nearest_longitude": "nearest_park_longitude"
        },
        inplace=True
    )

    return train_data, test_data, sample_submission