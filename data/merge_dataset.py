from typing import Union
import pandas as pd
from data.feature_engineering import find_nearest_haversine_distance

def merge_dataset(
    train_data: pd.DataFrame, 
    test_data: pd.DataFrame,
    interest_data: pd.DataFrame,
    subway_data: pd.DataFrame,
    school_data: pd.DataFrame,
    park_data: pd.DataFrame
) -> Union[pd.DataFrame, pd.DataFrame]:
    """학습(훈련), 테스트 데이터프레임에 새로운 변수를 추가하거나 병합한 데이터프레임으로 반환하는 함수

    Args:
        train_data (pd.DataFrame): 학습(훈련) 데이터프레임
        test_data (pd.DataFrame): 테스트 데이터프레임
        interest_data (pd.DataFrame): 금리 연월, 금리가 담긴 데이터프레임
        subway_data (pd.DataFrame): 지하철 위도, 경도가 담긴 데이터프레임
        school_data (pd.DataFrame): 학교 종류, 위도, 경도가 담긴 데이터프레임
        park_data (pd.DataFrame): 공원 위도, 경도, 면적이 담긴 데이터프레임

    Returns:
        Union[pd.DataFrame, pd.DataFrame]: 병합된 학습(훈련) 데이터프레임, 병합된 테스트 데이터프레임 
    """
    ### 금리: 계약 연월 기준으로 interest_data를 train_data로 병합 ###
    train_data: pd.DataFrame = pd.merge(train_data, interest_data, left_on="contract_year_month", right_on="year_month", how="left")
    train_data: pd.DataFrame = train_data.drop(columns=["year_month"])
    
    test_data: pd.DataFrame = pd.merge(test_data, interest_data, left_on="contract_year_month", right_on="year_month", how="left")
    test_data: pd.DataFrame = test_data.drop(columns=["year_month"])

    
    ### 최단거리 변수: find_nearest_haversine_distance 활용 ###
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
    merged_df: pd.DataFrame = find_nearest_haversine_distance(unique_loc_test, subway_data)
    merged_df: pd.DataFrame = pd.concat([merged_df, unique_loc_test], axis=1)
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
    merged_df: pd.DataFrame = find_nearest_haversine_distance(unique_loc_test, school_data)
    merged_df: pd.DataFrame = pd.concat([merged_df, unique_loc_test], axis=1)
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
    merged_df: pd.DataFrame = find_nearest_haversine_distance(unique_loc_test, park_data)
    merged_df: pd.DataFrame = pd.concat([merged_df, unique_loc_test], axis=1)
    test_data: pd.DataFrame = pd.merge(test_data, merged_df, on=["latitude", "longitude"], how="left")
    test_data.rename(columns={
        "nearest_distance": "nearest_park_distance",
        "nearest_latitude": "nearest_park_latitude",
        "nearest_longitude": "nearest_park_longitude"
        },
        inplace=True
    )

    
    ### subway_data, school_data, park_data에서 각각 위도와 경도로 그룹화하여 개수 세기 ###
    # 개수를 센 데이터를 subway_count, school_count, park_count로 반환
    subway_count: pd.DataFrame = subway_data.groupby(["latitude", "longitude"]).size().reset_index(name='nearest_subway_num')
    school_count: pd.DataFrame = school_data.groupby(["latitude", "longitude"]).size().reset_index(name='nearest_school_num')
    park_count: pd.DataFrame = park_data.groupby(["latitude", "longitude"]).size().reset_index(name='nearest_park_num')

    # train_data에 최단거리 지하철의 위도, 경도에 대한 지하철 개수 정보 추가
    train_data: pd.DataFrame = pd.merge(train_data, subway_count, left_on=["nearest_subway_latitude", "nearest_subway_longitude"],
                                        right_on=["latitude", "longitude"], how="left")
    train_data: pd.DataFrame = train_data.drop(columns=["latitude_y", "longitude_y"])
    train_data.rename(columns={
        "latitude_x": "latitude",
        "longitude_x": "longitude",
        },
        inplace=True
    )

    # test_data에 최단거리 지하철의 위도, 경도에 대한 지하철 개수 정보 추가
    test_data: pd.DataFrame = pd.merge(test_data, subway_count, left_on=["nearest_subway_latitude", "nearest_subway_longitude"],
                                        right_on=["latitude", "longitude"], how="left")
    test_data: pd.DataFrame = test_data.drop(columns=["latitude_y", "longitude_y"])
    test_data.rename(columns={
        "latitude_x": "latitude",
        "longitude_x": "longitude",
        },
        inplace=True
    )

    # train_data에 최단거리 학교의 위도, 경도에 대한 학교 개수 정보 추가
    train_data: pd.DataFrame = pd.merge(train_data, school_count, left_on=["nearest_school_latitude", "nearest_school_longitude"],
                                        right_on=["latitude", "longitude"], how="left")
    train_data: pd.DataFrame = train_data.drop(columns=["latitude_y", "longitude_y"])
    train_data.rename(columns={
        "latitude_x": "latitude",
        "longitude_x": "longitude",
        },
        inplace=True
    )

    # test_data에 최단거리 학교의 위도, 경도에 대한 학교 개수 정보 추가
    test_data: pd.DataFrame = pd.merge(test_data, school_count, left_on=["nearest_school_latitude", "nearest_school_longitude"],
                                        right_on=["latitude", "longitude"], how="left")
    test_data: pd.DataFrame = test_data.drop(columns=["latitude_y", "longitude_y"])
    test_data.rename(columns={
        "latitude_x": "latitude",
        "longitude_x": "longitude",
        },
        inplace=True
    )

    # train_data에 최단거리 공원의 위도, 경도에 대한 공원 개수 정보 추가
    train_data: pd.DataFrame = pd.merge(train_data, park_count, left_on=["nearest_park_latitude", "nearest_park_longitude"],
                                        right_on=["latitude", "longitude"], how="left")
    train_data: pd.DataFrame = train_data.drop(columns=["latitude_y", "longitude_y"])
    train_data.rename(columns={
        "latitude_x": "latitude",
        "longitude_x": "longitude",
        },
        inplace=True
    )

    # test_data에 최단거리 공원의 위도, 경도에 대한 공원 개수 정보 추가
    test_data: pd.DataFrame = pd.merge(test_data, park_count, left_on=["nearest_park_latitude", "nearest_park_longitude"],
                                        right_on=["latitude", "longitude"], how="left")
    test_data: pd.DataFrame = test_data.drop(columns=["latitude_y", "longitude_y"])
    test_data.rename(columns={
        "latitude_x": "latitude",
        "longitude_x": "longitude",
        },
        inplace=True
    )

    return train_data, test_data