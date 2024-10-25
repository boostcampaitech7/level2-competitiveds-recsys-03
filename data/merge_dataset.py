from typing import Union
import pandas as pd
from data.feature_engineering import *
    
def merge_dataset(
    train_data: pd.DataFrame, 
    test_data: pd.DataFrame,
    interest_data: pd.DataFrame,
    subway_data: pd.DataFrame,
    school_data: pd.DataFrame,
    park_data: pd.DataFrame
) -> Union[pd.DataFrame, pd.DataFrame]:
    """
    학습(훈련), 테스트 데이터프레임에 새로운 변수를 추가하거나 병합한 데이터프레임으로 반환하는 함수

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

    # 공원 면적이 100,000 제곱미터 이상인 공원 데이터프레임 생성
    radius_meter: int = 1000 # 탐색하려는 반경(단위: 미터)
    earth_radius_meter: int = 6371000 # 지구의 반경(단위: 미터)
    extend_radian: float = radius_meter / earth_radius_meter # 선택하려는 위도, 경도 범위에서 탐색하고자 하는 반경만큼 범위 확장(단위: 라디안)
    criterion = 100000 # 도시지역권 근린공원 면적 기준

    new_park_data: pd.DataFrame = park_data[
          (park_data["latitude"] >= (train_data["latitude"].min() - extend_radian))
        & (park_data["latitude"] <= (train_data["latitude"].max() + extend_radian))
        & (park_data["longitude"] >= (train_data["longitude"].min() - extend_radian))
        & (park_data["longitude"] <= (train_data["longitude"].max() + extend_radian))
    ]
    new_park_data = new_park_data.drop_duplicates().reset_index(drop=True) # 중복값 16개 drop
    new_park_data = new_park_data[new_park_data["area"] > criterion] # 공원 면적이 100,000 제곱미터 이상만 고려
    new_park_data = new_park_data.reset_index(drop=True)
    
    # train_data에 최단거리 공원 정보 추가 (조건: 공원 면적이 100,000 제곱미터 이상인 공원들만 생각)
    merged_df: pd.DataFrame = find_nearest_haversine_distance(unique_loc_train, new_park_data)
    merged_df: pd.DataFrame = pd.concat([merged_df, unique_loc_train], axis=1)
    train_data: pd.DataFrame = pd.merge(train_data, merged_df, on=["latitude", "longitude"], how="left")
    train_data.rename(columns={
        "nearest_distance": "nearest_park_distance",
        "nearest_latitude": "nearest_park_latitude",
        "nearest_longitude": "nearest_park_longitude"
        },
        inplace=True
    )

    # test_data에 최단거리 공원 정보 추가 (조건: 공원 면적이 100,000 제곱미터 이상인 공원들만 생각)
    merged_df: pd.DataFrame = find_nearest_haversine_distance(unique_loc_test, new_park_data)
    merged_df: pd.DataFrame = pd.concat([merged_df, unique_loc_test], axis=1)
    test_data: pd.DataFrame = pd.merge(test_data, merged_df, on=["latitude", "longitude"], how="left")
    test_data.rename(columns={
        "nearest_distance": "nearest_park_distance",
        "nearest_latitude": "nearest_park_latitude",
        "nearest_longitude": "nearest_park_longitude"
        },
        inplace=True
    )

    
    ### 최단거리 지하철, 학교, 공원 개수 변수: subway_data, school_data, park_data에서 각각 위도와 경도로 그룹화하여 개수 세기 ###
    # 개수를 센 데이터를 subway_count, school_count, park_count로 반환
    subway_count: pd.DataFrame = subway_data.groupby(["latitude", "longitude"]).size().reset_index(name="nearest_subway_num")
    school_count: pd.DataFrame = school_data.groupby(["latitude", "longitude"]).size().reset_index(name="nearest_school_num")
    park_count: pd.DataFrame = new_park_data.groupby(["latitude", "longitude"]).size().reset_index(name="nearest_park_num")

    # train_data에 최단거리 지하철의 위도, 경도에 대한 지하철 개수 정보 추가
    train_data: pd.DataFrame = pd.merge(train_data, subway_count, 
                                        left_on=["nearest_subway_latitude", "nearest_subway_longitude"],
                                        right_on=["latitude", "longitude"], 
                                        how="left")
    train_data: pd.DataFrame = train_data.drop(columns=["latitude_y", "longitude_y"])
    train_data.rename(columns={
        "latitude_x": "latitude",
        "longitude_x": "longitude",
        },
        inplace=True
    )

    # test_data에 최단거리 지하철의 위도, 경도에 대한 지하철 개수 정보 추가
    test_data: pd.DataFrame = pd.merge(test_data, subway_count, 
                                       left_on=["nearest_subway_latitude", "nearest_subway_longitude"],
                                       right_on=["latitude", "longitude"], 
                                       how="left")
    test_data: pd.DataFrame = test_data.drop(columns=["latitude_y", "longitude_y"])
    test_data.rename(columns={
        "latitude_x": "latitude",
        "longitude_x": "longitude",
        },
        inplace=True
    )

    # train_data에 최단거리 학교의 위도, 경도에 대한 학교 개수 정보 추가
    train_data: pd.DataFrame = pd.merge(train_data, school_count, 
                                        left_on=["nearest_school_latitude", "nearest_school_longitude"],
                                        right_on=["latitude", "longitude"], 
                                        how="left")
    train_data: pd.DataFrame = train_data.drop(columns=["latitude_y", "longitude_y"])
    train_data.rename(columns={
        "latitude_x": "latitude",
        "longitude_x": "longitude",
        },
        inplace=True
    )

    # test_data에 최단거리 학교의 위도, 경도에 대한 학교 개수 정보 추가
    test_data: pd.DataFrame = pd.merge(test_data, school_count, 
                                       left_on=["nearest_school_latitude", "nearest_school_longitude"],
                                       right_on=["latitude", "longitude"], 
                                       how="left")
    test_data: pd.DataFrame = test_data.drop(columns=["latitude_y", "longitude_y"])
    test_data.rename(columns={
        "latitude_x": "latitude",
        "longitude_x": "longitude",
        },
        inplace=True
    )

    # train_data에 최단거리 공원(면적 100,000 제곱미터 이상)의 위도, 경도에 대한 공원 개수 정보 추가
    train_data: pd.DataFrame = pd.merge(train_data, park_count, 
                                        left_on=["nearest_park_latitude", "nearest_park_longitude"],
                                        right_on=["latitude", "longitude"], 
                                        how="left")
    train_data: pd.DataFrame = train_data.drop(columns=["latitude_y", "longitude_y"])
    train_data.rename(columns={
        "latitude_x": "latitude",
        "longitude_x": "longitude",
        },
        inplace=True
    )

    # test_data에 최단거리 공원(면적 100,000 제곱미터 이상)의 위도, 경도에 대한 공원 개수 정보 추가
    test_data: pd.DataFrame = pd.merge(test_data, park_count, 
                                       left_on=["nearest_park_latitude", "nearest_park_longitude"],
                                       right_on=["latitude", "longitude"], 
                                       how="left")
    test_data: pd.DataFrame = test_data.drop(columns=["latitude_y", "longitude_y"])
    test_data.rename(columns={
        "latitude_x": "latitude",
        "longitude_x": "longitude",
        },
        inplace=True
    )


    ### 특정 반경 내 지하철, 학교, 공원 수 변수: find_places_within_radius 함수 활용 ###
    # subway_data, school_data, new_park_data에서 위도, 경도 중복 행을 제외하고 추출 (new_park_data에서 미리 중복도 제거)
    unique_loc_subway: pd.DataFrame = subway_data[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True) # 같은 역이 다른 노선을 지나면 중복해서 카운트하므로 제거
    unique_loc_school: pd.DataFrame = school_data[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)

    # train_data에 700m 반경 이내 지하철 역 개수 정보 추가
    merged_df: pd.DataFrame = find_places_within_radius(unique_loc_train, unique_loc_subway, 700)
    train_data: pd.DataFrame = pd.merge(train_data, merged_df, on=["latitude", "longitude"], how="left")
    train_data.rename(columns={"num_of_places_within_radius": "num_of_subways_within_radius"}, inplace=True)

    # test_data에 700m 반경 이내 지하철 역 개수 정보 추가
    merged_df: pd.DataFrame = find_places_within_radius(unique_loc_test, unique_loc_subway, 700)
    test_data: pd.DataFrame = pd.merge(test_data, merged_df, on=["latitude", "longitude"], how="left")
    test_data.rename(columns={"num_of_places_within_radius": "num_of_subways_within_radius"}, inplace=True)

    # train_data에 700m 반경 이내 학교 개수 정보 추가
    merged_df: pd.DataFrame = find_places_within_radius(unique_loc_train, unique_loc_school, 700)
    train_data: pd.DataFrame = pd.merge(train_data, merged_df, on=["latitude", "longitude"], how="left")
    train_data.rename(columns={"num_of_places_within_radius": "num_of_schools_within_radius"}, inplace=True)

    # test_data에 700m 반경 이내 학교 개수 정보 추가
    merged_df: pd.DataFrame = find_places_within_radius(unique_loc_test, unique_loc_school, 700)
    test_data: pd.DataFrame = pd.merge(test_data, merged_df, on=["latitude", "longitude"], how="left")
    test_data.rename(columns={"num_of_places_within_radius": "num_of_schools_within_radius"}, inplace=True)

    # train_data에 1000m 반경 이내 100,000 제곱미터 이상인 공원 개수 정보 추가
    merged_df: pd.DataFrame = find_places_within_radius(unique_loc_train, new_park_data, 1000)
    train_data: pd.DataFrame = pd.merge(train_data, merged_df, on=["latitude", "longitude"], how="left")
    train_data.rename(columns={"num_of_places_within_radius": "num_of_parks_within_radius"}, inplace=True)

    # test_data에 1000m 반경 이내 100,000 제곱미터 이상인 공원 개수 정보 추가
    merged_df: pd.DataFrame = find_places_within_radius(unique_loc_test, new_park_data, 1000)
    test_data: pd.DataFrame = pd.merge(test_data, merged_df, on=["latitude", "longitude"], how="left")
    test_data.rename(columns={"num_of_places_within_radius": "num_of_parks_within_radius"}, inplace=True)


    ### 공원 근접성 유무 변수: 1000m 반경 이내 100,000 제곱미터 이상인 공원이 있으면 1, 아니면 0인 변수
    train_data["park_exists"] = train_data["num_of_parks_within_radius"].apply(lambda x: 1 if x != 0 else 0)
    test_data["park_exists"] = test_data["num_of_parks_within_radius"].apply(lambda x: 1 if x != 0 else 0)


    ### 클러스터링 변수: kmeans_clustering 활용 ###
    # 클러스터링 학습에 사용할 feature 선택
    feature_columns = ["latitude",	"longitude"]
    coords = train_data[feature_columns]

    # 클러스터링 객체 생성
    cm = ClusteringModel(data=coords)

    # 클러스터 개수(k) 설정
    n_clusters = 25
    
    # k-means 클러스터링 수행 후 train_data, test_data에 region 변수 추가
    kmeans = cm.kmeans_clustering(n_clusters, train_data, test_data, 
						          feature_columns,
							      label_column="region"
    )


    ### 클러스터별 평균 전세가 변수
    average_prices_by_region = train_data.groupby("region")["deposit"].mean().reset_index()
    average_prices_by_region.columns = ["region", "region_mean"]

    # train_data와 test_data에 average_prices_by_region 병합 (test에는 train의 평균가격이 병합된다.)
    train_data = pd.merge(train_data, average_prices_by_region, on="region", how="left")
    test_data = pd.merge(test_data, average_prices_by_region, on="region", how="left")


    ### 건물-대장 아파트 최단거리 변수: find_nearest_haversine_distance 활용 ###
    # train_data 대장 아파트 데이터프레임 생성
    max_deposits = train_data.groupby(["region"])["deposit"].max()
    leader_train_data = pd.DataFrame()

    for i in max_deposits.index:
        tmp = train_data[
              (train_data["region"] == i) 
            & (train_data["deposit"] == max_deposits[i])
        ][["region", "deposit", "latitude", "longitude"]] 
        leader_train_data = pd.concat([leader_train_data, tmp], axis=0, ignore_index=True)

    leader_train_data = leader_train_data.drop_duplicates().reset_index(drop=True) # 위도, 경도 중복 제거
    
    # train_data에 대장 아파트 최단거리 추가
    merged_df: pd.DataFrame = find_nearest_haversine_distance(unique_loc_train, leader_train_data)
    merged_df: pd.DataFrame = pd.concat([merged_df, unique_loc_train], axis=1)
    train_data: pd.DataFrame = pd.merge(train_data, merged_df, on=["latitude", "longitude"], how="left")
    train_data.rename(columns={
        "nearest_distance": "nearest_leader_distance",
        "nearest_latitude": "nearest_leader_latitude",
        "nearest_longitude": "nearest_leader_longitude"
        }, 
        inplace=True
    )

    # test_data에선 대장 아파트를 따로 알 수 없는 거니까 train_data의 대장 아파트를 활용해서 거리 계산을 한다.
    merged_df: pd.DataFrame = find_nearest_haversine_distance(unique_loc_test, leader_train_data)
    merged_df: pd.DataFrame = pd.concat([merged_df, unique_loc_test], axis=1)
    test_data: pd.DataFrame = pd.merge(test_data, merged_df, on=["latitude", "longitude"], how="left")
    test_data.rename(columns={
        "nearest_distance": "nearest_leader_distance",
        "nearest_latitude": "nearest_leader_latitude",
        "nearest_longitude": "nearest_leader_longitude"
        }, 
        inplace=True
    )

    return train_data, test_data