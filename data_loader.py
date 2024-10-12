import os
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

def load_dataset(option: str) -> pd.DataFrame:
    data_path = "~/house/data"

    # 옵션 1 : 기존 데이터 프레임을 불러오는 방식
    if option == "original":
        train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
        test_data = pd.read_csv(os.path.join(data_path, "test.csv"))
        sample_submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))

        return train_data, test_data, sample_submission

    # 옵션 2 : 병합된 csv 파일을 불러오는 방식
    elif option == "merge":
        merged_csv_path_train = os.path.join(data_path, "merged_train.csv")
        merged_csv_path_test = os.path.join(data_path, "merged_test.csv")

        # 병합된 CSV 파일이 이미 존재하면 그 파일을 불러옴
        if os.path.exists(merged_csv_path_train) and os.path.exists(merged_csv_path_test):
            train_data = pd.read_csv(merged_csv_path_train)
            test_data = pd.read_csv(merged_csv_path_test)
            sample_submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
            return train_data, test_data, sample_submission
        
        else:
            train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
            test_data = pd.read_csv(os.path.join(data_path, "test.csv"))
            sample_submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))

            interest_data = pd.read_csv(os.path.join(data_path, "interestRate.csv"))
            subway_data = pd.read_csv(os.path.join(data_path, "subwayInfo.csv"))
            school_data = pd.read_csv(os.path.join(data_path, "schoolinfo.csv"))
            park_data = pd.read_csv(os.path.join(data_path, "parkInfo.csv"))


            # 계약 연월 기준으로 interest_data 병합
            merged_train = pd.merge(train_data, interest_data, left_on="contract_year_month", right_on="year_month", how="left")
            merged_train = merged_train.drop(columns=["year_month"])

            merged_test = pd.merge(test_data, interest_data, left_on="contract_year_month", right_on="year_month", how="left")
            merged_test = merged_test.drop(columns=["year_month"])

            # 거리 계산 함수 (지하철 예시)
            def find_closest_distance_haversine(train_data: pd.DataFrame, loc_df: pd.DataFrame) -> pd.DataFrame:
                train_coords = np.radians(train_data[["latitude", "longitude"]].values)
                loc_coords = np.radians(loc_df[["latitude", "longitude"]].values)
                tree = BallTree(loc_coords, metric="haversine")
                distances, indices = tree.query(train_coords, k=1)
                distances_meter = distances * 6371000
                closest_coords = loc_df[["latitude", "longitude"]].iloc[indices.flatten()].values

                result_df = pd.DataFrame({
                    "index": train_data.index,
                    "closest_distance": distances_meter.flatten(),
                    "closest_latitude": closest_coords[:, 0],
                    "closest_longtitude": closest_coords[:, 1]
                })

                return result_df

            # 지하철 거리 계산 및 병합
            subway_result = find_closest_distance_haversine(train_data, subway_data)
            subway_result.columns = ["index", "nearest_subway_distance", "nearest_subway_latitude", "nearest_subway_longtitude"]
            train_data = pd.merge(train_data, subway_result, on="index")

            subway_result = find_closest_distance_haversine(test_data, subway_data)
            subway_result.columns = ["index", "nearest_subway_distance", "nearest_subway_latitude", "nearest_subway_longtitude"]
            test_data = pd.merge(test_data, subway_result, on="index")

            # 학교 거리 계산 및 병합
            school_result = find_closest_distance_haversine(train_data, school_data)
            school_result.columns = ["index", "nearest_school_distance", "nearest_school_latitude", "nearest_school_longtitude"]
            train_data = pd.merge(train_data, school_result, on="index")

            school_result = find_closest_distance_haversine(test_data, school_data)
            school_result.columns = ["index", "nearest_school_distance", "nearest_school_latitude", "nearest_school_longtitude"]
            test_data = pd.merge(test_data, school_result, on="index")

            # 공원 거리 계산 및 병합
            park_result = find_closest_distance_haversine(train_data, park_data)
            park_result.columns = ["index", "nearest_park_distance", "nearest_park_latitude", "nearest_park_longtitude"]
            train_data = pd.merge(train_data, park_result, on="index")      

            park_result = find_closest_distance_haversine(test_data, park_data)
            park_result.columns = ["index", "nearest_park_distance", "nearest_park_latitude", "nearest_park_longtitude"]
            test_data = pd.merge(test_data, park_result, on="index")

            on = merged_train.columns.drop("interest_rate").tolist()
            train_data = pd.merge(merged_train, train_data, on=on, how="left")
            on = merged_test.columns.drop("interest_rate").tolist()
            test_data = pd.merge(merged_test, test_data, on=on, how="left")

            # 병합된 데이터를 CSV 파일로 저장
            train_data.to_csv(merged_csv_path_train, index=False)
            test_data.to_csv(merged_csv_path_test, index=False)

            return train_data, test_data, sample_submission

    else:
        raise ValueError("Invalid option. Please choose 1 or 2.")
