import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

def find_nearest_haversine_distance(
    train_data: pd.DataFrame, 
    loc_data: pd.DataFrame
) -> pd.DataFrame:
    """건물과 지하철/학교/공원 사이의 최단 haversine 거리와 위치 정보를 반환하는 함수

    Args:
        train_data (pd.DataFrame): 학습(훈련) 또는 테스트 데이터프레임
        loc_data (pd.DataFrame): 위도, 경도를 column으로 갖는 데이터프레임

    Returns:
        pd.DataFrame: 최단거리, 최단거리에 해당하는 지점의 위도, 경도를 column으로 갖는 데이터프레임
    """
    # haversine 거리 계산을 위해 degree -> radian 값으로 변환
    train_coords = np.radians(train_data[["latitude", "longitude"]].values)
    loc_coords = np.radians(loc_data[["latitude", "longitude"]].values)
    
    # Ball Tree 객체 생성 
    tree = BallTree(loc_coords, metric="haversine")

    # tree.query로 거리 계산
    distances, indices = tree.query(train_coords, k=1) # 가까운 1 지점만 
    distances_meter = distances * 6371000 # 단위를 meter로 변환
    nearest_coords = loc_data[["latitude", "longitude"]].iloc[indices.flatten()].values # 가장 가까운 지점들의 좌표 저장

    # 최단거리, 최단거리에 해당하는 지점의 위도, 경도로 이루어진 데이터프레임 생성
    result_df = pd.DataFrame({
        "nearest_distance" : distances_meter.flatten(),
        "nearest_latitude" : nearest_coords[:, 0],
        "nearest_longitude" : nearest_coords[:, 1]
    })

    return result_df