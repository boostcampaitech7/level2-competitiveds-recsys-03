import pandas as pd

# 이상치 탐지 함수
def outlier_detection(data: pd.Series) -> pd.Series:
    """안 울타리(inner fence) 밖에 있는 데이터(이상치, outlier)를 반환하는 함수

    Args:
        data (pd.Series): 이상치 탐지를 하고싶은 데이터의 column

    Returns:
        pd.Series: 이상치에 해당하는 데이터 Series 반환
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR


# 위치 중복도 선택/제거 함수
def delete_low_density(
    data: pd.DataFrame, 
    lower: int, 
    upper: int,
    drop: bool = True
) -> pd.DataFrame:
    """위치 중복도가 선택할 구간에 포함되면 이에 해당하는 데이터를 반환하거나 또는 해당하는 데이터를 제거하는 함수

    Args:
        data (pd.DataFrame): 위도, 경도를 column으로 갖는 데이터프레임
        lower (int): 선택할 구간의 하한
        upper (int): 선택할 구간의 상한
        drop (bool, optional): False이면 구간에 해당하는 데이터 반환. Default는 True.

    Returns:
        pd.DataFrame: 선택할 구간에 포함하는 위치 중복도를 갖는 데이터를 선택 또는 삭제 후 데이터프레임 반환
    """

    group = data.groupby(["latitude", "longitude"])["index"].count() # 위도, 경도 그룹별 개수 카운트
    drop_index = group[(group >= lower) & (group <= upper)].index # 구간에 해당하는 위도, 경도 중복을 갖는 데이터 인덱스 후보
    
    # 조건에 해당하는 데이터프레임 저장
    if drop == True:
        result_df = train_data[
            ~ (train_data["latitude"].isin(drop_index.get_level_values(0))
            & train_data["longitude"].isin(drop_index.get_level_values(1)))
        ].reset_index()
        result_df.drop(columns="level_0", inplace=True) # 필요없는 column 제거
        result_df["index"] = result_df.index # "index" column도 초기화
    else:
        result_df = train_data[
              (train_data["latitude"].isin(drop_index.get_level_values(0))
            & train_data["longitude"].isin(drop_index.get_level_values(1)))
        ].reset_index()
        result_df.drop(columns="level_0", inplace=True) # 필요없는 column 제거
        result_df["index"] = result_df.index # # "index" column도 초기화

    return result_df