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

    return data[(data < lower_bound) | (data > upper_bound)]