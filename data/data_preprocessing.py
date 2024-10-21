import pandas as pd
from typing import Union
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

### 이상치 탐지 함수 ###
def outlier_detection(data: pd.Series) -> pd.Series:
    """
    안 울타리(inner fence) 밖에 있는 데이터(이상치, outlier)를 반환하는 함수

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


### 위치 중복도 선택/제거 함수 ###
def delete_low_density(
    data: pd.DataFrame, 
    lower: int, 
    upper: int,
    drop: bool = True
) -> pd.DataFrame:
    """
    위치 중복도가 선택할 구간에 포함되면 이에 해당하는 데이터를 반환하거나 또는 해당하는 데이터를 제거하는 함수

    Args:
        data (pd.DataFrame): 위도, 경도를 column으로 갖는 데이터프레임
        lower (int): 선택할 구간의 하한
        upper (int): 선택할 구간의 상한
        drop (bool, optional): False이면 구간에 해당하는 데이터 반환. Default는 True.

    Returns:
        pd.DataFrame: 선택할 구간에 포함하는 위치 중복도를 갖는 데이터를 선택 또는 삭제 후 데이터프레임 반환
    """
    # 위도, 경도 그룹별 개수 카운트
    group = data.groupby(["latitude", "longitude"])["index"].count()
    
    # 구간에 해당하는 위도, 경도 중복을 갖는 데이터 인덱스 후보 저장
    drop_index = group[(group >= lower) & (group <= upper)].index
    
    # 조건(선택/삭제)에 해당하는 데이터프레임 저장
    if drop == True:
        result_df = data[
            ~ (data["latitude"].isin(drop_index.get_level_values(0))
            & data["longitude"].isin(drop_index.get_level_values(1)))
        ].reset_index()
        result_df.drop(columns="level_0", inplace=True) # 필요없는 column 제거
        result_df["index"] = result_df.index # "index" column도 초기화
    else:
        result_df = data[
              (data["latitude"].isin(drop_index.get_level_values(0))
            & data["longitude"].isin(drop_index.get_level_values(1)))
        ].reset_index()
        result_df.drop(columns="level_0", inplace=True) # 필요없는 column 제거
        result_df["index"] = result_df.index # # "index" column도 초기화

    return result_df


### 결측치 처리 클래스 ###
class MissingValueImputer:

    ### 평균 보간 함수 ###
    def mean_imputer(self, df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        Pandas의 내장함수인 .mean()을 사용하여 결측치를 처리 후 반환하는 함수

        Args:
            df (Union[pd.DataFrame, pd.Series]): 결측치가 있는 DataFrame 및 column

        Returns:
            Union[pd.DataFrame, pd.Series]: imputer한 DataFrame 또는 Series 반환
        """

        if isinstance(df, pd.DataFrame):
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            return df 
               
        elif isinstance(df, pd.Series):
            return df.fillna(df.mean())


    ### 중앙값 보간 함수 ###
    def median_imputer(self, df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        Pandas의 내장함수인 .median()을 사용하여 결측치를 처리 후 반환하는 함수

        Args:
            df (Union[pd.DataFrame, pd.Series]): 결측치가 있는 DataFrame 및 column

        Returns:
            Union[pd.DataFrame, pd.Series]: imputer한 DataFrame 또는 Series 반환
        """

        if isinstance(df, pd.DataFrame):
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            return df
        
        elif isinstance(df, pd.Series):
            return df.fillna(df.median())


    ### Iterative방법을 사용한 보간 함수 ###
    def iterative_imputer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        scikit-learn의 IterativeImputer에서 default값을 사용해서 결측치를 처리 후 반환하는 함수

        Args:
            df (pd.DataFrame): 결측치가 있는 DataFrame

        Returns:
            pd.DataFrame: imputer한 DataFrame 반환
        """

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        imputer = IterativeImputer()
        df_imputed_array = imputer.fit_transform(df[numeric_cols])
        imputed_df = pd.DataFrame(df_imputed_array, columns=numeric_cols)
        df[numeric_cols] = imputed_df.values
        return df


    ### knn방법을 사용한 보간 함수 ###
    def knn_imputer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        scikit-learn의 KNNImputer에서 default값을 사용하여 결측치를 처리 후 반환하는 함수

        Args:
            df (pd.DataFrame): 결측치가 있는 DataFrame

        Returns:
            pd.DataFrame: imputer한 DataFrame 반환
        """

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        imputer = KNNImputer()
        df_imputed_array = imputer.fit_transform(df[numeric_cols])
        imputed_df = pd.DataFrame(df_imputed_array, columns=numeric_cols)
        df[numeric_cols] = imputed_df.values
        return df