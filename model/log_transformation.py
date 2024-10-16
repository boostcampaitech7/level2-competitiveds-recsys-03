import numpy as np

def apply_log_transformation(train_data, test_data):
    """
    학습 데이터와 테스트 데이터에 로그 변환을 적용하는 함수.
    주로 크기가 큰 값들을 로그 변환하여 데이터 분포를 조정하는 데 사용됨.

    Args:
        train_data (DataFrame): 학습용 데이터프레임.
        test_data (DataFrame): 테스트용 데이터프레임.

    Returns:
        train_data (DataFrame): 로그 변환이 적용된 학습용 데이터프레임.
        test_data (DataFrame): 로그 변환이 적용된 테스트용 데이터프레임.
    """
    #train_data log 변환
    train_data["log_deposit"] = np.log1p(train_data["deposit"])
    train_data["log_area_m2"] = np.log1p(train_data["area_m2"])
    train_data["log_school_distance"] = np.log1p(train_data["nearest_school_distance"])
    train_data["log_park_distance"] = np.log1p(train_data["nearest_park_distance"])
    train_data["log_subway_distance"] = np.log1p(train_data["nearest_subway_distance"])
    
    #test_data log 변환
    test_data["log_area_m2"] = np.log1p(test_data["area_m2"])
    test_data["log_school_distance"] = np.log1p(test_data["nearest_school_distance"])
    test_data["log_park_distance"] = np.log1p(test_data["nearest_park_distance"])
    test_data["log_subway_distance"] = np.log1p(test_data["nearest_subway_distance"])
    
    return train_data, test_data
    