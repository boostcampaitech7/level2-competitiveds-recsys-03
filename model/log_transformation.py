import numpy as np

def apply_log_transformation(train_data, test_data):
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
    