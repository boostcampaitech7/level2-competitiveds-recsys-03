import os
import pandas as pd

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

    return train_data, test_data, sample_submission, interest_data, subway_data, school_data, park_data