import numpy as np
import pandas as pd

def save_csv(model, test_data: pd.DataFrame, sample_submission: pd.DataFrame):
    """
    학습된 모델을 사용하여 테스트 데이터에 대한 예측을 수행하고,
    예측 결과를 제출 파일 형식으로 저장하는 함수.

    Args:
        model (object): 학습된 모델 객체.
        test_data (pd.DataFrame): 예측에 사용할 테스트 데이터.
        sample_submission (pd.DataFrame): 제출 파일의 기본 형식을 가진 데이터프레임.

    Returns:
        None: 예측 결과를 "output.csv" 파일로 저장.
    """
    y_test_log = model.predict(test_data)
    y_test = np.expm1(y_test_log) # 지수변환 (로그변환의 역변환)
    sample_submission["deposit"] = y_test
    sample_submission.to_csv("output.csv", index=False)