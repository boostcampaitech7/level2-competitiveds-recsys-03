import numpy as np

def save_csv(model, test_data, sample_submission):
# 5. 테스트 데이터에 대한 예측 및 제출 파일 생성
    y_test_log = model.predict(test_data)
    y_test = np.expm1(y_test_log) # 지수변환 (로그변환의 역변환)
    sample_submission["deposit"] = y_test
    sample_submission.to_csv("output.csv", index=False)