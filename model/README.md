# Overview

모델링과 관련한 파일이 저장되어 있습니다.
<br><br>


# 🌳 File Tree 🌳

```
📂 model
 │
 ├── 📂 model_selection
 │    ├── Ensemble.py 
 │    └── TreeModel.py
 │
 ├── data_split.py
 ├── feature_select.py
 ├── inference.py
 ├── model_train.py
 └── README.md
```

# Descriptions

- **`model_selection/` : 모델 클래스 저장 폴더**
    - **`Ensemble.py` : Voting, Stacking**
        <details>
        <summary>train</summary>
          
        <h3> 함수 개요 </h3>
          
        - 모델 객체를 정의하고 훈련하는 함수
        <h3> 함수 파라미터 </h3>
        
        - `X_train (DataFrame)` : 독립 변수 데이터
          
        - `y_train (DataFrame)` : 예측 변수 데이터
        <h3> 함수 동작 방식 </h3>
          
        - 모델 객체 정의
          
        - 모델 훈련
          
        - 훈련된 모델 객체 반환
        </details><details>
        <summary>predict</summary>
        <h3> 함수 개요 </h3>
          
        - 훈련된 모델을 기반으로 예측값을 출력하는 함수
        <h3> 함수 파라미터 </h3>
        
        - `X_valid (DataFrame)` : 검증 데이터셋
        <h3> 함수 동작 방식 </h3>
        
        - 훈련하지 않고 함수 실행 시 ValueError 반환
          
        - 훈련된 모델로 예측 및 예측값 반환
        </details>
        
    - **`TreeModel.py` : XGBoost, LightGBM, CatBoost**
      <details>
      <summary>train</summary>
      <h3> 함수 개요 </h3>
        
      - 모델 객체를 정의하고 훈련하는 함수
      <h3> 함수 파라미터 </h3>
      
      - `X_train (DataFrame)` : 독립 변수 데이터
        
      - `y_train (DataFrame)` : 예측 변수 데이터
      <h3> 함수 동작 방식 </h3>
      
      - 모델 객체 정의
        
      - 모델 훈련
        
      - 훈련된 모델 객체 반환
      </details><details>
      <summary>predict</summary>
      <h3> 함수 개요 </h3>
        
      - 훈련된 모델을 기반으로 예측값을 출력하는 함수
      <h3> 함수 파라미터 </h3>
      
      - `X_valid (DataFrame)` : 검증 데이터셋
      <h3> 함수 동작 방식 </h3>
      
      - 훈련하지 않고 함수 실행 시 ValueError 반환
        
      - 훈련된 모델로 예측 및 예측값 반환
      </details><details>
      <summary>train_cls</summary>
      <h3> 함수 개요 </h3>
        
      - 모델 객체를 정의하고 훈련하는 함수입니다.
      </div>
      <h3> 함수 파라미터 </h3>
      
      - `X_train (DataFrame)` : 독립 변수
        
      - `y_train (DataFrame)` : 예측 변수 데이터
      <h3> 함수 동작 방식 </h3>
      
      - 모델 객체 정의
        
      - 모델 훈련
        
      - 훈련된 모델 객체 반환
      </details><details>
      <summary>predict_proba</summary>
      <h3> 함수 개요 </h3>
        
      - 훈련된 모델을 기반으로 예측값을 출력하는 함수입니다.
      </div>
      <h3> 함수 파라미터 </h3>
      
      - `X_valid (DataFrame)` : 검증 데이터셋
      <h3> 함수 동작 방식 </h3>
      
      - 훈련하지 않고 함수 실행 시 ValueError 반환
        
      - 훈련된 모델로 예측 및 예측값 반환
      </details>
    
- **`data_split.py` : target 데이터 분리**
  <details>
  <summary>split_features_and_target</summary>
  <h3> 함수 개요 </h3>
    
  - 학습 데이터에서 피처와 타겟 변수를 분리하는 함수
  <h3> 함수 파라미터 </h3>
  
  - `train_data (DataFrame)`: 학습용 데이터프레임 (deposit, log_deposit 열 포함해야)
  <h3> 함수 동작 방식 </h3>
  
  - 피처 데이터셋에 deposit, log_deposit을 제외한 열 할당
    
  - 타겟 데이터셋에 deposit, log_deposit 열 할당
    
  - 피처와 타겟 데이터셋 반환
  </details>
  
- **`deposit_group.py` : deposit_group 별로 학습 및 예측하기 위한 함수 모음**
  <details>
  <summary>categorize_deposit</summary>
  <h3> 함수 개요 </h3>
    
  - deposit을 기준으로 그룹을 분류하는 함수입니다.
  </div>
  <h3> 함수 파라미터 </h3>
  
  - `deposit: float`
  <h3> 함수 동작 방식 </h3>
  
  - deposit을 기준으로 그룹을 분류
    
  - deposit에 따라 분류된 그룹 번호 반환
  </details><details>
  <summary>`train_regressors_per_group`</summary>
  <h3> 함수 개요 </h3>
    
  - 각 deposit_group별로 모델을 훈련하고 성능을 평가하는 함수입니다.
  </div>
  <h3> 함수 파라미터 </h3>
  
  - `model_name: str` 사용할 모델 이름
    
  - `train_data: pd.DataFrame` 훈련 데이터
    
  - `selected_cols: list` 모델 훈련에 사용될 feature의 열 이름 리스트
    
  - `optuna: str` Optuna 최적화 사용 여부. "on" 또는 "off".
  <h3> 함수 동작 방식 </h3>
  
  - deposit_group 별로
  
    - 해당 deposit_group에 해당하는 데이터 할당
      
    - 데이터 독립변수(X_group), 종속변수(y_group)로 분류
      
    - 모델 훈련 (Optuna 사용하는 경우, 아닌 경우로 나누어 실행)
      
      - group_params에 그룹별 best_params 저장
        
      - group_scores에 그룹별 점수 저장
        
      - group_models에 그룹별 모델 저장
  - mean MAE 계산 (그룹별 점수 * 그룹별 개수 / 전체 개수)
    
  - group MAE, mean MAE 출력
    
  - group_models, group_params, mean_score 반환
  </details><details>
  <summary>predict_per_group</summary>
  <h3> 함수 개요 </h3>
    
  - 각 deposit_group별로 예측을 수행하는 함수입니다.
  </div>
  <h3> 함수 파라미터 </h3>
  
  - `test_data: pd.DataFrame` 훈련 데이터
    
  - `group_models: dict` 각 그룹에 대한 훈련된 모델을 포함하는 딕셔너리
    
  - `selected_cols: list` 모델 훈련에 사용될 feature의 열 이름 리스트
  <h3> 함수 동작 방식 </h3>
  
  - 예측값을 저장할 배열(y_pred) 초기화

  - deposit_group 별로
    - 해당 deposit_group에 해당하는 데이터 할당
    - 해당 그룹에 데이터가 있는 경우만 예측
    - y_pred에 그룹별 예측값 저장
  - y_pred 반환
  </details>
  
- **`feature_select.py` : 학습 및 테스트 데이터에서 사용할 feature 선택**
  <details>
  <summary>select_features</summary>
  <h3> 함수 개요 </h3>
    
  - 학습 데이터와 테스트 데이터에서 사용할 피처(컬럼)을 선택하는 함수
  <h3> 함수 파라미터 </h3>
  
  - `train_data (DataFrame)`: 학습용 데이터프레임
    
  - `test_data (DataFrame)`: 테스트용 데이터프레임
  <h3> 함수 동작 방식 </h3>
  
  - 학습용, 테스트용 피처 리스트로 선택
    
  - 선택된 피처 리스트로 지정한 데이터셋(학습, 테스트) 반환
  </details>
  
- **`inference.py` : 테스트 데이터에 대한 예측 및 제출 파일 생성**
  <details>
  <summary>save_csv</summary>
  <h3> 함수 개요 </h3>
    
  - 학습된 모델을 사용하여 테스트 데이터에 대한 예측을 수행하고
    
  - 예측 결과를 제출 파일 형식으로 저장하는 함수
  <h3> 함수 파라미터 </h3>
  
  - `model(object)` : 학습된 모델 객체
    
  - `test_data (Dataframe)` : 예측에 사용된 테스트 데이터
    
  - `sample_submission (DataFrame)` : 제출 파일의 기본 형식을 가진 데이터프레임
  <h3> 함수 동작 방식 </h3>
  
  - 학습된 모델로 테스트 데이터 예측
    
  - 예측결과 지수 변환
    
  - 지수 변환값 sample_submission 데이터셋에 할당
    
  - sample_submission 데이터셋 csv 파일로 반환
  </details>
- **`model_train.py` : 모델 훈련**
  <details>
  <summary>set_model</summary>
  <h3> 함수 개요 </h3>
    
  - 주어진 모델 이름에 따라 모델을 생성하고 반환하는 함수
  <h3> 함수 파라미터 </h3>
  
  - `model_name (str)` : 생성하려는 모델 이름
    
  - `*params (dict)` : 모델 생성 시 사용할 하이퍼파라미터
  <h3> 함수 동작 방식 </h3>
  
  - `match` 구문을 사용하여 `model_name`에 따라 적절한 모델 클래스를 선택
    
  - 각 모델 이름에 맞는 케이스를 정의하고, 해당 모델을 초기화
    
  - 생성된 모델 객체 반환
  </details><details>
  <summary>cv_train</summary>
  <h3> 함수 개요 </h3>
    
  - K-Fold를 이용하여 Cross Validation을 수행하는 함수
  <h3> 함수 파라미터 </h3>
  
  - `model (object)` : 수행하려는 모델
    
  - `X (DataFrame)` : 독립 변수
    
  - `y (DataFrame)` : 예측 변수. deposit과 log_deposit 열로 나뉨.
    
  - `verbose (bool, optional)` : Fold별 진행상황을 출력할 지 여부. Defaults to True.
  <h3> 함수 동작 방식 </h3>

  - K-Fold 객체 생성 (5-Fold)
    
  - 각 폴드에 대해
    
    - 현재 폴드 번호를 출력 (verbose가 True일 때)
      
    - 훈련 데이터와 검증 데이터를 인덱스를 이용해 분리
      
    - 모델을 훈련 데이터로 학습
      
    - 검증 데이터에 대한 예측을 수행
      
    - 예측 결과에 지수변환
      
    - MAE를 계산하고 리스트에 추가 (verbose가 True일 때 MAE 값을 출력)
  - 평균 MAE로 최종 MAE 계산
    
  - 전체 K-Fold 결과 출력 (verbose가 True일 때)
    
  - 최종 MAE 반환
  </details><details>
  <summary>optuna_train</summary>
  <h3> 함수 개요 </h3>
    
  - Optuna를 사용하여 주어진 모델의 하이퍼파라미터를 최적하는 함수
  <h3> 함수 파라미터 </h3>
  
  - `model (object)` : 최적화할 모델의 이름
    
  - `X (DataFrame)` : 독립 변수
    
  - `y (DataFrame)` : 예측 변수
  <h3> 함수 동작 방식 </h3>
  
  - 목표 함수 정의
    
    - 내부 함수 `objective(trial)`을 정의하여 각 하이퍼파라미터 조합을 평가
      
    - `match` 구문을 사용하여 모델 이름에 따른 하이퍼파라미터를 설정
    
    - 모델 생성 및 평가
    
    - `set_model` 함수를 호출하여 하이퍼파라미터를 적용한 모델 객체 생성
      
    - `cv_train` 함수를 호출하여 모델을 K-Fold 교차 검증을 통해 평가하고 MAE를 반환
  - Optuna 스터디 생성 및 최적화
    
  - `TPESampler`를 사용하여 샘플링 방법을 설정
    
  - `optuna.create_study`를 통해 스터디를 생성하고, 방향을 "minimize"로 설정하여 MAE 최소화를 목표
    
  - `study.optimize`를 호출하여 `objective` 함수 50번 실행하여 최적의 하이퍼파라미터 탐색
    
  - 최적의 하이퍼파라미터와 MAE 반환
  </details><details>
  <summary>voting_train</summary>
  <h3> 함수 개요 </h3>
    
  - Optuna를 사용하여 보팅 모델의 하이퍼파라미터를 최적하는 함수
  <h3> 함수 파라미터 </h3>
  
  - `models (list[str])` : 기본 모델의 이름을 담은 리스트
    
  - `X (DataFrame)` : 독립 변수
    
  - `y (DataFrame)` : 예측 변수
    
  - `n_trials (int)` : optuna 시행 횟수
  <h3> 함수 동작 방식 </h3>
  
  - 목표 함수 정의
    - 내부 함수 `objective(trial)`을 정의하여 각 하이퍼파라미터 조합을 평가
      
    - `match` 구문을 사용하여 모델 이름에 따른 하이퍼파라미터를 설정
      
    - 통합 모델 정의
      
    - 가중치 또한 튜닝할 파라미터로 지정 및 정규화 수행
      
    - `cv_train` 함수를 호출하여 모델을 K-Fold 교차 검증을 통해 평가하고 MAE를 반환
  - Optuna 스터디 생성 및 최적화
    
  - `TPESampler`를 사용하여 샘플링 방법을 설정
    
  - `optuna.create_study`를 통해 스터디를 생성하고, 방향을 "minimize"로 설정하여 MAE 최소화를 목표
    
  - `study.optimize`를 호출하여 `objective` 함수를 지정된 횟수만큼 실행하여 최적의 하이퍼파라미터 탐색
    
  - 최적의 하이퍼파라미터와 MAE 반환
  </details><details>
  <summary>stacking_train</summary>
  <h3> 함수 개요 </h3>
    
  - Optuna를 사용하여 스태킹 모델의 하이퍼파라미터를 최적하는 함수
  <h3> 함수 파라미터 </h3>
  
  - `models (list[str])` : 기본 모델의 이름을 담은 리스트
    
  - `meta_model (BaseEstimator)` : 메타 모델 객체
    
  - `X (DataFrame)` : 독립 변수
    
  - `y (DataFrame)` : 예측 변수
    
  - `n_trials (int)` : optuna 시행 횟수
  <h3> 함수 동작 방식 </h3>
  - 목표 함수 정의
    - 내부 함수 `objective(trial)`을 정의하여 각 하이퍼파라미터 조합을 평가
  
    - `match` 구문을 사용하여 모델 이름에 따른 하이퍼파라미터를 설정
      
    - 통합 모델 정의
      
    - `set_model` 함수를 호출하여 하이퍼파라미터를 적용한 모델 객체 생성
      
    - `cv_train` 함수를 호출하여 모델을 K-Fold 교차 검증을 통해 평가하고 MAE를 반환
  - Optuna 스터디 생성 및 최적화
    
  - `TPESampler`를 사용하여 샘플링 방법을 설정
    
  - `optuna.create_study`를 통해 스터디를 생성하고, 방향을 "minimize"로 설정하여 MAE 최소화를 목표
    
  - `study.optimize`를 호출하여 `objective` 함수를 지정된 횟수만큼 실행하여 최적의 하이퍼파라미터 탐색
    
  - 최적의 하이퍼파라미터와 MAE 반환
  </details><details>
  <summary>deposit_train</summary>
  <h3> 함수 개요 </h3>
    
  - 모델 객체를 정의하고 훈련하는 함수입니다.
  </div>
  <h3> 함수 파라미터 </h3>
  
  - `model_name (str)` : 최적화할 모델의 이름
    
  - `type (str)` : 모델의 타입을 나타내며, "cls"는 Classifier, "reg"는 Regressor를 의미
    
  - `X (DataFrame)` : 독립 변수
    
  - `y (DataFrame)` : 예측 변수
    
  - `params (dict)`: 하이퍼파라미터
  <h3> 함수 동작 방식 </h3>
  
  - 독립 변수와 예측 변수를 train, valid 데이터셋으로 분리
    
  - `set_model` 함수로 `model_name`에 맞는 model 객체 생성
    
  - Classifier인 경우와 Regressor인 경우로 나누어 학습 및 예측
    
  - valid MAE 반환
  </details><details>
  <summary>deposit_optuna_train</summary>
  <h3> 함수 개요 </h3>
    
  - Optuna를 사용하여 주어진 모델의 하이퍼파라미터를 최적화하는 함수입니다.
  </div>
  <h3> 함수 파라미터 </h3>
  
  - `model_name (str)` : 최적화할 모델의 이름
    
  - `type (str)` : 모델의 타입을 나타내며, "cls"는 Classifier, "reg"는 Regressor를 의미
    
  - `X (DataFrame)` : 독립 변수
    
  - `y (DataFrame)` : 예측 변수
    
  - `n_trials (int, optional)` : Optuna가 최적화를 위해 수행할 시험 횟수 (Defaults to 50)
  <h3> 함수 동작 방식 </h3>
  
  - 목표 함수 정의
    - 내부 함수 objective(trial) 을 정의하여 각 하이퍼파라미터 조합을 평가
      
    - match 구문을 사용하여 모델 이름에 따른 하이퍼파라미터를 설정
      
    - 모델 생성 및 평가
      
    - `deposit_train` 함수를 호출하여 모델을 평가하고 MAE를 반환
  - Optuna 스터디 생성 및 최적화
    
  - `TPESampler`를 사용하여 샘플링 방법을 설정
  
  - `optuna.create_study`를 통해 스터디를 생성하고, 방향을 "minimize"로 설정하여 MAE 최소화를 목표
    
  - `study.optimize`를 호출하여 `objective`함수 50번 실행하여 최적의 하이퍼파라미터 탐색
    
  - 최적의 하이퍼파라미터와 MAE 반환
  </details>

<br></br>
