# Overview

데이터 처리와 관련한 파일들을 저장한 폴더입니다.

# 🌳 File Tree 🌳

```
📁 data
 |
 ├── load_dataset.py
 ├── merge_dataset.py
 ├── data_preprocessing.py
 ├── feature_engineering.py
 └── README.md
```

# Descriptions
- `data_preprocessing.py` : 데이터 전처리를 위한 클래스, 함수가 담긴 모듈
    <details>
    <summary>outlier_detection()</summary>
    <h3> 함수 개요 </h3>
    - 주어진 Series에 대해 $\pm$ 1.5 $\times$ IQR 범위 밖의 데이터(이상치, outlier)를 찾는 함수
    - 데이터프레임에서 이상치를 찾고 싶은 column에 이 함수를 적용하면 된다.
    
    <h3> 함수 파라미터 </h3>
    
    `data: pd.Series` 입력 Series
    
    <h3> 함수 동작 방식 </h3>
    
    1. `quantile()` 함수로 입력받은 데이터의 Q1, Q3 계산
    2. 울타리 너비(step)로 1.5 IQR 설정
    3. 안 울타리 밖의 데이터를 반환
    
    </details>
    
    <details>
    <summary>delete_low_density()</summary>
    <h3> 함수 개요 </h3>
    
    위도, 경도가 같은 데이터를 그룹화하여 그 개수가 일정 개수 이상, 이하인 데이터들을 **데이터프레임에서 선택하거나 또는 제거** **후 반환**하는 함수 구현
    
    <h3> 함수 파라미터 </h3>
    
    - `data (pd.DataFrame)` : 위도, 경도를 column으로 갖는 데이터프레임
    - `lower (int)` : 선택할 구간의 하한
    - `upper (int)` : 선택할 구간의 상한
    - `drop (bool, optional)` : False이면 구간에 해당하는 데이터 반환. Default는 True.
    
    <h3> 함수 동작 방식 </h3>
    
    1. 인자(argument)로 받은 데이터프레임(`data`)을 위도, 경도별로 그룹화하여 중복 수를 센다.
    2. 제거(또는 선택)하고자 하는 구간에 중복 수가 들어가면, 해당하는 데이터의 인덱스를 저장한다.
    3. 데이터프레임에서 제거(또는 선택)하려는 인덱스에 해당하는 데이터를 버린 후(또는 선택 후) 반환한다.
    
    </details>
    
    <details>
    <summary>class MissingValueImputer</summary>
    <h3> 클래스 개요 </h3>
    주어진 데이터셋에 결측치 처리를 적용하는 메서드 제공
    
    - `mean_imputer()`: 평균 보간 함수
    - `median_imputer()`: 중앙값 보간 함수
    - `iterative_imputer()`: Iterative 방법을 사용한 보간 함수
    - `knn_imputer()`: KNN 방법을 사용한 보간 함수
    
    <h3> 메서드 파라미터 </h3>
    
    - `mean_imputer()`, `median_imputer()` :
        
        `df: Union[pd.DataFrame, pd.Series]` 입력 데이터프레임 또는 시리즈
        
    - `iterative_imputer()`, `knn_imputer()` :
        
        `df: pd.DataFrame` 입력 데이터프레임
        
    
    <h3> 메서드 동작 방식 </h3>
    
    - `mean_imputer()`: Pandas의 메서드 `pd.mean()`을 사용하여 결측치를 처리 후 반환하는 함수
    - `median_imputer()`: Pandas의 메서드 `pd.median()`을 사용하여 결측치를 처리 후 반환하는 함수
    - `iterative_imputer()`: sklearn.impute의 메서드 `IterativeImputer()`에서 default 값을 사용하여 결측치를 처리 후 반환하는 함수
    - `knn_imputer()`: sklearn.impute의 메서드 `KNNImputer()`에서 default 값을 사용하여 결측치를 처리 후 반환하는 함수
    
    </details>
    
    <details>
    <summary>urgent_sale_cut()</summary>
    <h3> 함수 개요 </h3>
    
    `train_data`에서 급처매물에 해당하는 인덱스를 원소로 갖는 리스트를 반환하는 함수
    
    <h3> 함수 파라미터 </h3>
    
    - `data: pd.DataFrame` 입력 데이터프레임. **무조건 `merge_dataset()` 함수를 통해서 불러온 `train_data`여야 한다.**
    - `sigma_weight: float` 표준편차에 대한 가중치
    
    <h3> 함수 동작 방식 </h3>
    
    1. `train_data`에서 위도, 경도를 그룹화 합니다.
    2. 그룹별 `deposit`의 평균과 표준편차를 구합니다.
    3. 평균에 `sigma_weight`와 표준편차를 곱해서 빼 `key_benchmark`를 구합니다.
    4. 그룹별 `deposit`이 각 key_benchmark 미만인 index를 반환하는 리스트를 만듭니다.
    
    </details>
    
- `feature_engineering.py` : 새로운 피처(파생변수) 생성하거나 변환하는 클래스, 함수를 모아놓은 모듈
    <details>
    <summary>find_nearest_haversine_distance()</summary>
    <h3> 함수 개요 </h3>
    건물과 지하철/학교/공원 사이의 최단 거리를 구하고, 그 거리에 해당하는 공공장소의 위치 정보(위도, 경도)를 데이터프레임으로 함께 반환하는 함수
    <h3> 함수 파라미터 </h3>
      
    - `train_data: pd.DataFrame` 건물의 위도와 경도를 포함하는 (학습/테스트) 데이터프레임
    - `loc_data: pd.DataFrame` 비교 대상(지하철/학교/공원)의 위도 경도를 담고 있는 데이터프레임
    
    <h3> 함수 동작 방식 </h3>
    
    1. 위도, 경도 데이터를 라디안 값으로 변환 (haversine 공식에서 삼각함수 계산에 사용)
    2. BallTree 구조를 활용한 haversine 거리 계산
    3. 가장 가까운 공공장소를 찾고, 거리를 미터($m$) 단위로 변환
    4. 위도, 경도, 최단거리를 포함한 데이터프레임 생성하고 반환
    
    </details>
    
    <details>
    <summary>find_places_within_radius()</summary>
    <h3> 함수 개요 </h3>
    특정 반경 이내 지하철, 학교, 공원의 개수와 위치 정보를 반환하는 함수
    <h3> 함수 파라미터 </h3>
    
    - `data: pd.DataFrame` 건물의 위도, 경도를 column으로 갖는 학습(훈련) 또는 테스트 데이터프레임
    - `loc_data: pd.DataFrame` 공공장소의 위도, 경도를 column으로 갖는 데이터프레임
    - `radius_meter: int` 탐색하려는 반경
    
    <h3> 함수 동작 방식 </h3>
    
    1. 위도, 경도 데이터를 라디안 값으로 변환 (haversine 공식에서 삼각함수 계산에 사용)
    2. `BallTree` 구조를 활용한 haversine 거리 계산
    3. `query_radius` 메서드로 주어진 반경 이내의 공공장소 개수 탐색
    4. 특정 반경 내 공공장소 수, 반경의 중심(위도, 경도)로 이루어진 데이터프레임 생성 후 반환
    
    </details>
    
    <details>
    <summary>apply_log_transformation()</summary>
    <h3> 함수 개요 </h3>
    
    - 주어진 데이터프레임에 로그 변환을 적용한 뒤 데이터프레임을 반환하는 함수
    - 0인 데이터를에 로그를 취했을 때 -inf 값이 나오는 걸 방지하기 위해 `np.log1p` 함수 활용
    
    <h3> 함수 파라미터 </h3>
    
    - `train_data: pd.DataFrame` 학습(훈련)용 데이터프레임
    - `test_data: pd.DataFrame` 테스트용 데이터프레임
    
    <h3> 함수 동작 방식 </h3>
    
    주어진 데이터프레임에 오른쪽으로 꼬리가 긴 분포를 갖는 column에 각각 `np.log1p` 함수를 적용한다.
    
    </details>
    
    <details>
    <summary>class ClusteringModel</summary>
    <h3> 클래스 개요 </h3>
    주어진 데이터셋에 대해 클러스터링 기법(K-means, DBSCAN, GMM)을 적용하고 각각 최적의 클러스터 수를 찾는 메서드를 제공
    <h3> 메서드 파라미터 </h3>
    
    - `data` : 클러스터링을 수행할 데이터셋 (ex. `train_data["latitude", "longitude"]`)
    - `max_clusters` : 시도할 최대 클러스터 수(default=20)
    - `n_clusters` : 클러스터링에서 사용할 클러스터 수
    - `min_samples` : DBSCAN에서의 최소 샘플 수(default=5)
    - `eps` : 두 샘플이 동일한 클러스터에 속하기 위한 최대 거리(DBSCAN)
    - `train_data`, `test_data`, `feature_columns`, `label_column`
    
    <h3> 메서드 동작 방식 </h3>
    
    1. K-Means 클러스터링
        - `find_kmeans_n_clusters(max_clusters=20)` : 최대 클러스터 수를 지정하여 K-Means의 최적 클러스터 수 결정 (Elbow 테스트 결과를 시각화하고 사용자에게 입력받는 형태)
        - `kmeans_clustering()` : 사용자가 선택한 클러스터 수로 수행하고 클러스터링 결과 반환
    2. DBSCAN 클러스터링
        - `find_dbscan_n_clusters(min_samples=5)` : Silhouette score를 계산해 최적의 eps값을 반환
        - `dbscan_clustering(eps, min_samples=5)` : 최적의 eps값으로 수행 후 클러스터링 결과 반환
    3. GMM 클러스터링
        - `find_gmm_n_clusters(max_clusters=20)` : BIC, AIC 점수를 계산해 시각화한 후, 더 작은 값을 기반으로 최적의 클러스터 수 결정
        - `gmm_clustering(n_clusters)` : 최적의 클러스터 수로 수행 후 클러스터링 결과 반환
    
    </details>
    
- `load_dataset.py` : 데이터셋 로드
    <details>
    <summary>load_dataset()</summary>
    
    <h3> 함수 개요 </h3>
    
    대회에서 제공한 데이터셋을 모두 데이터프레임으로 불러오는 함수
    
    <h3> 함수 파라미터 </h3>
    
    없음
    
    <h3> 함수 동작 방식 </h3>
    
    `data_path`를 파일 경로로 지정하고, `pd.read_csv()`를 통해 데이터셋을 모두 불러온 후 데이터프레임으로 반환
    (`train_data`, `test_data`, `sample_submission`, `interest_data`, `subway_data`, `school_data`, `park_data`)
    
    </details>
    
- `merge_dataset.py` : 새로운 피처를 기존 데이터셋에 추가
    <details>
    <summary>merge_dataset()</summary>
    <h3> 함수 개요 </h3>
    
    `train_data` 와 `test_data` 에 추가적인 변수를 병합하여 최종적으로 확장된 데이터로 반환하는 함수
    
    <h3> 함수 파라미터 </h3>
    
    - `train_data (pd.DataFrame)` : 학습 데이터프레임
    - `test_data (pd.DataFrame)` : 테스트 데이터프레임
    - `interest_data (pd.DataFrame)` : 금리 정보가 담긴 데이터프레임
    - `subway_data (pd.DataFrame)` : 지하철 위도, 경도가 담긴 데이터프레임
    - `school_data (pd.DataFrame)` : 학교 종류, 위도, 경도가 담긴 데이터프레임
    - `park_data (pd.DataFrame)` : 공원 위도, 경도, 면적이 담긴 데이터프레임
    
    <h3> 함수 동작 방식 </h3>
    
    1. `interest_data`(금리 정보)를 `train_data`, `test_data`의 `contract_year_month`를 기준으로 병합한 후 `interest_rate`를 추가
    2. `find_nearest_haversine_distance()` 함수를 사용해 건물과 지하철, 학교, 공원과의 최단 거리, 해당 위도・경도를 계산하여 새로운 변수로 추가
    (이 때, 공원의 경우 100,000 $m^2$ 이상인 공원과의 거리만 계산한다.)
    3. 최단거리에 위치한 지하철, 학교, 공원을 위도・경도를 기준으로 그룹화한 뒤, 개수를 각각 계산하여 새로운 변수로 추가
    (이 때, 공원의 경우 100,000 $m^2$ 이상인 공원의 수만 계산한다.)
    4. `find_places_within_radius()` 함수를 사용해 건물로부터 특정 반경 이내에 지하철, 학교, 공원 수를 계산하여 새로운 변수로 추가
    (이 때, 공원의 경우 100,000 $m^2$ 이상인 공원의 수만 계산한다.)
    5. 4번에서 만든 변수를 활용, 특정 반경 이내 100,000 $m^2$ 이상인 공원이 있으면 1, 없으면 0인 범주형 변수 추가
    6. `kmeans_clustering()` 함수를 사용해 위도, 경도를 기준으로 클러스터링한 결과를 새로운 변수로 추가
    7. 6번에서 만든 변수와 `find_nearest_haversine_distance()` 함수를 활용, 클러스터별 `deposit`이 최댓값인 데이터와의 최단거리를 계산하여 새로운 변수로 추가
    8. 병합된 `train_data`와 `test_data`를 반환
    
    </details>
