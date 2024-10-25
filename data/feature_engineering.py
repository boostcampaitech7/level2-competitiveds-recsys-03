import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from typing import Union


def find_nearest_haversine_distance(
    data: pd.DataFrame, 
    loc_data: pd.DataFrame
) -> pd.DataFrame:
    """
    건물과 지하철/학교/공원 사이의 최단 haversine 거리와 위치 정보를 반환하는 함수

    Args:
        data (pd.DataFrame): 위도, 경도를 column으로 갖는 학습(훈련) 또는 테스트 데이터프레임
        loc_data (pd.DataFrame): 거리를 재려고 하는 대상(학교, 지하철, 공원)의 위도, 경도를 column으로 갖는 데이터프레임

    Returns:
        pd.DataFrame: 최단거리, 최단거리에 해당하는 지점의 위도, 경도를 column으로 갖는 데이터프레임
    """
    # haversine 거리 계산을 위해 degree -> radian 값으로 변환
    data_coords = np.radians(data[["latitude", "longitude"]].values)
    loc_coords = np.radians(loc_data[["latitude", "longitude"]].values)
    
    # 지구 반경 미터 단위로 설정
    earth_radius_meter = 6371000

    # Ball Tree 객체 생성 
    tree = BallTree(loc_coords, metric="haversine")

    # tree.query로 거리 계산
    distances, indices = tree.query(data_coords, k=1) # 가까운 1 지점만 
    distances_meter = distances * earth_radius_meter # 단위를 meter로 변환
    nearest_coords = loc_data[["latitude", "longitude"]].iloc[indices.flatten()].values # 가장 가까운 지점들의 좌표 저장

    # 최단거리, 최단거리에 해당하는 지점의 위도, 경도로 이루어진 데이터프레임 생성
    result_df = pd.DataFrame({
        "nearest_distance" : distances_meter.flatten(),
        "nearest_latitude" : nearest_coords[:, 0],
        "nearest_longitude" : nearest_coords[:, 1]
    })

    return result_df



def find_places_within_radius(
		data: pd.DataFrame,
		loc_data: pd.DataFrame,
		radius_meter: int
) -> pd.DataFrame:
    """
    특정 반경 이내 공공장소의 개수와 위치 정보를 반환하는 함수

    Args:
        data (pd.DataFrame): 건물의 위도, 경도를 column으로 갖는 학습(훈련) 또는 테스트 데이터프레임
        loc_data (pd.DataFrame): 공공장소의 위도, 경도를 column으로 갖는 데이터프레임
        radius_meter (int): 탐색하려는 반경

    Returns:
        pd.DataFrame: 특정 반경 이내 공공장소 수와 반경의 중심(건물)의 위도, 경도를 column으로 갖는 데이터프레임
    """
    # degree -> radian 값으로 변환 for 삼각함수
    data_coords = np.radians(data[["latitude", "longitude"]].values)
    loc_coords = np.radians(loc_data[["latitude", "longitude"]].values)
    
    # 지구 반경 미터 단위로 설정
    earth_radius_meter = 6371000

    # Ball Tree 객체 생성
    tree = BallTree(loc_coords, metric="haversine")
    
    # query_radius 메서드로 주어진 반경 이내의 공공장소 개수 탐색
    places_within_radius = tree.query_radius(data_coords, r=radius_meter/earth_radius_meter, 
                                             count_only=True
    )

    # 특정 반경 내 공공장소 수, 반경의 중심(위도, 경도)로 이루어진 데이터프레임 생성
    result_df = pd.DataFrame({
        "num_of_places_within_radius": places_within_radius,
        "latitude": data["latitude"],
        "longitude": data["longitude"]
    })

    return result_df



def apply_log_transformation(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Union[pd.DataFrame, pd.DataFrame]:

    """
    학습 데이터와 테스트 데이터에 로그 변환을 적용하는 함수.
    주로 오른쪽으로 꼬리가 긴 분포를 갖고, 큰 값들을 갖는 변수를 로그 변환하여 데이터 분포를 조정하는 데 사용한다.

    Args:
        train_data (pd.DataFrame): 학습용 데이터프레임.
        test_data (pd.DataFrame): 테스트용 데이터프레임.

    Returns:
        Union[pd.DataFrame, pd.DataFrame]: 로그 변환이 적용된 학습용, 테스트용 데이터프레임.

    """
    #train_data log 변환: 가격, 면적, 지하철, 학교, 공원까지의 최단거리, 대장 아파트까지의 최단거리
    train_data["log_deposit"] = np.log1p(train_data["deposit"])
    train_data["log_area_m2"] = np.log1p(train_data["area_m2"])
    train_data["log_subway_distance"] = np.log1p(train_data["nearest_subway_distance"])
    train_data["log_school_distance"] = np.log1p(train_data["nearest_school_distance"])
    train_data["log_park_distance"] = np.log1p(train_data["nearest_park_distance"])
    train_data["log_leader_distance"] = np.log1p(train_data)["nearest_leader_distance"]
    
    #test_data log 변환: 면적, 지하철, 학교, 공원까지의 최단거리, 대장 아파트까지의 최단거리
    test_data["log_area_m2"] = np.log1p(test_data["area_m2"])
    test_data["log_subway_distance"] = np.log1p(test_data["nearest_subway_distance"])
    test_data["log_school_distance"] = np.log1p(test_data["nearest_school_distance"])
    test_data["log_park_distance"] = np.log1p(test_data["nearest_park_distance"])
    test_data["log_leader_distance"] = np.log1p(test_data)["nearest_leader_distance"]
    
    return train_data, test_data


class ClusteringModel:
    ### 초기화 메서드 ###
    def __init__(self, data):
        self.data = data

    ### K-means 최적의 클러스터 수 찾는 메서드 ###

    def find_kmeans_n_clusters(self, max_clusters: int = 20) -> int:
        """
        K-means 클러스터링에서 최적의 클러스터 수를 찾는 함수 (Elbow Method 사용)

        Args:
            max_clusters (int, optional): 최대 클러스터 수 Defaults to 20.

        Returns:
            int: 최적의 클러스터 수
        """
        wcss = []
        for i in tqdm(range(1, max_clusters + 1), desc="Elbow Method 진행 중...⏳"):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(self.data)
            wcss.append(kmeans.inertia_)

        plt.plot(range(1, max_clusters + 1), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.show()

        # 최적의 클러스터 수 선택
        optimal_clusters = int(input("적절한 n_clusters를 선택해주세요 :"))
        print(f'KMeans Optimal_clusters: {optimal_clusters}')

        return optimal_clusters


    ### K-means 클러스터링 수행 메서드 ###
    def kmeans_clustering(
        self, 
        n_clusters: int, 
        train_data: pd.DataFrame, 
        test_data: pd.DataFrame, 
        feature_columns: list[str], 
        label_column: str = "region"
    ) -> KMeans:
        """
        K-means 클러스터링을 수행하고, train_data와 test_data에 클러스터 레이블 추가

        Args:
            n_clusters (int): 클러스터 수
            train_data (pd.DataFrame): 학습 데이터
            test_data (pd.DataFrame): 테스트 데이터
            feature_columns (list[str]): 클러스터링에 사용할 피처 리스트
            label_column (str, optional): 레이블이 저장될 열 이름. Defaults to "region".

        Returns:
            KMeans: 학습된 KMeans 모델
        """
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        kmeans.fit(self.data)

        train_data[label_column] = kmeans.predict(train_data[feature_columns])
        test_data[label_column] = kmeans.predict(test_data[feature_columns]) 

        return kmeans


    ### DBSCAN 최적의 클러스터 수 찾는 메서드 ###
    def find_dbscan_n_clusters(self, min_samples : int = 5) -> float:
        """
        DBSCAN 클러스터링에서 최적의 eps 값을 찾는 함수

        Args:
            min_samples (int, optional): 최소 샘플 수 Defaults to 5.

        Returns:
            float: 최적의 eps 값. 클러스터가 2개 이상인 경우에만 반환
        """
        best_eps = None
        best_score = -1

        eps_values = [0.01, 0.1, 0.2]

        for eps in tqdm(eps_values, desc="Silhouette Score 계산 중...⏳"):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(self.data)

            if len(set(labels)) > 1:  # clusters 2개 이상
                score = silhouette_score(self.data, labels)
                if score > best_score:
                    best_score = score
                    best_eps = eps

        print(f'Best eps: {best_eps} with Silhouette Score: {best_score}')
        
        return best_eps


    ### DBSCAN 클러스터링 수행 메서드 ###
    def dbscan_clustering(
        self, 
        eps: float, 
        train_data: pd.DataFrame, 
        test_data: pd.DataFrame,
        feature_columns: list[str], 
        label_column: str = "region", 
        min_samples: int = 5
    ) -> DBSCAN:
        """
        DBSCAN 클러스터링을 수행하고, train_data와 test_data에 클러스터 레이블 추가

        Args:
            eps (float): DBSCAN의 eps 값
            train_data (pd.DataFrame): 학습 데이터
            test_data (pd.DataFrame): 테스트 데이터
            feature_columns (list[str]): 클러스터링에 사용할 피처 리스트
            label_column (str, optional): 레이블이 저장될 열 이름. Defaults to "region".
            min_samples (int, optional): 최소 샘플 수. Defaults to 5.

        Returns:
            학습된 DBSCAN 모델
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(self.data)
        
        train_data[label_column] = dbscan.fit_predict(train_data[feature_columns])
        test_data[label_column] = dbscan.fit_predict(test_data[feature_columns]) 

        return dbscan


    ### GMM 및 최적의 클러스터 수 찾는 메서드 ###
    def find_gmm_n_clusters(self, max_clusters: int = 20) -> int:
        """
        GMM에서 BIC와 AIC 값을 이용해 최적의 클러스터 수를 찾는 함수

        Args:
            max_clusters (int, optional): 최대 클러스터 수. Defaults to 20.

        Returns:
            optimal_n_clusters(int): 최적의 클러스터 수
        """
        aic_scores = []
        bic_scores = []

        n_components_range = range(1, max_clusters + 1)

        for n_components in tqdm(n_components_range, desc="BIC, AIC 계산 중...⏳"):
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(self.data)

            # BIC, AIC값 저장
            bic_scores.append(gmm.bic(self.data))
            aic_scores.append(gmm.aic(self.data))

        # BIC와 AIC 점수 시각화 (선택 사항)
        plt.plot(n_components_range, bic_scores, label='BIC')
        plt.plot(n_components_range, aic_scores, label='AIC')
        plt.xlabel('Number of Components')
        plt.ylabel('Scores')
        plt.title('BIC and AIC Scores for GMM')
        plt.legend()
        plt.show()

        # 최적의 클러스터 수 결정
        optimal_n_clusters = n_components_range[np.argmin(bic_scores)] if min(bic_scores) < min(aic_scores) else n_components_range[np.argmin(aic_scores)]
        print(f'Optimal_n_components: {optimal_n_clusters}')
        
        return optimal_n_clusters


    ### GMM 클러스터링 수행 메서드 ###
    def gmm_clustering(self, 
            n_clusters: int,
            train_data: pd.DataFrame, 
            test_data: pd.DataFrame, 
            feature_columns: list[str], 
            label_column: str = "region"
        ) -> GaussianMixture:
        """
        GaussianMixture 클러스터링을 수행하고, train_data와 test_data에 클러스터 레이블 추가

        Args:
            n_clusters (int): 최적의 클러스터 개수
            train_data (pd.DataFrame): 학습 데이터
            test_data (pd.DataFrame): 테스트 데이터
            feature_columns (list[str]): 클러스터링에 사용할 피처 리스트
            label_column (str, optional): 클러스터 결과를 저장할 컬럼 이름. Defaults to "region".

        Returns:
            GaussianMixture: 학습된 GaussianMixture 모델
        """
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(self.data)

        train_data[label_column] = gmm.predict(train_data[feature_columns])
        test_data[label_column] = gmm.predict(test_data[feature_columns]) 
        
        return gmm