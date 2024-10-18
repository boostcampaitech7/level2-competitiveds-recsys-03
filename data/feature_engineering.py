import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from tqdm import tqdm

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


class ClusteringModel:
    def __init__(self, data):
        self.data = data

    ### K-means 최적의 클러스터 수 찾는 메서드 ###
    def find_kmeans_n_clusters(self, max_clusters=20):
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
    def kmeans_clustering(self, n_clusters, train_data, test_data, feature_columns, label_column="region"):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        kmeans.fit(self.data)

        train_data[label_column] = kmeans.predict(train_data[feature_columns])
        test_data[label_column] = kmeans.predict(test_data[feature_columns]) 

        return kmeans


    ### DBSCAN 최적의 클러스터 수 찾는 메서드 ###
    def find_dbscan_n_clusters(self, min_samples=5):
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
    def dbscan_clustering(self, eps, train_data, test_data, feature_columns, label_column="region", min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(self.data)
        
        train_data[label_column] = dbscan.fit_predict(train_data[feature_columns])
        test_data[label_column] = dbscan.fit_predict(test_data[feature_columns]) 

        return dbscan


    ### GMM 및 최적의 클러스터 수 찾는 메서드 ###
    def find_gmm_n_clusters(self, max_clusters=20):
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
    def gmm_clustering(self, n_clusters, train_data, test_data, feature_columns, label_column="region"):
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(self.data)

        train_data[label_column] = gmm.predict(train_data[feature_columns])
        test_data[label_column] = gmm.predict(test_data[feature_columns]) 
        
        return gmm