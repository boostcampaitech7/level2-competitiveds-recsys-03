from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from data.load_dataset import load_dataset
from data.merge_dataset import merge_dataset
from data.feature_engineering import *
from model.inference import save_csv
from model.feature_select import select_features
from model.data_split import split_features_and_target
from model.model_train import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

train_data, test_data, sample_submission, interest_data, subway_data, school_data, park_data = load_dataset()
# 기존 데이터에 새로운 feature들을 병합한 데이터프레임 불러오기
train_data, test_data = merge_dataset(train_data, test_data, interest_data, subway_data, school_data, park_data)

### 3. Data Preprocessing

# 위치 중복도 낮은 행 삭제
groups = train_data.groupby(["latitude", "longitude"])["index"].count()
conditioned_groups_index = groups[(groups >= 2) & (groups <= 5)].index # 이 범위를 파라미터로 조정하는걸로
small_groups = train_data[
    train_data["latitude"].isin(conditioned_groups_index.get_level_values(0)) &
    train_data["longitude"].isin(conditioned_groups_index.get_level_values(1))
]
train_data.drop(small_groups.index, axis=0, inplace=True)

# built_year > 2024 행 삭제
train_data = train_data[train_data["built_year"] < 2024]
train_data.reset_index(drop=True, inplace=True)

# log 변환
train_data, test_data = apply_log_transformation(train_data, test_data)

selected_cols = [
    "log_area_m2",
    "built_year",
    "latitude",
    "longitude",
    "log_leader_distance",
    "log_subway_distance",
    "log_school_distance",
    "log_park_distance",
    "contract_year_month",
    # "contract_day",
    "num_of_subways_within_radius",
    "park",
    "region"
]
X_train, y_train = train_data[selected_cols], train_data["deposit"]
X_test = test_data[selected_cols]
log_y_train = train_data["log_deposit"]

# 1. 데이터 불러오기
X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 2. 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(12, 4)
        self.fc2 = nn.Linear(4, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        # out = self.dense3(out)
        # out = self.relu(out)
        return out

# 3. 모델 인스턴스 생성
model = SimpleModel()

# 가중치 초기화
for layer in model.children():
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)

# 4. 손실 함수와 옵티마이저 정의
criterion = nn.L1Loss()  # Mean Absolute Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adam 옵티마이저

# 5. 모델 학습
num_epochs = 5
loss_df = []
for epoch in tqdm(range(num_epochs), desc="💃Total Epoch🕺"):
    model.train()
    epoch_loss = 0
    for inputs, targets in tqdm(train_loader, desc=f"✨Epoch {epoch+1}✨:"):
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    average_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')
    loss_df.append(average_loss)


plt.plot(loss_df)

# 6. 예측 수행 및 배치 단위로 MAE 계산
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze()
print(y_pred)