from data.load_dataset import load_dataset
from data.merge_dataset import merge_dataset
from data.feature_engineering import *
from model.model_train import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import torch
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings
import random

warnings.filterwarnings("ignore", category=UserWarning)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

match torch.cuda.is_available():
    case True: device = "cuda"
    case _: device = "cpu"

# 데이터 로드
train_data, test_data, sample_submission, interest_data, subway_data, school_data, park_data = load_dataset()
train_data, test_data = merge_dataset(train_data, test_data, interest_data, subway_data, school_data, park_data)


### Data Preprocessing

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

# year, month 변수 분리
train_data["contract_year"] = train_data["contract_year_month"] // 100
train_data["contract_month"] = train_data["contract_year_month"] % 100
test_data["contract_year"] = test_data["contract_year_month"] // 100
test_data["contract_month"] = test_data["contract_year_month"] % 100


### Feature Selection

selected_cols = [
    "area_m2",
    "built_year",
    "latitude",
    "longitude",
    "log_leader_distance",
    "log_subway_distance",
    # "log_school_distance",
    "log_park_distance",
    # "contract_year_month",
    "num_of_subways_within_radius",
    "park",
    "region",
    # 추가 변수
    # "interest_rate",
    # "num_of_schools_within_radius",
    # "num_of_parks_within_radius",
    # "contract_type",
    "contract_year",
    "contract_month"
]

X_train, y_train = train_data[selected_cols], train_data["deposit"]
X_test = test_data[selected_cols]
log_y_train = train_data["log_deposit"]

# StandardScaler 적용
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.fit_transform(X_test)

# 다항식 변환 적용 (2차 다항식)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)

# 훈련, 검증 데이터 분리
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

# 자료형 변경
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train.values, dtype=torch.float32, device=device)
X_val = torch.tensor(X_valid, dtype=torch.float32, device=device)
y_val = torch.tensor(y_valid.values, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)

# 데이터로더 설정
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

### MLP 모델 정의

input_size = X_train.shape[1]

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.softplus(self.fc3(x))
        return out


### 모델 학습

# 모델 객체 생성
model = SimpleModel()
model.to(device)

# 가중치 초기화
for layer in model.children():
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)

# 손실 함수와 옵티마이저 정의
criterion = nn.L1Loss()  # Mean Absolute Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 옵티마이저
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# 조기 종료를 위한 파라미터 설정
patience = 20
best_val_loss = float('inf')
epochs_no_improve = 0

# 학습 과정
num_epochs = 1000
loss_df = []
for epoch in range(num_epochs): # tqdm(range(num_epochs), desc="💃Total Epoch🕺"):
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader: # tqdm(train_loader, desc=f"✨Epoch {epoch+1}✨:", leave=False):
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_loader)
    scheduler.step(average_loss)
    print(f'✨Epoch [{epoch+1}/{num_epochs}]✨ Train Loss: {average_loss:.4f}', end=', ')

    # 검증 과정
    model.eval()
    val_epoch_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs).squeeze()
            val_loss = criterion(outputs, targets)
            val_epoch_loss += val_loss.item()

    average_val_loss = val_epoch_loss / len(val_loader)

    # # NaN 예외 처리
    # if math.isnan(average_val_loss):
    #     print('Validation Loss is NaN. Skipping this epoch.')
    #     continue
    
    print(f'Valid Loss: {average_val_loss:.4f}')

    # 조기 종료 조건 체크
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        epochs_no_improve = 0  # 성능이 개선된 경우, 카운터를 리셋
    else:
        epochs_no_improve += 1  # 성능이 개선되지 않으면 카운터 증가

    # 설정한 patience를 초과하면 학습 중단
    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs. Best validation loss: {best_val_loss:.4f}")
        break

    # 손실 기록
    loss_df.append(average_val_loss)


### 예측, 시각화 및 제출 파일 생성

# 에포크 별 손실 시각화 (.py의 경우 생략)
sns.lineplot(loss_df, color="tomato")

plt.grid()
plt.title("MAE Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.show()


# 예측 수행 및 배치 단위로 MAE 계산
model.eval()
with torch.no_grad():
    y_pred = model(X_test).squeeze()
print(y_pred)


# 예측값 시각화 (.py의 경우 생략)
y_pred = y_pred.cpu()
y_pred = pd.Series(y_pred)
sns.histplot(y_pred)

# 통계량 확인
y_pred.describe()

# 제출 파일 생성
sample_submission["deposit"] = y_pred
sample_submission.to_csv("output.csv", index=False)


### (선택) 모델의 가중치 저장

# 현재 모델 가중치를 저장
torch.save(model.state_dict(), 'model_checkpoint.pth') # epoch 152에서 중단

# # 모델 인스턴스 생성 후 가중치 로드
# model.load_state_dict(torch.load('model_checkpoint.pth'))