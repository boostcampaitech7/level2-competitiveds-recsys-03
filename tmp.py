import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 랜덤 데이터 생성
torch.manual_seed(42)
X = torch.rand(1000, 10)
y = torch.rand(1000)  # 회귀 문제를 위해 연속형 타겟 생성

# 훈련 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# 데이터로더 생성
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 간단한 다중 퍼셉트론 모델 정의 (회귀)
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)  # 회귀 문제에서는 출력이 1개
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 회귀에서는 softmax를 사용하지 않음
        return x

# 모델, 손실 함수, 옵티마이저 정의
model = SimpleMLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()  # 출력 차원을 조정
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# 모델 평가
model.eval()
with torch.no_grad():
    y_pred = model(X_test).squeeze()

# 회귀 평가지표 계산
mse = mean_squared_error(y_test.numpy(), y_pred.numpy())
r2 = r2_score(y_test.numpy(), y_pred.numpy())

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
