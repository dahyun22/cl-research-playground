# 빠른 참조 가이드 (Quick Reference)

## 전체 작업 흐름

```
1. 초기화
   ├─ 데이터셋 로드 (Split MNIST / Split CIFAR-10)
   ├─ 모델 초기화 (MNIST_MLP / CIFAR10_CNN)
   └─ 방법별 학습기 생성 (Finetune / EWC / DER++)

2. 순차 작업 학습 (작업 0-4)
   ├─ Task i 학습 데이터 로드
   ├─ 모델 학습 (5 에포크)
   ├─ 모든 작업 j∈[0..i]에서 평가
   ├─ 파라미터 드리프트 계산
   ├─ 결과 저장
   └─ 메서드별 후처리:
      ├─ EWC: Fisher Information 계산 및 누적
      └─ DER++: 버퍼에 데이터 추가

3. 메트릭 계산 및 시각화
   ├─ 정확도 행렬 생성
   ├─ BWT (Backward Transfer) 계산
   ├─ 히트맵, 곡선, 막대 그래프 생성
   └─ 결과.pkl, 플롯.png 저장
```

## 주요 코드 패턴

### 1. 데이터 로드
```python
from datasets import SplitMNIST, SplitCIFAR10

# MNIST 데이터셋 생성
dataset = SplitMNIST(data_root="./data", batch_size=32)

# Task 0 학습 데이터 가져오기
train_loader = dataset.get_task_data(task_id=0, split="train")

# Task 0 테스트 데이터 가져오기
test_loader = dataset.get_task_data(task_id=0, split="test")

# 여러 작업 평가용 통합 로더
eval_loader = dataset.get_all_test_data(task_ids=[0, 1, 2])
```

### 2. 모델 사용
```python
from models import MNIST_MLP, CIFAR10_CNN
import torch

# 모델 생성
model = MNIST_MLP()

# 포워드 패스
logits = model(x)  # x shape: (batch_size, 784)

# 파라미터 스냅샷
params_snapshot = model.get_parameters()
# Returns: {name: tensor, ...}

# 파라미터 드리프트 계산
drift_per_layer, total_drift = model.compute_weight_drift(params_snapshot)
# drift_per_layer: {name: float, ...}
# total_drift: float (모든 파라미터의 L2 노름)
```

### 3. Fine-tuning (기준)
```python
from methods.finetune import FinetuneTrainer

# 학습기 생성
trainer = FinetuneTrainer(
    model,
    device="cpu",
    learning_rate=0.001,
    epochs=5
)

# Task 학습
train_loss = trainer.train_task(train_loader, verbose=True)

# 평가
accuracy = trainer.evaluate(test_loader)
```

### 4. EWC
```python
from methods.ewc import EWCTrainer

# 학습기 생성
trainer = EWCTrainer(
    model,
    device="cpu",
    learning_rate=0.001,
    epochs=5,
    ewc_lambda=5000  # 정규화 강도
)

# Task 학습 (EWC 손실 자동 포함)
train_loss = trainer.train_task(train_loader, verbose=True)

# Fisher Information 계산 및 파라미터 보존
trainer.consolidate_task(train_loader)

# 평가
accuracy = trainer.evaluate(test_loader)
```

### 5. DER++
```python
from methods.derpp import DERppTrainer

# 학습기 생성
trainer = DERppTrainer(
    model,
    device="cpu",
    learning_rate=0.001,
    epochs=5,
    buffer_size=200,   # 재현 버퍼 크기
    alpha=0.1,         # Dark ER 가중치
    beta=0.5           # 표준 ER 가중치
)

# Task 학습 (버퍼의 데이터와 새로운 데이터 혼합)
train_loss = trainer.train_task(train_loader, verbose=True)

# 버퍼 업데이트
trainer.update_buffer(train_loader)

# 평가
accuracy = trainer.evaluate(test_loader)
```

## 메트릭 계산

### 정확도 행렬
```python
accuracy_matrix = results["accuracy_matrix"]
# accuracy_matrix[i][j] = accuracy on task j after learning task i

# 예: MNIST Fine-tuning
# [[0.9995, 0.9995],           # Task 0 후: T0=99.95%, T1=99.95%
#  [0.7234, 0.8304],           # Task 1 후: T0=72.34% (포게팅!), T1=83.04%
#  [0.5421, 0.6231, 0.6631],   # Task 2 후: T0=54.21%, T1=62.31%, T2=66.31%
#  ...]
```

### Backward Transfer (BWT)
```python
# BWT 공식
T = len(accuracy_matrix)
bwt_sum = 0.0
for j in range(T - 1):
    final_acc = accuracy_matrix[-1][j]      # 모든 작업 후 T_j
    initial_acc = accuracy_matrix[j][j]     # T_j 처음 학습했을 때
    bwt_sum += final_acc - initial_acc

bwt = bwt_sum / (T - 1)

# 예: BWT = -0.5939 (MNIST Fine-tuning)
# 의미: 평균적으로 이전 작업을 ~59% 잊음
```

### 가중치 드리프트
```python
weight_drift = results["weight_drift"]
# weight_drift[layer_name] = [drift_task0, drift_task1, ..., drift_task4]

# 예: network.fc0.weight
# [10.076, 16.158, 9.836, 9.859, 10.336]  # Fine-tuning (높은 드리프트)
# [10.303, 14.670, 8.562, 7.495, 7.994]  # EWC (더 제약됨)
# [10.455, 15.172, 9.217, 12.400, 11.927] # DER++ (적응적)
```

## 결과 해석

### 히트맵 читання
```
         T0    T1    T2    T3    T4
    ┌─────────────────────────────┐
T0  │ 100%  ??    ??    ??    ??  │  대각선: 작업 학습 직후 성능
T1  │ 72%   83%   ??    ??    ??  │  
T2  │ 54%   62%   66%   ??    ??  │  비대각선: 포게팅 (왼쪽 아래로 갈수록 작으면 포게팅 심함)
T3  │ 52%   58%   61%   71%   ??  │
T4  │ 51%   56%   59%   65%   70% │  마지막 행: 최종 성능 (모든 작업 봤을 때)
    └─────────────────────────────┘
```

- **파란색 (High)**: 좋은 유지 성능
- **빨간색 (Low)**: 포게팅 발생

### 평균 정확도 곡선
```
Accuracy
   |     
 95%|  ●─────────────  DER++ (재현 버퍼로 강함)
   |   │ \
 80%|  ●   \──●────   EWC (중간 정도 감소)
   |   │      \   \
 50%|  ●        \   \─────  Fine-tuning (가파른 하락)
   |
    └─────────────────────────
      T1   T2   T3   T4   T5

Key: 오른쪽으로 갈수록 덜 떨어질수록 포게팅이 적음
```

### BWT 스코어 해석
```
BWT 값           의미
────────────────────────────
 +0.1            긍정적 이전 (새 작업 학습이 이전 작업 도움)
  0.0            영향 없음
 -0.1            약간의 포게팅
 -0.5            심각한 포게팅
 -0.9            거의 모든 이전 작업을 잊음
```

## 각 방법의 장단점

| 측면 | Fine-tuning | EWC | DER++ |
|------|-------------|-----|-------|
| **메모리** | 매우 낮음 | 낮음 | 중간 (200 샘플) |
| **계산 비용** | 낮음 | 중간 (Fisher 계산) | 중간 (버퍼 샘플링) |
| **포게팅 방지** | 나쁨 | 중간 | 매우 좋음 |
| **새 작업 성능** | 좋음 | 좋음 | 좋음 |
| **매개변수 수** | 1개 (λ) | 1개 (λ) | 3개 (buffer_size, α, β) |
| **데이터 필요** | 현재만 | 현재만 | 모든 이전 (버퍼) |

## 커스터마이징

### 다양한 람다 값 (EWC) 비교
```python
# train.py 수정:
ewc_lambdas = [100, 1000, 5000, 10000]
results = {}

for lambda_val in ewc_lambdas:
    trainer = EWCTrainer(..., ewc_lambda=lambda_val)
    # 학습 및 평가
    results[lambda_val] = accuracies
```

### 다양한 버퍼 크기 (DER++) 비교
```python
# train.py 수정:
buffer_sizes = [50, 100, 200, 500]
results = {}

for buffer_size in buffer_sizes:
    trainer = DERppTrainer(..., buffer_size=buffer_size)
    # 학습 및 평가
    results[buffer_size] = accuracies
```

### 더 많은 작업 (Split MNIST에 10개 작업)
```python
# datasets.py 수정 (SplitMNIST.__init__):
self.num_tasks = 10
self.task_labels = [
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 0),  # 추가
]
```

## 디버깅 팁

### 학습이 수렴하지 않음
1. 학습률 조정 (0.001 → 0.01 또는 0.0001)
2. 배치 크기 조정 (32 → 64 또는 16)
3. 에포크 증가 (5 → 10)

### EWC 손실이 NaN
1. 람다 값 감소 (5000 → 500 또는 100)
2. 데이터 정규화 확인
3. Fisher 계산 전 모델 평가 모드

### DER++ 버퍼가 너무 빨리 가득 찼음
- buffer_size 증가 또는
- 각 작업에서 더 적은 샘플 저장

## 성능 최적화

### GPU 사용
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

### 배치 처리 증가
```python
dataset = SplitMNIST(batch_size=128)  # 32 → 128
```

### HalfPrecision (32-bit → 16-bit)
```python
model.half()
# 메모리 50% 절감, 약간 빠름, 수치 안정성 감소
```

## 결과 분석 (사후)

### results.pkl 로드
```python
import pickle
import numpy as np

with open("results.pkl", "rb") as f:
    all_results = pickle.load(f)

# MNIST Fine-tuning의 정확도 행렬
mnist_ft = all_results["MNIST"]["finetune"]["accuracy_matrix"]
print(np.array(mnist_ft))

# 모든 방법의 BWT 비교
for method in ["finetune", "ewc", "derpp"]:
    bwt = all_results["MNIST"][method]["bwt"]
    print(f"{method}: BWT = {bwt:.4f}")
```

### 커스텀 플롯 생성
```python
import matplotlib.pyplot as plt
import numpy as np

# Fine-tuning의 대각선 정확도
accuracy_matrix = all_results["MNIST"]["finetune"]["accuracy_matrix"]
diagonal_accs = [accuracy_matrix[i][i] for i in range(len(accuracy_matrix))]

plt.plot(diagonal_accs, marker='o')
plt.xlabel("Task Index")
plt.ylabel("Accuracy (Diagonal)")
plt.title("Fine-tuning: Task Learning Performance")
plt.show()
```

---

더 많은 정보는 [README.md](README.md)를 참조하세요.
