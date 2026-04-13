# Continual Learning Experiment: Fine-tuning vs EWC vs DER++

완전한 PyTorch 기반 연속 학습(Continual Learning) 실험으로 세 가지 방법을 비교합니다:
- **Fine-tuning (기준)**
- **Elastic Weight Consolidation (EWC)**
- **Experience Replay with DER++**

## 개요

카타스트로픽 포게팅(Catastrophic Forgetting)을 측정하면서 모델 파라미터가 여러 작업에 걸쳐 어떻게 변하는지 시각화합니다.

## 프로젝트 구조

```
cl/
├── run_all.py                  # 진입점: 전체 실험 실행
├── datasets.py                 # Split MNIST & Split CIFAR-10 데이터셋 클래스
├── models.py                   # MNIST_MLP 및 CIFAR10_CNN 모델
├── train.py                    # 메인 학습 루프 (TaskIncrementalLearner)
├── evaluate.py                 # 평가 유틸리티
├── visualize.py                # 결과 시각화 (matplotlib)
├── methods/
│   ├── __init__.py
│   ├── finetune.py            # 기준: 단순 파인튜닝
│   ├── ewc.py                 # EWC with Fisher Information Matrix
│   └── derpp.py               # DER++ with Experience Replay Buffer
├── results.pkl                # 저장된 결과 (실행 후)
├── continual_learning_results.png # 시각화 (실행 후)
└── README.md                  # 이 파일
```

## 데이터셋 설정

### Split MNIST
- **5개 이진 작업**: (0vs1), (2vs3), (4vs5), (6vs7), (8vs9)
- 각 작업: 2개 클래스의 이진 분류
- 입력 크기: 784 (28×28 플래트)

### Split CIFAR-10
- **5개 이진 작업**: 2개 클래스씩
  - Task 0: {airplane, automobile}
  - Task 1: {bird, cat}
  - Task 2: {deer, dog}
  - Task 3: {frog, horse}
  - Task 4: {ship, truck}
- 각 작업: 2개 클래스의 이진 분류
- 입력 크기: 3072 (3×32×32)

## 모델 아키텍처

### MNIST_MLP (소규모 MLP)
```
Input (784)
  ↓
FC (256) + ReLU
  ↓
FC (256) + ReLU
  ↓
FC (2)  # 이진 출력
```

### CIFAR10_CNN (소규모 CNN)
```
Input (3, 32, 32)
  ↓
Conv2d(3→32, 3×3) + ReLU + MaxPool
  ↓
Conv2d(32→64, 3×3) + ReLU + MaxPool
  ↓
Flatten
  ↓
FC (512) + ReLU
  ↓
FC (2)  # 이진 출력
```

## 세 가지 방법의 개념

### 1. Fine-tuning (기준)
**개념**: 각 새 작업에서 표준 SGD/Adam으로 모델을 업데이트
- 메모리: 없음
- 정규화: 없음
- **예상**: 높은 카타스트로픽 포게팅

**손실 함수**:
```
L = CE(f(x), y)
```

### 2. Elastic Weight Consolidation (EWC)
**개념**: Fisher Information Matrix를 사용하여 이전 작업에 중요한 파라미터를 보호

각 작업 후:
1. 현재 작업의 Fisher 정보(대각) 계산
2. 중요 파라미터는 큰 Fisher 값을 가짐
3. 새 작업 학습 시, 중요 파라미터의 변화를 제약

**손실 함수**:
```
L = CE(f(x), y) + (λ/2) Σ_i F_i * (θ_i - θ_i*)²

where:
  F_i = Fisher 중요도
  θ_i = 현재 파라미터
  θ_i* = 이전 작업 최적 파라미터
  λ = 5000 (정규화 강도)
```

**예상**: 중간 정도의 포게팅 감소

### 3. Experience Replay with DER++
**개념**: 이전 작업의 샘플을 저장하고 새 작업 학습 중 재현(replay)

특징:
- 고정 크기 재현 버퍼 (200개 샘플)
- 클래스 균형 유지 (reservoir sampling)
- 두 가지 재현 손실:
  1. 표준 재현: 재현 샘플 간 CE 손실
  2. Dark Experience Replay: 현재 로짓 vs 저장된 로짓의 MSE

**손실 함수**:
```
L = CE(f(x_new), y_new)
    + α * MSE(f(x_replay), logits_replay)    # Dark ER
    + β * CE(f(x_replay), y_replay)          # Standard ER

where:
  α = 0.1 (dark ER 가중치)
  β = 0.5 (표준 ER 가중치)
```

**예상**: 가장 강한 포게팅 방지

## 추적되는 메트릭

### 1. 정확도 행렬 (Accuracy Matrix)
```
accuracy_matrix[i][j] = 작업 i를 학습한 후 작업 j의 정확도
```
- **대각선**: 각 작업을 처음 학습했을 때의 성능
- **비대각선**: 포게팅을 보여줌 (아래쪽 값이 작을수록 포게팅 심함)

### 2. 평균 정확도 (Average Accuracy)
각 작업 학습 후, 지금까지 본 모든 작업의 평균 정확도

### 3. 가중치 드리프트 (Weight Drift)
파라미터 변화량의 L2 노름 (각 작업 후)
- **높은 드리프트**: 큰 파라미터 변화
- **낮은 드리프트** (EWC/DER++): 제약된 변화

### 4. 역 이전 (Backward Transfer, BWT)
포게팅을 정량화하는 메트릭

```
BWT = (1/(T-1)) * Σ_{j=0}^{T-2} (A[T-1][j] - A[j][j])

where:
  T = 총 작업 수
  A[i][j] = 작업 i 후 작업 j의 정확도
```

- **BWT < 0**: 포게팅 (음수일수록 심함)
- **BWT = 0**: 포게팅 없음
- **BWT > 0**: 긍정적 이전 (드물음)

## 시각화

실행 후 `continual_learning_results.png`가 생성되며, 다음을 포함합니다:

### 1. 정확도 행렬 히트맵 (2개 데이터셋 × 3개 방법)
6개의 히트맵:
- 행 = 학습한 작업
- 열 = 평가한 작업
- 색상 = 정확도 (초록 = 높음, 빨강 = 낮음)

### 2. 평균 정확도 곡선
- X축: 작업 인덱스 (1-5)
- Y축: 평균 정확도
- 3개 선 (Fine-tuning / EWC / DER++)
- MNIST와 CIFAR-10 모두 표시

### 3. 가중치 드리프트
- X축: 작업 인덱스
- Y축: 파라미터 변화 (L2 노름)
- 각 방법의 드리프트 패턴 비교
- EWC가 드리프트를 제약하는 모습을 볼 수 있음

### 4. Backward Transfer (BWT) 막대 그래프
- X축: 방법 이름
- Y축: BWT 점수
- MNIST와 CIFAR-10의 그룹화된 막대
- 음수 = 포게팅 (DER++에서 가장 작은 음수 = 포게팅 최소)

## 실행 방법

### 요구사항
```
PyTorch >= 2.0
torchvision
matplotlib
numpy
tqdm
```

### 설치
```bash
pip install torch torchvision matplotlib numpy tqdm
```

### 실행
```bash
python run_all.py
```

또는 가상환경이 이미 설정되었다면:
```bash
/Users/idahyun/Desktop/cl/.venv/bin/python run_all.py
```

### 실행 시간
CPU 기준: ~17-20분
- MNIST: ~1분 (fine-tuning) + ~1.5분 (EWC) + ~1.5분 (DER++)
- CIFAR-10: ~3분 (fine-tuning) + ~4분 (EWC) + ~4.5분 (DER++)

## 결과 해석

### 예상되는 결과

| 메트릭 | Fine-tuning | EWC | DER++ |
|--------|------------|-----|-------|
| 최종 평균 정확도 (MNIST) | ~50-60% | ~60-65% | ~90%+ |
| 최종 평균 정확도 (CIFAR-10) | ~70% | ~73% | ~76%+ |
| BWT (MNIST) | ~-0.59 | ~-0.52 | ~-0.10 |
| BWT (CIFAR-10) | ~-0.27 | ~-0.21 | ~-0.19 |
| 가중치 드리프트 제약 | 없음 | 중간 | 버퍼 기반 |

### 해석

1. **Fine-tuning은 최악**: 각 새 작업 학습 시 이전 작업을 잊음

2. **EWC는 개선**: Fisher 행렬로 중요 파라미터 보호
   - 드리프트가 감소하는 경향
   - BWT가 개선됨

3. **DER++이 최고**: 재현 버퍼로 실제 데이터 유지
   - 가장 높은 평균 정확도
   - 가장 작은 음수 BWT (포게팅 최소)
   - 메모리 요구사항: 고정 200개 샘플

## 파일 설명

### run_all.py
- 진입점
- 두 데이터셋에서 모든 방법 실행
- 결과를 `results.pkl`로 저장
- 시각화 생성 및 저장

### datasets.py
- `SplitMNIST`: Split MNIST 데이터셋
- `SplitCIFAR10`: Split CIFAR-10 데이터셋
- `TaskDataset`: 개별 작업 데이터 래퍼
- `CombinedTaskDataset`: 여러 작업 평가 시 사용

### models.py
- `MNIST_MLP`: MNIST용 소규모 MLP
- `CIFAR10_CNN`: CIFAR-10용 소규모 CNN
- `get_parameters()`: 파라미터 스냅샷 저장
- `compute_weight_drift()`: 파라미터 변화 계산

### train.py
- `TaskIncrementalLearner`: 메인 학습 오케스트레이터
- `train_all_tasks()`: 순차적 작업 학습
- BWT 계산

### methods/finetune.py
- `FinetuneTrainer`: 기준 파인튜닝
- 표준 교차 엔트로피 손실만 사용

### methods/ewc.py
- `EWCTrainer`: Elastic Weight Consolidation
- `compute_fisher()`: Fisher Information Matrix 계산
- `consolidate_task()`: 작업 완료 후 Fisher 누적
- `ewc_loss()`: EWC 정규화 항

### methods/derpp.py
- `DERppTrainer`: DER++ 학습기
- `ReplayBuffer`: 경험 재현 버퍼
  - `add_task_data()`: Reservoir sampling으로 버퍼 업데이트
  - `sample()`: 재현 배치 샘플링
- Dark ER과 표준 ER 손실 결합

### evaluate.py
- 평가 유틸리티 함수
- 통계 계산 (대각선 정확도, final row 등)

### visualize.py
- `plot_results()`: 종합 시각화 생성
- 4개 서브플롯 타입 작성

## 고급 설정

## 하이퍼파라미터 조정

파라미터를 조정하려면 `run_all.py`의 `TaskIncrementalLearner` 초기화를 수정하세요:

```python
learner = TaskIncrementalLearner(
    model_class=model_class,
    num_tasks=num_tasks,
    dataset=dataset,
    device=device,
    learning_rate=0.001,  # 학습률 조정
    epochs=5              # 에포크 조정
)
```

### EWC 람다 조정

`train.py`에서:
```python
trainer = EWCTrainer(
    model, device=device,
    learning_rate=learning_rate, epochs=epochs,
    ewc_lambda=5000  # 정규화 강도 (높을수록 더 강한 제약)
)
```

### DER++ 하이퍼파라미터 조정

`train.py`에서:
```python
trainer = DERppTrainer(
    model, device=device,
    learning_rate=learning_rate, epochs=epochs,
    buffer_size=200,    # 재현 버퍼 크기
    alpha=0.1,          # Dark ER 가중치
    beta=0.5            # 표준 ER 가중치
)
```

## 재현성

모든 난수 시드는 `run_all.py`의 `set_seed(42)`로 설정됩니다:
- PyTorch 난수 생성기
- NumPy 난수 생성기
- Python random 모듈
- CUDA 난수 생성기 (if available)

## 문제 해결

### 메모리 부족
- 배치 크기 감소 (`datasets.py`에서 `batch_size` 파라미터)
- 에포크 수 감소 (`run_all.py`에서 `epochs` 파라미터)
- DER++ 버퍼 크기 감소 (derpp.py에서 buffer_size)

### 느린 실행
- CPU 대신 GPU 사용 (CUDA 가능한 경우)
- 에포크 수 감소
- 데이터셋 크기 감소

## 논문 참고

- **EWC**: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (PNAS 2017)
- **DER++**: Buzzega et al., "Dark Experience for General Continual Learning" (ECCV 2020)

## 라이선스

이 코드는 교육 목적으로 제공됩니다.

## 작성자

GitHub Copilot - 연속 학습 실험 구현
