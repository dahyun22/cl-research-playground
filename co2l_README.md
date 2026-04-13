# Co2L 구현 설명

> Cha et al., "Co2L: Contrastive Continual Learning", ICCV 2021

---

## 1. 핵심 아이디어

기존 Continual Learning 방법들의 문제:

| 방법 | 문제 |
|---|---|
| Fine-tuning | 이전 태스크 완전히 망각 |
| EWC | Fisher 기반 정규화, 중요 파라미터 보호 |
| Replay | 이전 데이터 저장 필요 (메모리 부담) |

**Co2L의 접근**: Replay 없이, 표현 공간(representation space) 자체를 보존한다.

- **SupConLoss**: 같은 클래스끼리 표현을 가깝게 → 새 태스크에서 좋은 표현 학습
- **AsymDistillLoss**: 현재 모델의 표현을 이전 모델의 표현에 가깝게 유지 → 망각 방지

---

## 2. 관련 파일 구조

```
models.py              ← MLP_Co2L 클래스
losses.py              ← SupConLoss, AsymDistillLoss
methods/co2l.py        ← Co2LTrainer (학습 루프)
tests/test_supcon.py          ← SupConLoss 단위 테스트
tests/test_co2l_convergence.py ← Split-MNIST 수렴 확인
```

---

## 3. 모델 구조: `MLP_Co2L`

```
입력 (784)
    ↓
backbone: fc0(784→256) → ReLU → fc1(256→256) → ReLU
    ↓ features (256-dim)
   ┌─────────────┐
   ↓             ↓
classifier     proj_head
(256→2)        (256→128) → L2 normalize
   ↓             ↓
 logits       projections
(태스크 분류)  (대조학습용)
```

`forward(x)` 는 `(features, projections)` 를 반환한다.

- `features` (B, 256): backbone 출력. `model.classifier(features)` 로 logits 획득
- `projections` (B, 128): L2 정규화된 proj_head 출력. loss 계산에 직접 사용

기존 `MNIST_MLP` 와 다른 점은 backbone/classifier 분리와 proj_head 추가뿐이다.

---

## 4. Loss 함수

### 4-1. 최종 Loss

```
태스크 1:  L = L_CE + L_SupCon
태스크 t:  L = L_CE + L_SupCon + λ · L_AsymDistill
```

각 항의 역할:

| Loss | 입력 | 역할 |
|---|---|---|
| L_CE | logits, labels | classifier head 학습 |
| L_SupCon | projections, labels | 같은 클래스 표현 집합 |
| L_AsymDistill | proj_cur, proj_prev | 이전 표현 공간 보존 |

---

### 4-2. SupConLoss

**수식** (anchor i 기준):

```
L_i = -1/|P(i)| · Σ_{p∈P(i)} log [ exp(z_i·z_p / τ) / Σ_{a∈A(i)} exp(z_i·z_a / τ) ]
```

- `P(i)` = 같은 클래스이면서 i가 아닌 샘플 집합
- `A(i)` = i가 아닌 모든 샘플 집합 (분모)
- `τ` = temperature (default: 0.1)

**직관**: 같은 클래스 샘플끼리는 cosine similarity를 높이고, 다른 클래스와는 낮춘다.

**구현 포인트** (`losses.py:SupConLoss`):

```python
sim = z @ z.T / τ                          # (B, B) 유사도 행렬
sim -= sim.max(dim=1).values               # 수치 안정 (log-sum-exp trick)
exp_sim = exp(sim).masked_fill(self_mask)  # 대각선(자기 자신) 제거
log_prob = sim - log(exp_sim.sum(dim=1))   # log-확률
loss = -mean(log_prob over P(i))
```

**엣지 케이스**: 배치 내 positive pair가 없으면 `(projections * 0).sum()` 반환
→ 값은 0이지만 computation graph 유지 → `.backward()` 안전하게 호출 가능

---

### 4-3. AsymDistillLoss

**수식**:

```
L_i = -log [ exp(z_cur_i · z_prev_i / τ) / Σ_{j=1}^{B} exp(z_cur_i · z_prev_j / τ) ]
```

- `z_cur_i`: 현재 모델의 projection (gradient 흐름)
- `z_prev_i`: 이전 모델의 projection (frozen, no grad)
- 대각선이 positive pair (같은 샘플, 다른 모델)

**직관**: 현재 모델이 이전 모델과 같은 방향으로 샘플을 표현하도록 강제한다.

**왜 "비대칭(Asymmetric)"인가?**

- 쿼리는 항상 현재 모델 (`z_cur`), key bank는 이전 모델 (`z_prev`)
- 반대 방향(이전→현재) loss는 없다
- 덕분에 현재 모델이 새 태스크에 맞게 표현을 조정할 자유도를 유지하면서, 이전 표현 구조는 보존한다

**SupConLoss vs AsymDistillLoss 분모 차이**:

| | SupConLoss | AsymDistillLoss |
|---|---|---|
| 분모 | `j ≠ i` (자기 자신 제외) | `j = 1..B` (자기 자신 포함) |
| 이유 | 같은 모델 내 self-similarity 제거 | 서로 다른 모델이므로 i=j가 진짜 positive |
| 구현 | `masked_fill(self_mask, 0)` | `F.cross_entropy(sim, arange(B))` |

---

## 5. 학습 루프: `Co2LTrainer`

### 태스크 간 흐름

```
┌─────────────────────────────────────────────┐
│  task t 시작                                 │
│                                             │
│  train_task(loader)                         │
│    for each batch:                          │
│      features, proj = model(x)             │
│      logits = model.classifier(features)   │
│                                             │
│      ce     = CrossEntropy(logits, y)       │
│      supcon = SupConLoss(proj, y)           │
│      if prev_model:                         │
│        proj_prev = prev_model(x) [no_grad] │
│        distill = AsymDistillLoss(proj,      │
│                                  proj_prev) │
│        loss = ce + supcon + λ·distill       │
│      else:                                  │
│        loss = ce + supcon                   │
│                                             │
│      loss.backward() / optimizer.step()     │
│                                             │
│  evaluate(loader)                           │
│                                             │
│  consolidate_task()                         │
│    prev_model = deepcopy(model)             │
│    prev_model.eval()                        │
│    prev_model.requires_grad_(False)         │
└─────────────────────────────────────────────┘
         ↓ task t+1 시작 (prev_model 존재)
```

### EWC와의 구조 대응

| | EWC | Co2L |
|---|---|---|
| 첫 태스크 판단 | `if self.fisher_dict:` | `if self.prev_model is not None:` |
| 이전 정보 저장 | `consolidate_task(loader)` → Fisher | `consolidate_task()` → deepcopy |
| 망각 방지 방식 | 파라미터 거리 정규화 | 표현 공간 거리 정규화 |
| 메모리 오버헤드 | Fisher matrix (파라미터 수만큼) | 모델 전체 복사 1개 |

---

## 6. 실행 방법

### 단독 실행

```python
from datasets import SplitMNIST
from models import MLP_Co2L
from train import TaskIncrementalLearner

dataset = SplitMNIST()
learner = TaskIncrementalLearner(
    model_class=MLP_Co2L,
    num_tasks=dataset.num_tasks,
    dataset=dataset,
    device="cpu",
    learning_rate=1e-3,
    epochs=5,
)
results = learner.train_all_tasks(method_name="co2l", verbose=True)
```

> `model_class`에 `MLP_Co2L`을 넘기더라도 `train_all_tasks("co2l")` 내부에서 자동으로 `MLP_Co2L()`로 초기화된다.

### CLI

```bash
python run_all.py --dataset mnist --methods finetune ewc co2l
```

### 테스트

```bash
python tests/test_supcon.py             # SupConLoss 단위 테스트 (7개 케이스)
python tests/test_co2l_convergence.py   # Split-MNIST 수렴 확인
```

---

## 7. 주요 하이퍼파라미터

| 파라미터 | 위치 | 기본값 | 설명 |
|---|---|---|---|
| `temperature` | `Co2LTrainer` | 0.1 | SupConLoss, AsymDistillLoss 공유 τ |
| `distill_lambda` | `Co2LTrainer` | 1.0 | AsymDistillLoss 가중치 λ |
| `proj_dim` | `MLP_Co2L` | 128 | projection head 출력 차원 |
| `learning_rate` | `Co2LTrainer` | 1e-3 | Adam optimizer lr |

`temperature`가 작을수록 분포가 sharp해져 harder negative를 강하게 밀어낸다.
`distill_lambda`를 높일수록 망각 방지에 집중하고, 낮출수록 새 태스크 적응에 자유도가 생긴다.
