# Continual Learning Research Playground

Task-Incremental Learning 환경에서 8가지 연속 학습(Continual Learning) 알고리즘을 직접 구현하고 비교하는 실험 프레임워크.

---

## 지원 알고리즘

| 키 | 이름 | 핵심 아이디어 | 데이터 저장 | importance 계산 |
|---|---|---|---|---|
| `finetune` | Fine-tuning | 제약 없는 순차 학습 (baseline) | ✗ | — |
| `ewc` | EWC | Fisher Information으로 중요 파라미터 보호 | ✗ | post-hoc (데이터 필요) |
| `si` | SI | 학습 중 파라미터 기여도를 온라인으로 누적 | ✗ | online (데이터 불필요) |
| `lwf` | LwF | Knowledge distillation으로 이전 task 지식 보존 | ✗ | — |
| `derpp` | DER++ | Replay buffer + logit distillation | ✓ 버퍼 | — |
| `gem` | GEM | Gradient projection으로 이전 task 손실 증가 차단 | ✓ 메모리 | — |
| `hat` | HAT | Task별 binary mask로 뉴런 소유권 분리 | ✗ | — |
| `co2l` | Co2L | Contrastive learning + asymmetric distillation | ✗ | — |

---

## 프로젝트 구조

```
cl-research-playground/
├── run_all.py          # 실험 진입점 (CLI)
├── train.py            # TaskIncrementalLearner — 학습 루프 오케스트레이터
├── models.py           # MNIST_MLP, CIFAR10_CNN, MLP_Co2L, CNN_Co2L
├── datasets.py         # SplitMNIST, SplitCIFAR10
├── losses.py           # SupConLoss, AsymDistillLoss (Co2L용)
├── evaluate.py         # 평가 유틸리티
├── visualize.py        # 결과 시각화 (matplotlib)
└── methods/
    ├── finetune.py     # FinetuneTrainer
    ├── ewc.py          # EWCTrainer
    ├── si.py           # SITrainer
    ├── lwf.py          # LwFTrainer
    ├── derpp.py        # DERppTrainer, ReplayBuffer
    ├── gem.py          # GEMTrainer, EpisodicMemory
    ├── hat.py          # HATTrainer
    └── co2l.py         # Co2LTrainer
```

---

## 실험 설정

### 데이터셋

**Split MNIST** — 5개 이진 분류 task
```
Task 0: 0 vs 1
Task 1: 2 vs 3
Task 2: 4 vs 5
Task 3: 6 vs 7
Task 4: 8 vs 9
```

**Split CIFAR-10** — 5개 이진 분류 task
```
Task 0: airplane vs automobile
Task 1: bird vs cat
Task 2: deer vs dog
Task 3: frog vs horse
Task 4: ship vs truck
```

### 모델 아키텍처

**MNIST_MLP**
```
784 → FC(256) → ReLU → FC(256) → ReLU → FC(2)
```

**CIFAR10_CNN**
```
3×32×32 → Conv(32) → MaxPool → Conv(64) → MaxPool → FC(512) → ReLU → FC(2)
```

> Co2L은 backbone + projection head가 분리된 `MLP_Co2L` / `CNN_Co2L` 변형을 자동 사용.

---

## 알고리즘 상세

### Fine-tuning (baseline)

제약 없이 각 task를 순차 학습. 이전 task를 빠르게 망각함.

```
L = CE(f(x), y)
```

---

### EWC — Elastic Weight Consolidation
> Kirkpatrick et al., PNAS 2017

각 task 완료 후 데이터로 Fisher Information Matrix (대각 근사)를 계산하고, 중요한 파라미터가 크게 변하지 않도록 정규화 항을 추가.

```
L = CE(f(x), y) + (λ/2) Σ_i F_i (θ_i − θ_i*)²

F_i  : Fisher 중요도 (기울기 제곱 평균) — task 완료 후 데이터로 계산
θ_i* : 이전 task 완료 시점의 최적 파라미터
λ    : 5000
```

**주요 메서드**
- `consolidate_task(loader)` — Fisher 계산 및 최적 파라미터 저장 (task 완료 후, 데이터 필요)

---

### SI — Synaptic Intelligence
> Zenke, Poole & Ganguli, ICML 2017

EWC와 동일한 정규화 구조이지만, 중요도를 학습 중 매 step마다 온라인으로 누적. task 완료 후 데이터 재사용 없음.

```
온라인 누적 (각 gradient step):
  ω_k += −g_k · (θ_k_new − θ_k_old)

task 완료 후 (consolidation):
  Ω_k += clamp( ω_k / (Δθ_k² + ξ),  min=0 )
  Δθ_k = task 전체 파라미터 변화량,  ξ = 0.1 (damping)

정규화 (EWC와 동일 구조):
  L = CE(f(x), y) + (λ/2) Σ_k Ω_k (θ_k − θ_k*)²

λ = 1.0,  ξ = 0.1
```

**EWC와의 차이**

| | EWC | SI |
|---|---|---|
| 중요도 계산 | task 후 데이터 재사용 | 학습 중 온라인 누적 |
| 추가 데이터 패스 | 필요 | 불필요 |
| 중요도 의미 | 손실 곡률 (Fisher) | 손실 감소 기여도 |

**주요 메서드**
- `consolidate_task()` — ω → Ω 변환, 리셋 (데이터 불필요)

---

### LwF — Learning without Forgetting
> Li & Hoiem, ECCV 2016 / TPAMI 2018

새로운 task 학습 전에 현재 모델을 frozen 참조 모델로 저장하고, soft output을 distillation target으로 사용. 이전 task 데이터를 전혀 저장하지 않는 memory-free 방식.

```
L = CE(f(x; θ), y)  +  λ · T² · KL( softmax(f(x;θ)/T)  ‖  softmax(f(x;θ_old)/T) )

θ_old : task 학습 전에 freeze된 이전 모델
T     : temperature (높을수록 soft한 분포)
λ = 1.0,  T = 2.0
```

**학습 순서 (pre-task 훅 존재)**
```
1. record_soft_labels(loader)   ← task 학습 전: 현재 모델 snapshot
2. train_task(loader)           ← CE + KD 결합 학습
```

**주요 메서드**
- `record_soft_labels(loader)` — task 학습 전 현재 모델을 `old_model`로 freeze
- `distillation_loss(logits, x)` — temperature T 적용 KL divergence (T² 스케일링)

---

### DER++ — Dark Experience Replay++
> Buzzega et al., NeurIPS 2020

고정 크기 replay buffer에 `(x, y, logit)` 트리플을 저장하고, 새 task 학습 중 두 가지 replay 손실을 함께 사용.

```
L = CE(f(x_new), y_new)
  + α · MSE(f(x_buf), logit_buf)     ← dark experience replay
  + β · CE(f(x_buf), y_buf)          ← standard replay

α = 0.1,  β = 0.5,  buffer_size = 200
```

**주요 메서드**
- `update_buffer(loader)` — reservoir sampling으로 버퍼 업데이트

---

### GEM — Gradient Episodic Memory
> Lopez-Paz & Ranzato, NeurIPS 2017

각 task의 episodic memory를 저장하고, 새 task 학습 시 gradient가 모든 이전 task의 memory gradient와 양의 내적을 가지도록 제약. 위반 시 QP로 projection.

```
제약 조건:  ⟨g̃, g_k⟩ ≥ 0    (모든 이전 task k에 대해)

Primal:  min_{g̃}  ½‖g̃ − g‖²    s.t.  G g̃ ≥ 0
Dual:    min_{v≥0} ½ v^T(GG^T)v + (Gg)^T v
복원:    g̃ = g + G^T v*

n_memories = 256,  margin = 0.0
```

**QP solver 의존성**
- `quadprog` 설치 시: 최적 QP projection
- 미설치 시: Gram-Schmidt projection으로 자동 fallback

```bash
pip install quadprog    # 권장
```

**주요 메서드**
- `consolidate_task(loader)` — episodic memory 저장 (task 완료 후)

---

### HAT — Hard Attention to the Task
> Serra et al., ICML 2018

Task별 학습 가능한 임베딩에서 sigmoid mask를 생성. Hard mask가 된 뉴런은 이후 task에서 gradient를 차단하여 "소유권"을 보존.

```
mask_t  = σ(s · e_t)           (s: 온도, e_t: task embedding)
L = CE + λ · capacity_reg

λ = 0.75,  s_max = 400
```

**주요 메서드**
- `set_eval_task(task_id)` — 평가 시 해당 task의 hard mask 적용
- `consolidate_task()` — 현재 task mask를 누적 mask에 병합

---

### Co2L — Contrastive Continual Learning
> Cha et al., CVPR 2021

Supervised Contrastive Loss로 표현을 학습하고, 이전 모델과의 Asymmetric Distillation Loss로 표현 공간이 변하지 않도록 제약.

```
L = CE(classifier(z), y)
  + L_SupCon(proj(z))
  + λ · L_AsymDistill(proj(z), proj_prev(z))    (task ≥ 1)

λ = 1.0,  temperature = 0.1
```

**주요 메서드**
- `consolidate_task()` — 현재 모델을 deepcopy하여 다음 task의 distillation 참조 모델로 보존

---

## 평가 지표

### Accuracy Matrix
```
A[i][j] = task i 학습 완료 후 task j에서의 정확도
```
- 대각선 `A[t][t]`: 각 task를 처음 학습했을 때의 성능
- 대각선 아래 값이 낮을수록 forgetting이 심함

### Backward Transfer (BWT)
```
BWT = (1/(T−1)) Σ_{j=0}^{T−2} (A[T−1][j] − A[j][j])
```
- `BWT < 0`: forgetting (음수가 클수록 심함)
- `BWT = 0`: forgetting 없음
- `BWT > 0`: positive backward transfer

### Weight Drift
각 task 완료 후 파라미터 변화량의 L2 norm. 정규화 방법일수록 드리프트가 억제되는 경향.

---

## 실행 방법

### 설치

```bash
pip install torch torchvision numpy matplotlib tqdm
pip install quadprog    # GEM QP projection (선택, 없으면 fallback 사용)
```

### 기본 실행

```bash
# 전체 실험 (MNIST + CIFAR-10, 8개 방법)
python run_all.py

# 특정 데이터셋만
python run_all.py --dataset mnist
python run_all.py --dataset cifar10

# 특정 방법만
python run_all.py --methods finetune ewc si lwf

# 시각화 없이 실행
python run_all.py --no-plot

# 저장된 결과로 시각화만 생성
python run_all.py --plot-only results.pkl
```

### CLI 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--dataset` | `all` | `mnist`, `cifar10`, `all` |
| `--methods` | 전체 8개 | 실행할 방법 목록 |
| `--epochs` | `5` | task당 에포크 수 |
| `--learning-rate` | `0.001` | 학습률 |
| `--device` | `auto` | `cpu`, `cuda`, `auto` |
| `--seed` | `42` | 재현성 시드 |
| `--output` | `results.pkl` | 결과 저장 경로 |

---

## 하이퍼파라미터 요약

| 방법 | 파라미터 | 기본값 |
|---|---|---|
| EWC | `ewc_lambda` | 5000 |
| SI | `si_lambda` / `xi` | 1.0 / 0.1 |
| LwF | `lwf_lambda` / `temperature` | 1.0 / 2.0 |
| DER++ | `buffer_size` / `alpha` / `beta` | 200 / 0.1 / 0.5 |
| GEM | `n_memories` / `margin` | 256 / 0.0 |
| HAT | `hat_lambda` / `s_max` | 0.75 / 400 |
| Co2L | `distill_lambda` / `temperature` | 1.0 / 0.1 |

하이퍼파라미터 변경은 `train.py`의 `train_all_tasks()` 내 각 trainer 초기화 블록에서 수정.

---

## 논문 참고

| 알고리즘 | 논문 |
|---|---|
| EWC | Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017 |
| SI | Zenke, Poole & Ganguli, "Continual Learning Through Synaptic Intelligence", ICML 2017 |
| LwF | Li & Hoiem, "Learning without Forgetting", ECCV 2016 / TPAMI 2018 |
| DER++ | Buzzega et al., "Dark Experience for General Continual Learning: a Strong, Simple Baseline", NeurIPS 2020 |
| GEM | Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning", NeurIPS 2017 |
| HAT | Serra et al., "Overcoming Catastrophic Forgetting with Hard Attention to the Task", ICML 2018 |
| Co2L | Cha et al., "Co2L: Contrastive Continual Learning", CVPR 2021 |
