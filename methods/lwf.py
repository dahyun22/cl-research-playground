"""
망각 없는 학습(LwF) 트레이너
===========================
새로운 태스크 데이터에 대한 지식 증류를 통해 이전 태스크의 지식을 유지하며,
에피소드 메모리나 이전 태스크 데이터는 저장하지 않는다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy


class LwFTrainer:
    """
    LwF(Learning without Forgetting) 트레이너.

    개념적 접근 방식:
      - 새로운 태스크를 학습하기 전에 현재 모델을 고정된 기준 모델
        (old_model)로 스냅샷한다. 이 모델은 지금까지 학습한 모든 지식을 담고 있다.
      - 학습 중에는 두 개의 손실을 동시에 최소화한다:
          1. 하드 라벨에 대한 교차 엔트로피  → 새로운 태스크 학습
          2. 학생 모델과 old_model의 소프트 출력 사이의 KL divergence
             → 이전 태스크 망각 방지
      - 이전 태스크의 데이터는 전혀 저장하지 않는다(메모리 프리).
        파인튜닝 대비 추가 비용은 배치마다 한 번의 고정된 forward pass와
        이전 모델 가중치 사본을 유지하는 것뿐이다.
      - Task-Incremental Learning 설정에서는 모델이 모든 태스크에 대해 하나의
        백본을 공유하고, 출력 헤드는 태스크별 이진 클래스 점수(0 대 1)로 매핑된다.
        지식 증류 손실은 이 공유 헤드에 적용되며, 이를 통해 백본이 이전 태스크에
        유용한 표현에서 멀어지지 않도록 한다.

    핵심 수식:
      L_task     = CE(f(x; θ), y)
      L_distill  = T² · KL( softmax(f(x; θ)/T)  ‖  softmax(f(x; θ_old)/T) )
      L_total    = L_task + λ · L_distill

      여기서:
        θ_old = 현재 태스크 학습 전에 고정된 모델 가중치
        T     = temperature  (클수록 더 부드러운 분포 → 더 풍부한 신호)
        λ     = distillation 가중치

    참고 문헌:
      Li & Hoiem, "Learning without Forgetting", ECCV 2016 / TPAMI 2018.
    """

    def __init__(self, model, device="cpu", learning_rate=0.001, epochs=5,
                 lwf_lambda=1.0, temperature=2.0):
        """
        Args:
            model: 신경망 모델
            device (str): 학습에 사용할 장치
            learning_rate (float): 옵티마이저의 학습률
            epochs (int): 태스크당 epoch 수
            lwf_lambda (float): 지식 증류 손실의 가중치 (λ)
            temperature (float): 증류에 사용할 softmax temperature (T ≥ 1).
                                 값이 클수록 타깃 분포가 더 부드러워지며,
                                 클래스 간 관계 정보가 더 잘 드러난다.
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lwf_lambda = lwf_lambda
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 이전 태스크의 정보를 담은 고정 기준 모델.
        # record_soft_labels()가 처음 호출되기 전까지는 None이다.
        self.old_model = None

    def record_soft_labels(self, train_loader):
        """
        새로운 태스크를 학습하기 전에 현재 모델을 스냅샷한다.

        원본 LwF 논문에서는 새로운 태스크 데이터에 대한 이전 네트워크의 "응답"을
        미리 계산해 고정된 증류 타깃으로 저장한다. 이 구현은 현재 모델을
        old_model로 고정하고 학습 중에 soft target을 즉석에서 계산함으로써
        같은 효과를 낸다. old_model은 태스크 내에서 변하지 않으므로 두 방식은
        수학적으로 동등하다.

        각 태스크마다 train.py에서 train_task() 바로 직전에 호출되어야 하며,
        이렇게 해야 old_model이 현재 태스크는 아직 학습하지 않았지만 이전
        모든 태스크는 반영한 가중치를 가지게 된다.

        Args:
            train_loader: 다음 태스크의 학습 데이터에 대한 DataLoader.
                          API 일관성과 향후 pre-caching 확장을 위해 받지만,
                          기본 구현에서는 사용하지 않는다.
        """
        self.old_model = deepcopy(self.model) # 현재 모델 복사
        self.old_model.eval() # 평가 모드로 고정
        for param in self.old_model.parameters():
            param.requires_grad_(False) # 그래디언트 업데이트 방지, 학습 X

    def distillation_loss(self, student_logits, x):
        """
        현재 모델과 old_model 사이의 지식 증류 손실.

        Temperature T는 두 분포의 부드러움을 조절한다:
          - T = 1  : 일반적인 softmax, argmax 클래스에 뾰족하게 집중됨
          - T > 1  : 더 부드러운 분포가 되어 모델이 모든 클래스에 대해 가지는
                     신뢰도 순서를 드러냄(dark knowledge)

        T²를 곱하는 이유는 softmax 전에 logits를 T로 나누면서 줄어든 gradient
        크기를 보정해, 증류 항이 교차 엔트로피 항과 비슷한 스케일을 유지하도록
        하기 위해서다.
        (Hinton et al., "Distilling the Knowledge in a Neural Network", 2015)

        Args:
            student_logits (torch.Tensor): 현재 모델의 logits, shape (B, C)
            x (torch.Tensor): device 위의 입력 배치, old_model 조회에 사용됨

        Returns:
            torch.Tensor: T²로 스케일된 스칼라 KL divergence 증류 손실
        """
        T = self.temperature

        with torch.no_grad(): # old_model의 출력은 고정된 타깃이므로 그래디언트 계산 불필요
            teacher_logits = self.old_model(x)

        student_log_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)

        # KL(teacher ‖ student); batchmean은 배치 크기로 정규화한다.
        kl = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean")
        return kl * (T ** 2)

    def train_task(self, train_loader, verbose=False):
        """
        LwF 증류를 사용해 단일 태스크에서 모델을 학습한다.

        첫 번째 태스크(old_model is None)에서는 교차 엔트로피만 사용한다.
        두 번째 태스크부터는 새로운 태스크의 하드 라벨에 대한 교차 엔트로피와
        고정된 old_model로부터의 지식 증류를 함께 사용해, 데이터를 저장하지
        않고도 catastrophic forgetting을 줄인다.

        Args:
            train_loader: 현재 태스크 학습 데이터에 대한 DataLoader
            verbose (bool): epoch별 학습 진행 상황 출력 여부

        Returns:
            float: 마지막 epoch의 평균 손실
        """
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0

            for x, y, _ in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(x)
                # LwF 손실 = CE(새로운 태스크) + λ * KL(현재 모델 ‖ old_model)
                ce_loss = self.criterion(logits, y)

                if self.old_model is not None: 
                    kd_loss = self.distillation_loss(logits, x) # 증류 손실
                    loss = ce_loss + self.lwf_lambda * kd_loss # lwf_lambda로 가중치 조절
                else: # if 0 Task, 증류 손실이 없음.
                    loss = ce_loss

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)

            if verbose:
                print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

        return avg_loss

    def evaluate(self, test_loader):
        """
        테스트 데이터에서 모델 정확도를 평가한다.

        Args:
            test_loader: 테스트 데이터에 대한 DataLoader

        Returns:
            float: 정확도 (0-1)
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y, _ in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                preds = torch.argmax(logits, dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        accuracy = correct / max(total, 1)
        return accuracy
