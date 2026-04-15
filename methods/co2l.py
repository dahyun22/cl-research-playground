"""
Co2L 트레이너: Contrastive Continual Learning
===========================================
Replay buffer 없이 catastrophic forgetting을 줄이기 위해
supervised contrastive loss와 asymmetric distillation을 결합한다.

손실 함수 (step당):
    Task 1       : L_CE + L_SupCon
    Task t (t>1) : L_CE + L_SupCon + λ * L_AsymDistill

참고 문헌:
    Cha et al., "Co2L: Contrastive Continual Learning", ICCV 2021
    Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
"""

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from losses import SupConLoss, AsymDistillLoss


class Co2LTrainer:
    """
    Co2L(Contrastive Continual Learning) 트레이너.

    개념적 접근 방식:
      - SupConLoss는 같은 클래스의 projection들을 임베딩 공간에서 가깝게 모아,
        태스크 전반에 걸쳐 일반화 가능한 표현을 만든다.
      - AsymDistillLoss는 같은 샘플에 대해 현재 모델의 projection이 이전 모델의
        projection과 가깝도록 강제하여, replay 데이터를 저장하지 않고도 이전
        태스크 지식을 보존한다.
      - CE loss는 평가 시 사용할 classifier head가 제대로 학습되도록 유지한다.
      - 이전 모델은 각 태스크 종료 시점(consolidate_task)에 deepcopy로 고정되며,
        이는 EWC의 consolidate 패턴과 유사하다.

    핵심 수식:
      L = L_CE + L_SupCon + λ * L_AsymDistill   (task t > 1)
      L = L_CE + L_SupCon                        (task 1, 이전 모델 없음)

      L_CE          = CrossEntropy(classifier(features), y)
      L_SupCon      = SupConLoss(projections, y)
      L_AsymDistill = AsymDistillLoss(proj_cur, proj_prev)
    """

    def __init__(self, model, device="cpu", learning_rate=0.001, epochs=5,
                 distill_lambda=1.0, temperature=0.1):
        """
        Args:
            model          : MLP_Co2L 인스턴스
            device (str)   : "cpu" 또는 "cuda"
            learning_rate  : Adam 학습률
            epochs         : 태스크당 epoch 수
            distill_lambda : AsymDistillLoss의 가중치 λ
            temperature    : 두 contrastive loss가 공유하는 τ
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.distill_lambda = distill_lambda

        self.criterion   = nn.CrossEntropyLoss()
        self.supcon_loss = SupConLoss(temperature=temperature)
        self.distill_loss = AsymDistillLoss(temperature=temperature)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 각 태스크 종료 후 consolidate_task()에서 설정됨.
        # 첫 번째 태스크에서는 None이다.
        self.prev_model = None

    # ──────────────────────────────────────────────────────────────────────────
    # 핵심 학습
    # ──────────────────────────────────────────────────────────────────────────

    def train_task(self, train_loader, verbose=False):
        """
        단일 태스크에서 모델을 학습한다.

        각 배치에서의 단계:
          1. Forward pass  → features (B, 256), projections (B, 128)
          2. classifier(features)와 라벨 사이의 CE loss 계산
          3. projections와 라벨 사이의 SupCon loss 계산
          4. projections와 prev_model의 projections 사이의 distill loss 계산
             [task t>1]
          5. Backward + Adam step

        Args:
            train_loader : (x, y, _) 튜플을 반환하는 DataLoader
            verbose      : epoch별 손실 출력 여부

        Returns:
            float: 마지막 epoch의 평균 손실
        """
        self.model.train()
        avg_loss = 0.0

        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0

            for x, y, _ in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                # ── 순전파 ────────────────────────────────────────────────
                features, projections = self.model(x)
                logits = self.model.classifier(features)

                # ── 손실 계산 ──────────────────────────────────────────────
                ce     = self.criterion(logits, y)
                supcon = self.supcon_loss(projections, y)

                if self.prev_model is not None:
                    with torch.no_grad():
                        _, proj_prev = self.prev_model(x)
                    distill = self.distill_loss(projections, proj_prev) # 표현 공간에서 이전 모델과 비교
                    loss = ce + supcon + self.distill_lambda * distill
                else:
                    loss = ce + supcon

                # ── 역전파 ────────────────────────────────────────────────
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)

            if verbose:
                print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

        return avg_loss

    # ──────────────────────────────────────────────────────────────────────────
    # 태스크 종료 후 정리(consolidation)
    # ──────────────────────────────────────────────────────────────────────────

    def consolidate_task(self):
        """
        현재 모델의 스냅샷을 고정하여 다음 태스크에서 증류 기준 모델로 사용한다.

        각 태스크마다 train_task()와 evaluate() 이후 호출되며,
        EWCTrainer.consolidate_task()와 유사한 역할을 한다.
        고정된 모델은 이후 다시 업데이트되지 않으며, 모든 파라미터는
        requires_grad=False 상태이고 모델은 eval 모드로 유지된다.
        """
        self.prev_model = deepcopy(self.model)
        self.prev_model.eval()
        for p in self.prev_model.parameters():
            p.requires_grad_(False)

    # ──────────────────────────────────────────────────────────────────────────
    # 평가
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(self, test_loader):
        """
        분리된 테스트셋에서 classifier 정확도를 평가한다.

        projection head가 아니라 backbone → classifier 경로를 사용하며,
        이는 다른 모든 trainer의 평가 방식과 동일하다.

        Args:
            test_loader: (x, y, _) 튜플을 반환하는 DataLoader

        Returns:
            float: [0, 1] 범위의 정확도
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y, _ in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                features, _ = self.model(x)
                logits = self.model.classifier(features)
                preds  = torch.argmax(logits, dim=1)

                correct += (preds == y).sum().item()
                total   += y.size(0)

        return correct / max(total, 1)
