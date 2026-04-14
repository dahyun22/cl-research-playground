"""
Synaptic Intelligence (SI) 트레이너
=================================
학습 중에 파라미터별 중요도를 온라인으로 누적하여 이전 태스크에 중요한
파라미터를 보호하며, 태스크 종료 후 추가 데이터 패스가 필요 없다.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class SITrainer:
    """
    SI(Synaptic Intelligence) 트레이너.

    개념적 접근 방식:
      - EWC는 학습이 끝난 뒤 데이터로 파라미터 중요도를 사후 측정한다.
        반면 SI는 학습 중 각 파라미터가 태스크 손실 감소에 얼마나 기여했는지를
        추적해 중요도를 온라인으로 누적한다.
      - 각 gradient step에서 파라미터 θ_k의 기여도는
        −g_k · Δθ_k로 근사한다. 즉, 이 파라미터의 이동으로 인해 손실이
        얼마나 줄어들었는지를 본다.
      - 태스크가 끝난 후 누적 기여도 ω_k를 전체 파라미터 이동량의 제곱으로
        정규화하여 파라미터별 중요도 Ω_k를 구한다.
      - 이 중요도들은 태스크가 바뀌어도 누적되며, 이후 업데이트를 EWC와
        동일한 방식으로 정규화하지만 추가 데이터 비용은 들지 않는다.

    핵심 수식:

      온라인 누적 (각 gradient step):
        ω_k  +=  −g_k · (θ_k_new − θ_k_old)

        여기서:
          g_k      = ∂L/∂θ_k  (현재 step의 gradient)
          θ_k_old  = optimizer.step() 이전의 파라미터 값
          θ_k_new  = optimizer.step() 이후의 파라미터 값

      태스크 τ 이후 (consolidation):
        Ω_k  +=  clamp( ω_k / (Δθ_k² + ξ),  min=0 )

        여기서:
          Δθ_k = θ_k(τ 종료 시점) − θ_k(τ 시작 시점)   (전체 이동량)
          ξ    = damping 상수  (0으로 나누는 문제 방지)

      정규화 손실 (구조는 EWC와 동일):
        L_SI = (λ/2) · Σ_k  Ω_k · (θ_k − θ_k*)²

        여기서 θ_k*는 직전 태스크 종료 시점의 최적 파라미터다.

    참고 문헌:
      Zenke, Poole & Ganguli, "Continual Learning Through Synaptic Intelligence",
      ICML 2017.
    """

    def __init__(self, model, device="cpu", learning_rate=0.001, epochs=5,
                 si_lambda=1.0, xi=0.1):
        """
        Args:
            model: 신경망 모델
            device (str): 학습에 사용할 장치
            learning_rate (float): 옵티마이저의 학습률
            epochs (int): 태스크당 epoch 수
            si_lambda (float): 정규화 강도(ewc_lambda와 유사한 역할)
            xi (float): damping 상수 ξ
                        태스크 동안 파라미터가 거의 움직이지 않았을 때
                        중요도가 과도하게 커지는 것을 방지한다 (기본값 0.1)
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.si_lambda = si_lambda
        self.xi = xi

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 완료된 모든 태스크에 대해 누적된 중요도 Ω
        # {param_name: tensor}  — 태스크가 끝날 때마다 누적 합산됨
        self.importance_dict = {}

        # 직전 태스크 종료 시점의 최적 파라미터 값 θ*
        self.optimal_params = {}

        # ── 온라인 추적 상태 (각 태스크 시작 시 초기화) ──────────
        # ω: 현재 태스크 내에서의 −g · dθ 누적값
        self._omega_running = {
            name: torch.zeros_like(param.data)
            for name, param in self.model.named_parameters()
        }
        # θ_prev: 직전 step의 파라미터 스냅샷 (step별 Δθ 계산용)
        self._theta_prev = {
            name: param.data.clone().detach()
            for name, param in self.model.named_parameters()
        }
        # θ_start: 현재 태스크 시작 시점의 파라미터 스냅샷
        # (consolidate_task에서 전체 이동량 Δθ 계산에 사용)
        self._theta_start = {
            name: param.data.clone().detach()
            for name, param in self.model.named_parameters()
        }

    # ------------------------------------------------------------------
    # 온라인 ω 누적
    # ------------------------------------------------------------------

    def _update_omega(self, saved_grads):
        """
        각 optimizer step 이후 누적 중요도 추정치 ω를 업데이트한다.

        파라미터 궤적을 따라가는 경로 적분
        ω_k ≈ −∫ g_k dθ_k 를, 이산 step들에서의 −g_k · Δθ_k 합으로 근사한다.

        새 파라미터 값이 있어야 하므로 optimizer.step() 이후에 호출되어야 한다.
        또한 다음 iteration의 zero_grad()가 gradient를 지우기 때문에,
        gradient는 optimizer.step() 이전에 저장해야 한다.

        Args:
            saved_grads (dict): 이번 iteration에서 optimizer step 직전에
                                저장한 {param_name: gradient tensor}
        """
        for name, param in self.model.named_parameters():
            if name not in saved_grads:
                continue
            # Δθ_step = 이번 optimizer step 한 번에서의 변화량
            delta_theta = param.data - self._theta_prev[name]
            # 손실 감소에 대한 기여도: −g · Δθ  (양수면 도움이 된 것)
            self._omega_running[name] -= saved_grads[name] * delta_theta
            # step 기준 스냅샷 갱신
            self._theta_prev[name] = param.data.clone().detach()

    # ------------------------------------------------------------------
    # SI 정규화 손실
    # ------------------------------------------------------------------

    def si_loss(self):
        """
        SI 정규화 페널티를 계산한다.

        구조적으로는 ewc_loss()와 동일하다. 직전 태스크의 최적 파라미터에서
        얼마나 벗어났는지에 대한 가중 제곱 편차를 계산하며, 가중치는
        Fisher diagonal이 아니라 누적된 시냅스 중요도 Ω를 사용한다.

        Returns:
            torch.Tensor: 스칼라 SI 정규화 손실
        """
        loss = 0.0

        for name, param in self.model.named_parameters():
            if name in self.importance_dict:
                importance = self.importance_dict[name]
                optimal = self.optimal_params[name]

                diff = param - optimal
                loss += (importance * diff ** 2).sum()

        return (self.si_lambda / 2) * loss

    # ------------------------------------------------------------------
    # 핵심 트레이너 인터페이스
    # ------------------------------------------------------------------

    def train_task(self, train_loader, verbose=False):
        """
        SI 온라인 중요도 추적과 함께 단일 태스크에서 모델을 학습한다.

        각 gradient step에서 다음을 수행한다:
          1. gradient를 얻기 위한 forward + backward
          2. gradient 저장 (optimizer step 이후에도 필요함)
          3. optimizer.step()
          4. −g · Δθ로 ω 업데이트

        SI 정규화 항은 두 번째 태스크부터 추가된다.
        (importance_dict가 consolidate_task()로 채워진 이후)

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
                task_loss = self.criterion(logits, y)

                # 이전 태스크들이 consolidate되었다면 SI 정규화 추가
                if self.importance_dict:
                    si_reg = self.si_loss()
                    total_task_loss = task_loss + si_reg
                else:
                    total_task_loss = task_loss

                total_task_loss.backward()

                # optimizer가 파라미터를 바꾸기 전에 gradient 저장
                saved_grads = {
                    name: param.grad.data.clone()
                    for name, param in self.model.named_parameters()
                    if param.grad is not None
                }

                self.optimizer.step()

                # 이번 step의 기여도로 온라인 ω 추정치 갱신
                self._update_omega(saved_grads)

                total_loss += total_task_loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)

            if verbose:
                print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

        return avg_loss

    def consolidate_task(self, train_loader=None):
        """
        누적된 ω를 태스크 중요도 Ω로 변환하고 θ*를 업데이트한다.

        각 태스크 학습이 끝난 뒤 한 번 호출된다. EWC와 달리 추가 데이터 패스가
        필요 없으며, 필요한 정보는 모두 train_task() 중 온라인으로 수집되었다.

        단계:
          1. 파라미터별 이동량 Δθ = θ_current − θ_start 계산
          2. 현재 태스크 중요도 계산: clamp(ω / (Δθ² + ξ), min=0)
          3. importance_dict의 누적값에 더함
          4. 현재 파라미터를 새로운 최적 θ*로 저장
          5. 다음 태스크를 위해 ω, θ_prev, θ_start 초기화

        Args:
            train_loader: 다른 트레이너와의 API 일관성을 위해 받지만,
                          사용하지 않는다 (SI는 태스크 후 데이터 패스가 필요 없음).
        """
        for name, param in self.model.named_parameters():
            # 완료된 태스크 전체 동안의 파라미터 이동량
            delta = param.data - self._theta_start[name]

            # 태스크별 중요도: 이동량 크기로 ω를 정규화
            # 0 이상으로 clamp: 음수면 오히려 anti-regularisation이 됨
            task_importance = torch.clamp(
                self._omega_running[name] / (delta ** 2 + self.xi),
                min=0.0
            )

            # 태스크 간 누적
            if name not in self.importance_dict:
                self.importance_dict[name] = task_importance.clone()
            else:
                self.importance_dict[name] += task_importance

            # 현재 파라미터를 다음 태스크용 기준점으로 저장
            self.optimal_params[name] = param.data.clone().detach()

            # 다음 태스크를 위해 온라인 추적 상태 초기화
            self._omega_running[name] = torch.zeros_like(param.data)
            self._theta_prev[name] = param.data.clone().detach()
            self._theta_start[name] = param.data.clone().detach()

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
