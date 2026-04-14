"""
Hard Attention to the Task (HAT) 학습기
=======================================
온도 annealing이 적용된 sigmoid gating을 통해 태스크별 이진 마스크를 학습한다.
그래디언트 마스킹으로 이전 태스크가 이미 점유한 뉴런이 덮어써지는 것을 막는다.

참고 문헌: Serra et al., "Overcoming Catastrophic Forgetting with Hard
Attention to the Task", ICML 2018.

핵심 아이디어
-------------
1. 각 태스크 t는 마스킹되는 각 레이어 l마다 학습 가능한 임베딩 벡터 e_t^l를 가진다.

2. 학습 중 soft mask (온도 s를 1/s_max → s_max로 annealing):
     m_t^l = sigmoid(s * e_t^l)
   학습 초반에는 s가 작아 마스크가 평평해지고(~0.5), 그래디언트가 자유롭게 흐른다.
   학습 후반에는 s가 커져 마스크가 이진값에 가까워진다.

3. 평가 / 통합 시 hard mask (s = s_max):
     h_t^l ≈ m_t^l > 0.5  (거의 이진값)

4. 누적 마스크는 과거 어떤 태스크든 "점유한" 뉴런을 추적한다:
     c^l = max over past tasks of h_t^l

5. 각 backward pass 이후 그래디언트 마스킹:
   - 임베딩 그래디언트: c^l = 1인 위치를 0으로 만든다(이미 점유된 뉴런은 이동 금지).
   - 가중치/편향 그래디언트: 점유된 뉴런과 연결된 행/열을 0으로 만든다.

6. 용량 정규화는 희소한 마스크를 유도한다:
     R = (1/N) * sum_l sum_i  m_t^l_i / (1 + c^l_i)
   전체 손실: L = L_CE + lambda * R
"""

import torch
import torch.nn as nn
import torch.optim as optim


class HATTrainer:
    """
    HAT (Hard Attention to the Task) 학습기.

    개념적 접근:
      - 태스크별, 레이어별로 학습 가능한 attention 임베딩을 유지한다.
      - 학습 중에는 soft mask가 각 은닉층 출력을 gate하고, 온도 annealing으로
        점차 hard mask에 가까워진다.
      - 각 태스크가 끝나면 이진 hard mask를 누적 마스크에 병합해, 해당 뉴런이
        이후 태스크에서 수정되지 않도록 보호한다.
      - 용량 정규화 항은 이미 점유된 뉴런의 재사용을 억제해, 새 태스크에 대해
        새로운 용량을 할당하도록 유도한다.

    인터페이스:
      train_task(train_loader)       – 하나의 태스크 학습(self.current_task 사용)
      consolidate_task(train_loader) – 누적 마스크 갱신, 태스크 카운터 증가
      set_eval_task(task_id)         – evaluate()에서 사용할 태스크 마스크 선택
      evaluate(test_loader)          – eval_task_id용 hard mask로 정확도 평가
    """

    def __init__(self, model, device="cpu", learning_rate=0.001, epochs=5,
                 num_tasks=5, hat_lambda=0.75, s_max=400):
        """
        매개변수:
            model: 신경망 모델(get_hat_layer_sizes, forward_hat,
                   get_hat_gradient_mask_info를 구현해야 함).
            device (str): "cpu" 또는 "cuda".
            learning_rate (float): Adam 학습률.
            epochs (int): 태스크당 학습 에폭 수.
            num_tasks (int): 전체 태스크 수(임베딩을 미리 할당).
            hat_lambda (float): 용량 패널티에 대한 정규화 가중치.
            s_max (float): 최대 온도; 마스크가 얼마나 이진화되는지 제어.
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_tasks = num_tasks
        self.hat_lambda = hat_lambda
        self.s_max = s_max

        self.layer_sizes = model.get_hat_layer_sizes()
        self.num_mask_layers = len(self.layer_sizes)

        # task_embeddings[t][l]: shape가 (layer_sizes[l],)인 학습 가능한 Parameter
        # 0으로 초기화하여 sigmoid 결과가 0.5가 되도록 함 → 초기 attention은 균등
        self.task_embeddings = nn.ModuleList([ # Task 수만큼
            nn.ParameterList([
                nn.Parameter(torch.zeros(size)) # 레이어마다 학습 가능한 실수값
                for size in self.layer_sizes
            ])
            for _ in range(num_tasks)
        ]).to(device)

        # cumulative_mask[l]: 과거 태스크가 점유한 뉴런을 표시하는 이진 텐서
        self.cumulative_mask = [
            torch.zeros(size, device=device)
            for size in self.layer_sizes
        ]

        # current_task는 consolidate_task()에서 증가하며 train_task()가 읽는다.
        self.current_task = 0
        # eval_task_id는 evaluate() 호출 전 set_eval_task()로 설정한다.
        self.eval_task_id = 0

        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # 마스크 유틸리티
    # ------------------------------------------------------------------

    def _get_masks(self, task_id, s):
        """
        온도 s에서 task_id에 대한 soft attention mask를 계산한다.

        매개변수:
            task_id (int): 사용할 태스크의 임베딩 인덱스.
            s (float): Sigmoid 온도(클수록 출력이 더 이진값에 가까워짐).

        반환값:
            list[Tensor]: 마스킹되는 레이어별 마스크 하나씩, shape (layer_size,).
        
        학습 후반부로 갈수록 s가 커져 마스크가 0 또는 1에 가까워진다.
        """
        return [
            torch.sigmoid(s * self.task_embeddings[task_id][l])
            for l in range(self.num_mask_layers)
        ]

    def set_eval_task(self, task_id):
        """
        evaluate()에서 사용할 태스크의 hard mask를 선택한다.

        올바른 태스크 전용 마스크로 네트워크를 gate하도록, evaluate()를
        호출하기 전에 반드시 실행해야 한다.

        매개변수:
            task_id (int): 평가할 태스크 인덱스.
        """
        self.eval_task_id = task_id

    # ------------------------------------------------------------------
    # 손실 함수
    # ------------------------------------------------------------------

    def _hat_reg_loss(self, task_id, s):
        """
        HAT 용량 정규화 손실.

        누적 마스크에 이미 포함된 뉴런을 사용하는 것을 패널티로 준다.
        태스크 0에서는 누적 마스크가 모두 0이므로, 이는 단순한 희소성 패널티가
        되어 처음부터 작은 마스크를 유도한다.

        식: R = (1/N) * sum_l sum_i  m_t^l_i / (1 + c^l_i)

        매개변수:
            task_id (int): 현재 태스크.
            s (float): 현재 온도.

        반환값:
            torch.Tensor: 스칼라 정규화 손실.
        """
        masks = self._get_masks(task_id, s)
        reg = torch.tensor(0.0, device=self.device)
        total = 0
        for mask, c_mask in zip(masks, self.cumulative_mask):
            reg = reg + (mask / (1.0 + c_mask)).sum()
            total += mask.numel()
        return reg / max(total, 1)

    # ------------------------------------------------------------------
    # 그래디언트 마스킹
    # ------------------------------------------------------------------

    def _clip_embedding_grads(self, task_id):
        """
        과거 태스크가 이미 점유한 뉴런에 대한 임베딩 그래디언트를 0으로 만든다.

        c^l = 1인 뉴런은 새 태스크를 위해 이동되면 안 되므로, optimizer step
        전에 해당 그래디언트 성분을 0으로 만든다.
        """
        for l, c_mask in enumerate(self.cumulative_mask):
            emb = self.task_embeddings[task_id][l]
            if emb.grad is not None:
                emb.grad.data.mul_(1.0 - c_mask) # 이미 점유된(c_mask=1) 뉴런의 그래디언트를 0으로 만들어 이동 금지

    def _clip_weight_grads(self):
        """
        이전 태스크의 뉴런을 보호하도록 가중치와 편향 그래디언트를 마스킹한다.

        모델의 get_hat_gradient_mask_info()가 반환하는 레이어-마스크 매핑을
        사용해 어떤 그래디언트 항목을 0으로 만들어야 하는지 결정한다.

        규칙:
          Bias (1-D):
            - post-layer 누적 마스크가 1인 항목을 0으로 만든다.

          FC weight (2-D, out × in):
            - post-layer 누적 마스크가 1인 행을 0으로 만든다.
            - pre-layer 누적 마스크가 1인 열을 0으로 만든다.
              in_features가 pre-mask 크기보다 크면(flatten된 conv 출력),
              repeat_interleave로 확장해 크기를 맞춘다.

          Conv weight (4-D, out_ch × in_ch × kH × kW):
            - post-layer 마스크가 1인 출력 채널 슬라이스를 0으로 만든다.
            - pre-layer 마스크가 1인 입력 채널 슬라이스를 0으로 만든다.
        """
        mask_info = self.model.get_hat_gradient_mask_info()
        param_dict = dict(self.model.named_parameters())

        for param_name, pre_idx, post_idx in mask_info:
            if param_name not in param_dict:
                continue
            param = param_dict[param_name]
            if param.grad is None:
                continue

            grad = param.grad.data

            if grad.dim() == 1:
                # 편향 벡터 (out_features,)
                if post_idx >= 0:
                    grad.mul_(1.0 - self.cumulative_mask[post_idx])

            elif grad.dim() == 2:
                # FC 가중치 (out_features × in_features)
                if post_idx >= 0:
                    c_post = self.cumulative_mask[post_idx]          # (out,)
                    grad.mul_((1.0 - c_post).unsqueeze(1))
                if pre_idx >= 0:
                    c_pre = self.cumulative_mask[pre_idx]            # (in_orig,)
                    if c_pre.size(0) < grad.size(1):
                        # 공간 차원 flattening에 맞게 확장:
                        # 각 채널은 펼쳐진 벡터에서 (H × W)개의 연속 feature를 차지
                        repeat_factor = grad.size(1) // c_pre.size(0)
                        c_pre = c_pre.repeat_interleave(repeat_factor)
                    grad.mul_((1.0 - c_pre).unsqueeze(0))

            elif grad.dim() == 4:
                # Conv 가중치 (out_ch × in_ch × kH × kW)
                if post_idx >= 0:
                    c_post = self.cumulative_mask[post_idx]          # (out_ch,)
                    grad.mul_((1.0 - c_post).view(-1, 1, 1, 1))
                if pre_idx >= 0:
                    c_pre = self.cumulative_mask[pre_idx]            # (in_ch,)
                    grad.mul_((1.0 - c_pre).view(1, -1, 1, 1))

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------

    def train_task(self, train_loader, verbose=False):
        """
        HAT 마스킹을 사용해 단일 태스크를 학습한다.

        배치별 흐름:
          1. 온도 s를 계산한다(1/s_max → s_max로 선형 annealing).
          2. 현재 self.current_task에 대한 soft mask를 온도 s로 계산한다.
          3. model.forward_hat(x, masks)로 forward pass를 수행한다.
          4. 전체 손실 = CE loss + hat_lambda * 용량 정규화 손실.
          5. Backward pass.
          6. 점유된 뉴런에 대한 임베딩 그래디언트를 잘라낸다.
          7. 점유된 뉴런과 연결된 가중치 그래디언트를 잘라낸다.
          8. Optimizer step.

        각 태스크마다 새 Adam optimizer를 생성하며, 공유 모델 파라미터와
        현재 태스크의 임베딩 파라미터만 포함한다.

        매개변수:
            train_loader: 현재 태스크 학습 데이터용 DataLoader.
            verbose (bool): 에폭별 평균 손실 출력 여부.

        반환값:
            float: 마지막 에폭의 평균 손실.
        """
        task_id = self.current_task
        self.model.train()

        # Optimizer: 공유 모델 가중치 + 현재 태스크의 attention 임베딩만 포함
        params = (
            list(self.model.parameters())
            + list(self.task_embeddings[task_id].parameters())
        )
        optimizer = optim.Adam(params, lr=self.learning_rate)

        num_batches = len(train_loader)
        total_steps = max(self.epochs * num_batches - 1, 1)
        avg_loss = 0.0

        for epoch in range(self.epochs):
            total_loss = 0.0
            num_seen = 0

            for batch_idx, (x, y, _) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)

                # 선형 온도 annealing: step 0에서 1/s_max, 마지막 step에서 s_max
                step = epoch * num_batches + batch_idx
                progress = step / total_steps
                s = (1.0 / self.s_max) + (self.s_max - 1.0 / self.s_max) * progress

                optimizer.zero_grad()

                masks = self._get_masks(task_id, s)
                logits = self.model.forward_hat(x, masks)
                task_loss = self.criterion(logits, y)
                reg_loss = self._hat_reg_loss(task_id, s)
                loss = task_loss + self.hat_lambda * reg_loss

                loss.backward()

                self._clip_embedding_grads(task_id) # backward 후 임베딩 그래디언트 차단
                self._clip_weight_grads() # backward 후 가중치 그래디언트 차단

                optimizer.step()

                total_loss += loss.item()
                num_seen += 1

            avg_loss = total_loss / max(num_seen, 1)
            if verbose:
                print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

        return avg_loss

    # ------------------------------------------------------------------
    # 태스크 종료 후 통합
    # ------------------------------------------------------------------

    def consolidate_task(self, train_loader=None):
        """
        self.current_task 학습이 끝난 뒤 누적 마스크를 갱신한다.

        방금 학습한 태스크의 hard mask(s = s_max 사용)를 계산한 뒤,
        기존 누적 마스크와 원소별 최댓값을 취해 해당 뉴런을 영구적으로
        점유 상태로 표시한다.

        다음 train_task() 호출이 다음 태스크의 올바른 임베딩을 사용하도록
        self.current_task를 증가시킨다.

        매개변수:
            train_loader: EWC와의 인터페이스 일관성을 위해 받지만 사용하지 않음.
        """
        task_id = self.current_task
        with torch.no_grad():
            hard_masks = self._get_masks(task_id, self.s_max) # s_max로 거의 0/1(hard mask) 마스크 생성
            for l, (h_mask, c_mask) in enumerate(zip(hard_masks, self.cumulative_mask)):
                self.cumulative_mask[l] = torch.max(c_mask, (h_mask > 0.5).float())
        self.current_task += 1

    # ------------------------------------------------------------------
    # 평가
    # ------------------------------------------------------------------

    def evaluate(self, test_loader):
        """
        self.eval_task_id에 해당하는 hard mask를 사용해 모델 정확도를 평가한다.

        올바른 태스크 전용 마스크를 선택하려면 이 메서드 전에
        set_eval_task(task_id)를 호출해야 한다. 관련 없는 태스크의 뉴런을
        활성화하면 정확도가 떨어질 수 있으므로 올바른 마스크 선택이 중요하다.

        매개변수:
            test_loader: 테스트 데이터용 DataLoader.

        반환값:
            float: [0, 1] 범위의 분류 정확도.
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            masks = self._get_masks(self.eval_task_id, self.s_max) # 해당 마스크 꺼내기 

            for x, y, _ in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model.forward_hat(x, masks) # 마스크 씌워서 예측
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return correct / max(total, 1)
