"""
Gradient Episodic Memory (GEM) 학습기
====================================
이전에 저장된 에피소드 메모리에서의 손실이 증가하지 않도록 그래디언트
업데이트를 제한하여 파국적 망각을 방지한다.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import quadprog
    _QUADPROG_AVAILABLE = True
except ImportError:
    _QUADPROG_AVAILABLE = False


class EpisodicMemory:
    """
    GEM을 위한 태스크별 에피소드 메모리 버퍼.

    완료된 각 태스크마다 분리된 메모리 뱅크를 유지하며, 무작위로 샘플링한
    학습 예제를 최대 n_memories개까지 저장한다. DER++의 평평한 reservoir
    버퍼와 달리, GEM은 태스크별 그래디언트를 계산해 제약식에 사용해야 하므로
    태스크가 분리된 메모리가 필요하다.

    개념적 역할:
      - 태스크 t가 끝난 뒤 add_task(t, loader)를 호출해 M_t를 채운다
      - 태스크 t+k를 학습하는 동안 저장된 모든 메모리를 순회하며 현재
        업데이트가 위반해서는 안 되는 기준 그래디언트를 얻는다
    """

    def __init__(self, n_memories=256):
        """
        매개변수:
            n_memories (int): 태스크별로 유지할 최대 샘플 수
        """
        self.n_memories = n_memories # 태스크 별로 이미지와 레이블을 최대 n_memories개까지 저장하는 버퍼 크기
        self.memory = {}  # {task_id: (x_tensor, y_tensor)}  — CPU에 저장

    def add_task(self, task_id, data_loader):
        """
        완료된 태스크에서 최대 n_memories개의 샘플을 수집해 저장한다.

        먼저 사용 가능한 모든 배치를 이어 붙인 뒤, 중복 없이 무작위 부분집합
        n_memories개를 뽑아 M_{task_id}를 만든다.

        매개변수:
            task_id (int): 완료된 태스크의 식별자
            data_loader: 완료된 태스크의 학습 데이터용 DataLoader
        """
        all_x, all_y = [], []

        for x, y, _ in data_loader:
            all_x.append(x.cpu())
            all_y.append(y.cpu())

        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)

        n = all_x.size(0)
        mem_size = min(self.n_memories, n)
        indices = torch.randperm(n)[:mem_size] # 무작위로 mem_size개 인덱스 선택 (256개)

        self.memory[task_id] = (all_x[indices], all_y[indices]) # 저장

    def num_tasks(self):
        """현재 메모리에 저장된 태스크 수를 반환한다."""
        return len(self.memory)


class GEMTrainer:
    """
    GEM (Gradient Episodic Memory) 학습기.

    개념적 접근:
      - 각 태스크를 학습한 뒤 M개의 샘플로 이루어진 작은 에피소드 메모리를 저장한다
      - 새로운 태스크 t를 학습할 때 다음 그래디언트 제약을 강제한다:
          ⟨g̃, g_k⟩ ≥ 0  for all previous tasks k
        여기서 g̃는 (투영된) 업데이트 그래디언트이고, g_k는 태스크 k의
        에피소드 메모리에서 계산된 그래디언트다
      - 어떤 k에서든 제약을 위반하면, 이차 계획법(QP)으로 가장 가까운
        feasible 그래디언트 g̃를 찾아 대신 적용한다
      - 구조 변경 없이(예: HAT과 달리) fine-tuning보다 망각을 크게 줄이는
        효과를 기대할 수 있다

    그래디언트 투영(QP 정식화):

      Primal:
        min_{g̃}  (1/2) ‖g̃ − g‖²
        s.t.      G g̃ ≥ −margin      (G: 메모리 그래디언트 행렬, 각 행 = g_k)

      Dual (quadprog로 풂):
        min_{v ≥ 0}  (1/2) v^T (G G^T) v  +  (G g + margin)^T v
        g̃ = g + G^T v*

    quadprog를 사용할 수 없으면, 학습기는 순차적인 Gram-Schmidt 투영으로
    대체한다. 이는 QP 최적해는 아니지만 내적 부호 제약은 유지하며 추가
    의존성이 필요 없다.

    참고 문헌:
      Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning",
      NeurIPS 2017.
    """

    def __init__(self, model, device="cpu", learning_rate=0.001, epochs=5,
                 n_memories=256, margin=0.0):
        """
        매개변수:
            model: 신경망 모델
            device (str): 학습에 사용할 디바이스
            learning_rate (float): 옵티마이저 학습률
            epochs (int): 태스크당 에폭 수
            n_memories (int): 에피소드 메모리에 태스크별로 저장할 샘플 수
            margin (float): 제약 여유값 ε; 제약식은 G g̃ ≥ −margin.
                            GEM 논문은 0을 사용하며, 작은 양수는 여유를 준다.
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n_memories = n_memories
        self.margin = margin

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 태스크별 에피소드 메모리 뱅크
        self.episodic_memory = EpisodicMemory(n_memories=n_memories)

        # 메모리가 저장된 태스크 수
        self.n_tasks_seen = 0

    # ------------------------------------------------------------------
    # 그래디언트 벡터 유틸리티
    # ------------------------------------------------------------------

    def _get_gradient_vector(self):
        """
        그래디언트는 파라미터마다 .grad 버퍼에 저장된다. 
        모든 파라미터의 그래디언트를 하나의 1차원 numpy 배열로 펼친다.
        그래디언트가 없는 파라미터는 0 블록으로 채운다.

        반환값:
            np.ndarray: 이어 붙인 그래디언트 벡터, shape (num_params,)
        """
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.view(-1).cpu())
            else:
                grads.append(torch.zeros(p.numel()))
        return torch.cat(grads).numpy()

    def _set_gradient_vector(self, grad_vec):
        """
        평평한 numpy 그래디언트 배열을 모델의 .grad 버퍼에 다시 써 넣는다.

        매개변수:
            grad_vec (np.ndarray): 길이가 num_params인 평탄화 그래디언트 벡터
        """
        grad_tensor = torch.from_numpy(grad_vec).float().to(self.device)
        offset = 0
        for p in self.model.parameters():
            n = p.numel()
            if p.grad is not None:
                p.grad.data.copy_(
                    grad_tensor[offset:offset + n].view_as(p.grad.data)
                )
            offset += n

    # ------------------------------------------------------------------
    # 메모리 그래디언트 계산
    # ------------------------------------------------------------------

    def _compute_memory_gradients(self):
        """
        이전 각 태스크의 에피소드 메모리를 사용해 태스크별 그래디언트 벡터를 계산한다.

        각 메모리 그래디언트 g_k = ∇_θ L(M_k; θ)는 태스크 k에 저장된 샘플 전체에
        대해 forward + backward를 한 번 수행해 얻는다. 반환 시점의 모델
        그래디언트 버퍼에는 마지막 태스크의 메모리 그래디언트가 남아 있으며,
        호출자는 _set_gradient_vector로 이를 덮어쓴다고 가정한다.

        반환값:
            np.ndarray or None: shape가 (n_prev_tasks, num_params)인 메모리
                                그래디언트 행렬 G. 아직 이전 태스크 메모리가
                                없으면 None.
        """
        if self.episodic_memory.num_tasks() == 0:
            return None

        memory_grads = []

        for task_id in sorted(self.episodic_memory.memory.keys()):
            x_mem, y_mem = self.episodic_memory.memory[task_id]
            x_mem = x_mem.to(self.device)
            y_mem = y_mem.to(self.device)

            self.model.zero_grad()
            logits = self.model(x_mem)
            loss = self.criterion(logits, y_mem)
            loss.backward()

            memory_grads.append(self._get_gradient_vector())

        return np.stack(memory_grads, axis=0)  # (n_prev_tasks, num_params)

    # ------------------------------------------------------------------
    # QP 기반 투영
    # ------------------------------------------------------------------

    def _qp_project(self, current_grad, memory_grads):
        """
        current_grad를 QP(quadprog)를 통해 feasible cone으로 투영한다.

        다음 dual 문제를 푼다:
          min_{v ≥ 0}  (1/2) v^T (G G^T) v  +  (G g + margin)^T v
          g̃ = g + G^T v*

        quadprog를 사용할 수 없거나 솔버가 수치적으로 실패하면 Gram-Schmidt
        방식으로 대체한다.

        매개변수:
            current_grad (np.ndarray): 현재 태스크 그래디언트 g, shape (d,)
            memory_grads (np.ndarray): 메모리 그래디언트 행렬 G, shape (K, d)

        반환값:
            np.ndarray: 투영된 그래디언트 g̃, shape (d,), dtype float32
        """
        if _QUADPROG_AVAILABLE:
            return self._qp_project_quadprog(current_grad, memory_grads)
        return self._gram_schmidt_project(current_grad, memory_grads)

    def _qp_project_quadprog(self, current_grad, memory_grads):
        """
        quadprog 라이브러리를 사용한 QP 투영.

        quadprog API: solve_qp(Q, a, C, b)는 다음을 푼다
          min (1/2) x^T Q x − a^T x   s.t. C^T x ≥ b

        dual 변수 v에 대한 대응:
          Q = G G^T  (메모리 그래디언트의 Gram 행렬, 안정화를 위해 정규화)
          a = −(G g + margin·1)
          C = I  (v ≥ 0 제약)
          b = 0

        매개변수:
            current_grad (np.ndarray): shape (d,)
            memory_grads (np.ndarray): shape (K, d)

        반환값:
            np.ndarray: 투영된 그래디언트, shape (d,), dtype float32
        """
        n_tasks = memory_grads.shape[0]

        # Gram 행렬(dual 계수 행렬), 안정성을 위해 정규화
        GGT = (memory_grads @ memory_grads.T).astype(np.float64)
        GGT += 1e-6 * np.eye(n_tasks)

        # 선형 항: G g + margin·1  (제약 여유값 반영)
        Gg = (memory_grads @ current_grad + self.margin).astype(np.float64)

        # dual 변수의 비음수 제약: I v ≥ 0
        C = np.eye(n_tasks, dtype=np.float64)
        b = np.zeros(n_tasks, dtype=np.float64)

        try:
            v = quadprog.solve_qp(GGT, -Gg, C, b)[0]
            projected = current_grad + (memory_grads.T @ v).astype(np.float32)
        except Exception:
            # 수치적 실패 시 Gram-Schmidt 방식으로 안전하게 대체
            projected = self._gram_schmidt_project(current_grad, memory_grads)

        return projected.astype(np.float32)

    def _gram_schmidt_project(self, current_grad, memory_grads):
        """
        순차적 Gram-Schmidt 그래디언트 투영(quadprog 없는 대체 경로).

        위반된 각 제약에 대해 메모리 그래디언트와 반대 방향을 가리키는
        그래디언트 성분을 제거한다:
          if ⟨g, g_k⟩ < 0:  g ← g − (⟨g, g_k⟩ / ‖g_k‖²) g_k

        QP 최적해는 아니지만 모든 k에 대해 ⟨g̃, g_k⟩ ≥ 0을 보장하며,
        외부 QP 솔버가 필요 없다. 수렴할 때까지 여러 번 반복한다.

        매개변수:
            current_grad (np.ndarray): shape (d,)
            memory_grads (np.ndarray): shape (K, d)

        반환값:
            np.ndarray: 투영된 그래디언트, shape (d,), dtype float32
        """
        g = current_grad.copy().astype(np.float64)

        for _ in range(memory_grads.shape[0]):
            dots = memory_grads @ g  # (K,) 각 메모리 그래디언트와의 내적 계산
            violated = np.where(dots < -self.margin)[0] # 위반된 task 인덱스 찾기
            if len(violated) == 0:
                break
            for k in violated:
                g_k = memory_grads[k].astype(np.float64)
                norm_sq = np.dot(g_k, g_k)
                if norm_sq > 1e-12:
                    g -= (np.dot(g, g_k) / norm_sq) * g_k # 반대 방향 제거

        return g.astype(np.float32)

    # ------------------------------------------------------------------
    # 그래디언트 투영 진입점
    # ------------------------------------------------------------------

    def _project_gradients(self):
        """
        GEM 제약을 확인하고, 위반 시 현재 그래디언트를 투영한다.

        단계:
          1. 현재 그래디언트 g를 저장한다(방금 끝난 backward pass 결과)
          2. 이전 각 태스크 k에 대한 메모리 그래디언트 G = {g_k}를 계산한다
             (이 과정에서 model.grad 버퍼가 덮어써지며, 4단계에서 처리한다)
          3. 모든 k에 대해 ⟨g, g_k⟩ ≥ −margin인지 확인한다
          4. 제약을 하나라도 위반하면 QP/GS 투영기를 호출해 g̃를 model.grad에
             다시 쓰고, 아니면 원래 g를 그대로 복원한다
        """
        # 1단계: 덮어써지기 전에 현재 그래디언트를 보관
        current_grad = self._get_gradient_vector()

        # 2단계: 메모리 그래디언트 계산(model.grad를 덮어씀)
        memory_grads = self._compute_memory_gradients()

        if memory_grads is None:
            # 아직 이전 태스크 메모리가 없으므로 제약할 것이 없음
            self._set_gradient_vector(current_grad)
            return

        # 3단계: 제약 확인 (내적 계산 -> 방향 확인)
        dots = memory_grads @ current_grad  # (n_prev_tasks,)

        if (dots + self.margin >= 0).all():
            # 모든 제약을 만족하므로 원래 그래디언트를 그대로 복원
            # 모두 같은 방향이면 그냥 진행
            self._set_gradient_vector(current_grad)
            return

        # 4단계: 제약 위반 발생 시 투영 후 다시 기록
        projected_grad = self._qp_project(current_grad, memory_grads)
        self._set_gradient_vector(projected_grad)

    # ------------------------------------------------------------------
    # 핵심 학습기 인터페이스
    # ------------------------------------------------------------------

    def train_task(self, train_loader, verbose=False):
        """
        GEM 그래디언트 제약을 적용해 단일 태스크를 학습한다.

        태스크 0에서는(아직 에피소드 메모리가 없으므로) 업데이트에 제약이 없다.
        태스크 1부터는 이전에 저장된 모든 태스크 메모리 k에 대해
        ⟨g̃, g_k⟩ ≥ 0을 만족하도록 각 그래디언트를 투영한다.

        매개변수:
            train_loader: 현재 태스크 학습 데이터용 DataLoader
            verbose (bool): 에폭별 학습 진행 상황 출력 여부

        반환값:
            float: 마지막 에폭의 평균 손실
        """
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0

            for x, y, _ in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward() # 그래디언트 계산 후,

                # 이전 태스크 메모리가 있을 때 GEM 투영 적용
                if self.n_tasks_seen > 0:
                    self._project_gradients() # 업데이트 직전에 그래디언트 방향 검사, 수정 (투영)

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)

            if verbose:
                print(
                    f"  Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f} "
                    f"(Tasks in memory: {self.episodic_memory.num_tasks()})"
                )

        return avg_loss

    def consolidate_task(self, train_loader):
        """
        방금 완료한 태스크의 에피소드 메모리를 저장한다.

        각 태스크 학습이 끝난 뒤 호출된다. 현재 태스크의 학습 데이터에서
        n_memories개를 무작위 부분집합으로 저장하고, 다음 태스크 학습에 제약이
        걸리도록 태스크 카운터를 증가시킨다.

        매개변수:
            train_loader: 완료된 태스크의 학습 데이터용 DataLoader
        """
        self.episodic_memory.add_task(self.n_tasks_seen, train_loader) # 데이터 저장
        self.n_tasks_seen += 1 

    def evaluate(self, test_loader):
        """
        테스트 데이터에서 모델 정확도를 평가한다.

        매개변수:
            test_loader: 테스트 데이터용 DataLoader

        반환값:
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
