"""
EWC with Separate Per-Task Output Heads
========================================
Shared head 방식에서는 모든 Task가 동일한 출력 뉴런 2개를 재사용하기 때문에
Task 간 출력층 충돌이 발생한다. 이 구현에서는 Task마다 별도의 classification
head (Linear layer)를 갖고, EWC는 공유 backbone만 보호한다.

표준 Task-Incremental Learning 설정:
  - Backbone (hidden layers): 모든 Task 공유, EWC로 보호
  - Task head: Task별 전용 Linear(feat_dim, num_classes), 학습 후 동결
  - 평가 시: 정답 Task의 head만 사용 (task_id를 알고 있는 TIL 설정)

EWC 역할 변화:
  - Shared head: 출력층 충돌 때문에 EWC가 새 Task 학습을 방해
  - Separate head: 출력층 충돌 없음, EWC가 backbone만 순수하게 보호
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class EWCSeparateHeadTrainer:
    """
    EWC trainer with per-task output heads.

    Key differences from EWCTrainer (shared head):
      - self.task_heads: nn.ModuleList of per-task Linear layers
      - Backbone forward stops before the model's original output layer
      - EWC Fisher and penalty computed on backbone parameters only
      - Previous task heads are excluded from the optimizer (frozen implicitly)
      - set_eval_task(task_id) must be called before evaluate()
    """

    def __init__(self, model, device="cpu", learning_rate=0.001, epochs=5,
                 ewc_lambda=5000, num_tasks=5):
        """
        Args:
            model     : MNIST_MLP (backbone + original output layer that we bypass)
            device    : 'cpu' or 'cuda'
            learning_rate: Adam lr
            epochs    : epochs per task
            ewc_lambda: EWC regularization strength
            num_tasks : total number of tasks (pre-allocate heads)
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.ewc_lambda = ewc_lambda
        self.num_tasks = num_tasks

        # Per-task heads: Task t gets its own Linear(feat_dim → num_classes)
        feat_dim = model.hidden_dims[-1]      # 256 for default MNIST_MLP
        num_classes = model.num_classes       # 2
        self.task_heads = nn.ModuleList([
            nn.Linear(feat_dim, num_classes) for _ in range(num_tasks)
        ]).to(device)

        self.criterion = nn.CrossEntropyLoss()
        self._eval_task_id = None

        # EWC storage: list of (fisher_dict, optimal_params) per consolidated task
        # fisher_dict  : param_name → Fisher diagonal tensor (backbone only)
        # optimal_params: param_name → anchor weights at task solution
        self.ewc_tasks = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _backbone_forward(self, x):
        """
        Forward through shared backbone only, bypassing the model's
        original output layer ('network.output').

        MNIST_MLP.network = Sequential(fc0, relu0, fc1, relu1, output)
        We run up to (but not including) 'output'.
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        for name, layer in self.model.network.named_children():
            if name == "output":
                break
            x = layer(x)
        return x

    def _forward(self, x, task_id):
        """Full forward: backbone → task-specific head."""
        return self.task_heads[task_id](self._backbone_forward(x))

    def _backbone_named_parameters(self):
        """
        Yield (name, param) for backbone parameters only.
        Excludes the original model output layer ('network.output.*')
        since we replaced it with per-task heads.
        """
        for name, param in self.model.named_parameters():
            if "network.output" not in name:
                yield name, param

    # ------------------------------------------------------------------
    # Public interface (compatible with TaskIncrementalLearner in train.py)
    # ------------------------------------------------------------------

    def set_eval_task(self, task_id):
        """Specify which task head to use during evaluate()."""
        self._eval_task_id = task_id

    def compute_fisher(self, data_loader, task_id):
        """
        Compute diagonal Fisher Information for backbone parameters
        using the specified task's head and data.

        Sampled Fisher (not empirical):
          - sample y ~ p(y|x, θ) instead of using the true label
          - prevents Fisher from collapsing to zero when the model is
            very confident (which happens on easy tasks like MNIST)
          - True Fisher = E_{y~model}[(∂log p/∂θ)²]
            vs Empirical Fisher = E_{y~data}[(∂log p/∂θ)²]

        Args:
            data_loader: DataLoader for the task (yields x, y, task_id)
            task_id    : which task head to use for gradient computation

        Returns:
            dict: param_name → Fisher diagonal tensor
        """
        fisher = {
            name: torch.zeros_like(param.data)
            for name, param in self._backbone_named_parameters()
        }

        self.model.eval()
        self.task_heads[task_id].eval()
        num_samples = 0

        for x, y, _ in data_loader:
            x = x.to(self.device)

            for i in range(x.size(0)):
                self.model.zero_grad()
                self.task_heads[task_id].zero_grad()

                logit = self._forward(x[i:i+1], task_id)

                # Sample from model's output distribution
                prob = F.softmax(logit, dim=1)
                sampled_y = torch.multinomial(prob[0], num_samples=1).squeeze()
                log_prob = F.log_softmax(logit, dim=1)[0, sampled_y]
                log_prob.backward()

                for name, param in self._backbone_named_parameters():
                    if param.grad is not None:
                        fisher[name] += param.grad.data ** 2

            num_samples += x.size(0)

        for name in fisher:
            fisher[name] /= max(num_samples, 1)

        return fisher

    def consolidate_task(self, data_loader):
        """
        Compute Fisher and save anchor weights after training on a task.
        Task id is inferred from the first batch of the loader.

        Called by TaskIncrementalLearner after each task's training.
        """
        _, _, task_id_t = next(iter(data_loader))
        task_id = int(task_id_t[0].item())

        new_fisher = self.compute_fisher(data_loader, task_id)

        optimal_params = {
            name: param.data.clone().detach()
            for name, param in self._backbone_named_parameters()
        }

        self.ewc_tasks.append((new_fisher, optimal_params))

    def ewc_loss(self):
        """
        EWC penalty over all previous tasks.

        Only backbone parameters are penalized — task heads are frozen
        by being excluded from the optimizer, so they never change.

        L_ewc = (λ/2) Σ_t Σ_i F^t_i (θ_i - θ*_i^t)²
        """
        loss = 0.0
        for fisher_dict, optimal_params in self.ewc_tasks:
            for name, param in self._backbone_named_parameters():
                if name in fisher_dict:
                    diff = param - optimal_params[name]
                    loss += (fisher_dict[name] * diff ** 2).sum()
        return (self.ewc_lambda / 2) * loss

    def train_task(self, train_loader, verbose=False):
        """
        Train on a single task.
        Task id is inferred from the first batch of the loader.

        Optimizer includes backbone + current task head only.
        Previous task heads are not in the optimizer → implicitly frozen.

        Args:
            train_loader: DataLoader for current task
            verbose     : print per-epoch loss

        Returns:
            float: final epoch average loss
        """
        _, _, task_id_t = next(iter(train_loader))
        task_id = int(task_id_t[0].item())

        self.model.train()
        self.task_heads[task_id].train()

        # Only backbone + current head are updated
        backbone_params = [p for _, p in self._backbone_named_parameters()]
        head_params = list(self.task_heads[task_id].parameters())
        optimizer = optim.Adam(backbone_params + head_params, lr=self.learning_rate)

        avg_loss = 0.0
        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0

            for x, y, _ in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()

                logits = self._forward(x, task_id)
                task_loss = self.criterion(logits, y)

                if self.ewc_tasks:
                    ewc_reg = self.ewc_loss()
                    total_task_loss = task_loss + ewc_reg
                else:
                    ewc_reg = torch.tensor(0.0)
                    total_task_loss = task_loss

                total_task_loss.backward()
                optimizer.step()

                total_loss += total_task_loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)

            if verbose:
                print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f} "
                      f"(task={task_loss.item():.4f}, "
                      f"ewc={ewc_reg.item() if self.ewc_tasks else 0.0:.4f})")

        return avg_loss

    def evaluate(self, test_loader):
        """
        Evaluate on a task. set_eval_task(task_id) must be called first.

        Args:
            test_loader: DataLoader for the task to evaluate

        Returns:
            float: accuracy (0–1)
        """
        assert self._eval_task_id is not None, \
            "Call set_eval_task(task_id) before evaluate()"
        task_id = self._eval_task_id

        self.model.eval()
        self.task_heads[task_id].eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y, _ in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self._forward(x, task_id)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return correct / max(total, 1)
