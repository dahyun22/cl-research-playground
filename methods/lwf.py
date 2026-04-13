"""
Learning without Forgetting (LwF) Trainer
==========================================
Retains knowledge of previous tasks via knowledge distillation on new task
data — no episodic memory or previous task data is stored.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy


class LwFTrainer:
    """
    LwF (Learning without Forgetting) trainer.

    Conceptual approach:
      - Before training on a new task, snapshot the current model as a frozen
        reference (old_model).  This captures all knowledge learned so far.
      - During training, minimise two losses simultaneously:
          1. Cross-entropy on hard labels  → learn the new task
          2. KL-divergence between student and old model's softened outputs
             → prevent forgetting of previous tasks
      - No data from previous tasks is ever stored (memory-free).
        The only overhead over fine-tuning is one extra frozen forward pass
        per batch and retaining a copy of the previous model's weights.
      - In a Task-Incremental Learning setting, the model shares a single
        backbone across all tasks while the output head maps to per-task
        binary class scores (0 vs 1).  The distillation loss is applied on
        this shared head, preventing the backbone from drifting away from
        representations useful for old tasks.

    Key equations:
      L_task     = CE(f(x; θ), y)
      L_distill  = T² · KL( softmax(f(x; θ)/T)  ‖  softmax(f(x; θ_old)/T) )
      L_total    = L_task + λ · L_distill

      where:
        θ_old = model weights frozen before training on current task
        T     = temperature  (higher → softer distribution → richer signal)
        λ     = distillation weight

    Reference:
      Li & Hoiem, "Learning without Forgetting", ECCV 2016 / TPAMI 2018.
    """

    def __init__(self, model, device="cpu", learning_rate=0.001, epochs=5,
                 lwf_lambda=1.0, temperature=2.0):
        """
        Args:
            model: Neural network model
            device (str): Device for training
            learning_rate (float): Learning rate for optimizer
            epochs (int): Number of epochs per task
            lwf_lambda (float): Weight for knowledge distillation loss (λ)
            temperature (float): Softmax temperature for distillation (T ≥ 1).
                                 Higher values produce softer targets and expose
                                 more inter-class relationship information.
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lwf_lambda = lwf_lambda
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Frozen reference model from the previous task.
        # None until record_soft_labels() is called for the first time.
        self.old_model = None

    def record_soft_labels(self, train_loader):
        """
        Snapshot the current model before training on a new task.

        In the original LwF paper, "responses" of the old network on new task
        data are pre-computed and stored as fixed distillation targets.  This
        implementation achieves the same effect by freezing the current model
        as old_model and computing soft targets on-the-fly during training.
        The two approaches are mathematically equivalent because old_model
        never changes within a task.

        Must be called in train.py immediately before train_task() for each
        task so that old_model reflects weights trained on all prior tasks
        but NOT yet on the current one.

        Args:
            train_loader: DataLoader for the upcoming task's training data.
                          Accepted for API symmetry and potential pre-caching
                          extensions; not consumed in the base implementation.
        """
        self.old_model = deepcopy(self.model)
        self.old_model.eval()
        for param in self.old_model.parameters():
            param.requires_grad_(False)

    def distillation_loss(self, student_logits, x):
        """
        Knowledge distillation loss between current model and old model.

        Temperature T controls the softness of both distributions:
          - T = 1  : standard softmax, peaks sharply at the argmax class
          - T > 1  : softer distribution, reveals the model's confidence
                     ordering over all classes (dark knowledge)

        Multiplying by T² compensates for the magnitude reduction in gradients
        caused by dividing logits by T before the softmax, keeping the
        distillation term at a comparable scale to the cross-entropy term.
        (Hinton et al., "Distilling the Knowledge in a Neural Network", 2015)

        Args:
            student_logits (torch.Tensor): Current model's logits, shape (B, C)
            x (torch.Tensor): Input batch on device, used to query old_model

        Returns:
            torch.Tensor: Scalar KL divergence distillation loss, scaled by T²
        """
        T = self.temperature

        with torch.no_grad():
            teacher_logits = self.old_model(x)

        student_log_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)

        # KL(teacher ‖ student); batchmean normalises by batch size
        kl = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean")
        return kl * (T ** 2)

    def train_task(self, train_loader, verbose=False):
        """
        Train model on a single task with LwF distillation.

        On the first task (old_model is None) only cross-entropy is used.
        From the second task onward, the total loss combines cross-entropy
        on new-task hard labels with knowledge distillation from the frozen
        old model, preventing catastrophic forgetting without storing any data.

        Args:
            train_loader: DataLoader for current task training data
            verbose (bool): Print per-epoch training progress

        Returns:
            float: Average loss over the final epoch
        """
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0

            for x, y, _ in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(x)
                ce_loss = self.criterion(logits, y)

                if self.old_model is not None:
                    kd_loss = self.distillation_loss(logits, x)
                    loss = ce_loss + self.lwf_lambda * kd_loss
                else:
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
        Evaluate model accuracy on test data.

        Args:
            test_loader: DataLoader for test data

        Returns:
            float: Accuracy (0-1)
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
