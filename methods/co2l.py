"""
Co2L Trainer: Contrastive Continual Learning
=============================================
Combines supervised contrastive loss with asymmetric distillation to resist
catastrophic forgetting without a replay buffer.

Loss (per step):
    Task 1       : L_CE + L_SupCon
    Task t (t>1) : L_CE + L_SupCon + λ * L_AsymDistill

References:
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
    Co2L (Contrastive Continual Learning) trainer.

    Conceptual approach:
      - SupConLoss clusters same-class projections together in embedding space,
        building representations that generalise across tasks.
      - AsymDistillLoss forces the current model's projections to stay close to
        the previous model's projections for the same samples, preserving old
        task knowledge without storing any replay data.
      - CE loss keeps the classifier head properly trained for evaluation.
      - The previous model is frozen via deepcopy at the end of each task
        (consolidate_task), mirroring EWC's consolidate pattern.

    Key equations:
      L = L_CE + L_SupCon + λ * L_AsymDistill   (task t > 1)
      L = L_CE + L_SupCon                        (task 1, no previous model)

      L_CE         = CrossEntropy(classifier(features), y)
      L_SupCon     = SupConLoss(projections, y)
      L_AsymDistill= AsymDistillLoss(proj_cur, proj_prev)
    """

    def __init__(self, model, device="cpu", learning_rate=0.001, epochs=5,
                 distill_lambda=1.0, temperature=0.1):
        """
        Args:
            model          : MLP_Co2L instance
            device (str)   : "cpu" or "cuda"
            learning_rate  : Adam learning rate
            epochs         : Epochs per task
            distill_lambda : Weight λ for AsymDistillLoss
            temperature    : Shared τ for both contrastive losses
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

        # Set by consolidate_task() after each task; None on the first task.
        self.prev_model = None

    # ──────────────────────────────────────────────────────────────────────────
    # Core training
    # ──────────────────────────────────────────────────────────────────────────

    def train_task(self, train_loader, verbose=False):
        """
        Train model on a single task.

        Step-by-step per batch:
          1. Forward pass  → features (B, 256), projections (B, 128)
          2. CE loss       on classifier(features) vs labels
          3. SupCon loss   on projections vs labels
          4. Distill loss  on projections vs prev_model projections  [task t>1]
          5. Backward + Adam step

        Args:
            train_loader : DataLoader yielding (x, y, _) triples
            verbose      : Print per-epoch loss

        Returns:
            float: Average loss of the final epoch
        """
        self.model.train()
        avg_loss = 0.0

        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0

            for x, y, _ in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                # ── Forward ────────────────────────────────────────────────
                features, projections = self.model(x)
                logits = self.model.classifier(features)

                # ── Losses ─────────────────────────────────────────────────
                ce     = self.criterion(logits, y)
                supcon = self.supcon_loss(projections, y)

                if self.prev_model is not None:
                    with torch.no_grad():
                        _, proj_prev = self.prev_model(x)
                    distill = self.distill_loss(projections, proj_prev)
                    loss = ce + supcon + self.distill_lambda * distill
                else:
                    loss = ce + supcon

                # ── Backward ───────────────────────────────────────────────
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)

            if verbose:
                print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

        return avg_loss

    # ──────────────────────────────────────────────────────────────────────────
    # Post-task consolidation
    # ──────────────────────────────────────────────────────────────────────────

    def consolidate_task(self):
        """
        Freeze a snapshot of the current model for use as the distillation
        reference during the next task.

        Called AFTER train_task() and evaluate() for each task, mirroring
        EWCTrainer.consolidate_task().  The frozen model is never updated again:
        all parameters have requires_grad=False and the model stays in eval mode.
        """
        self.prev_model = deepcopy(self.model)
        self.prev_model.eval()
        for p in self.prev_model.parameters():
            p.requires_grad_(False)

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(self, test_loader):
        """
        Evaluate classifier accuracy on a held-out test set.

        Uses backbone → classifier (not the projection head), identical to
        how all other trainers evaluate.

        Args:
            test_loader: DataLoader yielding (x, y, _) triples

        Returns:
            float: Accuracy in [0, 1]
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
