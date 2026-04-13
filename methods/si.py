"""
Synaptic Intelligence (SI) Trainer
====================================
Protects important parameters for previous tasks by accumulating per-parameter
importance online during training — no data pass required after task completion.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class SITrainer:
    """
    SI (Synaptic Intelligence) trainer.

    Conceptual approach:
      - EWC measures parameter importance POST-HOC (after training, using data).
        SI instead accumulates importance ONLINE during training by tracking
        how much each parameter contributed to reducing the task loss.
      - At each gradient step, the contribution of parameter θ_k is approximated
        by −g_k · Δθ_k (how much the loss decreased due to this parameter's move).
      - After a task, the accumulated contribution ω_k is normalised by the
        total parameter displacement squared to give per-parameter importance Ω_k.
      - These importances accumulate across tasks and regularise future updates
        in exactly the same way as EWC — but with zero extra data cost.

    Key equations:

      Online accumulation (each gradient step):
        ω_k  +=  −g_k · (θ_k_new − θ_k_old)

        where:
          g_k      = ∂L/∂θ_k  (gradient at this step)
          θ_k_old  = parameter value before optimizer.step()
          θ_k_new  = parameter value after  optimizer.step()

      After task τ (consolidation):
        Ω_k  +=  clamp( ω_k / (Δθ_k² + ξ),  min=0 )

        where:
          Δθ_k = θ_k(end of τ) − θ_k(start of τ)   (total displacement)
          ξ    = damping constant  (prevents division by zero)

      Regularisation loss (identical structure to EWC):
        L_SI = (λ/2) · Σ_k  Ω_k · (θ_k − θ_k*)²

        where θ_k* = optimal parameters at the end of the last task.

    Reference:
      Zenke, Poole & Ganguli, "Continual Learning Through Synaptic Intelligence",
      ICML 2017.
    """

    def __init__(self, model, device="cpu", learning_rate=0.001, epochs=5,
                 si_lambda=1.0, xi=0.1):
        """
        Args:
            model: Neural network model
            device (str): Device for training
            learning_rate (float): Learning rate for optimizer
            epochs (int): Number of epochs per task
            si_lambda (float): Regularisation strength (analogous to ewc_lambda)
            xi (float): Damping constant ξ — prevents importance blow-up when
                        a parameter barely moved during a task (default 0.1)
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.si_lambda = si_lambda
        self.xi = xi

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Accumulated importance Ω across all completed tasks
        # {param_name: tensor}  — grows additively after each task
        self.importance_dict = {}

        # Optimal parameter values θ* at the end of the last task
        self.optimal_params = {}

        # ── Online tracking state (reset at the start of each task) ──────────
        # ω: running integral of −g · dθ within the current task
        self._omega_running = {
            name: torch.zeros_like(param.data)
            for name, param in self.model.named_parameters()
        }
        # θ_prev: parameter snapshot from the previous step (for Δθ per step)
        self._theta_prev = {
            name: param.data.clone().detach()
            for name, param in self.model.named_parameters()
        }
        # θ_start: parameter snapshot from the very start of the current task
        # (used in consolidate_task to compute total displacement Δθ)
        self._theta_start = {
            name: param.data.clone().detach()
            for name, param in self.model.named_parameters()
        }

    # ------------------------------------------------------------------
    # Online ω accumulation
    # ------------------------------------------------------------------

    def _update_omega(self, saved_grads):
        """
        Update the running importance estimate ω after each optimizer step.

        Approximates the path integral ω_k ≈ −∫ g_k dθ_k along the parameter
        trajectory by summing −g_k · Δθ_k over discrete steps.

        Must be called AFTER optimizer.step() so that the new parameter values
        are available.  Gradients must be saved BEFORE optimizer.step() because
        zero_grad() at the next iteration would clear them.

        Args:
            saved_grads (dict): {param_name: gradient tensor} captured just
                                before the optimizer step for this iteration.
        """
        for name, param in self.model.named_parameters():
            if name not in saved_grads:
                continue
            # Δθ_step = change in this single optimizer step
            delta_theta = param.data - self._theta_prev[name]
            # Contribution to loss decrease: −g · Δθ  (positive = helped)
            self._omega_running[name] -= saved_grads[name] * delta_theta
            # Advance the per-step snapshot
            self._theta_prev[name] = param.data.clone().detach()

    # ------------------------------------------------------------------
    # SI regularisation loss
    # ------------------------------------------------------------------

    def si_loss(self):
        """
        Compute SI regularisation penalty.

        Structurally identical to ewc_loss(): a weighted squared deviation
        from the previous task's optimal parameters, where the weights are
        the accumulated synaptic importances Ω rather than Fisher diagonals.

        Returns:
            torch.Tensor: Scalar SI regularisation loss
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
    # Core trainer interface
    # ------------------------------------------------------------------

    def train_task(self, train_loader, verbose=False):
        """
        Train model on a single task with SI online importance tracking.

        At every gradient step:
          1. Forward + backward to obtain gradients
          2. Save gradients (needed after the optimizer step)
          3. optimizer.step()
          4. Update ω with −g · Δθ

        The SI regularisation term is added to the task loss from the second
        task onward (once importance_dict is populated by consolidate_task).

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
                task_loss = self.criterion(logits, y)

                # Add SI regularisation if previous tasks have been consolidated
                if self.importance_dict:
                    si_reg = self.si_loss()
                    total_task_loss = task_loss + si_reg
                else:
                    total_task_loss = task_loss

                total_task_loss.backward()

                # Capture gradients before the optimizer modifies parameters
                saved_grads = {
                    name: param.grad.data.clone()
                    for name, param in self.model.named_parameters()
                    if param.grad is not None
                }

                self.optimizer.step()

                # Update online ω estimate with this step's contribution
                self._update_omega(saved_grads)

                total_loss += total_task_loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)

            if verbose:
                print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

        return avg_loss

    def consolidate_task(self, train_loader=None):
        """
        Convert accumulated ω into task importance Ω and update θ*.

        Called once after finishing training on each task.  Unlike EWC, no
        data pass is needed — all information was gathered online during
        train_task().

        Steps:
          1. Compute per-parameter displacement Δθ = θ_current − θ_start
          2. Importance for this task: clamp(ω / (Δθ² + ξ), min=0)
          3. Add to running totals in importance_dict
          4. Save current parameters as new optimal θ*
          5. Reset ω, θ_prev, θ_start for the next task

        Args:
            train_loader: Accepted for API consistency with other trainers;
                          not used (SI requires no post-task data pass).
        """
        for name, param in self.model.named_parameters():
            # Total parameter displacement over the completed task
            delta = param.data - self._theta_start[name]

            # Per-task importance: normalise ω by displacement magnitude
            # clamp to ≥ 0: negative values would act as anti-regularisation
            task_importance = torch.clamp(
                self._omega_running[name] / (delta ** 2 + self.xi),
                min=0.0
            )

            # Accumulate across tasks
            if name not in self.importance_dict:
                self.importance_dict[name] = task_importance.clone()
            else:
                self.importance_dict[name] += task_importance

            # Store current parameters as optimal anchor for next task
            self.optimal_params[name] = param.data.clone().detach()

            # Reset online tracking state for the next task
            self._omega_running[name] = torch.zeros_like(param.data)
            self._theta_prev[name] = param.data.clone().detach()
            self._theta_start[name] = param.data.clone().detach()

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
