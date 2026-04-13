"""
Hard Attention to the Task (HAT) Trainer
=========================================
Learns per-task binary masks via sigmoid gating with temperature annealing.
Gradient masking prevents overwriting neurons already claimed by previous tasks.

Reference: Serra et al., "Overcoming Catastrophic Forgetting with Hard
Attention to the Task", ICML 2018.

Key ideas
---------
1. Each task t has learnable embedding vectors e_t^l for every masked layer l.

2. Soft mask during training (temperature s anneals 1/s_max → s_max):
     m_t^l = sigmoid(s * e_t^l)
   At the start of training s is small → masks are flat (~0.5), gradients
   flow freely. By the end s is large → masks approach binary.

3. Hard mask at evaluation / consolidation (s = s_max):
     h_t^l ≈ m_t^l > 0.5  (approximately binary)

4. Cumulative mask tracks which neurons are "claimed" by any past task:
     c^l = max over past tasks of h_t^l

5. Gradient masking after each backward pass:
   - Embedding gradients: zero where c^l = 1 (don't move claimed neurons).
   - Weight/bias gradients: zero rows/cols connected to claimed neurons.

6. Capacity regularization encourages sparse masks:
     R = (1/N) * sum_l sum_i  m_t^l_i / (1 + c^l_i)
   Total loss: L = L_CE + lambda * R
"""

import torch
import torch.nn as nn
import torch.optim as optim


class HATTrainer:
    """
    HAT (Hard Attention to the Task) trainer.

    Conceptual approach:
      - Maintains a learnable attention embedding per task per layer.
      - During training, soft masks gate each hidden layer's output and
        gradually harden via temperature annealing.
      - After each task, the binary hard mask is merged into a cumulative
        mask that protects those neurons from future modification.
      - A capacity regularizer penalises re-use of already-claimed neurons,
        pushing the model to allocate fresh capacity for new tasks.

    Interface:
      train_task(train_loader)       – train on one task (uses self.current_task)
      consolidate_task(train_loader) – update cumulative mask, advance task counter
      set_eval_task(task_id)         – select which task mask to use in evaluate()
      evaluate(test_loader)          – accuracy with hard mask for eval_task_id
    """

    def __init__(self, model, device="cpu", learning_rate=0.001, epochs=5,
                 num_tasks=5, hat_lambda=0.75, s_max=400):
        """
        Args:
            model: Neural network (must implement get_hat_layer_sizes,
                   forward_hat, get_hat_gradient_mask_info).
            device (str): "cpu" or "cuda".
            learning_rate (float): Adam learning rate.
            epochs (int): Training epochs per task.
            num_tasks (int): Total number of tasks (pre-allocates embeddings).
            hat_lambda (float): Regularization weight for the capacity penalty.
            s_max (float): Maximum temperature; controls how binary masks become.
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

        # task_embeddings[t][l]: learnable Parameter of shape (layer_sizes[l],)
        # Initialised to zero so sigmoid gives 0.5 → equal initial attention.
        self.task_embeddings = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(torch.zeros(size))
                for size in self.layer_sizes
            ])
            for _ in range(num_tasks)
        ]).to(device)

        # cumulative_mask[l]: binary tensor marking neurons claimed by past tasks.
        self.cumulative_mask = [
            torch.zeros(size, device=device)
            for size in self.layer_sizes
        ]

        # current_task increments in consolidate_task(); train_task() reads it.
        self.current_task = 0
        # eval_task_id is set via set_eval_task() before each evaluate() call.
        self.eval_task_id = 0

        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Mask utilities
    # ------------------------------------------------------------------

    def _get_masks(self, task_id, s):
        """
        Compute soft attention masks for task_id at temperature s.

        Args:
            task_id (int): Which task's embeddings to use.
            s (float): Sigmoid temperature (larger → more binary output).

        Returns:
            list[Tensor]: One mask per masked layer, shape (layer_size,).
        """
        return [
            torch.sigmoid(s * self.task_embeddings[task_id][l])
            for l in range(self.num_mask_layers)
        ]

    def set_eval_task(self, task_id):
        """
        Choose which task's hard mask to use during evaluate().

        Must be called before each evaluate() call so the correct
        task-specific mask gates the network.

        Args:
            task_id (int): Task index to evaluate.
        """
        self.eval_task_id = task_id

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _hat_reg_loss(self, task_id, s):
        """
        HAT capacity regularization.

        Penalises use of neurons already in the cumulative mask.
        For task 0 the cumulative mask is all-zero, so this becomes a plain
        sparsity penalty (encourages small masks from the start).

        Formula: R = (1/N) * sum_l sum_i  m_t^l_i / (1 + c^l_i)

        Args:
            task_id (int): Current task.
            s (float): Current temperature.

        Returns:
            torch.Tensor: Scalar regularization loss.
        """
        masks = self._get_masks(task_id, s)
        reg = torch.tensor(0.0, device=self.device)
        total = 0
        for mask, c_mask in zip(masks, self.cumulative_mask):
            reg = reg + (mask / (1.0 + c_mask)).sum()
            total += mask.numel()
        return reg / max(total, 1)

    # ------------------------------------------------------------------
    # Gradient masking
    # ------------------------------------------------------------------

    def _clip_embedding_grads(self, task_id):
        """
        Zero embedding gradients for neurons already claimed by past tasks.

        Neurons where c^l = 1 should not be moved for the new task, so we
        zero out the corresponding gradient components before the optimizer
        step.
        """
        for l, c_mask in enumerate(self.cumulative_mask):
            emb = self.task_embeddings[task_id][l]
            if emb.grad is not None:
                emb.grad.data.mul_(1.0 - c_mask)

    def _clip_weight_grads(self):
        """
        Mask weight and bias gradients to protect neurons of previous tasks.

        Uses the layer-to-mask mapping returned by the model's
        get_hat_gradient_mask_info() to determine which gradient entries to zero.

        Rules:
          Bias (1-D):
            - Zero entries where post-layer cumulative mask = 1.

          FC weight (2-D, out × in):
            - Zero rows   where post-layer cumulative mask = 1.
            - Zero columns where pre-layer cumulative mask = 1.
              If in_features > pre-mask size (flattened conv output), the
              pre-mask is expanded with repeat_interleave to match.

          Conv weight (4-D, out_ch × in_ch × kH × kW):
            - Zero output-channel slices where post-layer mask = 1.
            - Zero input-channel  slices where pre-layer  mask = 1.
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
                # Bias vector (out_features,)
                if post_idx >= 0:
                    grad.mul_(1.0 - self.cumulative_mask[post_idx])

            elif grad.dim() == 2:
                # FC weight (out_features × in_features)
                if post_idx >= 0:
                    c_post = self.cumulative_mask[post_idx]          # (out,)
                    grad.mul_((1.0 - c_post).unsqueeze(1))
                if pre_idx >= 0:
                    c_pre = self.cumulative_mask[pre_idx]            # (in_orig,)
                    if c_pre.size(0) < grad.size(1):
                        # Expand for spatial flattening: each channel occupies
                        # (H × W) consecutive features in the flattened vector.
                        repeat_factor = grad.size(1) // c_pre.size(0)
                        c_pre = c_pre.repeat_interleave(repeat_factor)
                    grad.mul_((1.0 - c_pre).unsqueeze(0))

            elif grad.dim() == 4:
                # Conv weight (out_ch × in_ch × kH × kW)
                if post_idx >= 0:
                    c_post = self.cumulative_mask[post_idx]          # (out_ch,)
                    grad.mul_((1.0 - c_post).view(-1, 1, 1, 1))
                if pre_idx >= 0:
                    c_pre = self.cumulative_mask[pre_idx]            # (in_ch,)
                    grad.mul_((1.0 - c_pre).view(1, -1, 1, 1))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_task(self, train_loader, verbose=False):
        """
        Train on a single task using HAT masking.

        Per-batch flow:
          1. Compute temperature s (linearly annealed 1/s_max → s_max).
          2. Compute soft masks for self.current_task at temperature s.
          3. Forward pass through model.forward_hat(x, masks).
          4. Total loss = CE loss + hat_lambda * capacity regularization.
          5. Backward pass.
          6. Clip embedding gradients for claimed neurons.
          7. Clip weight gradients for connections to claimed neurons.
          8. Optimizer step.

        A fresh Adam optimizer is created for each task, covering the shared
        model parameters plus only the current task's embedding parameters.

        Args:
            train_loader: DataLoader for the current task's training data.
            verbose (bool): Print per-epoch average loss.

        Returns:
            float: Average loss over the final epoch.
        """
        task_id = self.current_task
        self.model.train()

        # Optimizer: shared model weights + this task's attention embeddings only.
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

                # Linear temperature annealing: 1/s_max at step 0, s_max at final step.
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

                self._clip_embedding_grads(task_id)
                self._clip_weight_grads()

                optimizer.step()

                total_loss += loss.item()
                num_seen += 1

            avg_loss = total_loss / max(num_seen, 1)
            if verbose:
                print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

        return avg_loss

    # ------------------------------------------------------------------
    # Post-task consolidation
    # ------------------------------------------------------------------

    def consolidate_task(self, train_loader=None):
        """
        Update the cumulative mask after finishing self.current_task.

        Computes the hard mask (using s = s_max) for the just-trained task
        and takes an element-wise maximum with the existing cumulative mask,
        permanently marking those neurons as claimed.

        Increments self.current_task so the next train_task() call uses the
        correct embedding for the following task.

        Args:
            train_loader: Accepted for interface consistency with EWC but unused.
        """
        task_id = self.current_task
        with torch.no_grad():
            hard_masks = self._get_masks(task_id, self.s_max)
            for l, (h_mask, c_mask) in enumerate(zip(hard_masks, self.cumulative_mask)):
                self.cumulative_mask[l] = torch.max(c_mask, (h_mask > 0.5).float())
        self.current_task += 1

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_loader):
        """
        Evaluate model accuracy using the hard mask for self.eval_task_id.

        Call set_eval_task(task_id) before this method to select the correct
        task-specific mask. Using the right mask is important: activating
        neurons from unrelated tasks can degrade accuracy.

        Args:
            test_loader: DataLoader for test data.

        Returns:
            float: Classification accuracy in [0, 1].
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            masks = self._get_masks(self.eval_task_id, self.s_max)

            for x, y, _ in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model.forward_hat(x, masks)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return correct / max(total, 1)
