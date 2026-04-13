"""
Gradient Episodic Memory (GEM) Trainer
=======================================
Constrains gradient updates so that loss on any previously stored episodic
memory does not increase, preventing catastrophic forgetting.
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
    Per-task episodic memory buffer for GEM.

    Maintains a separate memory bank for each completed task, storing up to
    n_memories randomly-sampled training examples.  Unlike DER++'s flat
    reservoir buffer, GEM needs task-separated memories so that a
    task-specific gradient can be computed and used as a constraint.

    Conceptual role:
      - After finishing task t, call add_task(t, loader) to populate M_t
      - During training on task t+k, iterate over all stored memories to
        obtain reference gradients that the current update must not violate
    """

    def __init__(self, n_memories=256):
        """
        Args:
            n_memories (int): Maximum number of samples to keep per task
        """
        self.n_memories = n_memories
        self.memory = {}  # {task_id: (x_tensor, y_tensor)}  — stored on CPU

    def add_task(self, task_id, data_loader):
        """
        Collect and store up to n_memories samples from a completed task.

        All available batches are concatenated first, then a random subset
        of size n_memories is drawn without replacement to form M_{task_id}.

        Args:
            task_id (int): Identifier for the completed task
            data_loader: DataLoader for the completed task's training data
        """
        all_x, all_y = [], []

        for x, y, _ in data_loader:
            all_x.append(x.cpu())
            all_y.append(y.cpu())

        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)

        n = all_x.size(0)
        mem_size = min(self.n_memories, n)
        indices = torch.randperm(n)[:mem_size]

        self.memory[task_id] = (all_x[indices], all_y[indices])

    def num_tasks(self):
        """Return the number of tasks currently stored in memory."""
        return len(self.memory)


class GEMTrainer:
    """
    GEM (Gradient Episodic Memory) trainer.

    Conceptual approach:
      - After learning each task, store a small episodic memory of M samples
      - During training on a new task t, enforce gradient constraints:
          ⟨g̃, g_k⟩ ≥ 0  for all previous tasks k
        where g̃ is the (projected) update gradient and g_k is the gradient
        computed on the episodic memory of task k
      - When the constraint is violated for any k, find the closest feasible
        gradient g̃ via Quadratic Programming (QP) and apply it instead
      - Expected to show significantly reduced forgetting compared to
        fine-tuning while requiring no architectural changes (unlike HAT)

    Gradient projection (QP formulation):

      Primal:
        min_{g̃}  (1/2) ‖g̃ − g‖²
        s.t.      G g̃ ≥ −margin      (G: memory gradient matrix, rows = g_k)

      Dual (solved with quadprog):
        min_{v ≥ 0}  (1/2) v^T (G G^T) v  +  (G g + margin)^T v
        g̃ = g + G^T v*

    When quadprog is unavailable the trainer falls back to sequential
    Gram-Schmidt projection, which is not QP-optimal but still enforces
    the dot-product sign constraint and requires no extra dependency.

    Reference:
      Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning",
      NeurIPS 2017.
    """

    def __init__(self, model, device="cpu", learning_rate=0.001, epochs=5,
                 n_memories=256, margin=0.0):
        """
        Args:
            model: Neural network model
            device (str): Device for training
            learning_rate (float): Learning rate for optimizer
            epochs (int): Number of epochs per task
            n_memories (int): Samples stored per task in episodic memory
            margin (float): Constraint slack ε; constraint is G g̃ ≥ −margin.
                            GEM paper uses 0; small positive values add slack.
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n_memories = n_memories
        self.margin = margin

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Per-task episodic memory bank
        self.episodic_memory = EpisodicMemory(n_memories=n_memories)

        # Number of tasks whose memories have been stored
        self.n_tasks_seen = 0

    # ------------------------------------------------------------------
    # Gradient vector helpers
    # ------------------------------------------------------------------

    def _get_gradient_vector(self):
        """
        Flatten all parameter gradients into a single 1-D numpy array.
        Parameters without a gradient contribute a zero block.

        Returns:
            np.ndarray: Concatenated gradient vector, shape (num_params,)
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
        Write a flat numpy gradient array back into the model's .grad buffers.

        Args:
            grad_vec (np.ndarray): Flat gradient vector of length num_params
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
    # Memory gradient computation
    # ------------------------------------------------------------------

    def _compute_memory_gradients(self):
        """
        Compute one gradient vector per previous task using its episodic memory.

        Each memory gradient g_k = ∇_θ L(M_k; θ) is a full forward + backward
        pass over the stored samples for task k.  The model's gradient buffers
        are left containing the last task's memory gradient upon return
        (the caller is expected to overwrite them via _set_gradient_vector).

        Returns:
            np.ndarray or None: Memory gradient matrix G of shape
                                (n_prev_tasks, num_params), or None when no
                                previous task memory exists yet.
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
    # QP-based projection
    # ------------------------------------------------------------------

    def _qp_project(self, current_grad, memory_grads):
        """
        Project current_grad onto the feasible cone via QP (quadprog).

        Solves the dual:
          min_{v ≥ 0}  (1/2) v^T (G G^T) v  +  (G g + margin)^T v
          g̃ = g + G^T v*

        Falls back to Gram-Schmidt when quadprog is unavailable or the solver
        encounters a numerical failure.

        Args:
            current_grad (np.ndarray): Current task gradient g, shape (d,)
            memory_grads (np.ndarray): Memory gradient matrix G, shape (K, d)

        Returns:
            np.ndarray: Projected gradient g̃, shape (d,), dtype float32
        """
        if _QUADPROG_AVAILABLE:
            return self._qp_project_quadprog(current_grad, memory_grads)
        return self._gram_schmidt_project(current_grad, memory_grads)

    def _qp_project_quadprog(self, current_grad, memory_grads):
        """
        QP projection using quadprog library.

        quadprog API: solve_qp(Q, a, C, b) solves
          min (1/2) x^T Q x − a^T x   s.t. C^T x ≥ b

        Mapping from dual variables v:
          Q = G G^T  (Gram matrix of memory gradients, regularised)
          a = −(G g + margin·1)
          C = I  (v ≥ 0 constraint)
          b = 0

        Args:
            current_grad (np.ndarray): shape (d,)
            memory_grads (np.ndarray): shape (K, d)

        Returns:
            np.ndarray: Projected gradient, shape (d,), dtype float32
        """
        n_tasks = memory_grads.shape[0]

        # Gram matrix (dual coefficient matrix), regularised for stability
        GGT = (memory_grads @ memory_grads.T).astype(np.float64)
        GGT += 1e-6 * np.eye(n_tasks)

        # Linear term: G g + margin·1  (incorporates constraint slack)
        Gg = (memory_grads @ current_grad + self.margin).astype(np.float64)

        # Non-negativity constraint on dual variables: I v ≥ 0
        C = np.eye(n_tasks, dtype=np.float64)
        b = np.zeros(n_tasks, dtype=np.float64)

        try:
            v = quadprog.solve_qp(GGT, -Gg, C, b)[0]
            projected = current_grad + (memory_grads.T @ v).astype(np.float32)
        except Exception:
            # Numerical failure: gracefully fall back to Gram-Schmidt
            projected = self._gram_schmidt_project(current_grad, memory_grads)

        return projected.astype(np.float32)

    def _gram_schmidt_project(self, current_grad, memory_grads):
        """
        Sequential Gram-Schmidt gradient projection (quadprog-free fallback).

        For each violated constraint, removes the component of the gradient
        that points against the memory gradient:
          if ⟨g, g_k⟩ < 0:  g ← g − (⟨g, g_k⟩ / ‖g_k‖²) g_k

        Not QP-optimal, but guarantees ⟨g̃, g_k⟩ ≥ 0 for all k and requires
        no external QP solver.  Multiple passes are made until convergence.

        Args:
            current_grad (np.ndarray): shape (d,)
            memory_grads (np.ndarray): shape (K, d)

        Returns:
            np.ndarray: Projected gradient, shape (d,), dtype float32
        """
        g = current_grad.copy().astype(np.float64)

        for _ in range(memory_grads.shape[0]):
            dots = memory_grads @ g  # (K,)
            violated = np.where(dots < -self.margin)[0]
            if len(violated) == 0:
                break
            for k in violated:
                g_k = memory_grads[k].astype(np.float64)
                norm_sq = np.dot(g_k, g_k)
                if norm_sq > 1e-12:
                    g -= (np.dot(g, g_k) / norm_sq) * g_k

        return g.astype(np.float32)

    # ------------------------------------------------------------------
    # Gradient projection entry point
    # ------------------------------------------------------------------

    def _project_gradients(self):
        """
        Check GEM constraints and project the current gradient if violated.

        Steps:
          1. Save current gradient g (from the just-completed backward pass)
          2. Compute memory gradients G = {g_k} for each previous task k
             (this overwrites model.grad buffers — handled in step 4)
          3. Check ⟨g, g_k⟩ ≥ −margin for all k
          4. If any constraint is violated, call the QP/GS projector and
             write g̃ back into model.grad; otherwise restore g as-is
        """
        # Step 1: capture current gradient before it gets overwritten
        current_grad = self._get_gradient_vector()

        # Step 2: compute memory gradients (overwrites model.grad)
        memory_grads = self._compute_memory_gradients()

        if memory_grads is None:
            # No previous task memory yet — nothing to constrain
            self._set_gradient_vector(current_grad)
            return

        # Step 3: check constraints
        dots = memory_grads @ current_grad  # (n_prev_tasks,)

        if (dots + self.margin >= 0).all():
            # All constraints satisfied — restore original gradient unchanged
            self._set_gradient_vector(current_grad)
            return

        # Step 4: constraint(s) violated — project and write back
        projected_grad = self._qp_project(current_grad, memory_grads)
        self._set_gradient_vector(projected_grad)

    # ------------------------------------------------------------------
    # Core trainer interface
    # ------------------------------------------------------------------

    def train_task(self, train_loader, verbose=False):
        """
        Train model on a single task with GEM gradient constraints.

        On task 0 (no episodic memory yet) the update is unconstrained.
        From task 1 onward, each gradient is projected to satisfy
        ⟨g̃, g_k⟩ ≥ 0 for all previously stored task memories k.

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
                loss = self.criterion(logits, y)
                loss.backward()

                # Apply GEM projection when previous task memories exist
                if self.n_tasks_seen > 0:
                    self._project_gradients()

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
        Store episodic memory for the just-completed task.

        Called after finishing training on each task.  Stores a random subset
        of n_memories examples from the current task's training data and
        increments the task counter so the next task's training is constrained.

        Args:
            train_loader: DataLoader for the completed task's training data
        """
        self.episodic_memory.add_task(self.n_tasks_seen, train_loader)
        self.n_tasks_seen += 1

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
