"""
Loss Module: Co2L Loss Functions
=================================
SupConLoss      – Supervised Contrastive Loss (single-view, Khosla et al. 2020)
AsymDistillLoss – Asymmetric Distillation Loss (Co2L, Cha et al. 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for a single-view batch.

    Paper: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)

    Formula (per anchor i):
        L_i = -1/|P(i)| * sum_{p in P(i)} log [
                  exp(z_i · z_p / τ)
                  ─────────────────────────────
                  sum_{a in A(i)} exp(z_i · z_a / τ)
              ]

        L = mean over i of L_i   (only anchors with at least one positive)

    Definitions:
        P(i) = { j ≠ i  |  y_j == y_i }   (same-class, non-self)
        A(i) = { j ≠ i }                   (all non-self)
        τ    = temperature

    Args:
        temperature (float): Scaling factor τ. Smaller τ sharpens the
                             distribution (harder negatives). Default: 0.1.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def forward(self, projections: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            projections (torch.Tensor): Shape (B, dim).
                Assumed to be L2-normalised (MLP_Co2L.forward() guarantees this),
                but an internal re-normalisation is applied as a safety net.
            labels (torch.Tensor): Integer class labels, shape (B,).

        Returns:
            torch.Tensor: Scalar loss.
                Returns 0 (with gradient) if no anchor has a positive pair,
                so the caller can always call .backward() safely.
        """
        B = projections.size(0)
        device = projections.device

        # ── Safety normalisation ──────────────────────────────────────────────
        z = F.normalize(projections, dim=1)  # (B, dim)

        # ── Similarity matrix scaled by τ ────────────────────────────────────
        # sim[i, j] = z_i · z_j / τ
        sim = torch.matmul(z, z.T) / self.temperature  # (B, B)

        # ── Numerical stability: subtract per-row max before exp ─────────────
        # Equivalent to the log-sum-exp trick; does not change the result.
        # Note: sim[i,i] = 1/τ is always the row max (cosine-sim with self = 1),
        # so after subtraction the diagonal becomes 0 and all others are ≤ 0.
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        # ── Masks ─────────────────────────────────────────────────────────────
        self_mask = torch.eye(B, dtype=torch.bool, device=device)            # i == j
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask # same class, non-self

        # ── Denominator: sum exp(sim_ia) over a ∈ A(i) = all j ≠ i ──────────
        exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)   # zero out diagonal
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)  # (B, 1)

        # ── log-probability for every (anchor, positive) pair ────────────────
        # log_prob[i, j] = log [ exp(sim_ij) / sum_{a≠i} exp(sim_ia) ]
        #                = sim_ij - log_denom_i
        log_prob = sim - log_denom  # (B, B)

        # ── Average log_prob over P(i) for each anchor ───────────────────────
        n_pos = pos_mask.sum(dim=1).float()  # (B,)
        has_pos = n_pos > 0                  # anchors that have at least one positive

        # Edge case: no positive pairs in the batch (e.g. batch_size == 1,
        # or every sample is a singleton class).  Return 0 with gradient so
        # the caller can always call .backward() without special-casing.
        if not has_pos.any():
            return (projections * 0).sum()

        # Sum log_prob over positives, then divide by |P(i)|
        pos_log_prob_sum = (log_prob * pos_mask.float()).sum(dim=1)   # (B,)
        mean_log_prob_pos = pos_log_prob_sum[has_pos] / n_pos[has_pos]

        return -mean_log_prob_pos.mean()


class AsymDistillLoss(nn.Module):
    """
    Asymmetric Distillation Loss for Co2L.

    Paper: "Co2L: Contrastive Continual Learning" (Cha et al., ICCV 2021)

    Encourages the current model to preserve the previous model's representation
    space via a contrastive (InfoNCE-style) objective:

        L_i = -log [ exp(z_cur_i · z_prev_i / τ)
                     ─────────────────────────────────────────
                     sum_{j=1}^{B} exp(z_cur_i · z_prev_j / τ) ]

        L = (1/B) * sum_i L_i

    Why "asymmetric":
        - Current model projections are the queries (learn to align with old space).
        - Previous model projections are fixed keys (frozen reference bank).
        - The loss is computed in one direction only: current → previous.
          The reverse (previous → current) is intentionally omitted so that
          the current model is free to reshape its representation for new tasks,
          while still anchoring to the old model's structure.

    Positive pair: (z_cur_i, z_prev_i) — same sample, different models.
    Negatives    : (z_cur_i, z_prev_j) for all j ≠ i.

    Note: unlike SupConLoss the denominator includes the positive (j == i),
    following the standard InfoNCE / NT-Xent formulation.

    Args:
        temperature (float): Scaling factor τ. Default: 0.1.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def forward(self, z_cur: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_cur  (torch.Tensor): Current model projections,  shape (B, dim).
                                   Must carry gradients (requires_grad=True).
            z_prev (torch.Tensor): Previous model projections, shape (B, dim).
                                   Should be detached (frozen model output).

        Returns:
            torch.Tensor: Scalar loss.
        """
        # ── Safety normalisation ──────────────────────────────────────────────
        z_c = F.normalize(z_cur,  dim=1)   # (B, dim)
        z_p = F.normalize(z_prev, dim=1)   # (B, dim)

        # ── Cross-similarity matrix ───────────────────────────────────────────
        # sim[i, j] = z_cur_i · z_prev_j / τ
        # Rows = current (queries), Columns = previous (keys).
        sim = torch.matmul(z_c, z_p.T) / self.temperature  # (B, B)

        # ── Numerical stability ───────────────────────────────────────────────
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        # ── InfoNCE loss = cross-entropy with diagonal as ground-truth ────────
        # Positive pair for query i is key i  →  target class = i
        B = sim.size(0)
        targets = torch.arange(B, device=sim.device)
        return F.cross_entropy(sim, targets)
