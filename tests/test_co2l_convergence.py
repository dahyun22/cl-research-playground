"""
Co2L Convergence Check on Split-MNIST
======================================
λ=0.5, temperature=0.1 설정으로 실제 데이터셋을 돌리며
epoch별 loss 추이를 출력하고 수렴 여부를 확인한다.

체크 항목:
  1. Task 0 (prev_model 없음): epoch별 loss가 전반적으로 감소
  2. Task 1 (prev_model 있음): distillation loss 포함해도 loss가 감소
  3. 두 task 모두 loss가 유한하고 NaN/Inf 없음

실행:
    python tests/test_co2l_convergence.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader

from datasets import SplitMNIST
from models import MLP_Co2L
from losses import SupConLoss, AsymDistillLoss


# ─────────────────────────────────────────────────────────────────────────────
# 하이퍼파라미터
# ─────────────────────────────────────────────────────────────────────────────
DEVICE      = "cpu"
LR          = 1e-3
EPOCHS      = 5
LAMBDA      = 0.5   # AsymDistillLoss 가중치
TEMPERATURE = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# epoch 단위 loss 측정 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, supcon_fn, distill_fn,
              optimizer, prev_model, distill_lambda):
    """한 epoch 돌고 평균 loss와 컴포넌트별 값을 반환한다."""
    model.train()
    totals = {"ce": 0., "supcon": 0., "distill": 0., "total": 0.}
    n = 0

    for x, y, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        features, projections = model(x)
        logits = model.classifier(features)

        ce     = criterion(logits, y)
        supcon = supcon_fn(projections, y)

        if prev_model is not None:
            with torch.no_grad():
                _, proj_prev = prev_model(x)
            distill = distill_fn(projections, proj_prev)
            loss = ce + supcon + distill_lambda * distill
        else:
            distill = torch.zeros(1)
            loss = ce + supcon

        loss.backward()
        optimizer.step()

        totals["ce"]      += ce.item()
        totals["supcon"]  += supcon.item()
        totals["distill"] += distill.item()
        totals["total"]   += loss.item()
        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


def _check_finite(losses, task_id):
    for epoch_losses in losses:
        for k, v in epoch_losses.items():
            if not torch.isfinite(torch.tensor(v)):
                print(f"  [FAIL] Task {task_id}: {k} loss is not finite ({v})")
                sys.exit(1)


def _check_decreasing(losses, task_id):
    """첫 epoch loss > 마지막 epoch loss 여부 확인."""
    first = losses[0]["total"]
    last  = losses[-1]["total"]
    if last >= first:
        print(f"  [FAIL] Task {task_id}: loss did not decrease "
              f"(first={first:.4f}, last={last:.4f})")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)

    print("=" * 60)
    print(f"Co2L Convergence Check  (λ={LAMBDA}, τ={TEMPERATURE}, epochs={EPOCHS})")
    print("=" * 60)

    dataset = SplitMNIST()
    model   = MLP_Co2L().to(DEVICE)

    criterion  = nn.CrossEntropyLoss()
    supcon_fn  = SupConLoss(temperature=TEMPERATURE)
    distill_fn = AsymDistillLoss(temperature=TEMPERATURE)
    optimizer  = optim.Adam(model.parameters(), lr=LR)

    prev_model = None

    for task_id in range(2):   # task 0, 1만 확인
        loader = dataset.get_task_data(task_id, split="train")
        has_distill = prev_model is not None

        print(f"\n[Task {task_id}]  distillation={'ON' if has_distill else 'OFF (first task)'}")
        print(f"  {'epoch':>5}  {'total':>8}  {'CE':>8}  {'SupCon':>8}  {'Distill':>8}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

        epoch_losses = []
        for epoch in range(EPOCHS):
            ep = run_epoch(model, loader, criterion, supcon_fn, distill_fn,
                           optimizer, prev_model, LAMBDA)
            epoch_losses.append(ep)
            print(f"  {epoch+1:>5}  {ep['total']:>8.4f}  {ep['ce']:>8.4f}"
                  f"  {ep['supcon']:>8.4f}  {ep['distill']:>8.4f}")

        # ── 검증 ──────────────────────────────────────────────────────────
        _check_finite(epoch_losses, task_id)
        _check_decreasing(epoch_losses, task_id)

        delta = epoch_losses[0]["total"] - epoch_losses[-1]["total"]
        print(f"\n  [PASS] Task {task_id}: loss decreased by {delta:.4f}"
              f" ({epoch_losses[0]['total']:.4f} → {epoch_losses[-1]['total']:.4f})")

        # task 종료 후 prev_model 고정
        prev_model = deepcopy(model)
        prev_model.eval()
        for p in prev_model.parameters():
            p.requires_grad_(False)

    print("\n" + "=" * 60)
    print("All convergence checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
