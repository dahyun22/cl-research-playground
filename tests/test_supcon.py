"""
SupConLoss Unit Tests
=====================
각 케이스별로 기대 동작을 assert로 검증한다.

실행:
    python tests/test_supcon.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from losses import SupConLoss


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _make_proj(B, dim=128, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(B, dim)


def _pass(name):
    print(f"  [PASS] {name}")


def _fail(name, detail):
    print(f"  [FAIL] {name}: {detail}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 케이스 1: 기본 forward — 유한한 양수 loss
# ─────────────────────────────────────────────────────────────────────────────
def test_basic_forward():
    loss_fn = SupConLoss(temperature=0.1)
    z = _make_proj(8, seed=0)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    loss = loss_fn(z, y)

    if not loss.isfinite():
        _fail("basic_forward", f"loss is not finite: {loss.item()}")
    if loss.item() <= 0:
        _fail("basic_forward", f"loss should be > 0, got {loss.item():.4f}")

    _pass(f"basic_forward  loss={loss.item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 케이스 2: 완전히 분리된 표현 → loss ≈ 0
#   같은 클래스끼리 동일한 벡터, 다른 클래스는 직교
#   → positive sim = 1/τ, negative sim = 0  → log-softmax 최댓값 → loss 최솟값
# ─────────────────────────────────────────────────────────────────────────────
def test_perfect_separation():
    loss_fn = SupConLoss(temperature=0.1)

    # 4개 클래스, 각 클래스 2샘플, 클래스 간 완전 직교
    basis = torch.eye(4)           # (4, 4) orthonormal basis
    z = basis.repeat_interleave(2, dim=0)  # (8, 4): 각 행이 one-hot
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    loss = loss_fn(z, y)

    if not loss.isfinite():
        _fail("perfect_separation", f"loss is not finite: {loss.item()}")
    if loss.item() > 0.05:
        _fail("perfect_separation", f"loss should be ≈ 0, got {loss.item():.4f}")

    _pass(f"perfect_separation  loss={loss.item():.6f} (expected ≈ 0)")


# ─────────────────────────────────────────────────────────────────────────────
# 케이스 3: 모든 샘플이 singleton (positive 없음) → loss = 0
# ─────────────────────────────────────────────────────────────────────────────
def test_all_singletons():
    loss_fn = SupConLoss(temperature=0.1)
    z = _make_proj(4, seed=1)
    y = torch.tensor([0, 1, 2, 3])   # 클래스가 모두 다름

    loss = loss_fn(z, y)

    if loss.item() != 0.0:
        _fail("all_singletons", f"loss should be 0, got {loss.item()}")

    # backward도 문제없어야 함
    z2 = _make_proj(4, seed=1).requires_grad_(True)
    loss_fn(z2, y).backward()

    _pass(f"all_singletons  loss={loss.item():.4f}, backward OK")


# ─────────────────────────────────────────────────────────────────────────────
# 케이스 4: 모든 샘플이 같은 클래스 → 유한한 양수 loss (분모 ≠ 0)
# ─────────────────────────────────────────────────────────────────────────────
def test_all_same_class():
    loss_fn = SupConLoss(temperature=0.1)
    z = _make_proj(4, seed=2)
    y = torch.zeros(4, dtype=torch.long)

    loss = loss_fn(z, y)

    if not loss.isfinite():
        _fail("all_same_class", f"loss is not finite: {loss.item()}")

    _pass(f"all_same_class  loss={loss.item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 케이스 5: temperature 감도
#   낮은 τ → sharper distribution → 랜덤 입력에서 더 높은 loss
# ─────────────────────────────────────────────────────────────────────────────
def test_temperature_sensitivity():
    z = _make_proj(8, seed=3)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    loss_low  = SupConLoss(temperature=0.07)(z, y)
    loss_high = SupConLoss(temperature=0.5)(z, y)

    if loss_low.item() <= loss_high.item():
        _fail("temperature_sensitivity",
              f"expected loss(τ=0.07) > loss(τ=0.5), "
              f"got {loss_low.item():.4f} vs {loss_high.item():.4f}")

    _pass(f"temperature_sensitivity  τ=0.07→{loss_low.item():.4f}  τ=0.5→{loss_high.item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 케이스 6: gradient flow — 모든 파라미터에 grad 흐름
# ─────────────────────────────────────────────────────────────────────────────
def test_gradient_flow():
    loss_fn = SupConLoss(temperature=0.1)
    z = _make_proj(8, seed=4).requires_grad_(True)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    loss = loss_fn(z, y)
    loss.backward()

    if z.grad is None:
        _fail("gradient_flow", "grad is None after backward")
    if not z.grad.isfinite().all():
        _fail("gradient_flow", "grad contains NaN/Inf")

    _pass(f"gradient_flow  grad_norm={z.grad.norm().item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 케이스 7: 이미 L2-norm된 입력도 동일한 결과
# ─────────────────────────────────────────────────────────────────────────────
def test_prenormalized_input():
    loss_fn = SupConLoss(temperature=0.1)
    torch.manual_seed(5)
    z_raw  = torch.randn(8, 128)
    z_norm = F.normalize(z_raw, dim=1)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    loss_raw  = loss_fn(z_raw,  y).item()
    loss_norm = loss_fn(z_norm, y).item()

    if abs(loss_raw - loss_norm) > 1e-4:
        _fail("prenormalized_input",
              f"raw={loss_raw:.6f} vs norm={loss_norm:.6f}: should be identical")

    _pass(f"prenormalized_input  raw={loss_raw:.4f}  norm={loss_norm:.4f}  diff={abs(loss_raw-loss_norm):.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("SupConLoss Unit Tests")
    print("=" * 55)

    test_basic_forward()
    test_perfect_separation()
    test_all_singletons()
    test_all_same_class()
    test_temperature_sensitivity()
    test_gradient_flow()
    test_prenormalized_input()

    print("=" * 55)
    print("All tests passed.")
    print("=" * 55)
