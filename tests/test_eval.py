import numpy as np
import pytest
from src.eval import abs_rel, rmse, threshold_accuracy
from src.utils import align_scale_shift


# ---------------------------------------------------------------------------
# align_scale_shift
# ---------------------------------------------------------------------------

def test_align_scale_shift_identity():
    """When pred already matches gt, scale≈1 and shift≈0."""
    gt   = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pred = gt.copy()
    out  = align_scale_shift(pred, gt)
    np.testing.assert_allclose(out, gt, atol=1e-5)


def test_align_scale_shift_pure_scale():
    """pred = 2*gt → scale≈0.5, shift≈0, output ≈ gt."""
    gt   = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    pred = gt * 2.0
    out  = align_scale_shift(pred, gt)
    np.testing.assert_allclose(out, gt, atol=1e-5)


def test_align_scale_shift_pure_shift():
    """pred = gt + 10 → scale≈1, shift≈-10, output ≈ gt."""
    gt   = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    pred = gt + 10.0
    out  = align_scale_shift(pred, gt)
    np.testing.assert_allclose(out, gt, atol=1e-5)


# ---------------------------------------------------------------------------
# abs_rel
# ---------------------------------------------------------------------------

def test_abs_rel_perfect():
    gt   = np.array([1.0, 2.0, 4.0])
    pred = gt.copy()
    assert abs_rel(pred, gt) == pytest.approx(0.0, abs=1e-6)


def test_abs_rel_known_value():
    """pred = 2*gt → |pred-gt|/gt = 1.0 everywhere."""
    gt   = np.array([1.0, 2.0, 4.0])
    pred = gt * 2.0
    assert abs_rel(pred, gt) == pytest.approx(1.0, abs=1e-6)


def test_abs_rel_ignores_invalid():
    """Pixels with gt<=0 must not affect the result."""
    gt   = np.array([1.0, 0.0, 2.0])
    pred = np.array([1.0, 999.0, 2.0])
    assert abs_rel(pred, gt) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# rmse
# ---------------------------------------------------------------------------

def test_rmse_perfect():
    gt   = np.array([1.0, 2.0, 3.0])
    pred = gt.copy()
    assert rmse(pred, gt) == pytest.approx(0.0, abs=1e-6)


def test_rmse_known_value():
    """All errors = 1 → RMSE = 1."""
    gt   = np.array([1.0, 2.0, 3.0])
    pred = gt + 1.0
    assert rmse(pred, gt) == pytest.approx(1.0, abs=1e-6)


def test_rmse_ignores_invalid():
    gt   = np.array([1.0, 0.0, 3.0])
    pred = np.array([1.0, 999.0, 3.0])
    assert rmse(pred, gt) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# threshold_accuracy
# ---------------------------------------------------------------------------

def test_threshold_accuracy_perfect():
    """pred == gt → ratio = 1.0 < any threshold > 1."""
    gt   = np.array([1.0, 2.0, 4.0])
    pred = gt.copy()
    assert threshold_accuracy(pred, gt, 1.25) == pytest.approx(1.0, abs=1e-6)


def test_threshold_accuracy_all_outside():
    """pred = 2*gt → ratio = 2.0, all outside threshold=1.25."""
    gt   = np.array([1.0, 2.0, 4.0])
    pred = gt * 2.0
    assert threshold_accuracy(pred, gt, 1.25) == pytest.approx(0.0, abs=1e-6)


def test_threshold_accuracy_ignores_invalid():
    """Pixels with gt<=0 or pred<=0 must not affect the result."""
    gt   = np.array([1.0, 0.0, 2.0])
    pred = np.array([1.0, 999.0, 2.0])
    assert threshold_accuracy(pred, gt, 1.25) == pytest.approx(1.0, abs=1e-6)
