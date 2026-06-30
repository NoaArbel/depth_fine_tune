import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once (safe to call from scripts or notebooks)."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

def load_gt_depth(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)

def align_scale_shift(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Least-squares scale+shift alignment: fit  pred_aligned = scale * pred + shift
    so that pred_aligned matches gt in a least-squares sense.

    Depth Anything V2 outputs affine-invariant (relative) depth, so this alignment
    is required before computing AbsRel or RMSE against metric ground truth.
    """
    logger.debug("[utils] Aligning scale and shift")
    mask = gt > 0
    p = pred[mask].reshape(-1, 1)
    g = gt[mask].flatten()
    A = np.hstack([p, np.ones_like(p)])
    result, _, _, _ = np.linalg.lstsq(A, g, rcond=None)
    scale, shift = float(result[0]), float(result[1])
    return pred * scale + shift