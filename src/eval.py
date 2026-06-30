import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

def abs_rel(pred: np.ndarray, gt: np.ndarray) -> float:
    # absolute relative error, normalized by the true depth (aka - percentage error)
    mask = gt > 0
    return float(np.mean(np.abs(pred[mask] - gt[mask]) / gt[mask]))


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    # root mean square error, not normalized by the true depth
    mask = gt > 0
    return float(np.sqrt(np.mean((pred[mask] - gt[mask]) ** 2)))


def threshold_accuracy(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    mask = (gt > 0) & (pred > 0)
    ratio = np.maximum(pred[mask] / gt[mask], gt[mask] / pred[mask])
    return float(np.mean(ratio < threshold))