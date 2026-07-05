import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


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

def visualize_depth_predictions(
    model,
    test_ds,
    device: str,
    save_dir: Path,
    prefix: str,
    n: int = 5,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    n = min(n, len(test_ds))

    model.eval()
    with torch.no_grad():
        for i in range(n):
            sample = test_ds[i]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            gt = sample["depth"].numpy()
            stem = test_ds.stems[i]

            outputs = model(pixel_values=pixel_values)
            pred_raw = outputs.predicted_depth.cpu()  # (1, H', W')
            pred_np = F.interpolate(
                pred_raw.unsqueeze(1),
                size=gt.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()

            pred_aligned = align_scale_shift(pred_np, gt)
            pred_aligned = np.clip(pred_aligned, 0, None)

            error = np.zeros_like(gt)
            mask = gt > 0
            error[mask] = np.abs(pred_aligned[mask] - gt[mask])

            vmin, vmax = gt[mask].min(), gt[mask].max()
            rgb = Image.open(test_ds.rgb_dir / f"{stem}.png").convert("RGB")

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"{prefix} — {stem}")
            axes = axes.flatten()

            axes[0].imshow(rgb)
            axes[0].set_title("RGB")
            axes[0].axis("off")

            im1 = axes[1].imshow(pred_aligned, cmap="plasma", vmin=vmin, vmax=vmax)
            axes[1].set_title("Predicted Depth")
            axes[1].axis("off")
            plt.colorbar(im1, ax=axes[1])

            im2 = axes[2].imshow(gt, cmap="plasma", vmin=vmin, vmax=vmax)
            axes[2].set_title("GT Depth")
            axes[2].axis("off")
            plt.colorbar(im2, ax=axes[2])

            im3 = axes[3].imshow(error, cmap="hot")
            axes[3].set_title("Absolute Error")
            axes[3].axis("off")
            plt.colorbar(im3, ax=axes[3])

            plt.tight_layout()
            plt.savefig(save_dir / f"{prefix}_{stem}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def align_scale_shift(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Least-squares scale+shift alignment: fit  pred_aligned = scale * pred + shift
    so that pred_aligned matches gt in a least-squares sense.

    Depth Anything V2 outputs affine-invariant (relative) depth, so this alignment
    is required before computing AbsRel or RMSE against metric ground truth.
    """
    mask = gt > 0
    p = pred[mask].reshape(-1, 1)
    g = gt[mask].flatten()
    A = np.hstack([p, np.ones_like(p)])
    result, _, _, _ = np.linalg.lstsq(A, g, rcond=None)
    scale, shift = float(result[0]), float(result[1])
    return pred * scale + shift