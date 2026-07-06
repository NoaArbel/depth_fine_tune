import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import make_splits
from src.eval import abs_rel, rmse, threshold_accuracy
from src.model import load_model_with_lora
from src.utils import align_scale_shift, setup_logging, visualize_depth_predictions

logger = logging.getLogger(__name__)


def silog_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    ''' Sigmoid Logarithmic Error Loss '''
    mask = gt > 0
    d = torch.log(pred[mask].clamp(min=1e-6)) - torch.log(gt[mask].clamp(min=1e-6))
    return d.pow(2).mean() - 0.5 * d.mean().pow(2)

def gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    ''' Gradient Loss in log-space — pred/gt: (B, H, W) '''
    mask = (gt > 0).float()

    log_d_diff = torch.log(pred.clamp(min=1e-6)) - torch.log(gt.clamp(min=1e-6))
    log_d_diff = log_d_diff * mask  # zero out invalid pixels before differencing

    grad_x = (log_d_diff[:, :, 1:] - log_d_diff[:, :, :-1]).abs()
    grad_y = (log_d_diff[:, 1:, :] - log_d_diff[:, :-1, :]).abs()

    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]

    loss_x = (mask_x * grad_x).sum() / mask_x.sum().clamp(min=1)
    loss_y = (mask_y * grad_y).sum() / mask_y.sum().clamp(min=1)
    return loss_x + loss_y


def train_one_epoch(model, loader: DataLoader, optimizer, scheduler, device: str, lambda_grad: float = 0.0) -> float:
    model.train()
    ''' Single trainin step '''
    total_loss = 0.0
    for batch in tqdm(loader, desc="train"):
        pixel_values = batch["pixel_values"].to(device)
        gt = batch["depth"].to(device)

        outputs = model(pixel_values=pixel_values)
        pred = outputs.predicted_depth  # (B, H', W')

        # resize pred to match GT spatial size
        pred_resized = F.interpolate(
            pred.unsqueeze(1), size=gt.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)

        loss = silog_loss(pred_resized, gt) + lambda_grad * gradient_loss(pred_resized, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str) -> dict:
    model.eval()
    all_metrics: list[dict] = []

    for batch in tqdm(loader, desc="eval"):
        pixel_values = batch["pixel_values"].to(device)
        gt_batch = batch["depth"].numpy()

        outputs = model(pixel_values=pixel_values) # inference step
        pred_batch = outputs.predicted_depth.cpu()

        for i in range(len(gt_batch)):
            gt = gt_batch[i]
            pred = F.interpolate(
                pred_batch[i].unsqueeze(0).unsqueeze(0),
                size=gt.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()

            pred_aligned = align_scale_shift(pred, gt)
            pred_aligned = np.clip(pred_aligned, 0, None)

            all_metrics.append({
                "abs_rel": abs_rel(pred_aligned, gt),
                "rmse": rmse(pred_aligned, gt),
                "delta1": threshold_accuracy(pred_aligned, gt, 1.25),
                "delta2": threshold_accuracy(pred_aligned, gt, 1.25 ** 2),
                "delta3": threshold_accuracy(pred_aligned, gt, 1.25 ** 3),
            })

    return {
        k: float(np.mean([m[k] for m in all_metrics]))
        for k in all_metrics[0]
    }


def run_experiment(
    name: str,
    cfg: SimpleNamespace,
    train_ds,
    val_ds,
    test_ds,
    project_root: Path,
    device: str,
) -> dict:
    setup_logging()
    logger.info("=" * 60)
    logger.info("Experiment: %s", name)
    logger.info("=" * 60)

    model, _ = load_model_with_lora(cfg)
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.training.batch_size, shuffle=False, num_workers=2)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=cfg.training.warmup_steps)

    history = []
    for epoch in range(1, cfg.training.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, cfg.training.lambda_grad)
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        logger.info(
            "Epoch %d/%d  loss=%.4f  abs_rel=%.4f  delta1=%.4f",
            epoch, cfg.training.epochs, train_loss,
            val_metrics["abs_rel"], val_metrics["delta1"],
        )

    ckpt_dir = project_root / cfg.training.checkpoint_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)

    test_metrics = evaluate(model, test_loader, device)
    logger.info("Test metrics: %s", test_metrics)

    vis_dir = project_root / "results" / "visual"
    visualize_depth_predictions(model, test_ds, device, vis_dir, name)
    logger.info("Saved visual samples to %s", vis_dir)

    return {"name": name, "history": history, "test_metrics": test_metrics}


@torch.no_grad()
def run_baseline(
    cfg: SimpleNamespace,
    test_ds,
    project_root: Path,
    device: str,
) -> dict:
    from transformers import AutoModelForDepthEstimation

    setup_logging()
    logger.info("Running baseline (pretrained, no fine-tuning)")
    model = AutoModelForDepthEstimation.from_pretrained(cfg.model.name)
    model = model.to(device)

    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=2)
    test_metrics = evaluate(model, test_loader, device)
    logger.info("Baseline test metrics: %s", test_metrics)

    vis_dir = project_root / "results" / "visual"
    visualize_depth_predictions(model, test_ds, device, vis_dir, "baseline")
    logger.info("Saved baseline visual samples to %s", vis_dir)

    return {"name": "baseline", "test_metrics": test_metrics}


def _dict_to_ns(d: dict) -> SimpleNamespace:
    ''' Convert a dictionary to a namespace '''
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def main(config_path: str = "configs/config.yaml") -> None:
    setup_logging()

    with open(config_path) as f:
        cfg = _dict_to_ns(yaml.safe_load(f))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    model, processor = load_model_with_lora(cfg)
    model = model.to(device)

    rgb_dir = Path(cfg.data.raw_dir) / "rgb"
    depth_dir = Path(cfg.data.raw_dir) / "depth"
    train_ds, val_ds, _ = make_splits(rgb_dir, depth_dir, processor, train_ratio=0.8, val_ratio=0.2, seed=cfg.data.seed)
    logger.info("Train: %d samples  |  Val: %d samples", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=2)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=cfg.training.warmup_steps,
    )

    best_abs_rel = float("inf")
    epochs = cfg.training.epochs
    for epoch in range(1, epochs + 1):
        logger.info("=== Epoch %d/%d ===", epoch, epochs)
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, cfg.training.lambda_grad)
        logger.info("Train loss: %.4f", train_loss)

        metrics = evaluate(model, val_loader, device)
        for k, v in metrics.items():
            logger.info("Val %s: %.4f", k, v)

        if metrics["abs_rel"] < best_abs_rel:
            best_abs_rel = metrics["abs_rel"]
            ckpt_dir = Path(cfg.training.checkpoint_dir) / "best"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            (ckpt_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
            logger.info(
                "New best abs_rel=%.4f — checkpoint saved to %s",
                best_abs_rel, ckpt_dir,
            )


if __name__ == "__main__":
    main()
