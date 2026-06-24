import json
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
from src.utils import align_scale_shift

LAMBDA_GRAD = 0.1

def silog_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    ''' Sigmoid Logarithmic Error Loss '''
    mask = gt > 0
    d = torch.log(pred[mask].clamp(min=1e-6)) - torch.log(gt[mask].clamp(min=1e-6))
    return d.pow(2).mean() - 0.5 * d.mean().pow(2)

def gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    ''' Gradient Loss '''
    # compute only for x.y directions
    mask = gt > 0
    grad_pred_x = torch.gradient(pred[mask], dim=2)
    grad_pred_y = torch.gradient(pred[mask], dim=3)
    grad_gt_x = torch.gradient(gt[mask], dim=2)
    grad_gt_y = torch.gradient(gt[mask], dim=3)
    return torch.mean((grad_pred_x - grad_gt_x).pow(2)) + torch.mean((grad_pred_y - grad_gt_y).pow(2))


def train_one_epoch(model, loader: DataLoader, optimizer, scheduler, device: str) -> float:
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

        loss = silog_loss(pred_resized, gt) + LAMBDA_GRAD * gradient_loss(pred_resized, gt)
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

        outputs = model(pixel_values=pixel_values)
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
    print(f"\n{'='*60}\nExperiment: {name}\n{'='*60}")

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
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        print(f"Epoch {epoch}/{cfg.training.epochs}  loss={train_loss:.4f}  abs_rel={val_metrics['abs_rel']:.4f}  delta1={val_metrics['delta1']:.4f}")

    ckpt_dir = project_root / cfg.training.checkpoint_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)

    test_metrics = evaluate(model, test_loader, device)
    print(f"Test metrics: {test_metrics}")

    return {"name": name, "history": history, "test_metrics": test_metrics}


def _dict_to_ns(d: dict) -> SimpleNamespace:
    ''' Convert a dictionary to a namespace '''
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def main(config_path: str = "configs/config.yaml") -> None:
    with open(config_path) as f:
        cfg = _dict_to_ns(yaml.safe_load(f))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, processor = load_model_with_lora(cfg)
    model = model.to(device)

    rgb_dir = Path(cfg.data.raw_dir) / "rgb"
    depth_dir = Path(cfg.data.raw_dir) / "depth"
    train_ds, val_ds = make_splits(rgb_dir, depth_dir, processor, train_ratio=0.8, val_ratio=0.2, seed=cfg.data.seed)
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

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
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f}")

        metrics = evaluate(model, val_loader, device)
        print("Val metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        if metrics["abs_rel"] < best_abs_rel:
            best_abs_rel = metrics["abs_rel"]
            ckpt_dir = Path(cfg.training.checkpoint_dir) / "best"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            (ckpt_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
            print(f"New best (abs_rel={best_abs_rel:.4f}) — checkpoint saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
