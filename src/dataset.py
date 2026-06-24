import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DepthDataset(Dataset):
    def __init__(self, stems: list[str], rgb_dir: Path, depth_dir: Path, processor):
        self.stems = stems
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.processor = processor

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        image = Image.open(self.rgb_dir / f"{stem}.png").convert("RGB")
        depth = np.load(self.depth_dir / f"{stem}.npy").astype(np.float32)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (3, H, W)

        return {
            "pixel_values": pixel_values,
            "depth": torch.from_numpy(depth),
        }


def make_splits(
    rgb_dir: Path,
    depth_dir: Path,
    processor,
    split_csv: Path | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple["DepthDataset", "DepthDataset", "DepthDataset"]:
    if split_csv is not None and Path(split_csv).exists():
        train_stems, val_stems, test_stems = [], [], []
        buckets = {"train": train_stems, "val": val_stems, "test": test_stems}
        with open(split_csv) as f:
            for row in csv.DictReader(f):
                buckets[row["split"]].append(row["stem"])
    else:
        rgb_stems = {p.stem for p in rgb_dir.glob("*.png")}
        depth_stems = {p.stem for p in depth_dir.glob("*.npy")}
        stems = sorted(rgb_stems & depth_stems)

        rng = random.Random(seed)
        rng.shuffle(stems)

        # thre is oalways a ratio defined for test, val and train
        n_train = int(len(stems) * train_ratio)
        n_val = int(len(stems) * val_ratio)
        n_test = int(len(stems) * test_ratio)
        train_stems = stems[:n_train]
        val_stems = stems[n_train: n_train + n_val]
        test_stems = stems[n_train + n_val: n_train + n_val + n_test]

        if split_csv is not None:
            Path(split_csv).parent.mkdir(parents=True, exist_ok=True)
            with open(split_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["stem", "split"])
                writer.writeheader()
                writer.writerows({"stem": s, "split": "train"} for s in train_stems)
                writer.writerows({"stem": s, "split": "val"} for s in val_stems)
                writer.writerows({"stem": s, "split": "test"} for s in test_stems)

    return (
        DepthDataset(train_stems, rgb_dir, depth_dir, processor),
        DepthDataset(val_stems, rgb_dir, depth_dir, processor),
        DepthDataset(test_stems, rgb_dir, depth_dir, processor),
    )
