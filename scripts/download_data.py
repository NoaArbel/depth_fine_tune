"""
Download AirSim depth dataset from Kaggle.

Run with defaults:
    python scripts/download_data.py

Requires: kaggle API credentials (kagglehub will prompt on first run)
"""
import random
import shutil
from pathlib import Path

import kagglehub

DATASET = "lukpellant/droneflight-obs-avoidanceairsimrgbdepth10k-320x320"
RAW_DIR = Path("data/raw")
SUBSET_N = 100  # set to None to download the full dataset
SEED = 42


def download_subset(dataset: str, output_dir: Path, n: int, seed: int) -> None:
    print("Downloading full dataset to kagglehub cache...")
    cache_path = Path(kagglehub.dataset_download(dataset))

    # Locate the inner dataset folder (one level below the cache root)
    subdirs = [d for d in cache_path.iterdir() if d.is_dir()]
    dataset_root = subdirs[0] if len(subdirs) == 1 else cache_path

    rgb_dir = dataset_root / "rgb"
    depth_dir = dataset_root / "depth"
    cmd_dir = dataset_root / "commands"

    rgb_ids = {f.stem for f in rgb_dir.iterdir() if f.is_file()}
    depth_ids = {f.stem for f in depth_dir.iterdir() if f.is_file()}
    cmd_ids = {f.stem for f in cmd_dir.iterdir() if f.is_file()}

    common_ids = sorted(rgb_ids & depth_ids & cmd_ids)
    print(f"Matched triplets available: {len(common_ids)}")

    random.seed(seed)
    selected_ids = random.sample(common_ids, min(n, len(common_ids)))

    for split, src_dir in [("rgb", rgb_dir), ("depth", depth_dir), ("commands", cmd_dir)]:
        dst_dir = output_dir / split
        dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Copying {len(selected_ids)} triplets → {output_dir}")
    for stem in selected_ids:
        for split, src_dir in [("rgb", rgb_dir), ("depth", depth_dir), ("commands", cmd_dir)]:
            candidates = list(src_dir.glob(f"{stem}.*"))
            if candidates:
                src = candidates[0]
                shutil.copy2(src, output_dir / split / src.name)


def download_full(dataset: str, output_dir: Path) -> None:
    print("Downloading full dataset to kagglehub cache...")
    cache_path = Path(kagglehub.dataset_download(dataset))

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Copying full dataset → {output_dir}")
    shutil.copytree(cache_path, output_dir, dirs_exist_ok=True)


def main() -> None:
    if SUBSET_N is not None:
        download_subset(DATASET, RAW_DIR, SUBSET_N, SEED)
    else:
        download_full(DATASET, RAW_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
