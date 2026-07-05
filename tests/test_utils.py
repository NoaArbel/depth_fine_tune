import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from src.utils import visualize_depth_predictions


class _FakeDataset:
    def __init__(self, rgb_dir: Path, stems: list[str]):
        self.rgb_dir = rgb_dir
        self.stems = stems

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        return {
            "pixel_values": torch.zeros(3, 16, 16),
            "depth": torch.from_numpy(np.linspace(1.0, 5.0, 64).reshape(8, 8).astype("float32")),
        }


@pytest.fixture()
def fake_rgb_dir(tmp_path):
    for stem in ["img_0", "img_1", "img_2"]:
        img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        img.save(tmp_path / f"{stem}.png")
    return tmp_path


def test_visualize_depth_predictions_saves_files(fake_rgb_dir, tmp_path):
    stems = ["img_0", "img_1", "img_2"]
    dataset = _FakeDataset(fake_rgb_dir, stems)

    outputs = MagicMock()
    outputs.predicted_depth = torch.ones(1, 8, 8) * 3.0
    model = MagicMock(return_value=outputs)

    save_dir = tmp_path / "visual"
    visualize_depth_predictions(model, dataset, "cpu", save_dir, prefix="test_run", n=2)

    saved = list(save_dir.glob("*.png"))
    assert len(saved) == 2
    assert {f.name for f in saved} == {"test_run_img_0.png", "test_run_img_1.png"}
