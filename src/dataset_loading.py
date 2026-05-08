from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DatasetFiles:
    """Discovered files and calibration for a dataset layout."""

    name: str
    image_paths: list[Path]
    intrinsics: np.ndarray | None
    ground_truth: np.ndarray | None = None


KITTI_K = np.array(
    [
        [7.18856e02, 0.0, 6.071928e02],
        [0.0, 7.18856e02, 1.852157e02],
        [0.0, 0.0, 1.0],
    ]
)

MALAGA_K = np.array(
    [
        [621.18428, 0.0, 404.0076],
        [0.0, 621.18428, 309.05989],
        [0.0, 0.0, 1.0],
    ]
)

CUSTOM_K = np.array(
    [
        [1.05903465e03, 0.0, 6.29060709e02],
        [0.0, 1.06306400e03, 3.28563696e02],
        [0.0, 0.0, 1.0],
    ]
)


def list_images(directory: str | Path, pattern: str = "*.png", stride: int = 1) -> list[Path]:
    """Return sorted image paths from a directory, optionally keeping every nth image."""
    if stride < 1:
        raise ValueError("stride must be a positive integer")

    paths = sorted(Path(directory).glob(pattern))
    return paths[0::stride]


def load_kitti(root: str | Path = "kitti/kitti05/kitti") -> DatasetFiles:
    root = Path(root)
    image_paths = list_images(root / "05" / "image_0", "*.png")
    pose_path = root / "poses" / "05.txt"
    ground_truth = None
    if pose_path.exists():
        poses = np.loadtxt(pose_path)
        ground_truth = poses[:, [-9, -1]]

    return DatasetFiles("KITTI 05", image_paths, KITTI_K, ground_truth)


def load_malaga(root: str | Path = "malaga/malaga-urban-dataset-extract-07") -> DatasetFiles:
    root = Path(root)
    image_dir = root / "malaga-urban-dataset-extract-07_rectified_800x600_Images"
    return DatasetFiles("Malaga 07", list_images(image_dir, "*.jpg", stride=2), MALAGA_K)


def load_parking(root: str | Path = "parking/parking") -> DatasetFiles:
    root = Path(root)
    image_paths = list_images(root / "images", "*.png")
    intrinsics = None
    ground_truth = None

    k_path = root / "K.txt"
    if k_path.exists():
        intrinsics = np.loadtxt(k_path, delimiter=",", usecols=(0, 1, 2))

    pose_path = root / "poses.txt"
    if pose_path.exists():
        poses = np.loadtxt(pose_path)
        ground_truth = poses[:, [-9, -1]]

    return DatasetFiles("Parking", image_paths, intrinsics, ground_truth)


def load_custom(root: str | Path = "VAMR_Rome_dataset/VAMR_Rome_dataset") -> DatasetFiles:
    root = Path(root)
    return DatasetFiles("Custom Rome", list_images(root / "images", "*.png"), CUSTOM_K)
