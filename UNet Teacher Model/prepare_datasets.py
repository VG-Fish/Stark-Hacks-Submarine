from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    def __init__(
        self: "CrackDataset",
        image_paths: list[Path],
        mask_paths: list[Path],
        target_size: tuple[int, int],
        augment: bool = True,
    ) -> None:
        self.image_paths: list[Path] = image_paths
        self.mask_paths: list[Path] = mask_paths

        if augment:
            transforms: list = [
                A.Resize(*target_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    translate_percent=0.05,
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.RandomBrightnessContrast(
                    p=0.2, brightness_limit=0.2, contrast_limit=0.15
                ),
                # A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
                # A.CLAHE(p=0.3),
                # A.ElasticTransform(alpha=80, sigma=10, p=0.3),
            ]
        else:
            transforms = [
                A.Resize(*target_size),
            ]

        self.transform: A.Compose = A.Compose(
            transforms, additional_targets={"mask": "mask"}
        )

    def __len__(self: "CrackDataset") -> int:
        return len(self.image_paths)

    def __getitem__(
        self: "CrackDataset", idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        result: dict[str, np.ndarray] = self.transform(image=image, mask=mask)

        torch_image: torch.Tensor = (
            torch.tensor(result["image"], dtype=torch.float32) / 255.0
        )
        torch_mask: torch.Tensor = (
            torch.tensor(result["mask"], dtype=torch.float32) / 255.0 > 0.5
        ).float()

        torch_image = torch_image.unsqueeze(0)
        torch_mask = torch_mask.unsqueeze(0)

        return torch_image, torch_mask


def get_all_file_paths(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.*"))


def prepare_datasets(
    dataset_directory: Path, target_size: tuple[int, int]
) -> tuple[CrackDataset, CrackDataset, CrackDataset]:
    train_dataset: Path = dataset_directory / "train"
    train_image_paths: list[Path] = get_all_file_paths(train_dataset / "images")
    train_masks_paths: list[Path] = get_all_file_paths(train_dataset / "masks")

    assert len(train_image_paths) == len(train_masks_paths), (
        f"Train mismatch: {len(train_image_paths)} images vs {len(train_masks_paths)} masks"
    )

    val_dataset: Path = dataset_directory / "val"
    val_image_paths: list[Path] = get_all_file_paths(val_dataset / "images")
    val_masks_paths: list[Path] = get_all_file_paths(val_dataset / "masks")

    assert len(val_image_paths) == len(val_masks_paths), (
        f"Val mismatch: {len(val_image_paths)} images vs {len(val_masks_paths)} masks"
    )

    test_dataset: Path = dataset_directory / "test"
    test_image_paths: list[Path] = get_all_file_paths(test_dataset / "images")
    test_masks_paths: list[Path] = get_all_file_paths(test_dataset / "masks")

    assert len(test_image_paths) == len(test_masks_paths), (
        f"Test mismatch: {len(test_image_paths)} images vs {len(test_masks_paths)} masks"
    )

    return (
        CrackDataset(train_image_paths, train_masks_paths, target_size, augment=True),
        CrackDataset(val_image_paths, val_masks_paths, target_size, augment=False),
        CrackDataset(test_image_paths, test_masks_paths, target_size, augment=False),
    )
