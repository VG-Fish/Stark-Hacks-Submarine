from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
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

        spatial: list = [
            A.RandomCrop(*target_size),
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
        ]
        pixel_only: list = (
            [
                A.RandomBrightnessContrast(
                    p=0.2, brightness_limit=0.2, contrast_limit=0.15
                ),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
                A.CLAHE(p=0.3),
            ]
            if augment
            else []
        )

        self.transform: A.Compose = A.Compose(
            spatial + pixel_only, additional_targets={"mask": "mask"}
        )

    def __len__(self: "CrackDataset") -> int:
        return len(self.image_paths)

    def __getitem__(
        self: "CrackDataset", idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image: np.ndarray = np.array(
            Image.open(self.image_paths[idx]).convert("L")
        )  # grayscale
        mask: np.ndarray = np.array(Image.open(self.mask_paths[idx]).convert("L"))

        result: dict[str, np.ndarray] = self.transform(image=image, mask=mask)

        # Convert to tensors
        torch_image: torch.Tensor = (
            torch.tensor(result["image"], dtype=torch.float) / 255.0  # pyright: ignore[reportPrivateImportUsage]
        )
        torch_mask: torch.Tensor = (
            torch.tensor(result["mask"], dtype=torch.float) / 255.0  # pyright: ignore[reportPrivateImportUsage]
        )

        torch_image = torch_image.unsqueeze(0)
        torch_mask = torch_mask.unsqueeze(0)

        return torch_image, torch_mask


def get_all_file_paths(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.*"))


def prepare_datasets(
    dataset_directory: Path, target_size: tuple[int, int]
) -> tuple[CrackDataset, CrackDataset]:
    train_dataset: Path = dataset_directory / "train"
    train_image_paths: list[Path] = get_all_file_paths(train_dataset / "images")
    train_masks_paths: list[Path] = get_all_file_paths(train_dataset / "masks")

    assert len(train_image_paths) == len(train_masks_paths), (
        f"Train mismatch: {len(train_image_paths)} images vs {len(train_masks_paths)} masks"
    )

    test_dataset: Path = dataset_directory / "test"
    test_image_paths: list[Path] = get_all_file_paths(test_dataset / "images")
    test_masks_paths: list[Path] = get_all_file_paths(test_dataset / "masks")

    assert len(test_image_paths) == len(test_masks_paths), (
        f"Test mismatch: {len(test_image_paths)} images vs {len(test_masks_paths)} masks"
    )

    return CrackDataset(
        train_image_paths, train_masks_paths, target_size, augment=True
    ), CrackDataset(test_image_paths, test_masks_paths, target_size, augment=False)
