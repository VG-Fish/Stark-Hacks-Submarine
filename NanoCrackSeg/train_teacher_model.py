from pathlib import Path

import matplotlib.pyplot as plt
import torch
from prepare_datasets import prepare_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet_model import UNet, dice_bce_loss


def plot_metrics(
    train_losses: list[float],
    val_losses: list[float],
    val_f1s: list[float],
    save_dir: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs: range = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train Loss", color="steelblue")
    ax1.plot(epochs, val_losses, label="Val Loss", color="tomato")
    ax1.set_title("Model Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_f1s, label="Val F1", color="mediumseagreen")
    ax2.set_title("Validation F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "training_metrics.png", dpi=150)
    plt.close(fig)


def train_teacher(
    dataset_dir: Path,
    save_dir: Path,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 15,
) -> UNet:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path: Path = save_dir / "teacher_unet.pth"

    device: torch.device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Training on: {device}")

    train_dataset, val_dataset = prepare_datasets(dataset_dir, target_size=(96, 96))

    use_pin_memory = device.type == "cuda"
    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=use_pin_memory,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=use_pin_memory,
        persistent_workers=True,
        prefetch_factor=2,
    )

    model: UNet = UNet(in_channels=1, base_filters=64).to(device)
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss: float = float("inf")
    patience_counter: int = 0

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_f1s: list[float] = []

    for epoch in tqdm(range(epochs), desc="Model Training Time"):
        model.train()
        train_loss: float = 0.0

        for images, masks in train_loader:
            images, masks = (
                images.to(device, non_blocking=True),
                masks.to(device, non_blocking=True),
            )

            optimizer.zero_grad(set_to_none=True)

            predictions: torch.Tensor = model(images)
            loss: torch.Tensor = dice_bce_loss(predictions, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss: float = 0.0
        val_f1: float = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = (
                    images.to(device, non_blocking=True),
                    masks.to(device, non_blocking=True),
                )

                predictions = model(images)
                val_loss += dice_bce_loss(predictions, masks).item()

                predictions_bin: torch.Tensor = (predictions > 0.0).float()

                intersection: torch.Tensor = (predictions_bin * masks).sum()
                val_f1 += (
                    2 * intersection / (predictions_bin.sum() + masks.sum() + 1e-8)
                ).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_f1 /= len(val_loader)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model → {save_path}")
        else:
            patience_counter += 1
            print(f"  EarlyStopping counter: {patience_counter} out of {patience}")

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}! Training halted to save resources."
                )
                break

        plot_metrics(train_losses, val_losses, val_f1s, save_dir)

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def main() -> None:
    train_teacher(
        dataset_dir=Path("Dataset"),
        save_dir=Path("results/teacher"),
        patience=15,
    )


if __name__ == "__main__":
    main()
