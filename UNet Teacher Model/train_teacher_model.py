from pathlib import Path

import matplotlib.pyplot as plt
import torch
from prepare_datasets import CrackDataset, prepare_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet_model import UNet, dice_focal_loss


def plot_metrics(
    train_losses: list[float],
    val_losses: list[float],
    val_f1s: list[float],
    val_ious: list[float],
    val_precisions: list[float],
    val_recalls: list[float],
    val_accuracies: list[float],
    save_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs: range = range(1, len(train_losses) + 1)

    axes[0, 0].plot(epochs, train_losses, label="Train Loss", color="steelblue")
    axes[0, 0].plot(epochs, val_losses, label="Val Loss", color="tomato")
    axes[0, 0].set_title("Model Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, val_f1s, label="F1 (Dice)", color="mediumseagreen")
    axes[0, 1].plot(epochs, val_ious, label="IoU", color="darkorange")
    axes[0, 1].set_title("F1 & IoU")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, val_precisions, label="Precision", color="royalblue")
    axes[1, 0].plot(epochs, val_recalls, label="Recall", color="crimson")
    axes[1, 0].set_title("Precision & Recall")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(
        epochs, val_accuracies, label="Pixel Accuracy", color="mediumpurple"
    )
    axes[1, 1].set_title("Pixel Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "training_metrics.png", dpi=150)
    plt.close(fig)


def save_worst_predictions(worst_samples: list, save_dir: Path) -> None:
    """Plots the top 5 worst predictions based on IoU score."""
    if not worst_samples:
        return

    fig, axes = plt.subplots(
        len(worst_samples), 3, figsize=(10, 3 * len(worst_samples))
    )

    # Handle case where there's only 1 sample
    if len(worst_samples) == 1:
        axes = [axes]

    for i, (score, img, mask, pred) in enumerate(worst_samples):
        # Move tensors to CPU and convert to numpy for plotting
        img_np = img.cpu().numpy().squeeze()
        mask_np = mask.cpu().numpy().squeeze()
        pred_np = pred.cpu().numpy().squeeze()

        axes[i][0].imshow(img_np, cmap="gray")
        axes[i][0].set_title(f"Image (IoU: {score:.4f})")
        axes[i][0].axis("off")

        axes[i][1].imshow(mask_np, cmap="gray")
        axes[i][1].set_title("True Mask")
        axes[i][1].axis("off")

        axes[i][2].imshow(pred_np, cmap="gray")
        axes[i][2].set_title("Predicted Mask")
        axes[i][2].axis("off")

    plt.tight_layout()
    # Overwrite the same image each epoch to save disk space
    plt.savefig(save_dir / "worst_validation_errors.png", dpi=150)
    plt.close(fig)


def save_best_predictions(best_samples: list, save_dir: Path) -> None:
    """Plots the top 5 best predictions based on IoU score."""
    if not best_samples:
        return

    fig, axes = plt.subplots(len(best_samples), 3, figsize=(10, 3 * len(best_samples)))

    # Handle case where there's only 1 sample
    if len(best_samples) == 1:
        axes = [axes]

    for i, (score, img, mask, pred) in enumerate(best_samples):
        img_np = img.cpu().numpy().squeeze()
        mask_np = mask.cpu().numpy().squeeze()
        pred_np = pred.cpu().numpy().squeeze()

        axes[i][0].imshow(img_np, cmap="gray")
        axes[i][0].set_title(f"Image (IoU: {score:.4f})")
        axes[i][0].axis("off")

        axes[i][1].imshow(mask_np, cmap="gray")
        axes[i][1].set_title("True Mask")
        axes[i][1].axis("off")

        axes[i][2].imshow(pred_np, cmap="gray")
        axes[i][2].set_title("Predicted Mask")
        axes[i][2].axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / "best_validation_predictions.png", dpi=150)
    plt.close(fig)


def create_dataloader(
    dataset: CrackDataset,
    batch_size: int,
    device: torch.device,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        prefetch_factor=2,
    )


def train_teacher(
    dataset_dir: Path,
    save_dir: Path,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
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

    train_dataset, val_dataset, _ = prepare_datasets(dataset_dir, target_size=(96, 96))

    train_loader: DataLoader = create_dataloader(train_dataset, batch_size, device)
    val_loader: DataLoader = create_dataloader(
        val_dataset, batch_size, device, shuffle=False
    )

    model: UNet = UNet(in_channels=1, base_filters=48).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    best_val_f1: float = 0.0
    patience_counter: int = 0

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_f1s: list[float] = []
    val_ious: list[float] = []
    val_precisions: list[float] = []
    val_recalls: list[float] = []
    val_accuracies: list[float] = []

    for epoch in tqdm(range(epochs), desc="Model Training Time"):
        model.train()
        train_loss: float = 0.0

        for images, masks in tqdm(train_loader, desc="Training"):
            images, masks = (
                images.to(device, non_blocking=True),
                masks.to(device, non_blocking=True),
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                predictions = model(images)
                loss = dice_focal_loss(predictions, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss: float = 0.0
        total_tp: float = 0.0
        total_fp: float = 0.0
        total_fn: float = 0.0
        total_correct: float = 0.0
        total_pixels: float = 0.0

        worst_samples: list = []
        best_samples: list = []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating"):
                images, masks = (
                    images.to(device, non_blocking=True),
                    masks.to(device, non_blocking=True),
                )

                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    predictions = model(images)
                    val_loss += dice_focal_loss(predictions, masks).item()

                predictions_bin: torch.Tensor = (predictions > 0.5).float()

                for j in range(images.size(0)):
                    img_single = images[j]
                    mask_single = masks[j]
                    pred_single = predictions_bin[j]

                    intersection = (pred_single * mask_single).sum().item()
                    union = (
                        pred_single.sum().item()
                        + mask_single.sum().item()
                        - intersection
                    )

                    iou_img = intersection / (union + 1e-8)

                    worst_samples.append(
                        (
                            iou_img,
                            img_single.detach(),
                            mask_single.detach(),
                            pred_single.detach(),
                        )
                    )

                    worst_samples = sorted(worst_samples, key=lambda x: x[0])[:5]

                    best_samples.append(
                        (
                            iou_img,
                            img_single.detach(),
                            mask_single.detach(),
                            pred_single.detach(),
                        )
                    )

                    best_samples = sorted(
                        best_samples, key=lambda x: x[0], reverse=True
                    )[:5]

                total_tp += (predictions_bin * masks).sum().item()
                total_fp += (predictions_bin * (1 - masks)).sum().item()
                total_fn += ((1 - predictions_bin) * masks).sum().item()
                total_correct += (predictions_bin == masks).float().sum().item()
                total_pixels += masks.numel()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        precision: float = total_tp / (total_tp + total_fp + 1e-8)
        recall: float = total_tp / (total_tp + total_fn + 1e-8)
        val_f1: float = 2 * precision * recall / (precision + recall + 1e-8)
        iou: float = total_tp / (total_tp + total_fp + total_fn + 1e-8)
        accuracy: float = total_correct / (total_pixels + 1e-8)

        scheduler.step(epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        val_ious.append(iou)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_accuracies.append(accuracy)

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"F1: {val_f1:.4f} | "
            f"IoU: {iou:.4f} | "
            f"Prec: {precision:.4f} | "
            f"Rec: {recall:.4f} | "
            f"Acc: {accuracy:.4f}"
        )

        # Early Stopping Logic (based on F1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
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

        plot_metrics(
            train_losses,
            val_losses,
            val_f1s,
            val_ious,
            val_precisions,
            val_recalls,
            val_accuracies,
            save_dir,
        )

        save_worst_predictions(worst_samples, save_dir)
        save_best_predictions(best_samples, save_dir)

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def main() -> None:
    train_teacher(
        dataset_dir=Path("SubDataset"),
        save_dir=Path("results/teacher"),
        patience=15,
    )


if __name__ == "__main__":
    main()
