import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from nano_crack_seg import NanoCrackSeg, rfkd_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

TEACHER_PATH = Path(__file__).parent.parent / "UNet Teacher Model"
sys.path.insert(0, str(TEACHER_PATH))

try:
    from prepare_datasets import prepare_datasets
    from unet_model import UNet
except ImportError:
    print(f"Error: Ensure UNet files exist at {TEACHER_PATH}")
    sys.exit(1)

_AP_THRESHOLDS: list[float] = [i / 10 for i in range(1, 10)]


class FeatureExtractor:
    """Uses PyTorch Hooks to grab intermediate feature maps without changing model code."""

    def __init__(self, model: nn.Module, layers: list[str]):
        self.features = {}
        self.hooks = []
        for name, module in model.named_modules():
            if name in layers:
                self.hooks.append(
                    module.register_forward_hook(self.save_outputs_hook(name))
                )

    def save_outputs_hook(self, layer_id: str):
        def fn(_, __, output):
            self.features[layer_id] = output

        return fn

    def clear(self):
        """Clears stored features to prevent memory leaks on M3 Unified Memory."""
        self.features.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()


class FeatureAdaptors(nn.Module):
    """1x1 Convolutions to scale Student channels up to Teacher channels."""

    def __init__(self):
        super().__init__()
        self.adapt_e1 = nn.Conv2d(8, 48, kernel_size=1)
        self.adapt_e2 = nn.Conv2d(16, 96, kernel_size=1)
        self.adapt_e3 = nn.Conv2d(32, 192, kernel_size=1)

    def forward(self, student_features: dict) -> dict:
        return {
            "enc1": self.adapt_e1(student_features["enc1"]),
            "enc2": self.adapt_e2(student_features["enc2"]),
            "enc3": self.adapt_e3(student_features["enc3"]),
        }


def compute_ap(pred_probs: torch.Tensor, gt_mask: torch.Tensor) -> float:
    precisions, recalls = [], []
    for thresh in _AP_THRESHOLDS:
        pred_bin = (pred_probs > thresh).float()
        tp = (pred_bin * gt_mask).sum().item()
        fp = (pred_bin * (1 - gt_mask)).sum().item()
        fn = ((1 - pred_bin) * gt_mask).sum().item()
        precisions.append(tp / (tp + fp + 1e-8))
        recalls.append(tp / (tp + fn + 1e-8))

    paired = sorted(zip(recalls, precisions))
    ap = 0.0
    for i in range(1, len(paired)):
        ap += (paired[i][0] - paired[i - 1][0]) * paired[i][1]
    return ap


def plot_metrics(history: dict, save_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0, 0].plot(
        epochs, history["train_loss"], label="Train Loss", color="steelblue"
    )
    axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss", color="tomato")
    axes[0, 0].set_title("Model Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history["val_f1"], label="F1", color="mediumseagreen")
    axes[0, 1].plot(epochs, history["val_iou"], label="IoU", color="darkorange")
    axes[0, 1].set_title("F1 & IoU Scores")
    axes[0, 1].legend()

    axes[0, 2].plot(epochs, history["val_prec"], label="Precision", color="royalblue")
    axes[0, 2].plot(epochs, history["val_rec"], label="Recall", color="crimson")
    axes[0, 2].set_title("Precision & Recall")
    axes[0, 2].legend()

    axes[1, 0].plot(epochs, history["val_acc"], color="mediumpurple")
    axes[1, 0].set_title("Pixel Accuracy")

    axes[1, 1].plot(epochs, history["val_map"], color="teal")
    axes[1, 1].set_title("mAP")

    axes[1, 2].axis("off")
    plt.tight_layout()
    plt.savefig(save_dir / "training_metrics.png", dpi=150)
    plt.close(fig)


def save_prediction_samples(samples: list, save_dir: Path, filename: str) -> None:
    if not samples:
        return
    fig, axes = plt.subplots(len(samples), 3, figsize=(10, 3 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    for i, (score, img, mask, pred) in enumerate(samples):
        axes[i][0].imshow(img.cpu().numpy().squeeze(), cmap="gray")
        axes[i][0].set_title(f"IoU: {score:.4f}")
        axes[i][1].imshow(mask.cpu().numpy().squeeze(), cmap="gray")
        axes[i][1].set_title("True Mask")
        axes[i][2].imshow(pred.cpu().numpy().squeeze(), cmap="gray")
        axes[i][2].set_title("Prediction")
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / filename, dpi=150)
    plt.close(fig)


def train_student(
    dataset_dir: Path,
    teacher_weights_path: Path,
    save_dir: Path,
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 20,
) -> NanoCrackSeg:
    save_dir.mkdir(parents=True, exist_ok=True)

    # Optimized for M3 Pro
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

    train_dataset, val_dataset, _ = prepare_datasets(
        dataset_dir, target_size=(104, 104)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    teacher = UNet(in_channels=1, base_filters=48).to(device).eval()
    teacher.load_state_dict(torch.load(teacher_weights_path, map_location=device))
    for p in teacher.parameters():
        p.requires_grad = False

    student = NanoCrackSeg().to(device)
    adaptors = FeatureAdaptors().to(device)

    t_hooks = FeatureExtractor(teacher, ["enc1", "enc2", "enc3"])
    s_hooks = FeatureExtractor(student, ["enc1", "enc2", "enc3"])

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(adaptors.parameters()),
        lr=lr,
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        k: []
        for k in [
            "train_loss",
            "val_loss",
            "val_f1",
            "val_iou",
            "val_prec",
            "val_rec",
            "val_acc",
            "val_map",
        ]
    }
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        student.train()
        adaptors.train()
        train_loss_accum = 0.0

        for images, masks in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False
        ):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            t_hooks.clear()
            s_hooks.clear()

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)
            adapted_features = adaptors(s_hooks.features)

            loss = rfkd_loss(
                student_logits,
                teacher_logits,
                adapted_features,
                t_hooks.features,
                masks,
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_accum += loss.item()

        # Validation
        student.eval()
        v_loss, t_tp, t_fp, t_fn, t_corr, t_pix, t_ap, t_img_cnt = (
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        worst_samples, best_samples = [], []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="[Validating]", leave=False):
                images, masks = images.to(device), masks.to(device)
                logits = student(images)
                v_loss += F.binary_cross_entropy_with_logits(logits, masks).item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                t_tp += (preds * masks).sum().item()
                t_fp += (preds * (1 - masks)).sum().item()
                t_fn += ((1 - preds) * masks).sum().item()
                t_corr += (preds == masks).sum().item()
                t_pix += masks.numel()

                for j in range(images.size(0)):
                    intersection = (preds[j] * masks[j]).sum().item()
                    union = (preds[j] + masks[j]).gt(0).sum().item()
                    iou_score = intersection / (union + 1e-8)

                    sample = (
                        iou_score,
                        images[j].detach(),
                        masks[j].detach(),
                        preds[j].detach(),
                    )
                    worst_samples = sorted(
                        worst_samples + [sample], key=lambda x: x[0]
                    )[:5]
                    best_samples = sorted(
                        best_samples + [sample], key=lambda x: x[0], reverse=True
                    )[:5]

                    t_ap += compute_ap(probs[j], masks[j])
                    t_img_cnt += 1

        # Log Metrics
        prec = t_tp / (t_tp + t_fp + 1e-8)
        rec = t_tp / (t_tp + t_fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        history["train_loss"].append(train_loss_accum / len(train_loader))
        history["val_loss"].append(v_loss / len(val_loader))
        history["val_f1"].append(f1)
        history["val_iou"].append(t_tp / (t_tp + t_fp + t_fn + 1e-8))
        history["val_prec"].append(prec)
        history["val_rec"].append(rec)
        history["val_acc"].append(t_corr / t_pix)
        history["val_map"].append(t_ap / t_img_cnt)

        print(
            f"Epoch {epoch + 1:03d} | Train Loss: {history['train_loss'][-1]:.4f} | F1: {f1:.4f} | IoU: {history['val_iou'][-1]:.4f}"
        )

        # Plots and Samples
        plot_metrics(history, save_dir)
        save_prediction_samples(worst_samples, save_dir, "worst_validation_errors.png")
        save_prediction_samples(
            best_samples, save_dir, "best_validation_predictions.png"
        )

        # Checkpoint and Early Stopping
        if f1 > best_val_f1:
            best_val_f1 = f1
            patience_counter = 0
            torch.save(student.state_dict(), save_dir / "student_nano_crack_seg.pth")
            print("  [Saved Best Model]")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

    return student


if __name__ == "__main__":
    train_student(
        dataset_dir=Path("SubDataset"),
        teacher_weights_path=Path("results/teacher/teacher_unet.pth"),
        save_dir=Path("results/student"),
    )
