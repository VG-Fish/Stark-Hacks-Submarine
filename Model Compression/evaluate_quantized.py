"""
Evaluate all quantized TFLite NanoCrackSeg models (1-bit, 2-bit, 4-bit, 8-bit)
and generate comparison plots.

Dependencies:
    pip install tensorflow numpy tqdm torch matplotlib
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

TEACHER_PATH = Path(__file__).parent.parent / "UNet Teacher Model"
sys.path.insert(0, str(TEACHER_PATH))

try:
    from prepare_datasets import prepare_datasets
except ImportError:
    print(f"Error: Ensure prepare_datasets.py exists at {TEACHER_PATH}")
    sys.exit(1)


BIT_WIDTHS = [1, 2, 4, 8]


def tflite_filename(num_bits):
    if num_bits == 8:
        return "nano_crack_seg_int8.tflite"
    return f"nano_crack_seg_{num_bits}bit.tflite"


class TFLiteModel:
    """Wrapper around TFLite interpreter for batch inference."""

    def __init__(self, tflite_path: str):
        import tensorflow as tf

        # Disable XNNPACK — it fails on aggressively quantized (1/2-bit) models
        self.interpreter = tf.lite.Interpreter(
            model_path=tflite_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        self.input_scale = self.input_details["quantization_parameters"]["scales"][0]
        self.input_zero_point = self.input_details["quantization_parameters"]["zero_points"][0]
        self.output_scale = self.output_details["quantization_parameters"]["scales"][0]
        self.output_zero_point = self.output_details["quantization_parameters"]["zero_points"][0]

    def predict(self, image_np: np.ndarray) -> np.ndarray:
        """Run inference on a single image (H, W, C) float32 -> logits float32."""
        input_data = image_np / self.input_scale + self.input_zero_point
        input_data = np.clip(input_data, -128, 127).astype(np.int8)
        input_data = np.expand_dims(input_data, axis=0)

        self.interpreter.set_tensor(self.input_details["index"], input_data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details["index"])
        logits = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale
        return logits


def evaluate(tflite_model: TFLiteModel, loader: DataLoader, split_name: str) -> dict:
    """Compute segmentation metrics for a given split."""
    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_loss = 0.0
    n = 0

    for images, masks in tqdm(loader, desc=f"  {split_name}", leave=False):
        for i in range(images.size(0)):
            img_np = images[i].numpy().transpose(1, 2, 0)
            mask_np = masks[i].numpy().transpose(1, 2, 0)

            logits = tflite_model.predict(img_np).squeeze(0)

            sigmoid_logits = 1.0 / (1.0 + np.exp(-logits))
            loss = -np.mean(
                mask_np * np.log(sigmoid_logits + 1e-8)
                + (1 - mask_np) * np.log(1 - sigmoid_logits + 1e-8)
            )

            preds = (sigmoid_logits > 0.5).astype(np.float32)

            tp = (preds * mask_np).sum()
            fp = (preds * (1 - mask_np)).sum()
            fn = ((1 - preds) * mask_np).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)
            dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

            total_loss += loss
            total_iou += iou
            total_dice += dice
            total_precision += precision
            total_recall += recall
            n += 1

    return {
        "loss": float(total_loss / n),
        "iou": float(total_iou / n),
        "dice": float(total_dice / n),
        "precision": float(total_precision / n),
        "recall": float(total_recall / n),
    }


def get_model_statistics(tflite_path: Path) -> dict:
    """Calculate TFLite model size and tensor info."""
    import tensorflow as tf

    file_size_kb = tflite_path.stat().st_size / 1024

    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_path),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()

    total_params = 0
    for detail in interpreter.get_tensor_details():
        total_params += np.prod(detail["shape"])

    return {
        "file_size_kb": float(file_size_kb),
        "total_params": int(total_params),
    }


def print_metrics(split_name: str, metrics: dict) -> None:
    print(f"    {split_name:>5}: IoU={metrics['iou']:.4f}  Dice={metrics['dice']:.4f}  "
          f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}  Loss={metrics['loss']:.4f}")


# ── Plotting ─────────────────────────────────────────────────────────────


def plot_metrics_comparison(all_results, output_dir):
    """Bar chart of segmentation metrics (test set) across bit widths."""
    bit_widths = sorted(all_results.keys())
    metric_names = ["iou", "dice", "precision", "recall"]
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(bit_widths))
    width = 0.18

    for i, (metric, color) in enumerate(zip(metric_names, colors)):
        values = [all_results[b]["metrics"]["test"][metric] for b in bit_widths]
        bars = ax.bar(x + i * width, values, width, label=metric.upper(), color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Quantization Bit Width", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Segmentation Metrics vs. Quantization Level (Test Set)", fontsize=13)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{b}-bit" for b in bit_widths])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "metrics_comparison.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_iou_vs_size(all_results, output_dir):
    """Dual-axis plot: IoU (line) and model size (bars) vs bit width."""
    bit_widths = sorted(all_results.keys())
    ious = [all_results[b]["metrics"]["test"]["iou"] for b in bit_widths]
    sizes_kb = [all_results[b]["stats"]["file_size_kb"] for b in bit_widths]
    labels = [f"{b}-bit" for b in bit_widths]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    bar_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]
    bars = ax1.bar(labels, sizes_kb, color=bar_colors, alpha=0.7, label="Model Size")
    ax1.set_xlabel("Quantization Bit Width", fontsize=12)
    ax1.set_ylabel("TFLite File Size (KB)", fontsize=12, color="#555")
    ax1.tick_params(axis="y", labelcolor="#555")
    for bar, sz in zip(bars, sizes_kb):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{sz:.1f}", ha="center", va="bottom", fontsize=9, color="#555")

    ax2 = ax1.twinx()
    ax2.plot(labels, ious, "o-", color="#2c3e50", linewidth=2.5, markersize=8, label="IoU")
    ax2.set_ylabel("Test IoU", fontsize=12, color="#2c3e50")
    ax2.tick_params(axis="y", labelcolor="#2c3e50")
    for i, (label, iou) in enumerate(zip(labels, ious)):
        ax2.annotate(f"{iou:.4f}", (i, iou), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color="#2c3e50")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.grid(axis="y", alpha=0.2)
    plt.title("IoU vs. Model Size Across Quantization Levels", fontsize=13)
    plt.tight_layout()
    path = output_dir / "iou_vs_size.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_accuracy_degradation(all_results, output_dir):
    """Line plot showing how each metric degrades from 8-bit baseline."""
    bit_widths = sorted(all_results.keys())
    metric_names = ["iou", "dice", "precision", "recall"]
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    baseline = all_results[8]["metrics"]["test"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, color in zip(metric_names, colors):
        baseline_val = baseline[metric]
        if baseline_val > 0:
            pct = [
                (all_results[b]["metrics"]["test"][metric] / baseline_val) * 100
                for b in bit_widths
            ]
        else:
            pct = [0] * len(bit_widths)
        ax.plot([f"{b}-bit" for b in bit_widths], pct, "o-", color=color,
                linewidth=2, markersize=7, label=metric.upper())

    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="8-bit baseline")
    ax.set_xlabel("Quantization Bit Width", fontsize=12)
    ax.set_ylabel("% of 8-bit Baseline", fontsize=12)
    ax.set_title("Accuracy Retention vs. Quantization Level (Test Set)", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / "accuracy_degradation.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    model_dir = Path("results/student")
    dataset_dir = Path("SubDataset")

    # Check all models exist
    missing = []
    for num_bits in BIT_WIDTHS:
        p = model_dir / tflite_filename(num_bits)
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("Error: Missing TFLite models. Run convert_to_tflite.py first.")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    # Load dataset once
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        dataset_dir, target_size=(112, 112)
    )
    splits = {
        "train": DataLoader(train_dataset, batch_size=32, shuffle=False),
        "val": DataLoader(val_dataset, batch_size=32, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=32, shuffle=False),
    }
    print(
        f"Dataset sizes — Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Evaluate each bit width
    all_results = {}
    for num_bits in BIT_WIDTHS:
        model_path = model_dir / tflite_filename(num_bits)

        print(f"\n{'=' * 55}")
        print(f"  EVALUATING {num_bits}-BIT MODEL: {model_path.name}")
        print(f"{'=' * 55}")

        tflite_model = TFLiteModel(str(model_path))
        stats = get_model_statistics(model_path)
        print(f"  File size: {stats['file_size_kb']:.2f} KB  |  Params: {stats['total_params']:,}")

        metrics_by_split = {}
        for split_name, loader in splits.items():
            metrics = evaluate(tflite_model, loader, split_name)
            print_metrics(split_name, metrics)
            metrics_by_split[split_name] = metrics

        all_results[num_bits] = {
            "stats": stats,
            "metrics": metrics_by_split,
        }

    # Summary table
    print(f"\n{'=' * 70}")
    print("  SUMMARY — TEST SET RESULTS")
    print(f"{'=' * 70}")
    print(f"  {'Bits':>4}  {'Size (KB)':>9}  {'IoU':>7}  {'Dice':>7}  {'Prec':>7}  {'Recall':>7}  {'Loss':>7}")
    print(f"  {'—' * 62}")
    for num_bits in BIT_WIDTHS:
        r = all_results[num_bits]
        m = r["metrics"]["test"]
        print(
            f"  {num_bits:>4}  {r['stats']['file_size_kb']:>9.2f}  "
            f"{m['iou']:>7.4f}  {m['dice']:>7.4f}  "
            f"{m['precision']:>7.4f}  {m['recall']:>7.4f}  {m['loss']:>7.4f}"
        )
    print(f"{'=' * 70}")

    # Save results JSON
    results_path = model_dir / "quantization_results.json"
    with open(str(results_path), "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_metrics_comparison(all_results, model_dir)
    plot_iou_vs_size(all_results, model_dir)
    plot_accuracy_degradation(all_results, model_dir)

    print("\nEvaluation complete!")
