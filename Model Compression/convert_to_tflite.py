"""
Convert trained PyTorch NanoCrackSeg to TensorFlow Lite with multiple
quantization bit widths (1-bit, 2-bit, 4-bit, 8-bit).

Pipeline: PyTorch weights -> Keras model -> (optional N-bit weight quantization)
          -> TFLite full integer quantization

For sub-8-bit models, weights are uniformly quantized to N-bit precision
*before* TFLite conversion, simulating the accuracy impact of aggressive
quantization while keeping TFLite's INT8 runtime compatibility.

Dependencies:
    pip install tensorflow torch numpy
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

STUDENT_PATH = Path(__file__).parent.parent / "NanoCrackSeg"
sys.path.insert(0, str(STUDENT_PATH))

TEACHER_PATH = Path(__file__).parent.parent / "UNet Teacher Model"
sys.path.insert(0, str(TEACHER_PATH))

try:
    from nano_crack_seg import NanoCrackSeg
except ImportError:
    print(f"Error: Ensure NanoCrackSeg files exist at {STUDENT_PATH}")
    sys.exit(1)

try:
    from prepare_datasets import prepare_datasets
except ImportError:
    print(f"Error: Ensure prepare_datasets.py exists at {TEACHER_PATH}")
    sys.exit(1)


INPUT_SIZE = (112, 112)
NUM_CALIBRATION_SAMPLES = 200
BIT_WIDTHS = [1, 2, 4, 8]


def tflite_filename(num_bits):
    if num_bits == 8:
        return "nano_crack_seg_int8.tflite"
    return f"nano_crack_seg_{num_bits}bit.tflite"


# ── Keras model construction ────────────────────────────────────────────


def dw_conv_block(x, _, out_ch, name):
    """Depthwise separable conv + BatchNorm + ReLU (matches PyTorch DWConvBlock)."""
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3, padding="same", use_bias=False, name=f"{name}_dw"
    )(x)
    x = tf.keras.layers.Conv2D(
        out_ch, kernel_size=1, use_bias=False, name=f"{name}_pw"
    )(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f"{name}_bn")(x)
    x = tf.keras.layers.ReLU(name=f"{name}_relu")(x)
    return x


def build_keras_nanocrackseg():
    """Build the Keras equivalent of NanoCrackSeg (NHWC format)."""
    inp = tf.keras.layers.Input(shape=(*INPUT_SIZE, 1), name="input")

    # Encoder
    e1 = dw_conv_block(inp, 1, 8, "enc1")
    p1 = tf.keras.layers.MaxPool2D(2, name="pool1")(e1)

    e2 = dw_conv_block(p1, 8, 16, "enc2")
    p2 = tf.keras.layers.MaxPool2D(2, name="pool2")(e2)

    e3 = dw_conv_block(p2, 16, 32, "enc3")
    p3 = tf.keras.layers.MaxPool2D(2, name="pool3")(e3)

    # Bottleneck
    b = dw_conv_block(p3, 32, 64, "bottleneck")

    # Decoder with skip connections
    u3 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear", name="up3")(b)
    d3 = tf.keras.layers.Concatenate(name="cat3")([u3, e3])
    d3 = dw_conv_block(d3, 96, 32, "dec3")

    u2 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear", name="up2")(d3)
    d2 = tf.keras.layers.Concatenate(name="cat2")([u2, e2])
    d2 = dw_conv_block(d2, 48, 16, "dec2")

    u1 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear", name="up1")(d2)
    d1 = tf.keras.layers.Concatenate(name="cat1")([u1, e1])
    d1 = dw_conv_block(d1, 24, 8, "dec1")

    out = tf.keras.layers.Conv2D(1, kernel_size=1, name="output")(d1)

    return tf.keras.Model(inputs=inp, outputs=out, name="NanoCrackSeg")


# ── Weight transfer (PyTorch -> Keras) ───────────────────────────────────


def transfer_dw_conv_block(keras_model, block_name, pt_state, pt_prefix):
    """Transfer weights for one DWConvBlock from PyTorch to Keras.

    PyTorch layout:
      dw_conv.0 = DepthwiseConv2d  weight: (in_ch, 1, 3, 3)  [OIHW]
      dw_conv.1 = Conv2d (pointwise) weight: (out_ch, in_ch, 1, 1)
      dw_conv.2 = BatchNorm2d: weight, bias, running_mean, running_var

    Keras layout (NHWC):
      DepthwiseConv2D kernel: (3, 3, in_ch, 1)
      Conv2D kernel: (1, 1, in_ch, out_ch)
      BatchNormalization: gamma, beta, moving_mean, moving_variance
    """
    # Depthwise conv: PyTorch (C_in, 1, kH, kW) -> Keras (kH, kW, C_in, 1)
    dw_w = pt_state[f"{pt_prefix}.dw_conv.0.weight"].numpy()
    dw_w = np.transpose(dw_w, (2, 3, 0, 1))
    keras_model.get_layer(f"{block_name}_dw").set_weights([dw_w])

    # Pointwise conv: PyTorch (C_out, C_in, 1, 1) -> Keras (1, 1, C_in, C_out)
    pw_w = pt_state[f"{pt_prefix}.dw_conv.1.weight"].numpy()
    pw_w = np.transpose(pw_w, (2, 3, 1, 0))
    keras_model.get_layer(f"{block_name}_pw").set_weights([pw_w])

    # BatchNorm: gamma, beta, moving_mean, moving_var
    gamma = pt_state[f"{pt_prefix}.dw_conv.2.weight"].numpy()
    beta = pt_state[f"{pt_prefix}.dw_conv.2.bias"].numpy()
    mean = pt_state[f"{pt_prefix}.dw_conv.2.running_mean"].numpy()
    var = pt_state[f"{pt_prefix}.dw_conv.2.running_var"].numpy()
    keras_model.get_layer(f"{block_name}_bn").set_weights([gamma, beta, mean, var])


def transfer_all_weights(keras_model, pt_state):
    """Transfer all weights from PyTorch state dict to Keras model."""
    block_mapping = [
        ("enc1", "enc1"),
        ("enc2", "enc2"),
        ("enc3", "enc3"),
        ("bottleneck", "bottleneck"),
        ("dec3", "dec3"),
        ("dec2", "dec2"),
        ("dec1", "dec1"),
    ]

    for keras_name, pt_name in block_mapping:
        transfer_dw_conv_block(keras_model, keras_name, pt_state, pt_name)

    # Output conv: PyTorch (C_out, C_in, 1, 1) -> Keras (1, 1, C_in, C_out) + bias
    out_w = pt_state["output.weight"].numpy()
    out_w = np.transpose(out_w, (2, 3, 1, 0))
    out_b = pt_state["output.bias"].numpy()
    keras_model.get_layer("output").set_weights([out_w, out_b])


# ── N-bit weight quantization ───────────────────────────────────────────


def quantize_array(arr, num_bits):
    """Uniform quantization of a numpy array to num_bits precision.

    Maps the full range [min, max] to 2^num_bits discrete levels.
    """
    n_levels = 2 ** num_bits
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min < 1e-10:
        return arr
    scale = (arr_max - arr_min) / (n_levels - 1)
    return np.round((arr - arr_min) / scale) * scale + arr_min


def quantize_keras_weights(keras_model, num_bits):
    """Quantize all weights in a Keras model to num_bits precision."""
    for layer in keras_model.layers:
        weights = layer.get_weights()
        if weights:
            quantized = [quantize_array(w, num_bits) for w in weights]
            layer.set_weights(quantized)


# ── TFLite conversion ───────────────────────────────────────────────────


def convert_to_tflite(keras_model, tflite_path, dataset_dir, num_bits):
    """Convert Keras model to TFLite INT8 with N-bit effective weight precision.

    For num_bits < 8, weights are pre-quantized to N-bit before TFLite conversion.
    The resulting TFLite model uses INT8 runtime but effective weight precision is N bits.
    """
    if num_bits < 8:
        quantize_keras_weights(keras_model, num_bits)

    train_dataset, _, _ = prepare_datasets(dataset_dir, target_size=INPUT_SIZE)

    def representative_dataset():
        for i in range(min(NUM_CALIBRATION_SAMPLES, len(train_dataset))):
            img, _ = train_dataset[i]
            sample = img.numpy().transpose(1, 2, 0)
            sample = np.expand_dims(sample, axis=0).astype(np.float32)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(str(tflite_path), "wb") as f:
        f.write(tflite_model)

    return tflite_model


# ── Validation ───────────────────────────────────────────────────────────


def validate_conversion(pt_model, keras_model):
    """Verify Keras model produces the same output as PyTorch."""
    pt_model.eval()
    dummy_pt = torch.randn(1, 1, *INPUT_SIZE)

    with torch.no_grad():
        pt_out = pt_model(dummy_pt).numpy()

    dummy_tf = dummy_pt.numpy().transpose(0, 2, 3, 1)
    keras_out = keras_model.predict(dummy_tf, verbose=0)
    keras_out_nchw = keras_out.transpose(0, 3, 1, 2)

    max_diff = np.max(np.abs(pt_out - keras_out_nchw))
    mean_diff = np.mean(np.abs(pt_out - keras_out_nchw))
    print(f"  Validation — Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    if max_diff < 1e-4:
        print("  Weight transfer: PERFECT")
    elif max_diff < 1e-2:
        print("  Weight transfer: OK (minor floating point differences)")
    else:
        print("  WARNING: Large differences detected — check weight mapping!")


def count_weight_params(keras_model):
    """Count total trainable weight parameters."""
    total = 0
    for layer in keras_model.layers:
        for w in layer.get_weights():
            total += w.size
    return total


# ── Main ─────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    weights_path = Path("results/student/student_nano_crack_seg.pth")
    output_dir = Path("results/student")
    dataset_dir = Path("SubDataset")

    if not weights_path.exists():
        print(f"Error: Model weights not found at {weights_path}")
        sys.exit(1)

    # Load PyTorch model once
    print("Loading PyTorch model...")
    pt_model = NanoCrackSeg()
    pt_model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    pt_model.eval()

    pt_state = pt_model.state_dict()
    pth_size = weights_path.stat().st_size

    for num_bits in BIT_WIDTHS:
        print(f"\n{'=' * 55}")
        print(f"  CONVERTING TO {num_bits}-BIT QUANTIZATION")
        print(f"{'=' * 55}")

        tflite_path = output_dir / tflite_filename(num_bits)

        # Build fresh Keras model for each bit width (weights get modified)
        keras_model = build_keras_nanocrackseg()
        transfer_all_weights(keras_model, pt_state)

        # Validate weight transfer on the first (unquantized) model
        if num_bits == BIT_WIDTHS[-1]:
            validate_conversion(pt_model, keras_model)

        if num_bits < 8:
            print(f"  Pre-quantizing weights to {num_bits}-bit ({2**num_bits} levels)...")
            quantize_keras_weights(keras_model, num_bits)

        print(f"  Converting to TFLite INT8 (with calibration)...")
        tflite_model = convert_to_tflite(keras_model, tflite_path, dataset_dir, num_bits)
        print(f"  Saved: {tflite_path} ({len(tflite_model) / 1024:.2f} KB)")

    # Summary
    print(f"\n{'=' * 60}")
    print("  SIZE COMPARISON ACROSS ALL BIT WIDTHS")
    print(f"{'=' * 60}")
    print(f"  FP32 PyTorch (.pth): {pth_size:>8,} bytes  ({pth_size / 1024:.2f} KB)")
    print(f"  {'—' * 52}")

    # Build one model to count params for theoretical size
    keras_model = build_keras_nanocrackseg()
    transfer_all_weights(keras_model, pt_state)
    num_params = count_weight_params(keras_model)

    for num_bits in BIT_WIDTHS:
        p = output_dir / tflite_filename(num_bits)
        actual_kb = p.stat().st_size / 1024
        theoretical_kb = (num_params * num_bits) / (8 * 1024)
        label = f"{num_bits}-bit"
        print(
            f"  {label:>6} TFLite: {actual_kb:>8.2f} KB actual  "
            f"| {theoretical_kb:>6.2f} KB theoretical  "
            f"| {pth_size / 1024 / actual_kb:.1f}x compression"
        )

    print(f"{'=' * 60}")
    print("\nAll models saved. Run evaluate_quantized.py to compare accuracy.")
