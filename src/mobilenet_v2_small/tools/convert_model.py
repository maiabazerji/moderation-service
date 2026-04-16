#!/usr/bin/env python3
"""
Model format converter for EfficientNet food classifier.

Converts the trained Keras model to deployment formats:
  - TFLite  (mobile / edge)
  - TFJS    (browser)

The inference model strips data-augmentation layers so that
random augmentations are NOT applied at prediction time.

Usage:
    cd src/mobilenet_v2_small
    python -m tools.convert_model                          # convert all formats
    python -m tools.convert_model --format tflite          # TFLite only
    python -m tools.convert_model --format tfjs            # TFJS only
    python -m tools.convert_model --input path/to/model.keras
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "BestModelEfficientNetLite.keras"
OUTPUT_DIR = "exports"


# ── Build inference-only model ───────────────────────────────────────────────

def build_inference_model(model_path: str) -> tf.keras.Model:
    """
    Load the full training model and strip augmentation layers
    to build a clean inference graph.
    """
    print(f"Loading model: {model_path}")

    # The model uses a Lambda layer wrapping preprocess_input —
    # we must register it so Keras can deserialize it.
    # Support both EfficientNet and MobileNetV2 preprocessing.
    efficientnet_preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
    mobilenet_v2_preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input

    custom_objects = {
        "preprocess_input": mobilenet_v2_preprocess_fn,
        "efficientnet_preprocess": efficientnet_preprocess_fn,
        "backbone_preprocess": mobilenet_v2_preprocess_fn,
    }

    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except (TypeError, ValueError):
        # Models saved with newer Keras may contain keys (e.g. quantization_config)
        # that the current version doesn't recognise. Patch the config on-the-fly.
        import zipfile as _zf, tempfile as _tmp

        _STRIP = {"quantization_config"}

        def _strip(obj):
            if isinstance(obj, dict):
                return {k: _strip(v) for k, v in obj.items() if k not in _STRIP}
            if isinstance(obj, list):
                return [_strip(v) for v in obj]
            return obj

        td = _tmp.mkdtemp(prefix="keras_patch_")
        pp = os.path.join(td, "patched.keras")
        try:
            with _zf.ZipFile(model_path, "r") as zi, _zf.ZipFile(pp, "w") as zo:
                for it in zi.infolist():
                    d = zi.read(it.filename)
                    if it.filename == "config.json":
                        d = json.dumps(_strip(json.loads(d))).encode()
                    zo.writestr(it, d)
            model = tf.keras.models.load_model(pp, custom_objects=custom_objects)
        finally:
            shutil.rmtree(td, ignore_errors=True)
    model.summary()

    # Identify the augmentation layer(s) to skip.
    # Architecture:  Input -> preprocess -> data_augmentation -> backbone -> ...
    # We want to connect Input directly to the backbone, skipping augmentation.

    layer_names = [l.name for l in model.layers]
    print(f"Layer names: {layer_names}")

    # Strategy: find the data_augmentation layer and bypass it.
    aug_layer_idx = None
    for i, layer in enumerate(model.layers):
        if "data_augmentation" in layer.name or "augmentation" in layer.name:
            aug_layer_idx = i
            break

    if aug_layer_idx is not None:
        print(f"Stripping augmentation layer: {model.layers[aug_layer_idx].name} (index {aug_layer_idx})")

        # Rebuild the model skipping the augmentation layer.
        inputs = model.input
        x = inputs

        # Apply all layers except augmentation.
        for layer in model.layers[1:]:  # skip InputLayer
            if "data_augmentation" in layer.name or "augmentation" in layer.name:
                continue
            x = layer(x)

        inference_model = tf.keras.Model(inputs=inputs, outputs=x, name="food_classifier_inference")
    else:
        print("No augmentation layer found — using model as-is for inference.")
        inference_model = model

    print("\nInference model summary:")
    inference_model.summary()
    return inference_model


# ── TFLite conversion ────────────────────────────────────────────────────────

def convert_to_tflite(inference_model: tf.keras.Model, output_dir: Path) -> Path:
    """Convert to TFLite with DEFAULT + float16 quantization."""
    print("\n" + "=" * 60)
    print("Converting to TFLite...")
    print("=" * 60)

    tflite_dir = output_dir / "tflite"
    tflite_dir.mkdir(parents=True, exist_ok=True)

    # First export to SavedModel (needed for reliable TFLite conversion).
    saved_model_dir = output_dir / "_saved_model_tmp"
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)

    inference_model.export(str(saved_model_dir))
    print(f"SavedModel exported to: {saved_model_dir}")

    # Convert with DEFAULT optimizations.
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    tflite_path = tflite_dir / "model.tflite"
    tflite_path.write_bytes(tflite_model)
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite model saved: {tflite_path} ({size_mb:.1f} MB)")

    # Also create float16 quantized version (smaller).
    converter_f16 = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_f16.target_spec.supported_types = [tf.float16]

    tflite_f16 = converter_f16.convert()
    tflite_f16_path = tflite_dir / "model_float16.tflite"
    tflite_f16_path.write_bytes(tflite_f16)
    size_f16_mb = len(tflite_f16) / (1024 * 1024)
    print(f"TFLite float16 model saved: {tflite_f16_path} ({size_f16_mb:.1f} MB)")

    # Clean up temporary SavedModel.
    shutil.rmtree(saved_model_dir)

    # Write metadata.
    num_classes = inference_model.output_shape[-1]
    metadata = {
        "format": "tflite",
        "input_shape": [1, 224, 224, 3],
        "input_dtype": "float32",
        "output_shape": [1, num_classes],
        "quantization": "dynamic_range",
        "model_size_mb": round(size_mb, 2),
        "model_size_f16_mb": round(size_f16_mb, 2),
    }
    (tflite_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return tflite_dir


# ── TFJS conversion ─────────────────────────────────────────────────────────

def convert_to_tfjs(inference_model: tf.keras.Model, output_dir: Path) -> Path:
    """Convert to TensorFlow.js format via SavedModel + CLI converter."""
    print("\n" + "=" * 60)
    print("Converting to TFJS...")
    print("=" * 60)

    tfjs_dir = output_dir / "tfjs"
    if tfjs_dir.exists():
        shutil.rmtree(tfjs_dir)
    tfjs_dir.mkdir(parents=True, exist_ok=True)

    # Export to SavedModel first (TFJS converter works best from SavedModel).
    saved_model_dir = output_dir / "_saved_model_tfjs_tmp"
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)

    inference_model.export(str(saved_model_dir))
    print(f"SavedModel exported to: {saved_model_dir}")

    # Work around tensorflowjs importing tensorflow_decision_forests
    # (not needed for our CNN model but required by tensorflowjs 4.x).
    import types
    for mod_name in [
        "tensorflow_decision_forests",
        "tensorflow_decision_forests.keras",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    # Use tensorflowjs Python API directly (faster, avoids subprocess issues).
    try:
        from tensorflowjs.converters import converter as tfjs_converter
        tfjs_converter.convert([
            "--input_format=tf_saved_model",
            "--output_format=tfjs_graph_model",
            "--signature_name=serving_default",
            "--saved_model_tags=serve",
            str(saved_model_dir),
            str(tfjs_dir),
        ])
    except Exception as e:
        print(f"TFJS conversion error: {e}")
        # Fallback to CLI.
        cmd = [
            sys.executable, "-m", "tensorflowjs.converter",
            "--input_format=tf_saved_model",
            "--output_format=tfjs_graph_model",
            str(saved_model_dir),
            str(tfjs_dir),
        ]
        print(f"Fallback CLI: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"TFJS conversion failed: {result.stderr[-300:]}")

    # Clean up.
    shutil.rmtree(saved_model_dir, ignore_errors=True)

    # Verify output.
    model_json = tfjs_dir / "model.json"
    if model_json.exists():
        bin_files = list(tfjs_dir.glob("*.bin"))
        total_size = sum(f.stat().st_size for f in bin_files) / (1024 * 1024)
        print(f"TFJS model saved: {tfjs_dir}")
        print(f"  model.json + {len(bin_files)} weight file(s) ({total_size:.1f} MB)")
    else:
        print(f"WARNING: model.json not found in {tfjs_dir}")

    return tfjs_dir


# ── Class labels and config ─────────────────────────────────────────────────

def write_labels_and_config(output_dir: Path, model_path: str):
    """Write class labels and model config for deployment."""
    # Try to read class names from existing training config.
    class_names = None
    for config_path in [
        Path("train/results/training_results/training_config.json"),
        Path("train/results_sweep/exp_A/training_results/training_config.json"),
    ]:
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            class_names = cfg.get("class_names")
            if class_names:
                break

    if not class_names:
        # Fallback: read from dataset directory.
        train_dir = Path("train/dataset_merged/Train")
        if train_dir.exists():
            class_names = sorted(
                d.name for d in train_dir.iterdir() if d.is_dir()
            )

    if not class_names:
        class_names = [
            "Baked Potato", "Burger", "Crispy Chicken", "Donut",
            "Fries", "Hot Dog", "Other", "Pizza", "Sandwich"
        ]

    # Write labels.
    labels_path = output_dir / "labels.json"
    labels_data = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "id2label": {str(i): name for i, name in enumerate(class_names)},
        "label2id": {name: i for i, name in enumerate(class_names)},
    }
    labels_path.write_text(json.dumps(labels_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nLabels saved: {labels_path}")

    # Detect model type from config or default.
    model_type = "mobilenet-v2-035"
    preprocessing_note = "Use tf.keras.applications.mobilenet_v2.preprocess_input()"
    for config_path in [
        Path("train/results/training_results/training_config.json"),
    ]:
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            mc = cfg.get("model_config", {})
            if mc.get("model_name"):
                model_type = mc["model_name"]
                if model_type.startswith("efficientnet"):
                    preprocessing_note = "Use tf.keras.applications.efficientnet.preprocess_input()"
            break

    # Write model config.
    config = {
        "model_type": model_type,
        "task": "image-classification",
        "framework": "tensorflow",
        "input_size": 224,
        "input_shape": [1, 224, 224, 3],
        "num_classes": len(class_names),
        "class_names": class_names,
        "preprocessing": {
            "resize": [224, 224],
            "normalization": "mobilenet_v2" if model_type.startswith("mobilenet") else "efficientnet",
            "note": preprocessing_note,
        },
        "source_model": str(model_path),
        "formats_available": ["keras", "tflite", "tfjs"],
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Config saved: {config_path}")

    return class_names


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert EfficientNet model to deployment formats")
    parser.add_argument("--input", type=str, default=None,
                        help=f"Path to .keras or .h5 model (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--format", type=str, default="all",
                        choices=["all", "tflite", "tfjs"],
                        help="Which format(s) to convert")
    args = parser.parse_args()

    # Find model file.
    if args.input:
        model_path = Path(args.input)
    else:
        # Auto-detect: prefer .keras over .h5
        keras_path = Path(DEFAULT_MODEL)
        h5_path = Path(DEFAULT_MODEL.replace(".keras", ".h5"))
        if keras_path.exists():
            model_path = keras_path
        elif h5_path.exists():
            model_path = h5_path
        else:
            print(f"ERROR: No model found. Tried {keras_path} and {h5_path}")
            sys.exit(1)

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source model:   {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Output dir:     {output_dir}")
    print(f"Format(s):      {args.format}")
    print()

    # Copy source model to exports.
    src_copy = output_dir / model_path.name
    if not src_copy.exists():
        shutil.copy2(model_path, src_copy)
        print(f"Source model copied to: {src_copy}")

    # Build inference model (strip augmentation).
    inference_model = build_inference_model(str(model_path))

    # Convert.
    if args.format in ("all", "tflite"):
        convert_to_tflite(inference_model, output_dir)

    if args.format in ("all", "tfjs"):
        convert_to_tfjs(inference_model, output_dir)

    # Write labels and config.
    write_labels_and_config(output_dir, model_path)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)

    # Show output structure.
    print(f"\nOutput directory: {output_dir}/")
    for p in sorted(output_dir.rglob("*")):
        if p.is_file() and not p.name.startswith("_"):
            rel = p.relative_to(output_dir)
            size = p.stat().st_size
            if size > 1024 * 1024:
                print(f"  {rel}  ({size / 1024 / 1024:.1f} MB)")
            elif size > 1024:
                print(f"  {rel}  ({size / 1024:.1f} KB)")
            else:
                print(f"  {rel}  ({size} B)")


if __name__ == "__main__":
    main()
