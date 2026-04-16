# TensorFlow.js Model Conversion Guide

This project currently trains and exports models with TensorFlow/Keras, and also has a dedicated TFLite export script.

## 1) What was found in this repository

- Training pipeline saves a Keras model (`model.save(...)`) in `src/efficientnet_lite_gpu/train/train.py`.
- TFLite export pipeline in `src/efficientnet_lite_gpu/convert_to_tflite.py` does:
  1. Load Keras `.h5`
  2. Export to SavedModel directory
  3. Convert SavedModel to `.tflite`
- No actual model artifact files (`.h5`, `.keras`, `saved_model.pb`, `.tflite`) are currently committed in this repository snapshot.

Conclusion: a `.tflite` can exist downstream, but the reliable TFJS conversion source should be the upstream Keras/SavedModel model.

## 2) Supported conversion inputs for TFJS

The conversion script supports:

- Keras model file: `.h5` / `.keras` (directly supported)
- TensorFlow SavedModel directory (best-effort support)

The conversion script does **not** treat `.tflite` as a reliable standard input for TFJS conversion.

If input is `.tflite`, the script fails with a clear error:

`Only a .tflite file is available; it cannot be used as a standard TFJS conversion input. Please provide the upstream training model or the pre-export SavedModel.`

## 3) Script location

- `scripts/convert_to_tfjs.py`

## 4) Requirements

Install converter tool in your Python environment:

```bash
pip install tensorflowjs
```

You also need a model source file/folder:

- `.h5` or `.keras`, or
- SavedModel directory containing `saved_model.pb`

## 5) How to run

From repository root:

```bash
python scripts/convert_to_tfjs.py --input-model <MODEL_PATH> --output-dir <OUTPUT_DIR>
```

Examples:

```bash
python scripts/convert_to_tfjs.py --input-model ./models/food_classifier.h5 --output-dir ./artifacts/tfjs_model
python scripts/convert_to_tfjs.py --input-model ./models/saved_model_effnetlite_inference --output-dir ./artifacts/tfjs_model
```

## 6) Output files

On successful conversion, output directory contains standard TFJS artifacts:

- `model.json`
- one or more `*.bin` weight files

## 7) Most common failure reasons

- `tensorflowjs_converter` is not installed or not in PATH
- Input path does not exist
- Input is only `.tflite` (unsupported as standard reliable source)
- SavedModel contains unsupported ops/signature issues for TFJS converter
- Conversion command runs but output artifacts are incomplete

## 8) If you only have `.tflite`, what to add next

Provide one of these upstream artifacts:

- The trained Keras model (`.h5` or `.keras`)
- The pre-TFLite SavedModel export directory (`saved_model.pb` + variables)

Then rerun `scripts/convert_to_tfjs.py`.
