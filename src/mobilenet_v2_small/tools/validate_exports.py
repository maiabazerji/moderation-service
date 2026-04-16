#!/usr/bin/env python3
"""
Validate exported models against the real test dataset.

Runs inference on the full test set using each exported format:
  - Keras (.keras)
  - TFLite (dynamic range quantized)
  - TFLite float16

Generates a self-contained HTML report comparing:
  - Per-model accuracy, precision, recall, F1
  - Per-class metrics per model
  - Confusion matrices
  - Prediction agreement between formats
  - Inference speed

Usage:
    cd src/mobilenet_v2_small
    python -m tools.validate_exports
"""

import base64
import io
import json
import os
import sys
import time
import datetime
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

import tensorflow as tf
from PIL import Image


# ── Config ───────────────────────────────────────────────────────────────────

EXPORTS_DIR = Path("exports")
TEST_DIR = Path("train/dataset_merged/Test")
IMG_SIZE = (224, 224)

KERAS_MODEL = Path("BestModelMobileNetV2.keras")

MODELS = {
    "Keras": KERAS_MODEL,
    "TFLite": EXPORTS_DIR / "tflite" / "model.tflite",
    "TFLite-fp16": EXPORTS_DIR / "tflite" / "model_float16.tflite",
}


# ── Load test images ────────────────────────────────────────────────────────

def load_test_dataset(test_dir: Path) -> tuple:
    """Load all test images and labels from directory structure."""
    class_names = sorted(d.name for d in test_dir.iterdir() if d.is_dir())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    images = []
    labels = []
    paths = []

    for cls_name in class_names:
        cls_dir = test_dir / cls_name
        for img_path in sorted(cls_dir.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            try:
                img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                img_array = np.array(img, dtype=np.float32)
                images.append(img_array)
                labels.append(class_to_idx[cls_name])
                paths.append(str(img_path))
            except Exception as e:
                print(f"  WARN: skipping {img_path}: {e}")

    return np.array(images), np.array(labels), class_names, paths


# ── Inference functions ──────────────────────────────────────────────────────

def _load_keras_model_compat(model_path: Path) -> tf.keras.Model:
    """
    Load a .keras model with forward-compatibility handling.

    Models saved with newer Keras may contain keys (e.g. ``quantization_config``)
    that the current installed version doesn't recognise.  We patch the in-zip
    ``config.json`` on the fly to strip those keys before handing it to Keras.
    """
    import json
    import re
    import tempfile
    import shutil
    import zipfile

    mobilenet_v2_preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input

    custom_objects = {
        "preprocess_input": mobilenet_v2_preprocess_fn,
        "backbone_preprocess": mobilenet_v2_preprocess_fn,
    }

    # --- First, try a direct load (fast path) ---
    try:
        return tf.keras.models.load_model(
            str(model_path), custom_objects=custom_objects,
        )
    except (TypeError, ValueError):
        pass  # fall through to patched load

    print("  Direct load failed; patching config to strip unsupported keys...")

    # --- Patched load: rewrite config.json inside the .keras zip ---
    STRIP_KEYS = {"quantization_config"}

    def _strip_keys(obj):
        """Recursively remove keys that the current Keras doesn't know about."""
        if isinstance(obj, dict):
            return {k: _strip_keys(v) for k, v in obj.items() if k not in STRIP_KEYS}
        if isinstance(obj, list):
            return [_strip_keys(v) for v in obj]
        return obj

    tmp_dir = tempfile.mkdtemp(prefix="keras_patch_")
    patched_path = Path(tmp_dir) / "model_patched.keras"

    try:
        with zipfile.ZipFile(str(model_path), "r") as zin, \
             zipfile.ZipFile(str(patched_path), "w") as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "config.json":
                    cfg = json.loads(data)
                    cfg = _strip_keys(cfg)
                    data = json.dumps(cfg).encode("utf-8")
                zout.writestr(item, data)

        model = tf.keras.models.load_model(
            str(patched_path), custom_objects=custom_objects,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return model


def _build_inference_model(model_path: Path) -> tf.keras.Model:
    """
    Load the trained Keras model and strip the data_augmentation layer
    so that random augmentations are NOT applied at prediction time.

    Mirrors the logic in tools/convert_model.py.
    """
    model = _load_keras_model_compat(model_path)

    # Strip data_augmentation layer(s) — same approach as convert_model.py
    aug_layer_idx = None
    for i, layer in enumerate(model.layers):
        if "data_augmentation" in layer.name or "augmentation" in layer.name:
            aug_layer_idx = i
            break

    if aug_layer_idx is not None:
        print(f"  Stripping augmentation layer: {model.layers[aug_layer_idx].name}")
        inputs = model.input
        x = inputs
        for layer in model.layers[1:]:  # skip InputLayer
            if "data_augmentation" in layer.name or "augmentation" in layer.name:
                continue
            x = layer(x)
        inference_model = tf.keras.Model(inputs=inputs, outputs=x,
                                         name="food_classifier_inference")
    else:
        print("  No augmentation layer found — using model as-is.")
        inference_model = model

    return inference_model


def predict_keras(model_path: Path, images: np.ndarray) -> tuple:
    """Run inference with Keras model (augmentation stripped)."""
    inference_model = _build_inference_model(model_path)

    t0 = time.time()
    probs = inference_model.predict(images, batch_size=32, verbose=0)
    elapsed = time.time() - t0

    preds = np.argmax(probs, axis=1)
    return preds, probs, elapsed


def predict_tflite(model_path: Path, images: np.ndarray) -> tuple:
    """Run inference with TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_idx = input_details[0]["index"]
    output_idx = output_details[0]["index"]
    input_dtype = input_details[0]["dtype"]

    all_probs = []
    t0 = time.time()
    for i in range(len(images)):
        img = np.expand_dims(images[i], axis=0)
        if input_dtype == np.uint8:
            img = img.astype(np.uint8)
        else:
            img = img.astype(np.float32)

        interpreter.set_tensor(input_idx, img)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_idx)[0]
        all_probs.append(probs)
    elapsed = time.time() - t0

    all_probs = np.array(all_probs)
    preds = np.argmax(all_probs, axis=1)
    return preds, all_probs, elapsed


# ── Compute metrics ─────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "class_report": report,
    }


# ── Chart helpers ────────────────────────────────────────────────────────────

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def chart_accuracy_comparison(results: dict) -> str:
    names = list(results.keys())
    accs = [results[n]["metrics"]["accuracy"] for n in names]
    f1s = [results[n]["metrics"]["f1_weighted"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#4C72B0", "#55A868", "#C44E52"][:len(names)]

    axes[0].bar(names, accs, color=colors)
    axes[0].set_ylim(min(accs) - 0.05, 1.0)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Test Accuracy")
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

    axes[1].bar(names, f1s, color=colors)
    axes[1].set_ylim(min(f1s) - 0.05, 1.0)
    axes[1].set_ylabel("F1 (weighted)")
    axes[1].set_title("Weighted F1-score")
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Model Format Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_confusion_matrix(cm, class_names, title) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_per_class_f1(results: dict, class_names) -> str:
    n_cls = len(class_names)
    n_models = len(results)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(n_cls)
    w = 0.8 / n_models
    colors = ["#4C72B0", "#55A868", "#C44E52"][:n_models]

    for i, (name, r) in enumerate(results.items()):
        f1s = [r["metrics"]["class_report"].get(c, {}).get("f1-score", 0) for c in class_names]
        ax.bar(x + i * w - 0.4 + w / 2, f1s, w, label=name, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1-score by Model Format")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_agreement_heatmap(results: dict) -> str:
    names = list(results.keys())
    n = len(names)
    agreement = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            p_i = results[names[i]]["preds"]
            p_j = results[names[j]]["preds"]
            agreement[i, j] = np.mean(p_i == p_j) * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(agreement, cmap="Greens", vmin=90, vmax=100)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title("Prediction Agreement (%)")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{agreement[i,j]:.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_speed(results: dict) -> str:
    names = list(results.keys())
    times = [results[n]["elapsed"] for n in names]
    n_images = len(next(iter(results.values()))["preds"])
    ips = [n_images / t for t in times]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4C72B0", "#55A868", "#C44E52"][:len(names)]
    bars = ax.bar(names, ips, color=colors)
    ax.set_ylabel("Images / second")
    ax.set_title("Inference Speed (CPU)")
    for bar, v, t in zip(bars, ips, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.1f} img/s\n({t:.1f}s total)", ha="center", fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── HTML report ──────────────────────────────────────────────────────────────

def build_report(results: dict, class_names: list, n_images: int, output_path: Path):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    acc_chart = chart_accuracy_comparison(results)
    f1_chart = chart_per_class_f1(results, class_names)
    agree_chart = chart_agreement_heatmap(results)
    speed_chart = chart_speed(results)

    # Per-model confusion matrices.
    cm_sections = []
    for name, r in results.items():
        cm_b64 = chart_confusion_matrix(r["metrics"]["confusion_matrix"], class_names, f"Confusion Matrix — {name}")
        m = r["metrics"]

        cls_rows = []
        for c in class_names:
            cr = m["class_report"].get(c, {})
            cls_rows.append(
                f"<tr><td>{c}</td>"
                f"<td>{cr.get('precision',0):.4f}</td>"
                f"<td>{cr.get('recall',0):.4f}</td>"
                f"<td>{cr.get('f1-score',0):.4f}</td>"
                f"<td>{int(cr.get('support',0))}</td></tr>"
            )

        file_size = r.get("file_size_mb", "?")
        cm_sections.append(f"""
<div class="model-section">
  <h3>{name}</h3>
  <div class="meta">
    File: {r.get('file_path','')} | Size: {file_size} MB |
    Accuracy: <strong>{m['accuracy']:.4f}</strong> |
    F1(W): <strong>{m['f1_weighted']:.4f}</strong> |
    Speed: {r['elapsed']:.1f}s ({n_images/r['elapsed']:.1f} img/s)
  </div>
  <div class="chart-row">
    <div class="chart-half"><img src="data:image/png;base64,{cm_b64}" alt="CM"></div>
    <div class="chart-half">
      <table class="metrics-table">
        <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
        <tbody>{''.join(cls_rows)}</tbody>
      </table>
    </div>
  </div>
</div>""")

    # Summary table.
    summary_rows = []
    best_acc = max(r["metrics"]["accuracy"] for r in results.values())
    for name, r in results.items():
        m = r["metrics"]
        hl = ' class="best"' if abs(m["accuracy"] - best_acc) < 1e-6 else ""
        summary_rows.append(f"""<tr{hl}>
  <td>{name}</td>
  <td>{r.get('file_size_mb','?')} MB</td>
  <td><strong>{m['accuracy']:.4f}</strong></td>
  <td>{m['precision_weighted']:.4f}</td>
  <td>{m['recall_weighted']:.4f}</td>
  <td>{m['f1_weighted']:.4f}</td>
  <td>{m['f1_macro']:.4f}</td>
  <td>{r['elapsed']:.1f}s</td>
  <td>{n_images/r['elapsed']:.1f}</td>
</tr>""")

    # Disagreement analysis.
    names_list = list(results.keys())
    disagree_rows = []
    if len(names_list) >= 2:
        base = names_list[0]
        base_preds = results[base]["preds"]
        for other in names_list[1:]:
            other_preds = results[other]["preds"]
            diffs = np.where(base_preds != other_preds)[0]
            y_true = results[base]["y_true"]
            for idx in diffs[:20]:
                disagree_rows.append(
                    f"<tr><td>{idx}</td>"
                    f"<td>{class_names[y_true[idx]]}</td>"
                    f"<td>{class_names[base_preds[idx]]}</td>"
                    f"<td>{class_names[other_preds[idx]]}</td></tr>"
                )
        disagree_header = f"<th>#{base}</th><th>#{names_list[1]}</th>" if len(names_list) >= 2 else ""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Model Export Validation Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f7fa; color: #333; line-height: 1.6; }}
  .container {{ max-width: 1300px; margin: 0 auto; padding: 20px; }}
  h1 {{ text-align: center; color: #1a1a2e; margin: 30px 0 5px; }}
  .subtitle {{ text-align: center; color: #666; margin-bottom: 30px; }}
  h2 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px; margin: 35px 0 20px; }}
  h3 {{ color: #0f3460; margin: 15px 0 10px; }}
  .section {{ background: #fff; border-radius: 10px; padding: 25px; margin-bottom: 25px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .chart-center {{ text-align: center; margin: 15px 0; }}
  .chart-center img {{ max-width: 100%; border-radius: 6px; }}
  .chart-row {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 15px 0; }}
  .chart-half {{ flex: 1; min-width: 300px; }}
  .chart-half img {{ max-width: 100%; border-radius: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.85em; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: center; }}
  th {{ background: #0f3460; color: #fff; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .best {{ background: #d4edda !important; }}
  .model-section {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; margin: 20px 0; background: #fafbfc; }}
  .meta {{ color: #555; font-size: 0.88em; padding: 8px 12px; background: #eef2f7; border-radius: 5px; margin-bottom: 12px; }}
  .metrics-table {{ font-size: 0.82em; }}
  .verdict {{ font-size: 1.1em; padding: 15px; border-radius: 8px; margin: 15px 0; }}
  .verdict.pass {{ background: #d4edda; border: 1px solid #28a745; }}
  .verdict.warn {{ background: #fff3cd; border: 1px solid #ffc107; }}
  .footer {{ text-align: center; color: #999; font-size: 0.8em; margin: 40px 0 20px; }}
</style>
</head>
<body>
<div class="container">

<h1>Model Export Validation Report</h1>
<p class="subtitle">Generated: {now} | Test images: {n_images} | {len(class_names)} classes | Formats: {', '.join(results.keys())}</p>

<div class="section">
  <h2>1. Overall Comparison</h2>
  <div class="chart-center"><img src="data:image/png;base64,{acc_chart}" alt="Accuracy"></div>
</div>

<div class="section">
  <h2>2. Per-class F1 Comparison</h2>
  <div class="chart-center"><img src="data:image/png;base64,{f1_chart}" alt="F1"></div>
</div>

<div class="section">
  <h2>3. Prediction Agreement</h2>
  <div class="chart-row">
    <div class="chart-half"><img src="data:image/png;base64,{agree_chart}" alt="Agreement"></div>
    <div class="chart-half"><img src="data:image/png;base64,{speed_chart}" alt="Speed"></div>
  </div>
</div>

<div class="section">
  <h2>4. Per-Format Details</h2>
  {''.join(cm_sections)}
</div>

<div class="section">
  <h2>5. Summary Table</h2>
  <table>
    <thead><tr>
      <th>Format</th><th>Size</th><th>Accuracy</th><th>Precision(W)</th>
      <th>Recall(W)</th><th>F1(W)</th><th>F1(M)</th><th>Time</th><th>Img/s</th>
    </tr></thead>
    <tbody>{''.join(summary_rows)}</tbody>
  </table>
</div>

{"<div class='section'><h2>6. Disagreement Examples (Keras vs TFLite)</h2><table><thead><tr><th>Image #</th><th>True Label</th><th>Keras Pred</th><th>TFLite Pred</th></tr></thead><tbody>" + ''.join(disagree_rows) + "</tbody></table></div>" if disagree_rows else ""}

<div class="footer">
  Whispr Moderation Service — Model Export Validation<br>
  All formats tested on the same {n_images} test images
</div>

</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading test dataset...")
    images, labels, class_names, img_paths = load_test_dataset(TEST_DIR)
    n_images = len(images)
    print(f"Loaded {n_images} test images, {len(class_names)} classes: {class_names}")

    results = {}

    for name, model_path in MODELS.items():
        if not model_path.exists():
            print(f"  SKIP: {name} — file not found: {model_path}")
            continue

        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\nRunning {name} ({file_size_mb:.1f} MB)...")

        if name == "Keras":
            preds, probs, elapsed = predict_keras(model_path, images)
        else:
            preds, probs, elapsed = predict_tflite(model_path, images)

        metrics = compute_metrics(labels, preds, class_names)
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1(W):    {metrics['f1_weighted']:.4f}")
        print(f"  Time:     {elapsed:.1f}s ({n_images / elapsed:.1f} img/s)")

        results[name] = {
            "preds": preds,
            "probs": probs,
            "y_true": labels,
            "metrics": metrics,
            "elapsed": elapsed,
            "file_path": str(model_path),
            "file_size_mb": f"{file_size_mb:.1f}",
        }

    if not results:
        print("No models found. Run `python -m tools.convert_model` first.")
        sys.exit(1)

    # Generate report.
    output_path = EXPORTS_DIR / "validation_report.html"
    print(f"\nGenerating report...")
    path = build_report(results, class_names, n_images, output_path)
    print(f"Report saved: {path}")
    print(f"Open: file://{path.resolve()}")

    # Also dump JSON metrics.
    json_path = EXPORTS_DIR / "validation_metrics.json"
    json_data = {}
    for name, r in results.items():
        m = r["metrics"]
        json_data[name] = {
            "accuracy": m["accuracy"],
            "precision_weighted": m["precision_weighted"],
            "recall_weighted": m["recall_weighted"],
            "f1_weighted": m["f1_weighted"],
            "f1_macro": m["f1_macro"],
            "elapsed_seconds": r["elapsed"],
            "images_per_second": n_images / r["elapsed"],
            "file_size_mb": r["file_size_mb"],
        }

    # Agreement stats.
    names_list = list(results.keys())
    for i in range(len(names_list)):
        for j in range(i + 1, len(names_list)):
            a, b = names_list[i], names_list[j]
            agree = np.mean(results[a]["preds"] == results[b]["preds"]) * 100
            json_data[f"agreement_{a}_vs_{b}"] = f"{agree:.2f}%"

    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"Metrics JSON: {json_path}")

    # Print verdict.
    print("\n" + "=" * 60)
    for name, r in results.items():
        acc = r["metrics"]["accuracy"]
        status = "PASS" if acc > 0.90 else "WARN"
        print(f"  {name}: acc={acc:.4f} [{status}]")

    base = names_list[0]
    for other in names_list[1:]:
        agree = np.mean(results[base]["preds"] == results[other]["preds"]) * 100
        status = "PASS" if agree > 99 else "WARN" if agree > 95 else "FAIL"
        print(f"  {base} vs {other}: {agree:.1f}% agreement [{status}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
