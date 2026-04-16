#!/usr/bin/env python3
"""
Hyperparameter sweep report generator — REAL training.

Runs the actual two-stage training pipeline (MobileNetV2 default) multiple times
with different hyperparameter configurations, then generates a
self-contained HTML report from the real results.

Usage:
    cd src/mobilenet_v2_small
    python -m tools.generate_training_report          # run all experiments
    python -m tools.generate_training_report --report  # report only (skip training)
"""

import argparse
import base64
import copy
import io
import json
import datetime
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

# ── Experiment definitions ───────────────────────────────────────────────────
# Each entry overrides fields in the base config.yaml → train_config.

EXPERIMENTS = [
    {
        "id": "A",
        "name": "A: Baseline",
        "overrides": {
            "batch_size": 8,
            "stage1_learning_rate": 1e-3,
            "stage2_learning_rate": 2e-5,
            "stage1_epochs": 12,
            "stage2_epochs": 6,
            "fine_tune": True,
        },
        "model_overrides": {
            "model_name": "mobilenet-v2-035",
        },
    },
    {
        "id": "B",
        "name": "B: Large Batch",
        "overrides": {
            "batch_size": 32,
            "stage1_learning_rate": 1e-3,
            "stage2_learning_rate": 2e-5,
            "stage1_epochs": 12,
            "stage2_epochs": 6,
            "fine_tune": True,
        },
        "model_overrides": {
            "model_name": "mobilenet-v2-035",
        },
    },
    {
        "id": "C",
        "name": "C: High LR",
        "overrides": {
            "batch_size": 8,
            "stage1_learning_rate": 1e-2,
            "stage2_learning_rate": 1e-4,
            "stage1_epochs": 12,
            "stage2_epochs": 6,
            "fine_tune": True,
        },
        "model_overrides": {
            "model_name": "mobilenet-v2-035",
        },
    },
    {
        "id": "D",
        "name": "D: Low LR + Long",
        "overrides": {
            "batch_size": 8,
            "stage1_learning_rate": 5e-4,
            "stage2_learning_rate": 1e-5,
            "stage1_epochs": 20,
            "stage2_epochs": 8,
            "fine_tune": True,
        },
        "model_overrides": {
            "model_name": "mobilenet-v2-035",
        },
    },
    {
        "id": "E",
        "name": "E: MobileNetV2-050",
        "overrides": {
            "batch_size": 8,
            "stage1_learning_rate": 1e-3,
            "stage2_learning_rate": 2e-5,
            "stage1_epochs": 12,
            "stage2_epochs": 6,
            "fine_tune": True,
        },
        "model_overrides": {
            "model_name": "mobilenet-v2-050",
        },
    },
    {
        "id": "F",
        "name": "F: No Fine-tune",
        "overrides": {
            "batch_size": 8,
            "stage1_learning_rate": 1e-3,
            "stage2_learning_rate": 0,
            "stage1_epochs": 20,
            "stage2_epochs": 0,
            "fine_tune": False,
        },
        "model_overrides": {
            "model_name": "mobilenet-v2-035",
        },
    },
    {
        "id": "G",
        "name": "G: Batch 16 + Mid LR",
        "overrides": {
            "batch_size": 16,
            "stage1_learning_rate": 2e-3,
            "stage2_learning_rate": 5e-5,
            "stage1_epochs": 15,
            "stage2_epochs": 6,
            "fine_tune": True,
        },
        "model_overrides": {
            "model_name": "mobilenet-v2-035",
        },
    },
    {
        "id": "H",
        "name": "H: Aggressive Fine-tune",
        "overrides": {
            "batch_size": 8,
            "stage1_learning_rate": 1e-3,
            "stage2_learning_rate": 1e-4,
            "stage1_epochs": 10,
            "stage2_epochs": 12,
            "fine_tune": True,
        },
        "model_overrides": {
            "model_name": "mobilenet-v2-035",
        },
    },
]

SWEEP_RESULTS_ROOT = "train/results_sweep"


# ── Config manipulation ──────────────────────────────────────────────────────

def load_base_config() -> dict:
    cfg_path = Path.cwd() / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Base config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_experiment_config(base_cfg: dict, exp: dict) -> dict:
    """Deep-copy base config and apply experiment overrides."""
    cfg = copy.deepcopy(base_cfg)
    tc = cfg["train_config"]

    # Don't rebuild the split every time — dataset is already prepared.
    tc["rebuild_clean_split_before_train"] = False

    # Apply train_config overrides.
    for k, v in exp["overrides"].items():
        tc[k] = v

    # Apply model_config overrides.
    for k, v in exp.get("model_overrides", {}).items():
        tc["model_config"][k] = v

    # Redirect results to experiment-specific directory.
    tc["results_dir"] = f"{SWEEP_RESULTS_ROOT}/exp_{exp['id']}"

    return cfg


def experiment_is_complete(exp: dict) -> bool:
    """Check if an experiment already has complete results (for resume support)."""
    results_dir = Path.cwd() / SWEEP_RESULTS_ROOT / f"exp_{exp['id']}"
    required = [
        "evaluation_results/test_metrics.json",
        "evaluation_results/test_class_report.json",
        "training_logs/training_history.json",
        "training_logs/best_metrics.json",
        "training_results/training_config.json",
    ]
    return all((results_dir / f).exists() for f in required)


# ── Run real training ────────────────────────────────────────────────────────

def run_experiments(experiments: list, force: bool = False):
    """Run each experiment through the real training pipeline."""
    # Import the real training module.
    from train.train import run as train_run

    base_cfg = load_base_config()
    total = len(experiments)

    for i, exp in enumerate(experiments):
        tag = f"[{i+1}/{total}] {exp['name']}"

        if not force and experiment_is_complete(exp):
            print(f"{tag} — already complete, skipping (use --force to re-run)")
            continue

        print(f"\n{'='*70}")
        print(f"{tag} — STARTING")
        print(f"{'='*70}")

        cfg = build_experiment_config(base_cfg, exp)

        # Show key hyperparameters.
        tc = cfg["train_config"]
        print(f"  batch_size:  {tc['batch_size']}")
        print(f"  stage1_lr:   {tc.get('stage1_learning_rate', 'default')}")
        print(f"  stage2_lr:   {tc.get('stage2_learning_rate', 'default')}")
        print(f"  stage1_ep:   {tc.get('stage1_epochs', 'default')}")
        print(f"  stage2_ep:   {tc.get('stage2_epochs', 'default')}")
        print(f"  fine_tune:   {tc.get('fine_tune', True)}")
        print(f"  model:       {tc['model_config']['model_name']}")
        print(f"  results_dir: {tc['results_dir']}")

        try:
            train_run(cfg)
            print(f"{tag} — DONE")
        except Exception as e:
            print(f"{tag} — FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nAll experiments finished.")


# ── Collect real results ─────────────────────────────────────────────────────

def collect_results(experiments: list) -> list:
    """Read real training outputs from each experiment directory."""
    all_results = []

    for exp in experiments:
        results_dir = Path.cwd() / SWEEP_RESULTS_ROOT / f"exp_{exp['id']}"

        if not experiment_is_complete(exp):
            print(f"  WARN: {exp['name']} incomplete, skipping from report")
            continue

        def _load_json(rel_path):
            with open(results_dir / rel_path, "r", encoding="utf-8") as f:
                return json.load(f)

        test_metrics = _load_json("evaluation_results/test_metrics.json")
        class_report = _load_json("evaluation_results/test_class_report.json")
        history = _load_json("training_logs/training_history.json")
        best_metrics = _load_json("training_logs/best_metrics.json")
        training_config = _load_json("training_results/training_config.json")

        # Load confusion matrix if available.
        cm_path = results_dir / "evaluation_results/test_confusion_matrix.npy"
        cm = np.load(cm_path) if cm_path.exists() else None

        # Load existing chart PNGs if available.
        charts = {}
        for chart_name in ["confusion_matrix", "class_performance",
                           "performance_metrics", "class_distribution",
                           "training_history"]:
            for subdir in ["evaluation_results", "data_exploration", "training_results"]:
                png_path = results_dir / subdir / f"{chart_name}.png"
                if png_path.exists():
                    charts[chart_name] = png_path
                    break

        all_results.append({
            "exp": exp,
            "test_metrics": test_metrics,
            "class_report": class_report,
            "history": history,
            "best_metrics": best_metrics,
            "training_config": training_config,
            "confusion_matrix": cm,
            "charts": charts,
            "results_dir": results_dir,
        })

    return all_results


# ── Dataset stats (read from real directory) ─────────────────────────────────

def get_dataset_stats() -> dict:
    """Count real images per class per split."""
    dataset_root = Path.cwd() / "train" / "dataset_merged"
    stats = {"train": {}, "val": {}, "test": {}}
    split_dirs = {"train": "Train", "val": "Val", "test": "Test"}

    class_names = None
    for split_key, dir_name in split_dirs.items():
        split_dir = dataset_root / dir_name
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            count = sum(1 for p in cls_dir.rglob("*") if p.is_file())
            stats[split_key][cls_dir.name] = count
        if class_names is None:
            class_names = sorted(stats[split_key].keys())

    stats["class_names"] = class_names or []
    return stats


# ── Chart generation (returns base64 PNG) ────────────────────────────────────

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _png_file_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def chart_dataset_distribution(ds_stats: dict) -> str:
    class_names = ds_stats["class_names"]
    n = len(class_names)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(n)
    w = 0.25
    train_counts = [ds_stats["train"].get(c, 0) for c in class_names]
    val_counts = [ds_stats["val"].get(c, 0) for c in class_names]
    test_counts = [ds_stats["test"].get(c, 0) for c in class_names]

    axes[0].bar(x - w, train_counts, w, label="Train", color="#4C72B0")
    axes[0].bar(x,     val_counts,   w, label="Val",   color="#55A868")
    axes[0].bar(x + w, test_counts,  w, label="Test",  color="#C44E52")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylabel("Number of Images")
    axes[0].set_title("Class Distribution by Split")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    totals = [sum(train_counts), sum(val_counts), sum(test_counts)]
    axes[1].pie(totals, labels=["Train", "Val", "Test"],
                colors=["#4C72B0", "#55A868", "#C44E52"],
                autopct="%1.1f%%", startangle=90)
    axes[1].set_title(f"Split Ratio (Total: {sum(totals)})")

    fig.suptitle("Dataset Statistics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _fig_to_base64(fig)


def chart_training_curves(result: dict) -> str:
    history = result["history"]
    exp = result["exp"]

    s1 = history.get("stage1", {})
    s2 = history.get("stage2", {})

    acc1 = s1.get("accuracy", [])
    val_acc1 = s1.get("val_accuracy", [])
    loss1 = s1.get("loss", [])
    val_loss1 = s1.get("val_loss", [])

    acc2 = s2.get("accuracy", [])
    val_acc2 = s2.get("val_accuracy", [])
    loss2 = s2.get("loss", [])
    val_loss2 = s2.get("val_loss", [])

    epochs1 = list(range(1, len(acc1) + 1))
    epochs2 = list(range(len(acc1) + 1, len(acc1) + len(acc2) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs1, acc1, "b-o", ms=3, label="Train (S1)")
    axes[0].plot(epochs1, val_acc1, "b--s", ms=3, label="Val (S1)")
    if acc2:
        axes[0].axvline(len(acc1) + 0.5, color="gray", ls=":", alpha=0.6)
        axes[0].plot(epochs2, acc2, "r-o", ms=3, label="Train (S2)")
        axes[0].plot(epochs2, val_acc2, "r--s", ms=3, label="Val (S2)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs1, loss1, "b-o", ms=3, label="Train (S1)")
    axes[1].plot(epochs1, val_loss1, "b--s", ms=3, label="Val (S1)")
    if loss2:
        axes[1].axvline(len(loss1) + 0.5, color="gray", ls=":", alpha=0.6)
        axes[1].plot(epochs2, loss2, "r-o", ms=3, label="Train (S2)")
        axes[1].plot(epochs2, val_loss2, "r--s", ms=3, label="Val (S2)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Training Curves — {exp['name']}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _fig_to_base64(fig)


def chart_confusion_matrix(result: dict) -> str:
    cm = result["confusion_matrix"]
    exp = result["exp"]
    class_names = result["training_config"]["class_names"]

    if cm is None:
        # Fallback: use existing PNG if available.
        if "confusion_matrix" in result["charts"]:
            return _png_file_to_base64(result["charts"]["confusion_matrix"])
        return ""

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
    ax.set_title(f"Confusion Matrix — {exp['name']}")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    return _fig_to_base64(fig)


def chart_per_class_f1(all_results: list) -> str:
    if not all_results:
        return ""
    class_names = all_results[0]["training_config"]["class_names"]
    n_cls = len(class_names)
    n_exp = len(all_results)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(n_cls)
    w = 0.8 / n_exp
    colors = plt.cm.Set2(np.linspace(0, 1, n_exp))

    for i, r in enumerate(all_results):
        f1s = [r["class_report"].get(c, {}).get("f1-score", 0) for c in class_names]
        ax.bar(x + i * w - 0.4 + w / 2, f1s, w,
               label=r["exp"]["name"], color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1-score Comparison")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)


def chart_hyperparameter_comparison(all_results: list) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = [r["exp"]["name"] for r in all_results]
    accs  = [r["test_metrics"]["accuracy"] for r in all_results]
    f1s   = [r["test_metrics"]["f1_weighted"] for r in all_results]
    precs = [r["test_metrics"]["precision_weighted"] for r in all_results]
    recs  = [r["test_metrics"]["recall_weighted"] for r in all_results]

    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    axes[0, 0].barh(names, accs, color=colors)
    axes[0, 0].set_xlabel("Accuracy")
    axes[0, 0].set_title("Test Accuracy")
    lo = max(0.5, min(accs) - 0.05)
    axes[0, 0].set_xlim(lo, 1.0)
    for i, v in enumerate(accs):
        axes[0, 0].text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=8)

    axes[0, 1].barh(names, f1s, color=colors)
    axes[0, 1].set_xlabel("F1-score (weighted)")
    axes[0, 1].set_title("Weighted F1-score")
    axes[0, 1].set_xlim(lo, 1.0)
    for i, v in enumerate(f1s):
        axes[0, 1].text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=8)

    axes[1, 0].scatter(precs, recs, c=colors, s=100, edgecolors="black", zorder=5)
    for i, n in enumerate(names):
        axes[1, 0].annotate(n.split(":")[0], (precs[i], recs[i]),
                            fontsize=7, ha="center", va="bottom")
    axes[1, 0].set_xlabel("Precision (weighted)")
    axes[1, 0].set_ylabel("Recall (weighted)")
    axes[1, 0].set_title("Precision vs Recall")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axis("off")
    col_labels = ["Config", "Batch", "S1 LR", "S2 LR", "S1 Ep", "S2 Ep", "Model"]
    table_data = []
    for r in all_results:
        e = r["exp"]
        ov = e["overrides"]
        table_data.append([
            e["name"].split(":")[0],
            str(ov["batch_size"]),
            f"{ov['stage1_learning_rate']:.0e}",
            f"{ov['stage2_learning_rate']:.0e}" if ov.get("stage2_learning_rate") else "—",
            str(ov["stage1_epochs"]),
            str(ov["stage2_epochs"]),
            e.get("model_overrides", {}).get("model_name", "b0"),
        ])
    tbl = axes[1, 1].table(cellText=table_data, colLabels=col_labels,
                            loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)
    axes[1, 1].set_title("Hyperparameter Grid", pad=20)

    fig.suptitle("Hyperparameter Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _fig_to_base64(fig)


def chart_radar(all_results: list) -> str:
    categories = ["Accuracy", "Precision(W)", "Recall(W)", "F1(W)", "F1(M)"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))

    for i, r in enumerate(all_results):
        m = r["test_metrics"]
        values = [m["accuracy"], m["precision_weighted"], m["recall_weighted"],
                  m["f1_weighted"], m["f1_macro"]]
        values += values[:1]
        ax.plot(angles, values, "o-", ms=4, label=r["exp"]["name"], color=colors[i])
        ax.fill(angles, values, alpha=0.05, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    lo = max(0.5, min(r["test_metrics"]["f1_macro"] for r in all_results) - 0.05)
    ax.set_ylim(lo, 1.0)
    ax.set_title("Metrics Radar", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    fig.tight_layout()
    return _fig_to_base64(fig)


# ── HTML report assembly ─────────────────────────────────────────────────────

def _build_summary_table_html(all_results: list) -> str:
    best_acc = max(r["test_metrics"]["accuracy"] for r in all_results)
    rows = []
    for r in all_results:
        exp = r["exp"]
        ov = exp["overrides"]
        m = r["test_metrics"]
        bm = r["best_metrics"]
        highlight = ' class="best-row"' if abs(m["accuracy"] - best_acc) < 1e-6 else ""
        best_val = bm.get('best_val_acc_overall', 0)
        if isinstance(best_val, (int, float)):
            best_val_str = f"{best_val:.4f}"
        else:
            best_val_str = str(best_val)
        rows.append(f"""<tr{highlight}>
  <td>{exp['name']}</td>
  <td>{exp.get('model_overrides',{}).get('model_name','b0')}</td>
  <td>{ov['batch_size']}</td>
  <td>{ov['stage1_learning_rate']:.0e}</td>
  <td>{ov['stage2_learning_rate']:.0e}</td>
  <td>{ov['stage1_epochs']}</td>
  <td>{ov['stage2_epochs']}</td>
  <td>{best_val_str}</td>
  <td><strong>{m['accuracy']:.4f}</strong></td>
  <td>{m['precision_weighted']:.4f}</td>
  <td>{m['recall_weighted']:.4f}</td>
  <td>{m['f1_weighted']:.4f}</td>
  <td>{m['f1_macro']:.4f}</td>
</tr>""")
    return "\n".join(rows)


def _build_per_config_sections(all_results: list) -> str:
    sections = []
    for r in all_results:
        exp = r["exp"]
        ov = exp["overrides"]
        m = r["test_metrics"]
        class_names = r["training_config"]["class_names"]

        tc_b64 = chart_training_curves(r)
        cm_b64 = chart_confusion_matrix(r)

        cls_rows = []
        for c in class_names:
            cr = r["class_report"].get(c, {})
            cls_rows.append(
                f"<tr><td>{c}</td>"
                f"<td>{cr.get('precision',0):.4f}</td>"
                f"<td>{cr.get('recall',0):.4f}</td>"
                f"<td>{cr.get('f1-score',0):.4f}</td>"
                f"<td>{int(cr.get('support',0))}</td></tr>"
            )

        model_name = exp.get("model_overrides", {}).get("model_name", "b0")
        sections.append(f"""
<div class="config-section">
  <h3>{exp['name']}</h3>
  <div class="config-meta">
    Model: {model_name} | Batch: {ov['batch_size']} |
    S1 LR: {ov['stage1_learning_rate']:.0e} | S2 LR: {ov['stage2_learning_rate']:.0e} |
    Epochs: {ov['stage1_epochs']}+{ov['stage2_epochs']} |
    Fine-tune: {'Yes' if ov.get('fine_tune', True) else 'No'}
  </div>
  <div class="chart-row">
    <img src="data:image/png;base64,{tc_b64}" alt="Training curves">
  </div>
  <div class="chart-row">
    <div class="chart-half">
      <img src="data:image/png;base64,{cm_b64}" alt="Confusion matrix">
    </div>
    <div class="chart-half">
      <table class="metrics-table">
        <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
        <tbody>{''.join(cls_rows)}</tbody>
      </table>
      <p class="overall-metric">Test Accuracy: <strong>{m['accuracy']:.4f}</strong> |
         F1 (weighted): <strong>{m['f1_weighted']:.4f}</strong> |
         F1 (macro): <strong>{m['f1_macro']:.4f}</strong></p>
    </div>
  </div>
</div>""")
    return "\n".join(sections)


def build_html_report(all_results: list, ds_stats: dict, output_path: Path) -> Path:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    class_names = ds_stats["class_names"]
    n_classes = len(class_names)
    total_train = sum(ds_stats["train"].values())
    total_val = sum(ds_stats["val"].values())
    total_test = sum(ds_stats["test"].values())
    total_all = total_train + total_val + total_test

    dataset_b64 = chart_dataset_distribution(ds_stats)
    hp_compare_b64 = chart_hyperparameter_comparison(all_results)
    f1_compare_b64 = chart_per_class_f1(all_results)
    radar_b64 = chart_radar(all_results)
    summary_rows = _build_summary_table_html(all_results)
    config_sections = _build_per_config_sections(all_results)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MobileNetV2 Hyperparameter Sweep Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f7fa; color: #333; line-height: 1.6; }}
  .container {{ max-width: 1300px; margin: 0 auto; padding: 20px; }}
  h1 {{ text-align: center; color: #1a1a2e; margin: 30px 0 5px; font-size: 1.8em; }}
  .subtitle {{ text-align: center; color: #666; margin-bottom: 30px; font-size: 0.95em; }}
  h2 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px;
        margin: 35px 0 20px; font-size: 1.35em; }}
  h3 {{ color: #0f3460; margin: 15px 0 10px; font-size: 1.15em; }}
  .section {{ background: #fff; border-radius: 10px; padding: 25px;
              margin-bottom: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .chart-center {{ text-align: center; margin: 15px 0; }}
  .chart-center img {{ max-width: 100%; height: auto; border-radius: 6px; }}
  .chart-row {{ display: flex; gap: 20px; align-items: flex-start;
                flex-wrap: wrap; margin: 15px 0; }}
  .chart-row > img {{ max-width: 100%; height: auto; border-radius: 6px; }}
  .chart-half {{ flex: 1; min-width: 300px; }}
  .chart-half img {{ max-width: 100%; height: auto; border-radius: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.85em; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: center; }}
  th {{ background: #0f3460; color: #fff; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  tr:hover {{ background: #e8f0fe; }}
  .best-row {{ background: #d4edda !important; }}
  .best-row:hover {{ background: #c3e6cb !important; }}
  .config-section {{ border: 1px solid #e0e0e0; border-radius: 8px;
                     padding: 20px; margin: 20px 0; background: #fafbfc; }}
  .config-meta {{ color: #555; font-size: 0.88em; margin-bottom: 12px;
                  padding: 8px 12px; background: #eef2f7; border-radius: 5px; }}
  .metrics-table {{ font-size: 0.82em; }}
  .overall-metric {{ margin-top: 12px; font-size: 0.9em; color: #333; }}
  .footer {{ text-align: center; color: #999; font-size: 0.8em; margin: 40px 0 20px; }}
  .dataset-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 12px; margin: 15px 0; }}
  .stat-card {{ background: #eef2f7; border-radius: 8px; padding: 15px; text-align: center; }}
  .stat-card .value {{ font-size: 1.5em; font-weight: 700; color: #0f3460; }}
  .stat-card .label {{ font-size: 0.85em; color: #666; }}
</style>
</head>
<body>
<div class="container">

<h1>MobileNetV2 Food Classification — Hyperparameter Sweep</h1>
<p class="subtitle">Generated: {now} | {len(all_results)} configurations | Real training data</p>

<div class="section">
  <h2>1. Dataset Statistics</h2>
  <div class="dataset-stats">
    <div class="stat-card"><div class="value">{n_classes}</div><div class="label">Classes</div></div>
    <div class="stat-card"><div class="value">{total_train}</div><div class="label">Train</div></div>
    <div class="stat-card"><div class="value">{total_val}</div><div class="label">Val</div></div>
    <div class="stat-card"><div class="value">{total_test}</div><div class="label">Test</div></div>
    <div class="stat-card"><div class="value">{total_all}</div><div class="label">Total</div></div>
    <div class="stat-card"><div class="value">224x224</div><div class="label">Image Size</div></div>
  </div>
  <div class="chart-center">
    <img src="data:image/png;base64,{dataset_b64}" alt="Dataset distribution">
  </div>
</div>

<div class="section">
  <h2>2. Hyperparameter Comparison</h2>
  <div class="chart-center">
    <img src="data:image/png;base64,{hp_compare_b64}" alt="HP comparison">
  </div>
</div>

<div class="section">
  <h2>3. Per-class F1-score Comparison</h2>
  <div class="chart-center">
    <img src="data:image/png;base64,{f1_compare_b64}" alt="F1 comparison">
  </div>
</div>

<div class="section">
  <h2>4. Metrics Radar</h2>
  <div class="chart-center">
    <img src="data:image/png;base64,{radar_b64}" alt="Radar">
  </div>
</div>

<div class="section">
  <h2>5. Per-Configuration Details</h2>
  {config_sections}
</div>

<div class="section">
  <h2>6. Final Summary</h2>
  <div style="overflow-x:auto;">
  <table>
    <thead><tr>
      <th>Config</th><th>Model</th><th>Batch</th><th>S1 LR</th><th>S2 LR</th>
      <th>S1 Ep</th><th>S2 Ep</th><th>Best Val Acc</th><th>Test Acc</th>
      <th>Precision</th><th>Recall</th><th>F1(W)</th><th>F1(M)</th>
    </tr></thead>
    <tbody>
{summary_rows}
    </tbody>
  </table>
  </div>
</div>

<div class="footer">
  Whispr Moderation Service — MobileNetV2 Hyperparameter Sweep<br>
  Real training results from train/train.py pipeline
</div>

</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ── Main entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep + report")
    parser.add_argument("--report", action="store_true",
                        help="Only generate report from existing results (skip training)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run experiments even if results exist")
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated experiment IDs to run (e.g. A,B,C). Default: all")
    args = parser.parse_args()

    # Filter experiments if requested.
    experiments = EXPERIMENTS
    if args.experiments:
        ids = [x.strip().upper() for x in args.experiments.split(",")]
        experiments = [e for e in EXPERIMENTS if e["id"] in ids]
        if not experiments:
            print(f"No matching experiments for IDs: {ids}")
            sys.exit(1)

    # Ensure we're in the right working directory.
    expected_marker = Path.cwd() / "config.yaml"
    if not expected_marker.exists():
        alt = Path(__file__).resolve().parent.parent
        if (alt / "config.yaml").exists():
            os.chdir(alt)
            print(f"Changed working directory to: {alt}")
        else:
            print("ERROR: Run this script from src/mobilenet_v2_small/")
            sys.exit(1)

    if not args.report:
        print(f"Starting hyperparameter sweep ({len(experiments)} configs)...")
        print("NOTE: This runs REAL training. Each config may take 10-60+ min depending on hardware.")
        print(f"Results will be saved to: {SWEEP_RESULTS_ROOT}/\n")
        run_experiments(experiments, force=args.force)

    # Collect results and generate report.
    print("\nCollecting results...")
    all_results = collect_results(experiments)

    if not all_results:
        print("No completed experiments found. Run training first (without --report).")
        sys.exit(1)

    print(f"Found {len(all_results)} completed experiments.")

    ds_stats = get_dataset_stats()
    output_path = Path.cwd() / SWEEP_RESULTS_ROOT / "hyperparameter_sweep_report.html"
    path = build_html_report(all_results, ds_stats, output_path)
    print(f"\nReport saved to: {path}")
    print(f"Open in browser: file://{path}")

    # Dump raw metrics JSON.
    json_path = output_path.with_name("hyperparameter_sweep_metrics.json")
    json_data = []
    for r in all_results:
        json_data.append({
            "config": r["exp"],
            "test_metrics": r["test_metrics"],
            "best_metrics": {
                k: v for k, v in r["best_metrics"].items()
                if k != "leakage_report"
            },
        })
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metrics JSON: {json_path}")


if __name__ == "__main__":
    main()
