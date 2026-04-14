from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .calibration import calibration_bins, predict_test
from .utils import ensure_dir, write_json


def _write_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _metrics(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    probs = np.clip(probs.astype(np.float64), 1e-7, 1.0 - 1e-7)
    preds = (probs >= 0.5).astype(np.int32)
    _, ece = calibration_bins(labels, probs, n_bins)
    return {
        "auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "average_precision": float(average_precision_score(labels, probs)),
        "bce": float(log_loss(labels, probs, labels=[0, 1])),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "brier": float(np.mean((probs - labels) ** 2)),
        "ece": float(ece),
    }


def export_paired_predictions(
    data_dir: str | Path,
    mlp_run_dir: str | Path,
    gru_run_dir: str | Path,
    out_path: str | Path,
) -> dict:
    labels_mlp, probs_mlp, mlp_config = predict_test(data_dir, mlp_run_dir)
    labels_gru, probs_gru, gru_config = predict_test(data_dir, gru_run_dir)
    if len(labels_mlp) != len(labels_gru) or not np.array_equal(labels_mlp, labels_gru):
        raise RuntimeError("MLP and GRU predictions are not aligned on the same labels.")
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    rows = [
        {
            "index": int(i),
            "label": int(labels_mlp[i]),
            "prob_mlp": float(probs_mlp[i]),
            "prob_gru": float(probs_gru[i]),
        }
        for i in range(len(labels_mlp))
    ]
    _write_csv(rows, out_path)
    stats = {
        "n": int(len(labels_mlp)),
        "positive_rate": float(np.mean(labels_mlp)),
        "mlp_run": str(mlp_run_dir),
        "gru_run": str(gru_run_dir),
        "mlp_config": mlp_config,
        "gru_config": gru_config,
    }
    write_json(stats, out_path.with_suffix(".metadata.json"))
    print(f"[INFO] wrote paired predictions to {out_path} n={len(labels_mlp)}")
    return stats


def load_paired_predictions(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_csv(path)
    labels = np.asarray([int(r["label"]) for r in rows], dtype=np.int32)
    mlp = np.asarray([float(r["prob_mlp"]) for r in rows], dtype=np.float64)
    gru = np.asarray([float(r["prob_gru"]) for r in rows], dtype=np.float64)
    return labels, mlp, gru


def bootstrap_compare(
    predictions_path: str | Path,
    out_dir: str | Path,
    *,
    n_bootstrap: int = 500,
    seed: int = 42,
    n_bins: int = 10,
) -> dict:
    out_dir = ensure_dir(out_dir)
    labels, probs_mlp, probs_gru = load_paired_predictions(predictions_path)
    rng = np.random.default_rng(seed)
    n = len(labels)
    model_rows = []
    delta_rows = []
    model_boot = {"mlp": [], "gru": []}
    delta_boot = []

    full_mlp = _metrics(labels, probs_mlp, n_bins=n_bins)
    full_gru = _metrics(labels, probs_gru, n_bins=n_bins)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        m = _metrics(y, probs_mlp[idx], n_bins=n_bins)
        g = _metrics(y, probs_gru[idx], n_bins=n_bins)
        model_boot["mlp"].append(m)
        model_boot["gru"].append(g)
        delta_boot.append({key: g[key] - m[key] for key in m})
        if (b + 1) % 50 == 0:
            print(f"[BOOT] completed {b + 1}/{n_bootstrap}")

    def summarize(values: list[float]) -> tuple[float, float, float]:
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    for model_name, full_metrics in [("mlp", full_mlp), ("gru", full_gru)]:
        for metric in full_metrics:
            vals = [row[metric] for row in model_boot[model_name]]
            mean, lo, hi = summarize(vals)
            model_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "full_sample": full_metrics[metric],
                    "bootstrap_mean": mean,
                    "lower_95": lo,
                    "upper_95": hi,
                }
            )

    for metric in full_mlp:
        vals = [row[metric] for row in delta_boot]
        mean, lo, hi = summarize(vals)
        delta_rows.append(
            {
                "comparison": "gru_minus_mlp",
                "metric": metric,
                "full_sample_delta": full_gru[metric] - full_mlp[metric],
                "bootstrap_mean_delta": mean,
                "lower_95": lo,
                "upper_95": hi,
                "better_direction": "positive" if metric in {"auc", "average_precision", "accuracy", "f1"} else "negative",
            }
        )

    _write_csv(model_rows, out_dir / "bootstrap_model_ci.csv")
    _write_csv(delta_rows, out_dir / "paired_delta_summary.csv")
    write_json({"n": n, "n_bootstrap": n_bootstrap, "seed": seed, "n_bins": n_bins}, out_dir / "bootstrap_config.json")
    plot_bootstrap_delta(delta_boot, "auc", out_dir / "bootstrap_delta_auc.png")
    plot_bootstrap_delta(delta_boot, "bce", out_dir / "bootstrap_delta_bce.png")
    return {"model_ci": model_rows, "delta": delta_rows}


def plot_bootstrap_delta(delta_boot: list[dict], metric: str, out_path: str | Path) -> None:
    vals = np.asarray([row[metric] for row in delta_boot], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=40, color="#2f6f73", alpha=0.85)
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.axvline(vals.mean(), color="#9b4d3d", linewidth=2, label=f"mean={vals.mean():.4f}")
    ax.set_title(f"Paired Bootstrap: GRU - MLP {metric.upper()}")
    ax.set_xlabel(f"Delta {metric}")
    ax.set_ylabel("Bootstrap samples")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_roc_pr(predictions_path: str | Path, out_dir: str | Path) -> None:
    labels, probs_mlp, probs_gru = load_paired_predictions(predictions_path)
    out_dir = ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probs, color in [("MLP seq50", probs_mlp, "#2f6f73"), ("GRU seq50", probs_gru, "#9b4d3d")]:
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}", color=color)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("10k Test ROC Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve_10k.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probs, color in [("MLP seq50", probs_mlp, "#2f6f73"), ("GRU seq50", probs_gru, "#9b4d3d")]:
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        ax.plot(recall, precision, label=f"{name} AP={ap:.3f}", color=color)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("10k Test Precision-Recall Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve_10k.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 paired prediction and bootstrap robustness tools.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    export = sub.add_parser("export")
    export.add_argument("--data-dir", required=True)
    export.add_argument("--mlp-run-dir", required=True)
    export.add_argument("--gru-run-dir", required=True)
    export.add_argument("--out", required=True)

    boot = sub.add_parser("bootstrap")
    boot.add_argument("--predictions", required=True)
    boot.add_argument("--out-dir", required=True)
    boot.add_argument("--n-bootstrap", type=int, default=500)
    boot.add_argument("--seed", type=int, default=42)
    boot.add_argument("--bins", type=int, default=10)

    curves = sub.add_parser("curves")
    curves.add_argument("--predictions", required=True)
    curves.add_argument("--out-dir", required=True)

    args = parser.parse_args()
    if args.cmd == "export":
        export_paired_predictions(args.data_dir, args.mlp_run_dir, args.gru_run_dir, args.out)
    elif args.cmd == "bootstrap":
        bootstrap_compare(args.predictions, args.out_dir, n_bootstrap=args.n_bootstrap, seed=args.seed, n_bins=args.bins)
    elif args.cmd == "curves":
        plot_roc_pr(args.predictions, args.out_dir)


if __name__ == "__main__":
    main()
