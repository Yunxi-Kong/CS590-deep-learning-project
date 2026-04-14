from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader

from .dataset import EdNetSequenceDataset
from .models import build_model
from .utils import ensure_dir, read_json, write_json


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))


def calibration_bins(labels: np.ndarray, probs: np.ndarray, n_bins: int) -> tuple[list[dict], float]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    total = len(labels)
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        if count == 0:
            avg_conf = 0.5 * (lo + hi)
            avg_acc = np.nan
            gap = np.nan
        else:
            avg_conf = float(probs[mask].mean())
            avg_acc = float(labels[mask].mean())
            gap = abs(avg_acc - avg_conf)
            ece += (count / max(total, 1)) * gap
        rows.append(
            {
                "bin": i,
                "bin_low": float(lo),
                "bin_high": float(hi),
                "count": count,
                "avg_confidence": avg_conf,
                "empirical_accuracy": avg_acc,
                "abs_gap": gap,
            }
        )
    return rows, float(ece)


def _load_model(run_dir: Path, data_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    config = read_json(run_dir / "config.json")
    mapping = read_json(data_dir / "mappings.json")
    model = build_model(
        config["model"],
        num_questions=int(mapping["num_questions"]),
        num_parts=int(mapping["num_parts"]),
        num_tags=int(mapping["num_tags"]),
        embedding_dim=int(config["embedding_dim"]),
        hidden_size=int(config["hidden_size"]),
        dropout=float(config["dropout"]),
        include_metadata=bool(config.get("include_metadata", True)),
        sequence_use_agg=bool(config.get("sequence_use_agg", False)),
        interaction_history=bool(config.get("interaction_history", False)),
    ).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()
    return model, config


@torch.no_grad()
def predict_test(data_dir: str | Path, run_dir: str | Path, batch_size: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    data_dir = Path(data_dir)
    run_dir = Path(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = _load_model(run_dir, data_dir, device)
    dataset = EdNetSequenceDataset(
        data_dir,
        "test",
        int(config["seq_len"]),
        include_metadata=bool(config.get("include_metadata", True)),
    )
    loader = DataLoader(dataset, batch_size=batch_size or int(config.get("batch_size", 512)), shuffle=False, num_workers=0)
    labels_all = []
    probs_all = []
    for batch in loader:
        labels_all.append(batch["label"].numpy())
        logits = model(_to_device(batch, device)).detach().cpu().numpy()
        probs_all.append(_sigmoid(logits))
    return np.concatenate(labels_all), np.concatenate(probs_all), config


def evaluate_calibration(labels: np.ndarray, probs: np.ndarray, n_bins: int) -> tuple[dict, list[dict]]:
    preds = (probs >= 0.5).astype(np.int32)
    bins, ece = calibration_bins(labels, probs, n_bins)
    metrics = {
        "n": int(len(labels)),
        "pos_rate": float(labels.mean()),
        "auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "bce": float(log_loss(labels, np.clip(probs, 1e-7, 1.0 - 1e-7), labels=[0, 1])),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "brier": float(np.mean((probs - labels) ** 2)),
        "ece": ece,
        "mean_prob": float(probs.mean()),
    }
    return metrics, bins


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_reliability(all_bins: dict[str, list[dict]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", label="Perfect calibration")
    colors = ["#2f6f73", "#9b4d3d", "#4b6f9f", "#7a6a31"]
    for idx, (name, rows) in enumerate(all_bins.items()):
        xs = [float(row["avg_confidence"]) for row in rows if row["empirical_accuracy"] == row["empirical_accuracy"]]
        ys = [float(row["empirical_accuracy"]) for row in rows if row["empirical_accuracy"] == row["empirical_accuracy"]]
        ax.plot(xs, ys, marker="o", color=colors[idx % len(colors)], label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_calibration(
    data_dir: str | Path,
    runs: list[tuple[str, str | Path]],
    out_dir: str | Path,
    *,
    n_bins: int = 10,
) -> dict:
    out_dir = ensure_dir(out_dir)
    metrics_rows = []
    all_bins: dict[str, list[dict]] = {}
    for name, run_dir in runs:
        labels, probs, config = predict_test(data_dir, run_dir)
        metrics, bins = evaluate_calibration(labels, probs, n_bins)
        row = {"run_name": name, "model": config["model"], "seq_len": config["seq_len"], **metrics}
        metrics_rows.append(row)
        bin_rows = [{"run_name": name, **item} for item in bins]
        all_bins[name] = bins
        _write_csv(bin_rows, out_dir / f"{name}_calibration_bins.csv")
        write_json({"run_name": name, "config": config, "metrics": metrics}, out_dir / f"{name}_calibration_metrics.json")
        print(f"[CAL] {name} auc={metrics['auc']:.4f} bce={metrics['bce']:.4f} brier={metrics['brier']:.4f} ece={metrics['ece']:.4f}")
    _write_csv(metrics_rows, out_dir / "calibration_summary.csv")
    _plot_reliability(all_bins, out_dir / "reliability_diagram.png")
    return {"metrics": metrics_rows, "out_dir": str(out_dir)}


def _parse_runs(values: list[str]) -> list[tuple[str, str]]:
    runs = []
    for value in values:
        if "=" not in value:
            raise ValueError("--run entries must have the form name=path")
        name, path = value.split("=", 1)
        runs.append((name, path))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate calibration for trained EDNet models.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run", action="append", required=True, help="Model run in name=path format.")
    parser.add_argument("--out-dir", default="phase2_improvement/results/calibration")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    run_calibration(args.data_dir, _parse_runs(args.run), args.out_dir, n_bins=args.bins)


if __name__ == "__main__":
    main()
