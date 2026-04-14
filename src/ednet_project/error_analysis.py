from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import EdNetSequenceDataset
from .models import build_model
from .train import compute_metrics
from .utils import ensure_dir, read_json


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _logit_to_prob(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))


def _prob_to_logit(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return np.log(probs / (1.0 - probs))


def _length_bin(length: int) -> str:
    if length <= 5:
        return "01-05"
    if length <= 10:
        return "06-10"
    if length <= 20:
        return "11-20"
    if length <= 50:
        return "21-50"
    return "51+"


def _group_metrics(labels: np.ndarray, probs: np.ndarray, groups: np.ndarray) -> list[dict]:
    rows = []
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        metrics = compute_metrics(labels[mask], _prob_to_logit(probs[mask]))
        rows.append({"group": str(group), **metrics})
    return rows


def _write_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_rows(rows: list[dict], metric: str, title: str, path: Path) -> None:
    if not rows:
        return
    labels = [row["group"] for row in rows]
    values = [float(row[metric]) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color="#2f6f73")
    ax.set_title(title)
    ax.set_ylabel(metric.upper())
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_error_analysis(data_dir: str | Path, run_dir: str | Path, out_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    run_dir = Path(run_dir)
    out_dir = ensure_dir(out_dir)
    config = read_json(run_dir / "config.json")
    mapping = read_json(data_dir / "mappings.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    include_metadata = bool(config.get("include_metadata", True))
    dataset = EdNetSequenceDataset(data_dir, "test", int(config["seq_len"]), include_metadata=include_metadata)
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 512)), shuffle=False, num_workers=0)

    model = build_model(
        config["model"],
        num_questions=int(mapping["num_questions"]),
        num_parts=int(mapping["num_parts"]),
        num_tags=int(mapping["num_tags"]),
        embedding_dim=int(config["embedding_dim"]),
        hidden_size=int(config["hidden_size"]),
        dropout=float(config["dropout"]),
        include_metadata=include_metadata,
        sequence_use_agg=bool(config.get("sequence_use_agg", False)),
        interaction_history=bool(config.get("interaction_history", False)),
    ).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()

    labels_all = []
    probs_all = []
    lengths_all = []
    parts_all = []
    with torch.no_grad():
        for batch in loader:
            labels_all.append(batch["label"].numpy())
            lengths_all.append(batch["length"].numpy())
            parts_all.append(batch["target_part"].numpy())
            batch = _to_device(batch, device)
            probs_all.append(_logit_to_prob(model(batch).detach().cpu().numpy()))

    labels = np.concatenate(labels_all)
    probs = np.concatenate(probs_all)
    lengths = np.concatenate(lengths_all)
    parts = np.concatenate(parts_all)
    length_bins = np.asarray([_length_bin(int(x)) for x in lengths])
    part_groups = np.asarray([f"part_{int(x)}" for x in parts])

    length_rows = _group_metrics(labels, probs, length_bins)
    part_rows = _group_metrics(labels, probs, part_groups)
    _write_rows(length_rows, out_dir / "test_metrics_by_history_length.csv")
    _write_rows(part_rows, out_dir / "test_metrics_by_part.csv")
    _plot_rows(length_rows, "auc", "Test AUC by Available History Length", out_dir / "auc_by_history_length.png")
    _plot_rows(part_rows, "auc", "Test AUC by Question Part", out_dir / "auc_by_part.png")
    print(f"[INFO] wrote error analysis to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a trained model by history length and question part.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default="results/error_analysis")
    args = parser.parse_args()
    run_error_analysis(args.data_dir, args.run_dir, args.out_dir)


if __name__ == "__main__":
    main()
