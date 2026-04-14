from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir, read_json


def collect_runs(runs_dir: str | Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(Path(runs_dir).glob("*/final_metrics.json")):
        item = read_json(metrics_path)
        config = item.get("config", {})
        test = item.get("test", {})
        val = item.get("val", {})
        rows.append(
            {
                "run_name": item.get("run_name", metrics_path.parent.name),
                "model": item.get("model", config.get("model", "")),
                "seq_len": config.get("seq_len", ""),
                "hidden_size": config.get("hidden_size", ""),
                "embedding_dim": config.get("embedding_dim", ""),
                "dropout": config.get("dropout", ""),
                "lr": config.get("lr", ""),
                "include_metadata": config.get("include_metadata", ""),
                "sequence_use_agg": config.get("sequence_use_agg", ""),
                "interaction_history": config.get("interaction_history", ""),
                "best_epoch": item.get("best_epoch", ""),
                "val_auc": val.get("auc", np.nan),
                "val_bce": val.get("bce", np.nan),
                "test_auc": test.get("auc", np.nan),
                "test_bce": test.get("bce", np.nan),
                "test_accuracy": test.get("accuracy", np.nan),
                "test_f1": test.get("f1", np.nan),
                "test_pos_rate": test.get("pos_rate", np.nan),
                "test_n": test.get("n", np.nan),
            }
        )
    return rows


def write_summary(rows: list[dict], out_csv: str | Path) -> None:
    if not rows:
        return
    out_csv = Path(out_csv)
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict], metric: str, out_path: str | Path) -> None:
    valid = [r for r in rows if r.get(metric) == r.get(metric)]
    if not valid:
        return
    labels = [r["run_name"] for r in valid]
    values = [float(r[metric]) for r in valid]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4))
    ax.bar(labels, values, color="#2f6f73")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(f"Model Comparison: {metric.replace('_', ' ').title()}")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize EDNet experiment runs.")
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--out-csv", default="results/summary.csv")
    parser.add_argument("--figures-dir", default="results/figures")
    args = parser.parse_args()
    rows = collect_runs(args.runs_dir)
    write_summary(rows, args.out_csv)
    if rows:
        plot_metric(rows, "test_auc", Path(args.figures_dir) / "model_comparison_auc.png")
        plot_metric(rows, "test_bce", Path(args.figures_dir) / "model_comparison_bce.png")
        print(f"[INFO] wrote {args.out_csv} and figures for {len(rows)} runs")
    else:
        print("[WARN] no completed runs found")


if __name__ == "__main__":
    main()
