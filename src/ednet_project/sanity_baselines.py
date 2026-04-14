from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .dataset import EdNetSequenceDataset
from .train import compute_metrics
from .utils import ensure_dir, read_json, write_json


def _logit(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return np.log(probs / (1.0 - probs))


def _targets(dataset: EdNetSequenceDataset) -> tuple[np.ndarray, np.ndarray]:
    positions = dataset.target_positions
    labels = dataset.correct[positions].astype(np.float32)
    questions = dataset.question_idx[positions].astype(np.int64)
    return labels, questions


def run_sanity_baselines(
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    run_name: str = "sanity_global_question",
    seq_len: int = 20,
    smoothing: float = 20.0,
) -> dict:
    data_dir = Path(data_dir)
    run_dir = ensure_dir(Path(out_dir) / run_name)
    mapping = read_json(data_dir / "mappings.json")
    stats = read_json(data_dir / "dataset_stats.json")

    train_ds = EdNetSequenceDataset(data_dir, "train", seq_len)
    val_ds = EdNetSequenceDataset(data_dir, "val", seq_len)
    test_ds = EdNetSequenceDataset(data_dir, "test", seq_len)
    y_train, q_train = _targets(train_ds)
    global_prob = float(y_train.mean())

    num_questions = int(mapping["num_questions"])
    sums = np.zeros(num_questions, dtype=np.float64)
    counts = np.zeros(num_questions, dtype=np.float64)
    np.add.at(sums, q_train, y_train)
    np.add.at(counts, q_train, 1.0)
    question_probs = (sums + smoothing * global_prob) / (counts + smoothing)

    def eval_global(ds: EdNetSequenceDataset) -> dict[str, float]:
        y, _ = _targets(ds)
        probs = np.full_like(y, global_prob, dtype=np.float64)
        return compute_metrics(y, _logit(probs))

    def eval_question(ds: EdNetSequenceDataset) -> dict[str, float]:
        y, q = _targets(ds)
        probs = question_probs[q]
        return compute_metrics(y, _logit(probs))

    final = {
        "run_name": run_name,
        "model": "sanity",
        "best_epoch": 0,
        "elapsed_seconds": 0.0,
        "config": {
            "data_dir": str(data_dir),
            "out_dir": str(out_dir),
            "run_name": run_name,
            "seq_len": seq_len,
            "smoothing": smoothing,
            "include_metadata": False,
            "dataset_stats": stats,
            "model_parameters": 0,
        },
        "global_val": eval_global(val_ds),
        "global_test": eval_global(test_ds),
        "question_val": eval_question(val_ds),
        "question_test": eval_question(test_ds),
        "val": eval_question(val_ds),
        "test": eval_question(test_ds),
    }
    write_json(final, run_dir / "final_metrics.json")
    write_json({"global_prob": global_prob, "smoothing": smoothing}, run_dir / "config.json")
    print(
        f"[RESULT] {run_name} question_test_auc={final['test']['auc']:.4f} "
        f"question_test_bce={final['test']['bce']:.4f} global_test_bce={final['global_test']['bce']:.4f}"
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-neural sanity baselines.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="results/runs")
    parser.add_argument("--run-name", default="sanity_global_question")
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--smoothing", type=float, default=20.0)
    args = parser.parse_args()
    if math.isnan(args.smoothing) or args.smoothing < 0:
        raise ValueError("--smoothing must be non-negative")
    run_sanity_baselines(args.data_dir, args.out_dir, run_name=args.run_name, seq_len=args.seq_len, smoothing=args.smoothing)


if __name__ == "__main__":
    main()
