from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .dataset import EdNetSequenceDataset
from .models import build_model
from .utils import ensure_dir, read_json, set_seed, write_json


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def compute_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    labels = labels.astype(np.float32)
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    preds = (probs >= 0.5).astype(np.int32)
    metrics = {
        "n": float(len(labels)),
        "pos_rate": float(labels.mean()) if len(labels) else math.nan,
        "bce": float(log_loss(labels, probs, labels=[0, 1])) if len(labels) else math.nan,
        "accuracy": float(accuracy_score(labels, preds)) if len(labels) else math.nan,
        "f1": float(f1_score(labels, preds, zero_division=0)) if len(labels) else math.nan,
        "mean_prob": float(probs.mean()) if len(labels) else math.nan,
    }
    metrics["auc"] = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else math.nan
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for batch in loader:
        batch = _to_device(batch, device)
        logits = model(batch)
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(batch["label"].detach().cpu().numpy())
    if not logits_all:
        return compute_metrics(np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32))
    return compute_metrics(np.concatenate(labels_all), np.concatenate(logits_all))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = loss_fn(logits, batch["label"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        n = int(batch["label"].shape[0])
        total_loss += float(loss.detach().cpu()) * n
        total_n += n
    return total_loss / max(total_n, 1)


def train_model(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    run_dir = ensure_dir(Path(args.out_dir) / args.run_name)
    mapping = read_json(data_dir / "mappings.json")
    stats = read_json(data_dir / "dataset_stats.json")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    include_metadata = not args.no_metadata

    train_ds = EdNetSequenceDataset(
        data_dir,
        "train",
        args.seq_len,
        include_metadata=include_metadata,
        limit_samples=args.limit_train_samples,
        seed=args.seed,
    )
    val_ds = EdNetSequenceDataset(
        data_dir,
        "val",
        args.seq_len,
        include_metadata=include_metadata,
        limit_samples=args.limit_eval_samples,
        seed=args.seed,
    )
    test_ds = EdNetSequenceDataset(
        data_dir,
        "test",
        args.seq_len,
        include_metadata=include_metadata,
        limit_samples=args.limit_eval_samples,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        args.model,
        num_questions=int(mapping["num_questions"]),
        num_parts=int(mapping["num_parts"]),
        num_tags=int(mapping["num_tags"]),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        include_metadata=include_metadata,
        sequence_use_agg=args.sequence_use_agg,
        interaction_history=args.interaction_history,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "include_metadata": include_metadata,
            "sequence_use_agg": args.sequence_use_agg,
            "interaction_history": args.interaction_history,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "dataset_stats": stats,
            "model_parameters": int(sum(p.numel() for p in model.parameters())),
        }
    )
    write_json(config, run_dir / "config.json")

    best_score = -float("inf")
    best_epoch = 0
    patience_left = args.patience
    rows: list[dict[str, Any]] = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, device)
        score = val_metrics["auc"]
        if math.isnan(score):
            score = -val_metrics["bce"]

        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        rows.append(row)
        print(
            f"[EPOCH {epoch:02d}] train_loss={train_loss:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_bce={val_metrics['bce']:.4f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience_left = args.patience
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[INFO] early stopping at epoch {epoch}")
                break

    if rows:
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    final = {
        "run_name": args.run_name,
        "model": args.model,
        "best_epoch": best_epoch,
        "elapsed_seconds": time.time() - start_time,
        "config": config,
        "val": val_metrics,
        "test": test_metrics,
    }
    write_json(final, run_dir / "final_metrics.json")
    print(
        f"[RESULT] {args.run_name} test_auc={test_metrics['auc']:.4f} "
        f"test_bce={test_metrics['bce']:.4f} test_acc={test_metrics['accuracy']:.4f}"
    )
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an EDNet next-response model.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="results/runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", choices=["mlp", "gru", "lstm"], default="gru")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-eval-samples", type=int, default=None)
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--sequence-use-agg", action="store_true")
    parser.add_argument("--interaction-history", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    train_model(parse_args())


if __name__ == "__main__":
    main()
