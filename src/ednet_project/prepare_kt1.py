from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import ensure_dir, write_json


REQUIRED_KT1_COLUMNS = {"timestamp", "question_id", "user_answer"}


def _numeric_sort_key(text: str) -> tuple[int, str]:
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits) if digits else 10**18, text)


def _file_sort_key(path: Path) -> tuple[int, str]:
    return _numeric_sort_key(path.stem)


def _first_tag(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text.split(";")[0].strip() if text else ""


def load_question_metadata(questions_path: str | Path) -> tuple[pd.DataFrame, dict]:
    questions = pd.read_csv(
        questions_path,
        dtype={
            "question_id": "string",
            "correct_answer": "string",
            "part": "Int64",
            "tags": "string",
        },
    )
    questions = questions.dropna(subset=["question_id", "correct_answer"]).copy()
    questions["question_id"] = questions["question_id"].astype(str).str.strip()
    questions["correct_answer"] = questions["correct_answer"].astype(str).str.strip().str.lower()
    questions["part_idx"] = questions["part"].fillna(0).astype(int).clip(lower=0)
    questions["first_tag"] = questions["tags"].map(_first_tag)

    question_ids = sorted(questions["question_id"].unique(), key=_numeric_sort_key)
    question_to_idx = {qid: i + 1 for i, qid in enumerate(question_ids)}
    tag_values = sorted(t for t in questions["first_tag"].unique() if t)
    tag_to_idx = {tag: i + 1 for i, tag in enumerate(tag_values)}

    questions["question_idx"] = questions["question_id"].map(question_to_idx).astype(int)
    questions["first_tag_idx"] = questions["first_tag"].map(tag_to_idx).fillna(0).astype(int)
    meta = questions[
        ["question_id", "correct_answer", "part_idx", "first_tag", "first_tag_idx", "question_idx"]
    ].drop_duplicates("question_id")
    mapping = {
        "num_questions": int(max(question_to_idx.values(), default=0) + 1),
        "num_parts": int(meta["part_idx"].max() + 1),
        "num_tags": int(max(tag_to_idx.values(), default=0) + 1),
        "question_to_idx": question_to_idx,
        "tag_to_idx": tag_to_idx,
    }
    return meta, mapping


def _read_user_file(path: Path, question_meta: pd.DataFrame) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(
            path,
            dtype={
                "question_id": "string",
                "user_answer": "string",
                "timestamp": "Int64",
                "solving_id": "Int64",
                "elapsed_time": "Float64",
            },
        )
    except Exception as exc:
        print(f"[WARN] failed to read {path.name}: {exc}")
        return None

    if not REQUIRED_KT1_COLUMNS.issubset(df.columns):
        print(f"[WARN] skipping {path.name}: missing required columns")
        return None

    sort_cols = [c for c in ["timestamp", "solving_id"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort")
    if "elapsed_time" not in df.columns:
        df["elapsed_time"] = 0.0

    df = df[["timestamp", "question_id", "user_answer", "elapsed_time"]].copy()
    df["question_id"] = df["question_id"].astype(str).str.strip()
    df["user_answer"] = df["user_answer"].astype(str).str.strip().str.lower()
    df = df.replace({"user_answer": {"<NA>": np.nan, "nan": np.nan, "": np.nan}})
    df = df.dropna(subset=["question_id", "user_answer"])
    df = df.merge(question_meta, on="question_id", how="inner", validate="many_to_one")
    if df.empty:
        return None

    df["label"] = (df["user_answer"] == df["correct_answer"]).astype(np.int8)
    elapsed_ms = pd.to_numeric(df["elapsed_time"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["elapsed_log"] = np.log1p(elapsed_ms / 1000.0).astype(np.float32)
    return df


def _split_users(num_users: int, seed: int, train_frac: float, val_frac: float) -> dict[str, list[int]]:
    users = list(range(num_users))
    rng = random.Random(seed)
    rng.shuffle(users)
    n_train = int(math.floor(num_users * train_frac))
    n_val = int(math.floor(num_users * val_frac))
    return {
        "train": users[:n_train],
        "val": users[n_train : n_train + n_val],
        "test": users[n_train + n_val :],
    }


def _iter_candidate_files(kt1_dir: Path, seed: int, random_order: bool) -> Iterable[Path]:
    files = sorted(kt1_dir.glob("*.csv"), key=_file_sort_key)
    if random_order:
        rng = random.Random(seed)
        rng.shuffle(files)
    return files


def build_kt1_dataset(
    kt1_dir: str | Path,
    questions_path: str | Path,
    out_dir: str | Path,
    *,
    target_users: int = 2000,
    max_raw_users: int = 10000,
    min_responses: int = 10,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_order: bool = True,
) -> dict:
    kt1_dir = Path(kt1_dir)
    questions_path = Path(questions_path)
    out_dir = ensure_dir(out_dir)
    question_meta, mapping = load_question_metadata(questions_path)

    rows: list[pd.DataFrame] = []
    user_ids: list[str] = []
    offsets: list[int] = []
    lengths: list[int] = []
    raw_seen = filtered_short = failed = total_events = 0

    print(f"[INFO] reading question metadata from {questions_path}")
    print(f"[INFO] scanning KT1 user files in {kt1_dir}")
    for path in _iter_candidate_files(kt1_dir, seed=seed, random_order=random_order):
        if raw_seen >= max_raw_users or len(user_ids) >= target_users:
            break
        raw_seen += 1
        user_df = _read_user_file(path, question_meta)
        if user_df is None:
            failed += 1
            continue
        if len(user_df) < min_responses:
            filtered_short += 1
            continue

        user_idx = len(user_ids)
        user_df.insert(0, "user_idx", user_idx)
        user_df.insert(0, "user_id", path.stem)
        offsets.append(total_events)
        lengths.append(len(user_df))
        total_events += len(user_df)
        user_ids.append(path.stem)
        rows.append(user_df)
        if len(user_ids) % 250 == 0:
            print(f"[INFO] kept {len(user_ids)} users after reading {raw_seen} raw files")

    if not rows:
        raise RuntimeError("No usable KT1 users were found. Check paths or filtering thresholds.")

    events = pd.concat(rows, ignore_index=True)
    events = events[
        [
            "user_id",
            "user_idx",
            "timestamp",
            "question_id",
            "question_idx",
            "user_answer",
            "correct_answer",
            "label",
            "part_idx",
            "first_tag",
            "first_tag_idx",
            "elapsed_time",
            "elapsed_log",
        ]
    ]
    split = _split_users(len(user_ids), seed=seed, train_frac=train_frac, val_frac=val_frac)
    split_with_ids = {
        key: [{"user_idx": int(i), "user_id": user_ids[i]} for i in value] for key, value in split.items()
    }

    np.savez_compressed(
        out_dir / "events.npz",
        user_idx=events["user_idx"].to_numpy(np.int32),
        question_idx=events["question_idx"].to_numpy(np.int32),
        correct=events["label"].to_numpy(np.float32),
        part_idx=events["part_idx"].to_numpy(np.int16),
        tag_idx=events["first_tag_idx"].to_numpy(np.int16),
        elapsed_log=events["elapsed_log"].to_numpy(np.float32),
        offsets=np.asarray(offsets, dtype=np.int64),
        lengths=np.asarray(lengths, dtype=np.int32),
    )
    events.to_csv(out_dir / "cleaned_responses.csv.gz", index=False, compression="gzip")
    mapping["user_ids"] = user_ids
    write_json(mapping, out_dir / "mappings.json")
    write_json(split_with_ids, out_dir / "split_users.json")

    stats = {
        "kt1_dir": str(kt1_dir),
        "questions_path": str(questions_path),
        "out_dir": str(out_dir),
        "seed": seed,
        "target_users": target_users,
        "max_raw_users": max_raw_users,
        "min_responses": min_responses,
        "raw_files_seen": raw_seen,
        "kept_users": len(user_ids),
        "filtered_short_users": filtered_short,
        "failed_users": failed,
        "responses": int(len(events)),
        "sequence_samples": int(len(events) - len(user_ids)),
        "label_mean": float(events["label"].mean()),
        "split_user_counts": {k: len(v) for k, v in split.items()},
        "split_sample_counts": {k: int(sum(lengths[i] - 1 for i in v)) for k, v in split.items()},
        "num_questions": mapping["num_questions"],
        "num_parts": mapping["num_parts"],
        "num_tags": mapping["num_tags"],
    }
    write_json(stats, out_dir / "dataset_stats.json")
    print("[INFO] dataset build complete")
    print(f"[INFO] kept_users={stats['kept_users']} responses={stats['responses']} samples={stats['sequence_samples']}")
    print(f"[INFO] label_mean={stats['label_mean']:.4f} split_samples={stats['split_sample_counts']}")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a KT1 next-response dataset.")
    parser.add_argument("--kt1-dir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out-dir", default="data/processed/kt1_tiny")
    parser.add_argument("--target-users", type=int, default=2000)
    parser.add_argument("--max-raw-users", type=int, default=10000)
    parser.add_argument("--min-responses", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--ordered", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_kt1_dataset(
        kt1_dir=args.kt1_dir,
        questions_path=args.questions,
        out_dir=args.out_dir,
        target_users=args.target_users,
        max_raw_users=args.max_raw_users,
        min_responses=args.min_responses,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        random_order=not args.ordered,
    )


if __name__ == "__main__":
    main()
