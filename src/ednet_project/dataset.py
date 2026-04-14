from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import read_json


AGG_FEATURE_DIM = 8


class EdNetSequenceDataset(Dataset):
    """Windowed next-response dataset built from prepared KT1 arrays."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        seq_len: int,
        *,
        include_metadata: bool = True,
        limit_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = int(seq_len)
        self.include_metadata = include_metadata

        arrays = np.load(self.data_dir / "events.npz")
        self.user_idx = arrays["user_idx"]
        self.question_idx = arrays["question_idx"]
        self.correct = arrays["correct"]
        self.part_idx = arrays["part_idx"]
        self.tag_idx = arrays["tag_idx"]
        self.elapsed_log = arrays["elapsed_log"]
        self.offsets = arrays["offsets"]
        self.lengths = arrays["lengths"]

        split_users = read_json(self.data_dir / "split_users.json")
        users = [int(item["user_idx"]) for item in split_users[split]]
        target_positions = []
        for user in users:
            start = int(self.offsets[user])
            length = int(self.lengths[user])
            if length > 1:
                target_positions.append(np.arange(start + 1, start + length, dtype=np.int64))
        if target_positions:
            self.target_positions = np.concatenate(target_positions)
        else:
            self.target_positions = np.asarray([], dtype=np.int64)

        if limit_samples is not None and limit_samples > 0 and len(self.target_positions) > limit_samples:
            rng = np.random.default_rng(seed)
            choice = rng.choice(len(self.target_positions), size=limit_samples, replace=False)
            self.target_positions = self.target_positions[np.sort(choice)]

    def __len__(self) -> int:
        return int(len(self.target_positions))

    def _agg_features(self, hist_start: int, target_idx: int) -> np.ndarray:
        corr = self.correct[hist_start:target_idx].astype(np.float32)
        elapsed = self.elapsed_log[hist_start:target_idx].astype(np.float32)
        hist_len = len(corr)
        recent5 = corr[-5:]
        recent20 = corr[-20:]
        elapsed5 = elapsed[-5:]
        return np.asarray(
            [
                float(corr.mean()) if hist_len else 0.0,
                float(recent5.mean()) if len(recent5) else 0.0,
                float(recent20.mean()) if len(recent20) else 0.0,
                float(np.log1p(hist_len) / np.log1p(200.0)),
                float(elapsed.mean()) if hist_len else 0.0,
                float(elapsed5.mean()) if len(elapsed5) else 0.0,
                float(corr[-1]) if hist_len else 0.0,
                float(elapsed[-1]) if hist_len else 0.0,
            ],
            dtype=np.float32,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        target_idx = int(self.target_positions[index])
        user = int(self.user_idx[target_idx])
        user_start = int(self.offsets[user])
        hist_start = max(user_start, target_idx - self.seq_len)
        hist_len = target_idx - hist_start

        q = np.zeros(self.seq_len, dtype=np.int64)
        c = np.zeros(self.seq_len, dtype=np.int64)
        p = np.zeros(self.seq_len, dtype=np.int64)
        t = np.zeros(self.seq_len, dtype=np.int64)
        e = np.zeros(self.seq_len, dtype=np.float32)

        sl = slice(hist_start, target_idx)
        q[:hist_len] = self.question_idx[sl]
        c[:hist_len] = self.correct[sl].astype(np.int64) + 1
        if self.include_metadata:
            p[:hist_len] = self.part_idx[sl]
            t[:hist_len] = self.tag_idx[sl]
        e[:hist_len] = self.elapsed_log[sl]

        target_part = int(self.part_idx[target_idx]) if self.include_metadata else 0
        target_tag = int(self.tag_idx[target_idx]) if self.include_metadata else 0

        return {
            "hist_question": torch.from_numpy(q),
            "hist_correct": torch.from_numpy(c),
            "hist_part": torch.from_numpy(p),
            "hist_tag": torch.from_numpy(t),
            "hist_elapsed": torch.from_numpy(e),
            "length": torch.tensor(hist_len, dtype=torch.long),
            "target_question": torch.tensor(int(self.question_idx[target_idx]), dtype=torch.long),
            "target_part": torch.tensor(target_part, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
            "agg": torch.from_numpy(self._agg_features(hist_start, target_idx)),
            "label": torch.tensor(float(self.correct[target_idx]), dtype=torch.float32),
        }
