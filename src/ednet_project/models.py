from __future__ import annotations

import torch
from torch import nn

from .dataset import AGG_FEATURE_DIM


class MLPBaseline(nn.Module):
    def __init__(
        self,
        *,
        num_questions: int,
        num_parts: int,
        num_tags: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        dropout: float = 0.2,
        include_metadata: bool = True,
    ) -> None:
        super().__init__()
        self.include_metadata = include_metadata
        meta_dim = max(8, embedding_dim // 4)
        self.question_emb = nn.Embedding(num_questions, embedding_dim, padding_idx=0)
        self.part_emb = nn.Embedding(num_parts, meta_dim, padding_idx=0)
        self.tag_emb = nn.Embedding(num_tags, meta_dim, padding_idx=0)

        input_dim = embedding_dim + AGG_FEATURE_DIM
        if include_metadata:
            input_dim += 2 * meta_dim
        mid = max(32, hidden_size // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [self.question_emb(batch["target_question"]), batch["agg"]]
        if self.include_metadata:
            parts.append(self.part_emb(batch["target_part"]))
            parts.append(self.tag_emb(batch["target_tag"]))
        x = torch.cat(parts, dim=-1)
        return self.net(x).squeeze(-1)


class SequenceModel(nn.Module):
    def __init__(
        self,
        *,
        num_questions: int,
        num_parts: int,
        num_tags: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        dropout: float = 0.2,
        include_metadata: bool = True,
        rnn_type: str = "gru",
        use_agg_features: bool = False,
        interaction_history: bool = False,
    ) -> None:
        super().__init__()
        self.include_metadata = include_metadata
        self.rnn_type = rnn_type.lower()
        self.use_agg_features = use_agg_features
        self.interaction_history = interaction_history
        self.num_questions = num_questions
        meta_dim = max(8, embedding_dim // 4)

        self.question_emb = nn.Embedding(num_questions, embedding_dim, padding_idx=0)
        self.interaction_emb = nn.Embedding(num_questions * 2, embedding_dim, padding_idx=0)
        self.correct_emb = nn.Embedding(3, 8, padding_idx=0)
        self.part_emb = nn.Embedding(num_parts, meta_dim, padding_idx=0)
        self.tag_emb = nn.Embedding(num_tags, meta_dim, padding_idx=0)

        rnn_input_dim = embedding_dim + 1
        if not interaction_history:
            rnn_input_dim += 8
        if include_metadata:
            rnn_input_dim += 2 * meta_dim
        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=rnn_input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        target_dim = embedding_dim
        if include_metadata:
            target_dim += 2 * meta_dim
        if use_agg_features:
            target_dim += AGG_FEATURE_DIM
        self.head = nn.Sequential(
            nn.Linear(hidden_size + target_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.interaction_history:
            corr_offset = (batch["hist_correct"] - 1).clamp(min=0) * self.num_questions
            interaction_id = batch["hist_question"] + corr_offset
            interaction_id = torch.where(batch["hist_correct"] == 0, torch.zeros_like(interaction_id), interaction_id)
            hist_parts = [self.interaction_emb(interaction_id), batch["hist_elapsed"].unsqueeze(-1)]
        else:
            hist_parts = [
                self.question_emb(batch["hist_question"]),
                self.correct_emb(batch["hist_correct"]),
                batch["hist_elapsed"].unsqueeze(-1),
            ]
        if self.include_metadata:
            hist_parts.append(self.part_emb(batch["hist_part"]))
            hist_parts.append(self.tag_emb(batch["hist_tag"]))
        hist_x = torch.cat(hist_parts, dim=-1)

        lengths = batch["length"].detach().cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            hist_x,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.rnn(packed)
        hidden_state = hidden[0][-1] if self.rnn_type == "lstm" else hidden[-1]

        target_parts = [self.question_emb(batch["target_question"])]
        if self.include_metadata:
            target_parts.append(self.part_emb(batch["target_part"]))
            target_parts.append(self.tag_emb(batch["target_tag"]))
        if self.use_agg_features:
            target_parts.append(batch["agg"])
        target_x = torch.cat(target_parts, dim=-1)
        return self.head(torch.cat([hidden_state, target_x], dim=-1)).squeeze(-1)


def build_model(
    model_name: str,
    *,
    num_questions: int,
    num_parts: int,
    num_tags: int,
    embedding_dim: int,
    hidden_size: int,
    dropout: float,
    include_metadata: bool,
    sequence_use_agg: bool = False,
    interaction_history: bool = False,
) -> nn.Module:
    name = model_name.lower()
    if name == "mlp":
        return MLPBaseline(
            num_questions=num_questions,
            num_parts=num_parts,
            num_tags=num_tags,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            include_metadata=include_metadata,
        )
    if name in {"gru", "lstm"}:
        return SequenceModel(
            num_questions=num_questions,
            num_parts=num_parts,
            num_tags=num_tags,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            include_metadata=include_metadata,
            rnn_type=name,
            use_agg_features=sequence_use_agg,
            interaction_history=interaction_history,
        )
    raise ValueError(f"Unknown model: {model_name}")
