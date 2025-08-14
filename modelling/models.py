"""
Model definitions:
- BiLSTM tagger with categorical embeddings and optional numeric projection
- Transformer tagger with positional encodings
"""

from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding added to token embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D]
        B, T, D = x.size()
        device = x.device
        position = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)  # [T,1]
        div_term = torch.exp(torch.arange(0, D, 2, device=device).float() * (-torch.log(torch.tensor(10000.0, device=device)) / D))
        pe = torch.zeros(T, D, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        x = x + pe.unsqueeze(0)
        return self.dropout(x)


class MultiFeatureBiLSTMTagger(nn.Module):
    """BiLSTM token tagger with event+categorical embeddings and optional numeric projection."""

    def __init__(self,
                 vocab_size: int,
                 num_classes: int,
                 cat_vocab_sizes: Dict[str, int],
                 emb_event_dim: int = 64,
                 emb_cat_dim: int = 16,
                 num_cols: List[str] | None = None,
                 num_proj_dim: int = 16,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 pad_idx: int = 0,
                 dropout_p: float = 0.3,
                 rnn_dropout: float = 0.2):
        super().__init__()
        self.pad_idx = pad_idx
        self.num_cols = num_cols or []
        self.cat_cols = list(cat_vocab_sizes.keys())

        # embeddings
        self.emb_event = nn.Embedding(vocab_size, emb_event_dim, padding_idx=pad_idx)
        self.emb_cats = nn.ModuleDict({
            c: nn.Embedding(cat_vocab_sizes[c], emb_cat_dim, padding_idx=pad_idx)
            for c in self.cat_cols
        })

        # numeric projection
        self.num_in_dim = len(self.num_cols)
        self.num_proj = nn.Linear(self.num_in_dim, num_proj_dim) if self.num_in_dim > 0 else None

        in_dim = emb_event_dim + len(self.cat_cols) * emb_cat_dim + (num_proj_dim if self.num_proj else 0)

        # BiLSTM
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, tokens, mask, cat_feats: dict, num_feats: dict):
        x = self.emb_event(tokens)  # [B,T,Ee]
        parts = [x]

        for c in self.cat_cols:
            parts.append(self.emb_cats[c](cat_feats[c]))  # [B,T,Ec]

        if self.num_proj:
            stack = torch.stack([num_feats[c] for c in self.num_cols], dim=-1)  # [B,T,K]
            stack = torch.nan_to_num(stack, nan=0.0, posinf=1e6, neginf=-1e6)
            parts.append(self.num_proj(stack))

        x = torch.cat(parts, dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        lengths = mask.sum(dim=1).to("cpu")
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B,T,2H]

        out = self.dropout(out)
        logits = self.fc(out)
        return logits


class MultiFeatureTransformerTagger(nn.Module):
    """TransformerEncoder-based token tagger with embeddings + optional numeric projection."""

    def __init__(self,
                 vocab_size: int,
                 num_classes: int,
                 cat_vocab_sizes: Dict[str, int],
                 emb_event_dim: int = 64,
                 emb_cat_dim: int = 16,
                 num_cols: List[str] | None = None,
                 num_proj_dim: int = 16,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.2,
                 pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.num_cols = num_cols or []
        self.cat_cols = list(cat_vocab_sizes.keys())

        self.emb_event = nn.Embedding(vocab_size, emb_event_dim, padding_idx=pad_idx)
        self.emb_cats = nn.ModuleDict({
            c: nn.Embedding(cat_vocab_sizes[c], emb_cat_dim, padding_idx=pad_idx)
            for c in self.cat_cols
        })

        self.num_in_dim = len(self.num_cols)
        self.num_proj = nn.Linear(self.num_in_dim, num_proj_dim) if self.num_in_dim > 0 else None

        in_dim = emb_event_dim + len(self.cat_cols) * emb_cat_dim + (num_proj_dim if self.num_proj else 0)
        self.input_proj = nn.Linear(in_dim, d_model)

        self.posenc = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, tokens, mask, cat_feats: dict, num_feats: dict):
        x = self.emb_event(tokens)  # [B,T,Ee]
        parts = [x]

        for c in self.cat_cols:
            parts.append(self.emb_cats[c](cat_feats[c]))

        if self.num_proj:
            stack = torch.stack([num_feats[c] for c in self.num_cols], dim=-1)  # [B,T,K]
            stack = torch.nan_to_num(stack, nan=0.0, posinf=1e6, neginf=-1e6)
            parts.append(self.num_proj(stack))

        x = torch.cat(parts, dim=-1)           # [B,T,in_dim]
        x = self.input_proj(x)                 # [B,T,d_model]
        x = self.posenc(x)

        key_padding_mask = ~mask.bool()
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits
