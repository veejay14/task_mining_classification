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
    def __init__(
        self,
        vocab_size,
        num_classes,
        cat_vocab_sizes: dict,
        emb_event_dim=64,
        emb_cat_dim=16,
        num_cols=None,
        num_proj_dim=16,
        # NEW: text & image dims (raw input)
        txt_in_dim=0, txt_proj_dim=32,
        img_in_dim=0, img_proj_dim=32,
        # LSTM
        hidden_dim=128, num_layers=2,
        pad_idx=0, dropout_p=0.3, rnn_dropout=0.2,
        # NEW: fusion config
        fusion_mode="concat",        # 'concat' or 'gated'
        fusion_out_dim=None          # if None, we’ll match LSTM input size to computed value
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.num_cols = num_cols or []
        self.cat_cols = list(cat_vocab_sizes.keys())

        # Embeddings (same as before)
        self.emb_event = nn.Embedding(vocab_size, emb_event_dim, padding_idx=pad_idx)
        self.emb_cats = nn.ModuleDict({
            c: nn.Embedding(cat_vocab_sizes[c], emb_cat_dim, padding_idx=pad_idx)
            for c in self.cat_cols
        })

        # Numeric projection (optional)
        self.num_in_dim = len(self.num_cols)
        self.num_proj = nn.Linear(self.num_in_dim, num_proj_dim) if self.num_in_dim > 0 else None

        # Compute base dim = event + all cats + (numeric proj)
        self.base_dim = emb_event_dim + len(self.cat_cols)*emb_cat_dim + (num_proj_dim if self.num_proj else 0)

        # ---- NEW: fusion module (base + txt + img) ----
        self.fusion = MultiModalFusion(
            base_dim=self.base_dim,
            txt_in_dim=txt_in_dim,
            img_in_dim=img_in_dim,
            txt_proj_dim=txt_proj_dim,
            img_proj_dim=img_proj_dim,
            out_dim=fusion_out_dim or self.base_dim,   # keep same width unless you want a different one
            dropout_p=dropout_p,
            mode=fusion_mode
        )
        lstm_in = fusion_out_dim or self.base_dim

        # LSTM as before
        self.rnn = nn.LSTM(
            input_size=lstm_in, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=rnn_dropout if num_layers>1 else 0.0
        )
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, tokens, mask, cat_feats: dict, num_feats: dict, txt_feats: dict, img_feats: dict):
        # ----- build base embedding (same as before) -----
        x_event = self.emb_event(tokens)             # [B,T,Ee]
        parts = [x_event]

        for c in self.cat_cols:
            parts.append(self.emb_cats[c](cat_feats[c]))  # [B,T,Ec]

        if self.num_proj:
            num_stack = torch.stack([num_feats[c] for c in self.num_cols], dim=-1)  # [B,T,K]
            num_stack = torch.nan_to_num(num_stack, nan=0.0, posinf=1e6, neginf=-1e6)
            parts.append(self.num_proj(num_stack))                                  # [B,T,En]

        base = torch.cat(parts, dim=-1)                                             # [B,T,base_dim]
        base = torch.nan_to_num(base, nan=0.0, posinf=1e6, neginf=-1e6)

        # ----- NEW: fuse with text/image -----
        txt = txt_feats.get("__text__", None)
        img = img_feats.get("__image__", None)
        x = self.fusion(base, txt, img)                                            # [B,T,lstm_in]

        # packed LSTM as before
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        out = self.dropout(out)
        logits = self.fc(out)                                                       # [B,T,C]
        return logits


class MultiFeatureTransformerTagger(nn.Module):
    def __init__(
        self,
        vocab_size, num_classes, cat_vocab_sizes: dict,
        emb_event_dim=64, emb_cat_dim=16,
        num_cols=None, num_proj_dim=16,
        txt_in_dim=0, txt_proj_dim=32,
        img_in_dim=0, img_proj_dim=32,
        d_model=128, nhead=4, num_layers=2, dim_feedforward=256,
        dropout=0.2, pad_idx=0,
        fusion_mode="concat", fusion_out_dim=None
    ):
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

        self.base_dim = emb_event_dim + len(self.cat_cols)*emb_cat_dim + (num_proj_dim if self.num_proj else 0)

        # ---- NEW: fusion ----
        self.fusion = MultiModalFusion(
            base_dim=self.base_dim,
            txt_in_dim=txt_in_dim, img_in_dim=img_in_dim,
            txt_proj_dim=txt_proj_dim, img_proj_dim=img_proj_dim,
            out_dim=fusion_out_dim or self.base_dim,
            dropout_p=dropout,
            mode=fusion_mode
        )
        enc_in = fusion_out_dim or self.base_dim

        self.input_proj = nn.Linear(enc_in, d_model)
        self.posenc = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, tokens, mask, cat_feats: dict, num_feats: dict, txt_feats: dict, img_feats: dict):
        # base
        x_event = self.emb_event(tokens)
        parts = [x_event]
        for c in self.cat_cols:
            parts.append(self.emb_cats[c](cat_feats[c]))
        if self.num_proj:
            num_stack = torch.stack([num_feats[c] for c in self.num_cols], dim=-1)
            num_stack = torch.nan_to_num(num_stack, nan=0.0, posinf=1e6, neginf=-1e6)
            parts.append(self.num_proj(num_stack))
        base = torch.cat(parts, dim=-1)
        base = torch.nan_to_num(base, nan=0.0, posinf=1e6, neginf=-1e6)

        # fuse
        txt = txt_feats.get("__text__", None)
        img = img_feats.get("__image__", None)
        x = self.fusion(base, txt, img)                   # [B,T,enc_in]

        # transformer
        x = self.input_proj(x)                            # [B,T,d_model]
        x = self.posenc(x)
        key_padding_mask = ~mask.bool()
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


class MultiModalFusion(nn.Module):
    """
    Fuses base sequence features with optional text/image features.
    Modes:
      - 'concat':  x = LN( Drop( Linear( [base | text | image] ) ) )
      - 'gated':   x = base + α_text * P(text) + α_img * P(image), with learned gates per timestep
    Handles missing modalities gracefully (tensors can be absent or zero-length).
    """
    def __init__(
        self,
        base_dim: int,                 # dim of base sequence features before fusion
        txt_in_dim: int = 0,
        img_in_dim: int = 0,
        txt_proj_dim: int = 32,
        img_proj_dim: int = 32,
        out_dim: int = 128,            # output feature size to pass to LSTM/Transformer
        dropout_p: float = 0.2,
        mode: str = "concat"           # 'concat' or 'gated'
    ):
        super().__init__()
        assert mode in {"concat", "gated"}
        self.mode = mode

        # Optional projections for modalities
        self.use_text = txt_in_dim > 0
        self.use_image = img_in_dim > 0

        self.txt_proj = nn.Linear(txt_in_dim, txt_proj_dim) if self.use_text else None
        self.img_proj = nn.Linear(img_in_dim, img_proj_dim) if self.use_image else None

        # Normalizers (helpful with heterogeneous scales)
        self.base_norm = nn.LayerNorm(base_dim)
        self.txt_norm  = nn.LayerNorm(txt_proj_dim) if self.use_text else None
        self.img_norm  = nn.LayerNorm(img_proj_dim) if self.use_image else None

        self.dropout = nn.Dropout(dropout_p)

        if self.mode == "concat":
            fused_in = base_dim \
                       + (txt_proj_dim if self.use_text else 0) \
                       + (img_proj_dim if self.use_image else 0)
            self.out = nn.Sequential(
                nn.Linear(fused_in, out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim),
                nn.Dropout(dropout_p),
            )
        else:
            # gated: output stays at base_dim (or we can project to out_dim afterwards)
            # We’ll project to out_dim for consistency
            gate_in = base_dim \
                      + (txt_proj_dim if self.use_text else 0) \
                      + (img_proj_dim if self.use_image else 0)
            self.gate = nn.Sequential(
                nn.Linear(gate_in, 64),
                nn.GELU(),
                nn.Linear(64, (1 if self.use_text else 0) + (1 if self.use_image else 0)),
                nn.Sigmoid()  # α in [0,1]
            )
            self.out = nn.Sequential(
                nn.Linear(base_dim + (txt_proj_dim if self.use_text else 0) + (img_proj_dim if self.use_image else 0),
                          out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim),
                nn.Dropout(dropout_p),
            )

    def forward(self, base, txt=None, img=None):
        """
        base: [B, T, D_base]
        txt:  [B, T, D_txt_in]  or None
        img:  [B, T, D_img_in]  or None
        returns: [B, T, out_dim]
        """
        base = self.base_norm(base)

        parts = [base]
        if self.use_text and txt is not None:
            txt = self.txt_proj(txt)
            txt = self.txt_norm(txt)
            parts.append(txt)
        if self.use_image and img is not None:
            img = self.img_proj(img)
            img = self.img_norm(img)
            parts.append(img)

        if len(parts) == 1:
            # no extra modalities available → just project base to out_dim for downstream model
            fused = self.out(base) if isinstance(self.out, nn.Sequential) else base
            return self.dropout(fused)

        if self.mode == "concat":
            x = torch.cat(parts, dim=-1)        # [B,T, sum_dims]
            return self.out(self.dropout(x))    # [B,T,out_dim]

        # gated
        x = torch.cat(parts, dim=-1)            # [B,T, base+txt+img]
        gates = self.gate(self.dropout(x))      # [B,T,#modalities_present]
        idx = 0
        fused = base
        if self.use_text and len(parts) >= 2:
            txt = parts[1] if not self.use_image else (parts[1] if parts[1].shape[-1] != parts[-1].shape[-1] else parts[-2])
            fused = fused + gates[..., idx:idx+1] * txt
            idx += 1
        if self.use_image:
            img = parts[-1]
            fused = fused + gates[..., idx:idx+1] * img
        return self.out(self.dropout(fused))    # [B,T,out_dim]

