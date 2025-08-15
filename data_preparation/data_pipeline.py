"""
Data preparation utilities:
- splitting by session
- building vocabularies/encoders
- sequence dataset & dynamic padding collate
- flexible handling when numeric features are absent
- OPTIONAL text & image feature encoding (pretrained encoders)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from PIL import Image

# torch
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# Optional heavy libs (ok if missing when you don't use text/images)
try:
    from sentence_transformers import SentenceTransformer  # for text embeddings
except Exception:  # pragma: no cover
    SentenceTransformer = None

try:
    from torchvision import models
except Exception:
    models = None

PAD_IDX = 0
UNK_IDX = 1


# -------- splitting & generators --------
def split_by_session(df: pd.DataFrame,
                     session_col: str,
                     test_size: float,
                     val_size: float,
                     seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into Train/Val/Test by unique session IDs."""
    session_ids = df[session_col].unique()
    train_ids, test_ids = train_test_split(session_ids, test_size=test_size, random_state=seed)
    train_ids, val_ids = train_test_split(train_ids, test_size=val_size, random_state=seed)  # 60/20/20

    def pick(ids): return df[df[session_col].isin(ids)].copy()
    return pick(train_ids), pick(val_ids), pick(test_ids)


def session_sequence_generator(df: pd.DataFrame,
                               session_col: str,
                               event_col: str,
                               label_col: str) -> Iterable[Tuple[List[str], List[str], pd.DataFrame]]:
    """Yield per-session (event_sequence, label_sequence, group_df)."""
    for _, g in df.groupby(session_col, sort=False):
        yield g[event_col].astype(str).tolist(), g[label_col].astype(str).tolist(), g


# -------- vocab/encoders for events/labels/categoricals/numerics --------
def build_event_vocab(train_event_sequences: List[List[str]]) -> Dict[str, int]:
    """Create token vocabulary for event strings; reserves 0=<PAD>, 1=<UNK>."""
    vocab = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    for seq in train_event_sequences:
        for e in seq:
            if e not in vocab:
                vocab[e] = len(vocab)
    return vocab


def encode_events(seqs: List[List[str]], vocab: Dict[str, int]) -> List[List[int]]:
    """Map each event string to an integer id (UNK for unseen)."""
    get_id = lambda s: vocab.get(s, UNK_IDX)
    return [[get_id(tok) for tok in seq] for seq in seqs]


def fit_label_encoder(all_labels_flat: List[str]) -> LabelEncoder:
    """Fit label encoder on all label strings."""
    le = LabelEncoder()
    le.fit(all_labels_flat)
    return le


def encode_label_sequences(label_seqs: List[List[str]], le: LabelEncoder) -> List[List[int]]:
    """Encode each label sequence using a fitted LabelEncoder."""
    return [le.transform(seq).tolist() for seq in label_seqs]


def fit_cat_encoder(train_df: pd.DataFrame, cat_cols: List[str]) -> OrdinalEncoder | None:
    """Fit an OrdinalEncoder for categorical feature columns."""
    if not cat_cols:
        return None
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
    enc.fit(train_df[cat_cols].astype(str))
    return enc


def transform_cat_sequences(df: pd.DataFrame,
                            session_col: str,
                            enc: OrdinalEncoder | None,
                            cat_cols: List[str]) -> Dict[str, List[List[int]]]:
    """
    Transform categorical columns to integer IDs; group back by session.
    Shift values by +2 so PAD=0, UNK=1, real categories start at 2.
    """
    if not cat_cols or enc is None:
        return {}
    arr = enc.transform(df[cat_cols].astype(str)).astype(np.int64)  # [N,C]
    arr = arr + 2
    out_by_session = {c: [] for c in cat_cols}
    for _, g in df.groupby(session_col, sort=False):
        idx = g.index
        vals = arr[df.index.get_indexer(idx), :]
        for j, c in enumerate(cat_cols):
            out_by_session[c].append(vals[:, j].tolist())
    return out_by_session


def cat_vocab_sizes_from_encoder(enc: OrdinalEncoder | None, cat_cols: List[str]) -> Dict[str, int]:
    """Compute embedding vocab sizes for each categorical column (includes PAD & UNK)."""
    if not cat_cols or enc is None:
        return {}
    sizes = {}
    for col_idx, col in enumerate(cat_cols):
        n_cats = len(enc.categories_[col_idx])
        sizes[col] = n_cats + 2  # PAD, UNK
    return sizes


def fit_numeric_scaler(train_df: pd.DataFrame, num_cols: List[str]) -> StandardScaler | None:
    """Fit a StandardScaler for numeric features (or None if none)."""
    if not num_cols:
        return None
    scaler = StandardScaler()
    scaler.fit(train_df[num_cols].astype(float))
    return scaler


def transform_numeric_sequences(df: pd.DataFrame,
                                session_col: str,
                                scaler: StandardScaler | None,
                                num_cols: List[str]) -> Dict[str, List[List[float]]]:
    """Scale numeric features and group back by session (or return {} if none)."""
    if not num_cols or scaler is None:
        return {}
    arr = scaler.transform(df[num_cols].astype(float))  # [N,K]
    out_by_session = {c: [] for c in num_cols}
    for _, g in df.groupby(session_col, sort=False):
        idx = g.index
        vals = arr[df.index.get_indexer(idx), :]
        for j, c in enumerate(num_cols):
            out_by_session[c].append(vals[:, j].astype(np.float32).tolist())
    return out_by_session


# -------- OPTIONAL TEXT: encoder + per-session regroup --------
def fit_text_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                     device: str = "cpu"):
    """
    Returns a SentenceTransformer or None if unavailable.
    """
    if SentenceTransformer is None:
        return None
    try:
        enc = SentenceTransformer(model_name, device=device)
        enc.eval()
        return enc
    except Exception:
        return None


@torch.no_grad()
def transform_text_sequences_with_encoder(df: pd.DataFrame,
                                          session_col: str,
                                          text_cols: List[str],
                                          encoder,
                                          batch_size: int = 512):
    """
    Concatenate rows' text columns → encode with SentenceTransformer → regroup per session.
    Returns (by_session_dict, dim). If encoder or columns are missing, returns ({}, 0).
    """
    if not text_cols or encoder is None:
        return {}, 0
    df_pos = df.reset_index(drop=True)
    texts = df_pos[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()

    embs: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        embs.append(torch.as_tensor(encoder.encode(chunk, convert_to_tensor=True)).cpu())
    if not embs:
        return {}, 0
    embs = torch.cat(embs, dim=0)  # [N, D]
    D = embs.shape[1]

    out = {"__text__": []}
    start = 0
    for _, g in df_pos.groupby(session_col, sort=False):
        L = len(g)
        out["__text__"].append(
            embs[start:start + L, :].to(torch.float32).cpu().tolist()
        )
        start += L
    return out, int(D)


# -------- OPTIONAL IMAGE: backbone + per-session regroup --------
def fit_image_encoder(device: str = "cpu"):
    """
    Returns (backbone, preprocess) or (None, None) if torchvision is unavailable.
    Uses ResNet50 penultimate features (2048-d).
    """
    if models is None:
        return None, None
    try:
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()             # take penultimate features
        backbone.eval().to(device)
        preprocess = models.ResNet50_Weights.IMAGENET1K_V2.transforms()
        return backbone, preprocess
    except Exception:
        return None, None


@torch.no_grad()
def transform_image_sequences_with_encoder(df: pd.DataFrame,
                                           session_col: str,
                                           image_path_col: Optional[str],
                                           backbone,
                                           preprocess,
                                           device: str = "cpu"):
    """
    image_path_col: column with file paths (str). Missing/invalid rows → zero vector.
    Returns (by_session_dict, dim). If unavailable, returns ({}, 0).
    """
    if not image_path_col or image_path_col not in df.columns or backbone is None or preprocess is None:
        return {}, 0
    df_pos = df.reset_index(drop=True)
    vecs: List[torch.Tensor] = []
    for pth in df_pos[image_path_col].tolist():
        feat = torch.zeros(2048, dtype=torch.float32, device=device)
        try:
            if isinstance(pth, str) and len(pth) > 0:
                img = Image.open(pth).convert("RGB")
                x = preprocess(img).unsqueeze(0).to(device)  # [1,3,224,224]
                feat = backbone(x).squeeze(0)                # [2048]
        except Exception:
            pass  # keep zero vector on failure
        vecs.append(feat)
    if not vecs:
        return {}, 0
    embs = torch.stack(vecs, dim=0).cpu().numpy().astype("float32")  # [N, 2048]

    out = {"__image__": []}
    start = 0
    for _, g in df_pos.groupby(session_col, sort=False):
        L = len(g)
        out["__image__"].append(embs[start:start + L, :].tolist())
        start += L
    return out, 2048


# -------- Build split (now returns text/image lists & dims too) --------
def build_split(df_split: pd.DataFrame,
                cfg_data: dict,
                vocab: Dict[str, int],
                le: LabelEncoder,
                cat_enc: OrdinalEncoder | None,
                num_scaler: StandardScaler | None,
                text_encoder=None, text_cols: Optional[List[str]] = None,
                image_path_col: Optional[str] = None, image_backbone=None, image_preprocess=None):
    """
    Build encoded sequences and per-session feature dicts for a dataframe split.

    Returns:
        ev_ids:        List[List[int]]
        lab_ids:       List[List[int]]
        cat_list:      List[Dict[str, List[int]]]
        num_list:      List[Dict[str, List[float]]]
        txt_list:      List[Dict['__text__', List[List[float]]]]
        img_list:      List[Dict['__image__', List[List[float]]]]
        txt_dim:       int (0 if none)
        img_dim:       int (0 if none)
    """
    session_col = cfg_data["session_col"]
    event_col   = cfg_data["event_col"]
    label_col   = cfg_data["label_col"]
    cat_cols    = cfg_data.get("cat_features", []) or []
    num_cols    = cfg_data.get("num_features", []) or []

    ev_seqs, lbl_seqs = [], []
    for ev, lbl, _ in session_sequence_generator(df_split, session_col, event_col, label_col):
        ev_seqs.append(ev)
        lbl_seqs.append(lbl)
    ev_ids  = encode_events(ev_seqs, vocab)
    lab_ids = encode_label_sequences(lbl_seqs, le)

    cat_by_session = transform_cat_sequences(df_split, session_col, cat_enc, cat_cols)
    num_by_session = transform_numeric_sequences(df_split, session_col, num_scaler, num_cols)

    cat_list = [{c: cat_by_session[c][i] for c in cat_cols} if cat_by_session else {}
                for i in range(len(ev_seqs))]
    num_list = [{c: num_by_session[c][i] for c in num_cols} if num_by_session else {}
                for i in range(len(ev_seqs))]

    # TEXT
    if text_cols and text_encoder is not None:
        txt_by_session, txt_dim = transform_text_sequences_with_encoder(
            df_split, session_col, text_cols, text_encoder
        )
        txt_list = [{"__text__": txt_by_session["__text__"][i]} for i in range(len(ev_seqs))]
    else:
        txt_list = [{} for _ in range(len(ev_seqs))]
        txt_dim = 0

    # IMAGE
    if image_path_col and image_backbone is not None and image_preprocess is not None:
        device = str(next(image_backbone.parameters()).device)
        img_by_session, img_dim = transform_image_sequences_with_encoder(
            df_split, session_col, image_path_col, image_backbone, image_preprocess, device=device
        )
        img_list = [{"__image__": img_by_session["__image__"][i]} for i in range(len(ev_seqs))]
    else:
        img_list = [{} for _ in range(len(ev_seqs))]
        img_dim = 0

    return ev_ids, lab_ids, cat_list, num_list, txt_list, img_list, txt_dim, img_dim


# -------- Dataset & collate (now include text/image) --------
class MultiFeatureSequenceDataset(Dataset):
    """PyTorch Dataset that stores per-session sequences and feature dicts."""

    def __init__(self,
                 token_seqs: List[List[int]],
                 label_seqs: List[List[int]],
                 cat_feat_seqs: List[Dict[str, List[int]]],
                 num_feat_seqs: List[Dict[str, List[float]]],
                 txt_feat_seqs: List[Dict[str, List[List[float]]]] | None = None,
                 img_feat_seqs: List[Dict[str, List[List[float]]]] | None = None):
        self.tokens = token_seqs
        self.labels = label_seqs
        self.cat = cat_feat_seqs
        self.num = num_feat_seqs
        self.txt = txt_feat_seqs or [{} for _ in range(len(token_seqs))]
        self.img = img_feat_seqs or [{} for _ in range(len(token_seqs))]

        self.cat_cols = list(self.cat[0].keys()) if self.cat and self.cat[0] else []
        self.num_cols = list(self.num[0].keys()) if self.num and self.num[0] else []
        self.txt_cols = list(self.txt[0].keys()) if self.txt and self.txt[0] else []  # ['__text__'] or []
        self.img_cols = list(self.img[0].keys()) if self.img and self.img[0] else []  # ['__image__'] or []

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int):
        item = {
            "tokens": torch.tensor(self.tokens[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "cat": {c: torch.tensor(self.cat[idx][c], dtype=torch.long) for c in self.cat_cols},
            "num": {c: torch.tensor(self.num[idx][c], dtype=torch.float32) for c in self.num_cols},
        }
        if self.txt_cols:
            item["txt"] = {c: torch.tensor(self.txt[idx][c], dtype=torch.float32) for c in self.txt_cols}
        else:
            item["txt"] = {}
        if self.img_cols:
            item["img"] = {c: torch.tensor(self.img[idx][c], dtype=torch.float32) for c in self.img_cols}
        else:
            item["img"] = {}
        return item


def collate_batch(batch, pad_idx: int = PAD_IDX, ignore_index: int = -100):
    """
    Dynamically pad a list of samples into a batch.
    Returns:
      - tokens[B,T], labels[B,T], mask[B,T]
      - cat_feats{col->B,T}, num_feats{col->B,T}
      - txt_feats{'__text__'->B,T,D_txt} (if present)
      - img_feats{'__image__'->B,T,D_img} (if present)
    """
    tokens = [b["tokens"] for b in batch]
    labels = [b["labels"] for b in batch]
    cat_cols = list(batch[0]["cat"].keys())
    num_cols = list(batch[0]["num"].keys())
    txt_cols = list(batch[0].get("txt", {}).keys())
    img_cols = list(batch[0].get("img", {}).keys())

    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=pad_idx)  # [B,T]
    mask = (padded_tokens != pad_idx).long()
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=ignore_index)

    cat_feats: Dict[str, torch.Tensor] = {}
    for c in cat_cols:
        seqs = [b["cat"][c] for b in batch]
        cat_feats[c] = pad_sequence(seqs, batch_first=True, padding_value=pad_idx)  # long

    num_feats: Dict[str, torch.Tensor] = {}
    for c in num_cols:
        seqs = [b["num"][c] for b in batch]
        num_feats[c] = pad_sequence(seqs, batch_first=True, padding_value=0.0)      # float

    txt_feats: Dict[str, torch.Tensor] = {}
    for c in txt_cols:
        seqs = [b["txt"][c] for b in batch]  # each is [T, D_txt]
        txt_feats[c] = pad_sequence(seqs, batch_first=True, padding_value=0.0)      # [B,T,D_txt] float

    img_feats: Dict[str, torch.Tensor] = {}
    for c in img_cols:
        seqs = [b["img"][c] for b in batch]  # each is [T, D_img]
        img_feats[c] = pad_sequence(seqs, batch_first=True, padding_value=0.0)      # [B,T,D_img] float

    return padded_tokens, padded_labels, mask, cat_feats, num_feats, txt_feats, img_feats
