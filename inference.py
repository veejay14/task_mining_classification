# inference.py
"""
End-to-end inference on RAW event logs:
- Load cfg + artifacts (vocab/encoders/scaler) + trained checkpoint
- Build features from raw CSV (event-name + time + optional text/image)
- Encode sequences with saved encoders
- Run model to get predictions
- Optional selective labeling via probability threshold
- Save predictions aligned to input rows
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, List
from pathlib import Path
import json
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import joblib
from config.config import cfg
from modelling.models import MultiFeatureBiLSTMTagger, MultiFeatureTransformerTagger
from data_preparation.data_pipeline import (
    build_split, MultiFeatureSequenceDataset, collate_batch,
    cat_vocab_sizes_from_encoder,
    fit_text_encoder, fit_image_encoder,   # <-- NEW
)
from data_preparation.data_loading import EventLogDataBuilder


# ----------------------------- Artifact I/O -----------------------------

def load_artifacts(art_dir: str | Path) -> Dict[str, Any]:
    """
    Load artifacts produced during training:
      - event_vocab.json
      - label_encoder.pkl
      - cat_encoder.pkl (optional if no categorical features)
      - num_scaler.pkl  (optional if no numeric features)
    """
    art_dir = Path(art_dir)
    artifacts: Dict[str, Any] = {}

    with open(art_dir / "event_vocab.json", "r") as f:
        artifacts["event_vocab"] = json.load(f)

    le_path = art_dir / "label_encoder.pkl"
    artifacts["label_enc"] = joblib.load(le_path) if le_path.exists() else None

    ce_path = art_dir / "cat_encoder.pkl"
    artifacts["cat_enc"] = joblib.load(ce_path) if ce_path.exists() else None

    ns_path = art_dir / "num_scaler.pkl"
    artifacts["num_scaler"] = joblib.load(ns_path) if ns_path.exists() else None

    return artifacts


def build_model_from_cfg(event_vocab: Dict[str, int],
                         cat_enc,
                         device: torch.device,
                         num_classes: int,
                         # NEW dims to match training
                         txt_in_dim: int,
                         img_in_dim: int):
    """
    Rebuild a model instance that matches training-time config,
    including multimodal dimensions.
    """
    model_type = cfg["model"]["model_type"]
    cat_vocab_sizes = cat_vocab_sizes_from_encoder(cat_enc, cfg["data"]["cat_features"])

    fusion_mode = cfg["model"].get("fusion_mode", "concat")
    txt_proj_dim = cfg["model"].get("txt_proj_dim", 32)
    img_proj_dim = cfg["model"].get("img_proj_dim", 32)

    if model_type == "bilstm":
        model = MultiFeatureBiLSTMTagger(
            vocab_size=len(event_vocab),
            num_classes=num_classes,
            cat_vocab_sizes=cat_vocab_sizes,
            emb_event_dim=cfg["model"]["emb_event_dim"],
            emb_cat_dim=cfg["model"]["emb_cat_dim"],
            num_cols=cfg["data"]["num_features"],
            num_proj_dim=cfg["model"]["num_proj_dim"],
            # multimodal
            txt_in_dim=txt_in_dim,
            txt_proj_dim=txt_proj_dim,
            img_in_dim=img_in_dim,
            img_proj_dim=img_proj_dim,
            # rnn
            hidden_dim=cfg["model"]["lstm_hidden_dim"],
            num_layers=cfg["model"]["lstm_num_layers"],
            pad_idx=cfg["model"]["pad_idx"],
            dropout_p=cfg["model"]["lstm_dropout"],
            rnn_dropout=cfg["model"]["lstm_rnn_dropout"],
            fusion_mode=fusion_mode,
        ).to(device)

    elif model_type == "transformer":
        model = MultiFeatureTransformerTagger(
            vocab_size=len(event_vocab),
            num_classes=num_classes,
            cat_vocab_sizes=cat_vocab_sizes,
            emb_event_dim=cfg["model"]["emb_event_dim"],
            emb_cat_dim=cfg["model"]["emb_cat_dim"],
            num_cols=cfg["data"]["num_features"],
            num_proj_dim=cfg["model"]["num_proj_dim"],
            # multimodal
            txt_in_dim=txt_in_dim,
            txt_proj_dim=txt_proj_dim,
            img_in_dim=img_in_dim,
            img_proj_dim=img_proj_dim,
            # transformer
            d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"],
            num_layers=cfg["model"]["tf_num_layers"],
            dim_feedforward=cfg["model"]["tf_ff_dim"],
            dropout=cfg["model"]["tf_dropout"],
            pad_idx=cfg["model"]["pad_idx"],
            fusion_mode=fusion_mode,
        ).to(device)
    else:
        raise ValueError("cfg['model']['model_type'] must be 'bilstm' or 'transformer'")

    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str | Path, device: torch.device) -> None:
    """Load the trained weights into the model."""
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)


# -------------------------- Data → Features → DL -----------------------

def load_and_prepare_raw_dataframe() -> pd.DataFrame:
    """
    Load raw CSV and run full feature pipeline:
      - event-name features (action/app/path/etc.)
      - time features (elapsed/session_duration)
      - sorting / NA handling consistent with training
      - optional text/image columns as configured
    """
    builder = EventLogDataBuilder(cfg)
    df = builder.load_and_prepare("data/predictions.csv")
    return df


def ensure_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    At inference you might not have true labels; build_split expects a label column.
    We add an empty/dummy one if missing to keep shapes consistent.
    """
    label_col = cfg["data"]["label_col"]
    if label_col not in df.columns:
        df = df.copy()
        df[label_col] = ""   # dummy
    return df


def build_inference_loader(df_prepared: pd.DataFrame,
                           artifacts: Dict[str, Any],
                           device: torch.device) -> Tuple[DataLoader, int, int]:
    """
    Encode the prepared dataframe using saved encoders/scalers and return a DataLoader.
    Also returns (txt_dim, img_dim) so the model can be rebuilt with matching shapes.
    """
    df_prepared = ensure_label_column(df_prepared)

    # Optional encoders must be recreated (we didn't persist huge models)
    TEXT_FEATURES = cfg["data"].get("text_features", []) or []
    IMAGE_PATH_COL = cfg["data"].get("image_path_col", None)

    text_encoder = fit_text_encoder(device=str(device)) if TEXT_FEATURES else None
    image_backbone, image_preprocess = (fit_image_encoder(device=str(device)) if IMAGE_PATH_COL else (None, None))

    ev_ids, lab_ids, cat_list, num_list, txt_list, img_list, txt_dim, img_dim = build_split(
        df_split=df_prepared,
        cfg_data=cfg["data"],
        vocab=artifacts["event_vocab"],
        le=artifacts["label_enc"],      # trained label encoder (even if labels are dummy)
        cat_enc=artifacts["cat_enc"],
        num_scaler=artifacts["num_scaler"],
        text_encoder=text_encoder, text_cols=TEXT_FEATURES,
        image_path_col=IMAGE_PATH_COL, image_backbone=image_backbone, image_preprocess=image_preprocess
    )

    ds = MultiFeatureSequenceDataset(ev_ids, lab_ids, cat_list, num_list, txt_list, img_list)
    loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False, collate_fn=collate_batch)
    return loader, txt_dim, img_dim


# ---------------------------- Inference API ----------------------------

@torch.no_grad()
def predict_with_threshold(model: torch.nn.Module,
                           loader: DataLoader,
                           device: torch.device,
                           prob_threshold: float = 0.95,
                           unlabeled_code: int = -1) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Predict on a loader with abstention:
    Returns lists (per batch) of (preds[B,T], conf[B,T], mask[B,T]).
    Preds == unlabeled_code for abstained tokens.
    """
    model.eval()
    out_preds, out_conf, out_mask = [], [], []

    for tokens, labels, mask, cat_feats, num_feats, txt_feats, img_feats in loader:
        tokens, mask = tokens.to(device), mask.to(device)
        cat_feats = {k: v.to(device) for k, v in cat_feats.items()}
        num_feats = {k: v.to(device) for k, v in num_feats.items()}
        txt_feats = {k: v.to(device) for k, v in txt_feats.items()}
        img_feats = {k: v.to(device) for k, v in img_feats.items()}

        logits = model(tokens, mask, cat_feats, num_feats, txt_feats, img_feats)   # [B,T,C]
        probs = F.softmax(logits, dim=-1)                                          # [B,T,C]
        conf, pred = probs.max(dim=-1)                                             # [B,T], [B,T]
        take = (conf >= prob_threshold) & mask.bool()
        pred_out = torch.where(take, pred, torch.full_like(pred, unlabeled_code))

        out_preds.append(pred_out.cpu())
        out_conf.append(conf.cpu())
        out_mask.append(mask.cpu())

    return out_preds, out_conf, out_mask


def flatten_batches(batched_tensors: List[torch.Tensor]) -> torch.Tensor:
    """Concat tensors from a list of batches along batch dimension."""
    return torch.cat(batched_tensors, dim=0) if batched_tensors else torch.empty(0)


def attach_predictions_to_df(df_prepared: pd.DataFrame,
                             preds_bt: List[torch.Tensor],
                             conf_bt: List[torch.Tensor],
                             mask_bt: List[torch.Tensor],
                             label_enc) -> pd.DataFrame:
    """
    Map predictions (per-session sequences) back to the row order in df_prepared.
    Assumes df_prepared was sorted by [session_id, timestamp] like in training.
    """
    preds = flatten_batches(preds_bt)   # [N,T] padded
    confs = flatten_batches(conf_bt)    # [N,T]
    masks = flatten_batches(mask_bt)    # [N,T]

    valid_idx = masks.bool()
    flat_preds = preds[valid_idx]       # [M]
    flat_confs = confs[valid_idx]       # [M]

    pred_ids = flat_preds.tolist()
    conf_vals = flat_confs.tolist()

    out = df_prepared.copy()
    out["pred_step_id"] = np.nan
    out["pred_step_name"] = None
    out["pred_confidence"] = np.nan

    session_col = cfg["data"]["session_col"]

    # Walk sessions in the same order used to encode
    offset = 0
    for sid, g in out.groupby(session_col, sort=False):
        L = len(g)
        pred_slice = pred_ids[offset: offset + L]
        conf_slice = conf_vals[offset: offset + L]
        offset += L

        idx = g.index
        out.loc[idx, "pred_step_id"] = pred_slice
        out.loc[idx, "pred_confidence"] = conf_slice

        names: List[str | None] = []
        for p in pred_slice:
            if p is None or (isinstance(p, (int, float)) and int(p) < 0):
                names.append(None)
            else:
                names.append(label_enc.inverse_transform([int(p)])[0])
        out.loc[idx, "pred_step_name"] = names

    return out


# ------------------------------- Script --------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) artifacts
    artifacts = load_artifacts("trained_artifacts")
    label_enc = artifacts["label_enc"]
    if label_enc is None:
        raise RuntimeError("label_encoder.pkl not found — required to size classifier and decode predictions.")
    num_classes = len(label_enc.classes_)

    # 2) RAW → FEATURES (+ get txt/img dims for model shape)
    df_prepared = load_and_prepare_raw_dataframe()
    df_prepared = df_prepared.sort_values([cfg["data"]["session_col"], cfg["data"]["timestamp_col"]])
    loader, txt_dim, img_dim = build_inference_loader(df_prepared, artifacts, device)

    # 3) Build model with dims matching training (txt/img) and load weights
    model = build_model_from_cfg(
        event_vocab=artifacts["event_vocab"],
        cat_enc=artifacts["cat_enc"],
        device=device,
        num_classes=num_classes,
        txt_in_dim=txt_dim,
        img_in_dim=img_dim
    )
    load_checkpoint(model, "trained_artifacts/best.pt", device)

    # 4) INFERENCE with abstention
    preds_bt, conf_bt, mask_bt = predict_with_threshold(
        model=model,
        loader=loader,
        device=device,
        prob_threshold=cfg["train"]["precision_threshold"],
        unlabeled_code=-1
    )

    # 5) ALIGN predictions back to rows & SAVE
    scored_df = attach_predictions_to_df(
        df_prepared=df_prepared,
        preds_bt=preds_bt,
        conf_bt=conf_bt,
        mask_bt=mask_bt,
        label_enc=label_enc
    )
    out_path = Path("predictions.csv")
    scored_df.to_csv(out_path, index=False)
    coverage = (scored_df["pred_step_id"] != -1).mean()
    print(f"Saved predictions → {out_path} | coverage @ {cfg['train']['precision_threshold']:.2f}: {coverage:.3f}")


if __name__ == "__main__":
    main()
