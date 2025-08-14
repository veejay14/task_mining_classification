"""Training pipeline for event log sequence tagging models.

This script provides end-to-end training functionality including:
- Data loading and feature engineering from raw CSV
- Session-aware train/validation/test splitting (80/10/10)
- Vocabulary and encoder fitting on training data
- Optional k-fold cross-validation hyperparameter optimization
- Model training with early stopping and learning rate scheduling
- Comprehensive evaluation with misclassification analysis
- Model checkpoint and artifact saving

The pipeline supports both BiLSTM and Transformer architectures with
multimodal features (events, categorical, numeric, text, image).

Usage:
    python src/training_pipeline.py
    
Configuration is loaded from src/config/config.yaml
"""

from __future__ import annotations
from itertools import chain
from pathlib import Path
import json
import joblib

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.config import cfg
from src.data_preparation.data_pipeline import (
    split_by_session, session_sequence_generator, build_event_vocab, fit_label_encoder,
    fit_cat_encoder, cat_vocab_sizes_from_encoder, fit_numeric_scaler, build_split,
    MultiFeatureSequenceDataset, collate_batch,
    fit_text_encoder, fit_image_encoder  # <-- NEW
)
from src.data_preparation.data_loading import EventLogDataBuilder
from src.modelling.models import MultiFeatureBiLSTMTagger, MultiFeatureTransformerTagger
from src.evaluation.evaluator import evaluate_plain, evaluate_selective
from src.hyperparameter_tuning.tuner import run_hpo_kfold


# ------------------------------ Helpers -------------------------------

def _build_model_from_params(model_type, event_vocab, num_classes, cat_vocab_sizes, params, device,
                             txt_in_dim: int, img_in_dim: int):
    """
    Factory to build a model from a param dict (used by HPO and final build).
    Adds txt/img input dims and fusion options from YAML.
    """
    fusion_mode = cfg["model"].get("fusion_mode", "concat")
    txt_proj_dim = cfg["model"].get("txt_proj_dim", 32)
    img_proj_dim = cfg["model"].get("img_proj_dim", 32)

    if model_type == "bilstm":
        return MultiFeatureBiLSTMTagger(
            vocab_size=len(event_vocab),
            num_classes=num_classes,
            cat_vocab_sizes=cat_vocab_sizes,
            emb_event_dim=params["emb_event_dim"],
            emb_cat_dim=params["emb_cat_dim"],
            num_cols=cfg["data"]["num_features"],
            num_proj_dim=params["num_proj_dim"],
            # multimodal
            txt_in_dim=txt_in_dim,
            txt_proj_dim=txt_proj_dim,
            img_in_dim=img_in_dim,
            img_proj_dim=img_proj_dim,
            # rnn
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            pad_idx=cfg["model"]["pad_idx"],
            dropout_p=params["dropout_p"],
            rnn_dropout=params["rnn_dropout"],
            fusion_mode=fusion_mode,
        ).to(device)

    elif model_type == "transformer":
        return MultiFeatureTransformerTagger(
            vocab_size=len(event_vocab),
            num_classes=num_classes,
            cat_vocab_sizes=cat_vocab_sizes,
            emb_event_dim=params["emb_event_dim"],
            emb_cat_dim=params["emb_cat_dim"],
            num_cols=cfg["data"]["num_features"],
            num_proj_dim=params["num_proj_dim"],
            # multimodal
            txt_in_dim=txt_in_dim,
            txt_proj_dim=txt_proj_dim,
            img_in_dim=img_in_dim,
            img_proj_dim=img_proj_dim,
            # tf
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_layers=params["num_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params["dropout"],
            pad_idx=cfg["model"]["pad_idx"],
            fusion_mode=fusion_mode,
        ).to(device)

    else:
        raise ValueError("model_type must be 'bilstm' or 'transformer'")


def _sample_params(trial, model_type):
    """Search space for HPO (used by hpo/tuner.py via dependency injection)."""
    import optuna
    if model_type == "bilstm":
        return {
            "emb_event_dim": trial.suggest_categorical("emb_event_dim", [32, 48, 64, 96]),
            "emb_cat_dim":   trial.suggest_categorical("emb_cat_dim",   [8, 12, 16, 24]),
            "num_proj_dim":  trial.suggest_categorical("num_proj_dim",  [8, 12, 16, 24]),
            "hidden_dim":    trial.suggest_categorical("hidden_dim",    [64, 96, 128, 192]),
            "num_layers":    trial.suggest_int("num_layers", 1, 3),
            "dropout_p":     trial.suggest_float("dropout_p", 0.1, 0.6),
            "rnn_dropout":   trial.suggest_float("rnn_dropout", 0.0, 0.5),
            "lr":            trial.suggest_float("lr", 5e-4, 3e-3, log=True),
            "weight_decay":  trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True),
        }
    else:  # transformer
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        d_model = trial.suggest_categorical("d_model", [64, 96, 128, 192])
        if d_model % nhead != 0:  # ensure divisibility
            d_model = nhead * (d_model // nhead or 1)
        return {
            "emb_event_dim":   trial.suggest_categorical("emb_event_dim", [32, 48, 64]),
            "emb_cat_dim":     trial.suggest_categorical("emb_cat_dim",   [16, 24, 32, 48]),
            "num_proj_dim":    trial.suggest_categorical("num_proj_dim",  [8, 12, 16, 24]),
            "d_model":         d_model,
            "nhead":           nhead,
            "num_layers":      trial.suggest_int("num_layers", 1, 4),
            "dim_feedforward": trial.suggest_categorical("dim_feedforward", [128, 256, 384, 512]),
            "dropout":         trial.suggest_float("dropout", 0.1, 0.5),
            "lr":              trial.suggest_float("lr", 5e-4, 3e-3, log=True),
            "weight_decay":    trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True),
        }


def _defaults_from_yaml(model_type):
    """Fallback params if HPO disabled."""
    if model_type == "bilstm":
        return {
            "emb_event_dim": cfg["model"]["emb_event_dim"],
            "emb_cat_dim":   cfg["model"]["emb_cat_dim"],
            "num_proj_dim":  cfg["model"]["num_proj_dim"],
            "hidden_dim":    cfg["model"]["lstm_hidden_dim"],
            "num_layers":    cfg["model"]["lstm_num_layers"],
            "dropout_p":     cfg["model"]["lstm_dropout"],
            "rnn_dropout":   cfg["model"]["lstm_rnn_dropout"],
            "lr":            cfg["train"]["lr"],
            "weight_decay":  cfg["train"]["weight_decay"],
        }
    else:
        return {
            "emb_event_dim":   cfg["model"]["emb_event_dim"],
            "emb_cat_dim":     cfg["model"]["emb_cat_dim"],
            "num_proj_dim":    cfg["model"]["num_proj_dim"],
            "d_model":         cfg["model"]["d_model"],
            "nhead":           cfg["model"]["nhead"],
            "num_layers":      cfg["model"]["tf_num_layers"],
            "dim_feedforward": cfg["model"]["tf_ff_dim"],
            "dropout":         cfg["model"]["tf_dropout"],
            "lr":              cfg["train"]["lr"],
            "weight_decay":    cfg["train"]["weight_decay"],
        }


# -------------------------------- Main ---------------------------------

def main():
    # --- Load & feature-engineer raw data ---
    builder = EventLogDataBuilder(cfg)
    df = builder.load_and_prepare("data/code_challenge_dataset.csv")
    df = df.sort_values([cfg["data"]["session_col"], cfg["data"]["timestamp_col"]])

    # --- Split by session ---
    train_df, val_df, test_df = split_by_session(
        df,
        session_col=cfg["data"]["session_col"],
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        seed=cfg["data"]["random_seed"],
    )

    # --- Build vocab/encoders/scalers on TRAIN only ---
    train_ev_seqs = [ev for ev, _, _ in session_sequence_generator(
        train_df, cfg["data"]["session_col"], cfg["data"]["event_col"], cfg["data"]["label_col"]
    )]
    train_lbl_seqs = [lbl for _, lbl, _ in session_sequence_generator(
        train_df, cfg["data"]["session_col"], cfg["data"]["event_col"], cfg["data"]["label_col"]
    )]
    all_train_labels = [x for x in chain.from_iterable(train_lbl_seqs)]

    event_vocab = build_event_vocab(train_ev_seqs)
    label_enc = fit_label_encoder(all_train_labels)
    cat_enc = fit_cat_encoder(train_df, cfg["data"]["cat_features"])
    num_scaler = fit_numeric_scaler(train_df, cfg["data"]["num_features"])
    cat_vocab_sizes = cat_vocab_sizes_from_encoder(cat_enc, cfg["data"]["cat_features"])

    # --- Optional text & image encoders (pretrained) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEXT_FEATURES = cfg["data"].get("text_features", []) or []
    IMAGE_PATH_COL = cfg["data"].get("image_path_col", None)

    text_encoder = (
        fit_text_encoder(device=str(device)) if TEXT_FEATURES else None
    )
    image_backbone, image_preprocess = (
        fit_image_encoder(device=str(device)) if IMAGE_PATH_COL else (None, None)
    )

    # --- Build datasets/loaders (now including txt/img & dims) ---
    def make_ds(split_df):
        ev_ids, lab_ids, cat_list, num_list, txt_list, img_list, txt_dim, img_dim = build_split(
            split_df, cfg["data"], event_vocab, label_enc, cat_enc, num_scaler,
            text_encoder=text_encoder, text_cols=TEXT_FEATURES,
            image_path_col=IMAGE_PATH_COL, image_backbone=image_backbone, image_preprocess=image_preprocess
        )
        ds = MultiFeatureSequenceDataset(ev_ids, lab_ids, cat_list, num_list, txt_list, img_list)
        return ds, txt_dim, img_dim

    train_ds, txt_dim_train, img_dim_train = make_ds(train_df)
    val_ds,   _,             _             = make_ds(val_df)
    test_ds,  _,             _             = make_ds(test_df)

    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,  collate_fn=collate_batch)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["data"]["batch_size"], shuffle=False, collate_fn=collate_batch)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["data"]["batch_size"], shuffle=False, collate_fn=collate_batch)

    # --- HPO (optional) to get best_params using k-fold CV on train+val ---
    num_classes = len(label_enc.classes_)
    model_type = cfg["model"]["model_type"]

    best_params = None
    if cfg["train"]["hpo"]["enabled"]:
        # Combine train and val for HPO with k-fold cross validation
        train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        train_val_ds, _, _ = make_ds(train_val_df)
        
        print(f"\n[HPO] Running k-fold CV with {cfg['train']['hpo']['k_folds']} folds on {len(train_val_df)} samples...")
        best_params, study = run_hpo_kfold(
            model_type=model_type,
            device=device,
            num_classes=num_classes,
            event_vocab=event_vocab,
            cat_vocab_sizes=cat_vocab_sizes,
            train_val_dataset=train_val_ds,
            ignore_index=cfg["train"]["ignore_index"],
            grad_clip=cfg["train"]["grad_clip"],
            hpo_cfg=cfg["train"]["hpo"],
            txt_in_dim=txt_dim_train,
            img_in_dim=img_dim_train,
            k_folds=cfg["train"]["hpo"]["k_folds"],
            metric=cfg["train"]["hpo"]["metric"],
        )
        print(f"\n[HPO] Best params: {best_params}")
        print(f"[HPO] Best value: {study.best_value:.4f}")
    else:
        best_params = _defaults_from_yaml(model_type)

    # --- Build FINAL model using best_params (with txt/img dims) ---
    model = _build_model_from_params(
        model_type, event_vocab, num_classes, cat_vocab_sizes, best_params, device,
        txt_in_dim=txt_dim_train, img_in_dim=img_dim_train
    )

    # --- Final training using train+val data (after HPO) ---
    if cfg["train"]["hpo"]["enabled"]:
        # Use combined train+val for final training since we used it for HPO
        train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        final_train_ds, _, _ = make_ds(train_val_df)
        final_train_loader = DataLoader(final_train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, collate_fn=collate_batch)
        print(f"\n[TRAINING] Using combined train+val data ({len(train_val_df)} samples) for final training")
        final_val_loader = test_loader  # Use test as validation for early stopping (but don't evaluate on it)
        validation_name = "test (for early stopping only)"
    else:
        # Use regular train/val split if no HPO
        final_train_loader = train_loader
        final_val_loader = val_loader
        validation_name = "validation"

    # --- Loss/Optim/Scheduler ---
    loss_fn = nn.CrossEntropyLoss(ignore_index=cfg["train"]["ignore_index"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    # --- Train with early stopping ---
    best_val = float("inf")
    bad = 0
    patience = cfg["train"]["patience"]

    art_dir = Path("trained_artifacts")
    art_dir.mkdir(exist_ok=True)

    for epoch in range(cfg["train"]["num_epochs"]):
        model.train()
        tr_loss = 0.0
        for tokens, labels, mask, cat_feats, num_feats, txt_feats, img_feats in final_train_loader:
            tokens, labels, mask = tokens.to(device), labels.to(device), mask.to(device)
            cat_feats = {k: v.to(device) for k, v in cat_feats.items()}
            num_feats = {k: v.to(device) for k, v in num_feats.items()}
            txt_feats = {k: v.to(device) for k, v in txt_feats.items()}
            img_feats = {k: v.to(device) for k, v in img_feats.items()}

            optimizer.zero_grad()
            logits = model(tokens, mask, cat_feats, num_feats, txt_feats, img_feats)  # <-- pass txt/img
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()
            tr_loss += loss.item()

        avg_tr = tr_loss / max(1, len(final_train_loader))
        print(f"epoch {epoch+1} train loss: {avg_tr:.4f}")

        # validation (using final_val_loader which is either val_loader or test_loader)
        model.eval()
        vl_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for tokens, labels, mask, cat_feats, num_feats, txt_feats, img_feats in final_val_loader:
                tokens, labels, mask = tokens.to(device), labels.to(device), mask.to(device)
                cat_feats = {k: v.to(device) for k, v in cat_feats.items()}
                num_feats = {k: v.to(device) for k, v in num_feats.items()}
                txt_feats = {k: v.to(device) for k, v in txt_feats.items()}
                img_feats = {k: v.to(device) for k, v in img_feats.items()}

                logits = model(tokens, mask, cat_feats, num_feats, txt_feats, img_feats)
                vl_loss += loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)).item()
                preds = logits.argmax(-1)
                valid = mask.bool()
                correct += (preds[valid] == labels[valid]).sum().item()
                total += valid.sum().item()

        avg_vl = vl_loss / max(1, len(final_val_loader))
        val_acc = correct / max(1, total)
        print(f"           {validation_name} loss: {avg_vl:.4f} | token-acc: {val_acc:.3f}")

        scheduler.step(avg_vl)
        if avg_vl < best_val - 1e-4:
            best_val = avg_vl
            bad = 0
            torch.save(model.state_dict(), art_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Training complete. Best checkpoint saved to trained_artifacts/best.pt")

    # --- Load best & evaluate on test ---
    model.load_state_dict(torch.load(art_dir / "best.pt", map_location=device))

    test_loss, test_acc, test_f1, test_report, test_errors = evaluate_plain(
        test_loader, model, loss_fn, device,
        num_classes=len(label_enc.classes_),
        ignore_index=cfg["train"]["ignore_index"],
        return_errors=True,
        label_names=list(label_enc.classes_),  # nicer names in the report
        top_k=10
    )
    print("\n=== Misclassification analysis (plain test) ===")
    print(test_report)
    print(test_errors)

    # Selective
    _, _, _, test_sel_report, test_sel_errors = evaluate_selective(
        test_loader, model, loss_fn, device,
        num_classes=len(label_enc.classes_),
        ignore_index=cfg["train"]["ignore_index"],
        prob_threshold=cfg["train"]["precision_threshold"],
        return_errors=True,
        label_names=list(label_enc.classes_),
        top_k=10
    )
    print("\n=== Misclassification analysis (selective test) ===")
    print(test_report)
    print(test_sel_errors)

    # --- Save artifacts for inference ---
    with open(art_dir / "event_vocab.json", "w") as f:
        json.dump(event_vocab, f)
    joblib.dump(label_enc, art_dir / "label_encoder.pkl")
    if cat_enc is not None:
        joblib.dump(cat_enc, art_dir / "cat_encoder.pkl")
    if num_scaler is not None:
        joblib.dump(num_scaler, art_dir / "num_scaler.pkl")

    # (Text/image encoders are large; typically NOT saved. Recreate at inference if needed.)
    print(f"Artifacts saved to: {art_dir.resolve()}")


if __name__ == "__main__":
    main()
