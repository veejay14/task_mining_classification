# hpo/tuner.py
from __future__ import annotations
from typing import Dict, Any
import optuna
from optuna.pruners import MedianPruner
from torch import nn
import torch
from sklearn.metrics import f1_score


def run_hpo(
    model_type: str,
    device,
    num_classes: int,
    event_vocab: Dict[str,int],
    cat_vocab_sizes: Dict[str,int],
    train_loader,
    val_loader,
    ignore_index: int,
    grad_clip: float,
    hpo_cfg: Dict[str,Any],
    build_model_from_params,     # func(model_type, vocab, classes, cat_sizes, params, device) -> nn.Module
    sample_params,               # func(trial, model_type) -> dict
    metric: str = "val_loss",    # or "val_macro_f1"
):

    def objective(trial: optuna.Trial):
        params = sample_params(trial, model_type)
        model = build_model_from_params(model_type, event_vocab, num_classes, cat_vocab_sizes, params, device)
        loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        optim = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=1, verbose=False)

        best_val = float("inf"); best_f1 = 0.0; bad = 0; patience = 3
        for epoch in range(hpo_cfg["epochs_per_trial"]):
            # train
            model.train()
            for tokens, labels, mask, cat_feats, num_feats in train_loader:
                tokens, labels, mask = tokens.to(device), labels.to(device), mask.to(device)
                cat_feats = {k: v.to(device) for k, v in cat_feats.items()}
                num_feats = {k: v.to(device) for k, v in num_feats.items()}
                optim.zero_grad()
                logits = model(tokens, mask, cat_feats, num_feats)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()
            # val
            model.eval()
            vl_loss, correct, total = 0.0, 0, 0
            y_true, y_pred = [], []
            with torch.no_grad():
                for tokens, labels, mask, cat_feats, num_feats in val_loader:
                    tokens, labels, mask = tokens.to(device), labels.to(device), mask.to(device)
                    cat_feats = {k: v.to(device) for k, v in cat_feats.items()}
                    num_feats = {k: v.to(device) for k, v in num_feats.items()}
                    logits = model(tokens, mask, cat_feats, num_feats)
                    vl_loss += loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)).item()
                    preds = logits.argmax(-1)
                    valid = mask.bool() & (labels != ignore_index)
                    correct += (preds[valid] == labels[valid]).sum().item()
                    total += valid.sum().item()
                    if valid.any():
                        y_true.extend(labels[valid].tolist())
                        y_pred.extend(preds[valid].tolist())
            avg_vl = vl_loss / max(1, len(val_loader))
            sched.step(avg_vl)

            if metric == "val_loss":
                score = avg_vl
                trial.report(score, epoch)
                if trial.should_prune(): raise optuna.TrialPruned()
                if avg_vl < best_val - 1e-4: best_val, bad = avg_vl, 0
                else: bad += 1
            else:
                f1 = f1_score(y_true, y_pred, average="macro") if len(set(y_true)) > 1 else 0.0
                score = -f1  # maximize F1
                trial.report(score, epoch)
                if trial.should_prune(): raise optuna.TrialPruned()
                if f1 > best_f1 + 1e-4: best_f1, bad = f1, 0
                else: bad += 1

            if bad >= patience: break

        return score

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(direction=hpo_cfg["direction"], pruner=pruner)
    timeout = hpo_cfg.get("timeout_minutes")
    study.optimize(
        objective,
        n_trials=hpo_cfg["n_trials"],
        timeout=(timeout * 60) if timeout else None,
        show_progress_bar=False
    )
    return study.best_trial.params, study


def _build_model_from_params(model_type, event_vocab, num_classes, cat_vocab_sizes, params, device):
    if model_type == "bilstm":
        return MultiFeatureBiLSTMTagger(
            vocab_size=len(event_vocab),
            num_classes=num_classes,
            cat_vocab_sizes=cat_vocab_sizes,
            emb_event_dim=params["emb_event_dim"],
            emb_cat_dim=params["emb_cat_dim"],
            num_cols=cfg["data"]["num_features"],
            num_proj_dim=params["num_proj_dim"],
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            pad_idx=cfg["model"]["pad_idx"],
            dropout_p=params["dropout_p"],
            rnn_dropout=params["rnn_dropout"],
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
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_layers=params["num_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params["dropout"],
            pad_idx=cfg["model"]["pad_idx"],
        ).to(device)
    else:
        raise ValueError("model_type must be 'bilstm' or 'transformer'")


def _sample_params(trial, model_type):
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
        # ensure d_model divisible by nhead
        if d_model % nhead != 0:
            d_model = nhead * (d_model // nhead)
        return {
            "emb_event_dim": trial.suggest_categorical("emb_event_dim", [32, 48, 64, 96]),
            "emb_cat_dim": trial.suggest_categorical("emb_cat_dim", [8, 12, 16, 24, 32]),
            "num_proj_dim": trial.suggest_categorical("num_proj_dim", [8, 12, 16, 24]),
            "d_model": trial.suggest_categorical("d_model", [64, 96, 128]),
            "nhead": trial.suggest_categorical("nhead", [2, 4]),  # ensure d_model % nhead == 0
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dim_feedforward": trial.suggest_categorical("dim_feedforward", [128, 256, 384]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 5e-4, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True),
        }

