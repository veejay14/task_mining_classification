"""Hyperparameter optimization with k-fold cross validation."""
from __future__ import annotations
from typing import Dict, Any, Tuple
import optuna
from optuna.pruners import MedianPruner
from torch import nn
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import numpy as np
from ..config.config import cfg
from ..modelling.models import (
    MultiFeatureBiLSTMTagger,
    MultiFeatureTransformerTagger,
)


def run_hpo_kfold(
    model_type: str,
    device: torch.device,
    num_classes: int,
    event_vocab: Dict[str, int],
    cat_vocab_sizes: Dict[str, int],
    train_val_dataset,  # Combined train+val dataset for k-fold CV
    ignore_index: int,
    grad_clip: float,
    hpo_cfg: Dict[str, Any],
    txt_in_dim: int = 0,
    img_in_dim: int = 0,
    k_folds: int = 5,
    metric: str = "val_loss"
) -> Tuple[Dict[str, Any], optuna.Study]:
    """Run hyperparameter optimization using k-fold cross validation.
    
    Args:
        model_type: Either 'bilstm' or 'transformer'
        device: PyTorch device
        num_classes: Number of output classes
        event_vocab: Event vocabulary mapping
        cat_vocab_sizes: Categorical feature vocabulary sizes
        train_val_dataset: Combined train+validation dataset for CV
        ignore_index: Index to ignore in loss computation
        grad_clip: Gradient clipping value
        hpo_cfg: Hyperparameter optimization configuration
        txt_in_dim: Text feature input dimension
        img_in_dim: Image feature input dimension
        k_folds: Number of folds for cross validation
        metric: Optimization metric ('val_loss' or 'val_macro_f1')
        
    Returns:
        Tuple of (best_params, study)
    """

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, model_type)
        
        # K-fold cross validation
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_indices, val_indices) in enumerate(kfold.split(train_val_dataset)):
            # Create fold datasets
            train_subset = Subset(train_val_dataset, train_indices)
            val_subset = Subset(train_val_dataset, val_indices)
            
            from ..data_preparation.data_pipeline import collate_batch
            
            train_loader = DataLoader(
                train_subset, 
                batch_size=cfg["data"]["batch_size"], 
                shuffle=True,
                collate_fn=collate_batch
            )
            val_loader = DataLoader(
                val_subset, 
                batch_size=cfg["data"]["batch_size"], 
                shuffle=False,
                collate_fn=collate_batch
            )
            
            # Build model for this fold
            model = _build_model_from_params(
                model_type, event_vocab, num_classes, cat_vocab_sizes, 
                params, device, txt_in_dim, img_in_dim
            )
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=1
            )
            
            # Train for limited epochs per fold
            fold_epochs = hpo_cfg.get("epochs_per_fold", 10)
            best_val_score = float("inf") if metric == "val_loss" else 0.0
            
            for epoch in range(fold_epochs):
                # Training phase
                model.train()
                for batch in train_loader:
                    tokens, labels, mask, cat_feats, num_feats = batch[:5]
                    txt_feats = batch[5] if len(batch) > 5 else {}
                    img_feats = batch[6] if len(batch) > 6 else {}
                    
                    tokens, labels, mask = tokens.to(device), labels.to(device), mask.to(device)
                    cat_feats = {k: v.to(device) for k, v in cat_feats.items()}
                    num_feats = {k: v.to(device) for k, v in num_feats.items()}
                    txt_feats = {k: v.to(device) for k, v in txt_feats.items()}
                    img_feats = {k: v.to(device) for k, v in img_feats.items()}
                    
                    optimizer.zero_grad()
                    logits = model(tokens, mask, cat_feats, num_feats, txt_feats, img_feats)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                y_true, y_pred = [], []
                
                with torch.no_grad():
                    for batch in val_loader:
                        tokens, labels, mask, cat_feats, num_feats = batch[:5]
                        txt_feats = batch[5] if len(batch) > 5 else {}
                        img_feats = batch[6] if len(batch) > 6 else {}
                        
                        tokens, labels, mask = tokens.to(device), labels.to(device), mask.to(device)
                        cat_feats = {k: v.to(device) for k, v in cat_feats.items()}
                        num_feats = {k: v.to(device) for k, v in num_feats.items()}
                        txt_feats = {k: v.to(device) for k, v in txt_feats.items()}
                        img_feats = {k: v.to(device) for k, v in img_feats.items()}
                        
                        logits = model(tokens, mask, cat_feats, num_feats, txt_feats, img_feats)
                        val_loss += loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)).item()
                        
                        preds = logits.argmax(-1)
                        valid = mask.bool() & (labels != ignore_index)
                        if valid.any():
                            y_true.extend(labels[valid].tolist())
                            y_pred.extend(preds[valid].tolist())
                
                avg_val_loss = val_loss / max(1, len(val_loader))
                scheduler.step(avg_val_loss)
                
                # Update best score for this fold
                if metric == "val_loss":
                    if avg_val_loss < best_val_score:
                        best_val_score = avg_val_loss
                else:  # val_macro_f1
                    if len(set(y_true)) > 1:
                        f1 = f1_score(y_true, y_pred, average="macro")
                        if f1 > best_val_score:
                            best_val_score = f1
            
            # Store fold score
            fold_scores.append(best_val_score)
            
            # Report intermediate results
            trial.report(np.mean(fold_scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Return average score across all folds
        avg_score = np.mean(fold_scores)
        return avg_score if metric == "val_loss" else -avg_score  # Minimize loss, maximize F1

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


def _build_model_from_params(
    model_type: str, 
    event_vocab: Dict[str, int], 
    num_classes: int, 
    cat_vocab_sizes: Dict[str, int], 
    params: Dict[str, Any], 
    device: torch.device,
    txt_in_dim: int = 0,
    img_in_dim: int = 0
) -> nn.Module:
    """Build model from hyperparameters."""
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
            txt_in_dim=txt_in_dim,
            txt_proj_dim=txt_proj_dim,
            img_in_dim=img_in_dim,
            img_proj_dim=img_proj_dim,
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            pad_idx=cfg["model"]["pad_idx"],
            dropout_p=params["dropout_p"],
            rnn_dropout=params["rnn_dropout"],
            fusion_mode=fusion_mode,
        ).to(device)
    
    if model_type == "transformer":
        return MultiFeatureTransformerTagger(
            vocab_size=len(event_vocab),
            num_classes=num_classes,
            cat_vocab_sizes=cat_vocab_sizes,
            emb_event_dim=params["emb_event_dim"],
            emb_cat_dim=params["emb_cat_dim"],
            num_cols=cfg["data"]["num_features"],
            num_proj_dim=params["num_proj_dim"],
            txt_in_dim=txt_in_dim,
            txt_proj_dim=txt_proj_dim,
            img_in_dim=img_in_dim,
            img_proj_dim=img_proj_dim,
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_layers=params["num_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params["dropout"],
            pad_idx=cfg["model"]["pad_idx"],
            fusion_mode=fusion_mode,
        ).to(device)
    
    raise ValueError("model_type must be 'bilstm' or 'transformer'")


def _sample_params(trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
    """Sample hyperparameters for optimization."""
    if model_type == "bilstm":
        return {
            "emb_event_dim": trial.suggest_categorical("emb_event_dim", [32, 48, 64, 96]),
            "emb_cat_dim": trial.suggest_categorical("emb_cat_dim", [8, 12, 16, 24]),
            "num_proj_dim": trial.suggest_categorical("num_proj_dim", [8, 12, 16, 24]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 96, 128, 192]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout_p": trial.suggest_float("dropout_p", 0.1, 0.6),
            "rnn_dropout": trial.suggest_float("rnn_dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 5e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True),
        }
    
    # transformer
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    d_model = trial.suggest_categorical("d_model", [64, 96, 128, 192])
    # Ensure d_model is divisible by nhead
    if d_model % nhead != 0:
        d_model = nhead * (d_model // nhead)
    
    return {
        "emb_event_dim": trial.suggest_categorical("emb_event_dim", [32, 48, 64, 96]),
        "emb_cat_dim": trial.suggest_categorical("emb_cat_dim", [8, 12, 16, 24, 32]),
        "num_proj_dim": trial.suggest_categorical("num_proj_dim", [8, 12, 16, 24]),
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [128, 256, 384]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 5e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True),
    }

