from typing import Tuple, List, Dict, Any  # <-- add Dict, Any if not present
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score

def _unpack_batch(batch):
    """
    Support both legacy 5-tuple (tokens, labels, mask, cat_feats, num_feats)
    and new 7-tuple (..., txt_feats, img_feats).
    """
    if len(batch) == 5:
        tokens, labels, mask, cat_feats, num_feats = batch
        txt_feats, img_feats = {}, {}
    else:
        tokens, labels, mask, cat_feats, num_feats, txt_feats, img_feats = batch
    return tokens, labels, mask, cat_feats, num_feats, txt_feats, img_feats


def evaluate_plain(loader, model, loss_fn, device, num_classes: int, ignore_index: int) -> Tuple[float, float, float, str]:
    """Standard evaluation on all valid tokens (no abstention)."""
    model.eval()
    tot_loss, tot_tokens, tot_correct = 0.0, 0, 0
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for batch in loader:
            tokens, labels, mask, cat_feats, num_feats, txt_feats, img_feats = _unpack_batch(batch)

            tokens, labels, mask = tokens.to(device), labels.to(device), mask.to(device)
            cat_feats = {k: v.to(device) for k, v in cat_feats.items()}
            num_feats = {k: v.to(device) for k, v in num_feats.items()}
            txt_feats = {k: v.to(device) for k, v in txt_feats.items()}
            img_feats = {k: v.to(device) for k, v in img_feats.items()}

            logits = model(tokens, mask, cat_feats, num_feats, txt_feats, img_feats)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            tot_loss += loss.item()

            preds = logits.argmax(-1)
            valid = (labels != ignore_index) & mask.bool()

            tot_correct += (preds[valid] == labels[valid]).sum().item()
            tot_tokens  += valid.sum().item()

            y_true_all.append(labels[valid].detach().cpu())
            y_pred_all.append(preds[valid].detach().cpu())

    avg_loss = tot_loss / max(1, len(loader))
    acc = (tot_correct / tot_tokens) if tot_tokens > 0 else 0.0

    if y_true_all:
        y_true = torch.cat(y_true_all).tolist()
        y_pred = torch.cat(y_pred_all).tolist()
        macro_f1 = f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)))
        report = classification_report(
            y_true, y_pred,
            labels=list(range(num_classes)),
            zero_division=0,
            target_names=[str(c) for c in range(num_classes)]
        )
    else:
        macro_f1, report = 0.0, "No valid tokens."

    return avg_loss, acc, macro_f1, report


def evaluate_selective(loader, model, loss_fn, device,
                       num_classes: int, ignore_index: int,
                       prob_threshold: float = 0.95) -> Tuple[float, float, float, str]:
    """
    Selective evaluation:
    - Loss is over all tokens (reference)
    - Metrics are computed only on tokens with max prob >= threshold
    - Report includes coverage (fraction of valid tokens that were labeled)
    """
    model.eval()
    tot_loss, total_batches = 0.0, 0
    sel_correct, sel_count, valid_count = 0, 0, 0
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for batch in loader:
            tokens, labels, mask, cat_feats, num_feats, txt_feats, img_feats = _unpack_batch(batch)

            tokens, labels, mask = tokens.to(device), labels.to(device), mask.to(device)
            cat_feats = {k: v.to(device) for k, v in cat_feats.items()}
            num_feats = {k: v.to(device) for k, v in num_feats.items()}
            txt_feats = {k: v.to(device) for k, v in txt_feats.items()}
            img_feats = {k: v.to(device) for k, v in img_feats.items()}

            logits = model(tokens, mask, cat_feats, num_feats, txt_feats, img_feats)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            tot_loss += loss.item()
            total_batches += 1

            probs = F.softmax(logits, dim=-1)
            conf, preds = probs.max(dim=-1)

            valid = (labels != ignore_index) & mask.bool()
            selected = valid & (conf >= prob_threshold)

            valid_count += valid.sum().item()
            sel_correct += (preds[selected] == labels[selected]).sum().item()
            sel_count   += selected.sum().item()

            if selected.any():
                y_true_all.append(labels[selected].detach().cpu())
                y_pred_all.append(preds[selected].detach().cpu())

    avg_loss = tot_loss / max(1, total_batches)
    acc = (sel_correct / sel_count) if sel_count > 0 else 0.0
    coverage = (sel_count / valid_count) if valid_count > 0 else 0.0

    if y_true_all:
        y_true = torch.cat(y_true_all).tolist()
        y_pred = torch.cat(y_pred_all).tolist()
        macro_f1 = f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)))
        report = (
            f"Coverage (fraction labeled at ≥{prob_threshold:.2f}): {coverage:.3f}\n" +
            classification_report(
                y_true, y_pred,
                labels=list(range(num_classes)),
                zero_division=0,
                target_names=[str(c) for c in range(num_classes)]
            )
        )
    else:
        macro_f1 = 0.0
        report = f"Coverage (fraction labeled at ≥{prob_threshold:.2f}): {coverage:.3f}\nNo predictions above threshold."

    return avg_loss, acc, macro_f1, report
