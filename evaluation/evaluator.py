import torch
import torch.nn.functional as F
from typing import Tuple, List
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_recall_fscore_support


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


def _unpack_batch(batch):
    if len(batch) == 5:
        tokens, labels, mask, cat_feats, num_feats = batch
        txt_feats, img_feats = {}, {}
    else:
        tokens, labels, mask, cat_feats, num_feats, txt_feats, img_feats = batch
    return tokens, labels, mask, cat_feats, num_feats, txt_feats, img_feats


def _misclassification_summary(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
    label_names: List[str] | None = None,
    top_k: int = 10
) -> str:
    """
    Build a readable summary of confusions:
      - top off-diagonal confusions overall
      - per-class 'most confused with'
      - per-class precision/recall/f1 (numbers complement the text report)
    """
    if not y_true or not y_pred:
        return "No valid tokens to analyze misclassifications."

    if label_names is None:
        label_names = [str(i) for i in range(num_classes)]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    # Off-diagonal pairs (i!=j)
    pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue
            cnt = int(cm[i, j])
            if cnt > 0:
                pairs.append((cnt, i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])
    top_pairs = pairs[:top_k]

    lines = []
    lines.append("Top confusions (True → Predicted) [count, rate in true-class]:")
    for cnt, i, j in top_pairs:
        row_sum = cm[i].sum()
        rate = (cnt / row_sum) if row_sum > 0 else 0.0
        lines.append(f"  {label_names[i]} → {label_names[j]} : {cnt}  ({rate:.2%})")

    # Per-class: most confused with
    lines.append("\nPer-class 'most confused with':")
    for i in range(num_classes):
        row = cm[i].copy()
        row[i] = 0
        j = row.argmax()
        cnt = int(row[j])
        total = cm[i].sum()
        if total > 0 and cnt > 0:
            lines.append(f"  {label_names[i]} most often confused with {label_names[j]} : {cnt} ({cnt/total:.2%})")
        elif total > 0:
            lines.append(f"  {label_names[i]}: no confusions (perfect on observed examples).")
        else:
            lines.append(f"  {label_names[i]}: no true examples in evaluation.")

    # Per-class metrics for quick glance
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    lines.append("\nPer-class metrics (precision | recall | f1 | support):")
    for i in range(num_classes):
        lines.append(f"  {label_names[i]}: {prec[i]:.3f} | {rec[i]:.3f} | {f1[i]:.3f} | {int(support[i])}")

    return "\n".join(lines)


def evaluate_plain(
    loader, model, loss_fn, device, num_classes: int, ignore_index: int,
    return_errors: bool = False, label_names: List[str] | None = None, top_k: int = 10
) -> Tuple[float, float, float, str] | Tuple[float, float, float, str, str]:
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
            target_names=(label_names if label_names else [str(c) for c in range(num_classes)])
        )
        errors = _misclassification_summary(
            y_true, y_pred, num_classes,
            label_names=(label_names if label_names else [str(c) for c in range(num_classes)]),
            top_k=top_k
        )
    else:
        macro_f1, report = 0.0, "No valid tokens."
        errors = "No valid tokens to analyze misclassifications."

    return (avg_loss, acc, macro_f1, report, errors) if return_errors else (avg_loss, acc, macro_f1, report)


def evaluate_selective(
    loader, model, loss_fn, device,
    num_classes: int, ignore_index: int,
    prob_threshold: float = 0.95,
    return_errors: bool = False, label_names: List[str] | None = None, top_k: int = 10
) -> Tuple[float, float, float, str] | Tuple[float, float, float, str, str]:
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
        base_report = classification_report(
            y_true, y_pred,
            labels=list(range(num_classes)),
            zero_division=0,
            target_names=(label_names if label_names else [str(c) for c in range(num_classes)])
        )
        report = f"Coverage (fraction labeled at ≥{prob_threshold:.2f}): {coverage:.3f}\n" + base_report
        errors = _misclassification_summary(
            y_true, y_pred, num_classes,
            label_names=(label_names if label_names else [str(c) for c in range(num_classes)]),
            top_k=top_k
        )
    else:
        macro_f1 = 0.0
        report = f"Coverage (fraction labeled at ≥{prob_threshold:.2f}): {coverage:.3f}\nNo predictions above threshold."
        errors = "No predictions above threshold to analyze misclassifications."

    return (avg_loss, acc, macro_f1, report, errors) if return_errors else (avg_loss, acc, macro_f1, report)

