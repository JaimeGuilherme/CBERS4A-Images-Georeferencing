import numpy as np

def calculate_metrics(preds, targets):
    preds_np = preds.detach().cpu().numpy().astype(np.uint8) if hasattr(preds, 'detach') else np.array(preds, dtype=np.uint8)
    targets_np = targets.detach().cpu().numpy().astype(np.uint8) if hasattr(targets, 'detach') else np.array(targets, dtype=np.uint8)

    if preds_np.ndim == 4 and preds_np.shape[1] == 1:
        preds_np = np.squeeze(preds_np, axis=1)
    if targets_np.ndim == 4 and targets_np.shape[1] == 1:
        targets_np = np.squeeze(targets_np, axis=1)

    ious, precisions, recalls = [], [], []

    for p, t in zip(preds_np, targets_np):
        p_flat = p.flatten()
        t_flat = t.flatten()

        intersection = np.logical_and(p_flat, t_flat).sum()
        union = np.logical_or(p_flat, t_flat).sum()

        iou = (intersection / union) if union > 0 else 0.0
        precision = (intersection / p_flat.sum()) if p_flat.sum() > 0 else 0.0
        recall = (intersection / t_flat.sum()) if t_flat.sum() > 0 else 0.0

        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)

    return float(np.mean(ious)), float(np.mean(precisions)), float(np.mean(recalls))
