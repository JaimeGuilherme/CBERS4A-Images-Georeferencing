import numpy as np

def calculate_metrics(preds, targets):
    try:
        preds_np = preds.cpu().numpy() if hasattr(preds, 'cpu') else np.array(preds)
        targets_np = targets.cpu().numpy() if hasattr(targets, 'cpu') else np.array(targets)
    except:
        preds_np = np.array(preds); targets_np = np.array(targets)
    preds_flat = preds_np.flatten(); targets_flat = targets_np.flatten()
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection
    iou = (intersection / union) if union > 0 else 0.0
    precision = (intersection / preds_flat.sum()) if preds_flat.sum() > 0 else 0.0
    recall = (intersection / targets_flat.sum()) if targets_flat.sum() > 0 else 0.0
    return float(iou), float(precision), float(recall)
