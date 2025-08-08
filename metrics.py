import torch

def calculate_metrics(preds, targets):
    """
    preds, targets: tensores booleanos ou float 0/1, shape [batch, 1, H, W]
    Retorna: IoU, precisão, recall (float)
    """
    preds = preds.flatten()
    targets = targets.flatten()

    intersection = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item() - intersection
    iou = intersection / union if union > 0 else 0.0

    precision = intersection / preds.sum().item() if preds.sum().item() > 0 else 0.0
    recall = intersection / targets.sum().item() if targets.sum().item() > 0 else 0.0

    return iou, precision, recall
