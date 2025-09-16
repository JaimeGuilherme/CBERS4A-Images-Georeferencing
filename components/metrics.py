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

def association_metrics(distances: np.ndarray, n_detectados: int, n_osm: int, max_distance: float):
    '''
    distances: vetor (N_pares,) de distâncias dos pares válidos
    n_detectados, n_osm: totais disponíveis em cada conjunto
    max_distance: limiar aplicado para aceitar o par

    Retorna dict com contagens e estatísticas de proximidade.
    '''
    n_pares = int(distances.size)
    perc_cobertura_detect = (n_pares / max(1, n_detectados)) * 100.0
    perc_cobertura_osm = (n_pares / max(1, n_osm)) * 100.0

    if n_pares == 0:
        return {
            'n_detectados': int(n_detectados),
            'n_osm': int(n_osm),
            'n_pares': 0,
            'cobertura_detectados_%': 0.0,
            'cobertura_osm_%': 0.0,
            'mean_m': None,
            'median_m': None,
            'std_m': None,
            'rmse_m': None,
            'pct_<=5m': 0.0,
            'pct_<=10m': 0.0,
            'pct_<=20m': 0.0,
            'max_distance_m': float(max_distance)
        }

    mean = float(np.mean(distances))
    median = float(np.median(distances))
    std = float(np.std(distances))
    rmse = float(np.sqrt(np.mean(distances**2)))

    def pct(th):
        return float((distances <= th).mean() * 100.0)

    return {
        'n_detectados': int(n_detectados),
        'n_osm': int(n_osm),
        'n_pares': n_pares,
        'cobertura_detectados_%': perc_cobertura_detect,
        'cobertura_osm_%': perc_cobertura_osm,
        'mean_m': mean,
        'median_m': median,
        'std_m': std,
        'rmse_m': rmse,
        'pct_<=5m': pct(5.0),
        'pct_<=10m': pct(10.0),
        'pct_<=20m': pct(20.0),
        'max_distance_m': float(max_distance)
    }
