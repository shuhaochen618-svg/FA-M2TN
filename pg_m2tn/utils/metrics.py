"""Metrics used by PG-M2TN validation and held-out test evaluation."""

from collections import defaultdict

import numpy as np


def compute_metrics(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    errors = predictions - targets
    nonzero = np.abs(targets) > 1e-6
    filtered = targets >= 0.5
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "mape": (
            float(np.mean(np.abs(errors[nonzero] / targets[nonzero])) * 100.0)
            if nonzero.any()
            else 0.0
        ),
        "mape_filtered": (
            float(np.mean(np.abs(errors[filtered] / targets[filtered])) * 100.0)
            if filtered.any()
            else 0.0
        ),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0,
    }


def per_dataset_metrics(cell_ids, predictions, targets, datasets):
    groups = defaultdict(lambda: {"predictions": [], "targets": []})
    for cell_id, prediction, target in zip(cell_ids, predictions, targets):
        normalized = str(cell_id).upper().replace("-", "_")
        dataset = next((name for name in datasets if normalized.startswith(name)), None)
        if dataset is None:
            dataset = next((name for name in datasets if name in normalized), "OTHER")
        groups[dataset]["predictions"].append(prediction)
        groups[dataset]["targets"].append(target)

    output = {}
    for dataset, values in sorted(groups.items()):
        output[dataset] = {
            "n": len(values["targets"]),
            **compute_metrics(values["predictions"], values["targets"]),
        }
    return output


__all__ = ["compute_metrics", "per_dataset_metrics"]
