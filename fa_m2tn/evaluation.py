"""Evaluation helpers shared by training and checkpoint evaluation."""

from collections import defaultdict

import numpy as np
import torch

from fa_m2tn.utils.metrics import compute_metrics


@torch.no_grad()
def evaluate_soh(model, loader, device, return_samples=False):
    model.eval()
    predictions, targets, cell_ids = [], [], []
    for batch in loader:
        inputs = batch["x_masked"].to(device, non_blocking=True)
        _, soh_prediction, _ = model(inputs)
        predictions.append(soh_prediction.squeeze(-1).float().cpu().numpy())
        targets.append(batch["soh"].numpy())
        if return_samples:
            cell_ids.extend(str(value) for value in batch["cell_id"])

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    metrics = compute_metrics(predictions, targets)
    if not return_samples:
        return metrics
    return metrics, {
        "cell_ids": cell_ids,
        "soh_prediction": predictions,
        "soh_target": targets,
    }


@torch.no_grad()
def evaluate_tasks(model, loader, device, enable_mae=True):
    model.eval()
    pooled = defaultdict(list)
    grouped = defaultdict(lambda: defaultdict(list))

    for batch in loader:
        inputs = batch["x_masked"].to(device, non_blocking=True)
        targets = batch["x_full"].to(device, non_blocking=True)
        mask = batch["binary_mask"].to(device, non_blocking=True)
        reconstruction, soh_prediction, vdr_prediction = model(inputs)

        values = {
            "soh_prediction": soh_prediction.squeeze(-1).float().cpu().numpy(),
            "soh_target": batch["soh"].numpy(),
            "vdr_prediction": vdr_prediction.squeeze(-1).float().cpu().numpy(),
            "vdr_target": batch["vdr"].numpy(),
        }
        if enable_mae and reconstruction is not None:
            squared_error = (reconstruction.float() - targets.float()) ** 2
            expanded_mask = mask.unsqueeze(-1).expand_as(squared_error)
            numerator = (squared_error * expanded_mask).sum(dim=(1, 2))
            denominator = expanded_mask.sum(dim=(1, 2)).clamp_min(1)
            values["reconstruction_rmse"] = torch.sqrt(
                numerator / denominator
            ).cpu().numpy()

        for key, value in values.items():
            pooled[key].extend(np.asarray(value).tolist())
        for index, dataset in enumerate(batch["dataset"]):
            for key, value in values.items():
                grouped[str(dataset)][key].append(float(value[index]))

    def summarize(values):
        result = {
            "n": len(values["soh_target"]),
            "soh": compute_metrics(
                values["soh_prediction"], values["soh_target"]
            ),
            "vdr": compute_metrics(
                values["vdr_prediction"], values["vdr_target"]
            ),
        }
        if values.get("reconstruction_rmse"):
            reconstruction_rmse = np.asarray(
                values["reconstruction_rmse"], dtype=np.float64
            )
            result["masked_reconstruction"] = {
                "mean_per_sample_rmse": float(reconstruction_rmse.mean()),
                "median_per_sample_rmse": float(np.median(reconstruction_rmse)),
            }
        return result

    return summarize(pooled), {
        dataset: summarize(values)
        for dataset, values in sorted(grouped.items())
    }


__all__ = ["evaluate_soh", "evaluate_tasks"]
