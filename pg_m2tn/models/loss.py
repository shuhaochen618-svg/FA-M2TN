"""Fixed-weight objective used by the August 2026 PG-M2TN revision."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedWeightedLoss(nn.Module):
    """Combine SOH, VDR, and masked reconstruction losses with fixed weights."""

    def __init__(
        self,
        soh_weight=0.50,
        vdr_weight=0.50,
        lambda_mae=0.10,
        scale_vdr=1.0,
        reconstruction_scope="masked",
    ):
        super().__init__()
        self.soh_weight = float(soh_weight)
        self.vdr_weight = float(vdr_weight)
        self.lambda_mae = float(lambda_mae)
        self.scale_vdr = float(scale_vdr)
        self.reconstruction_scope = reconstruction_scope

    def reconstruction_loss(self, batch, reconstruction, device):
        if reconstruction is None:
            return torch.zeros((), device=device)

        target = batch["x_full"].to(device)
        squared_error = (reconstruction - target) ** 2
        if self.reconstruction_scope == "masked":
            mask = batch["binary_mask"].to(device)
            expanded_mask = mask.unsqueeze(-1).expand_as(squared_error)
            if expanded_mask.any():
                return squared_error[expanded_mask].mean()
        return squared_error.mean()

    def forward(self, batch, reconstruction, soh_prediction, vdr_prediction):
        soh_target = batch["soh"].to(soh_prediction.device).view_as(soh_prediction)
        vdr_target = batch["vdr"].to(vdr_prediction.device).view_as(vdr_prediction)

        soh_loss = F.mse_loss(soh_prediction, soh_target)
        vdr_loss = F.mse_loss(vdr_prediction, vdr_target) * self.scale_vdr
        mae_loss = self.reconstruction_loss(
            batch, reconstruction, soh_prediction.device
        )
        total = (
            self.soh_weight * soh_loss
            + self.vdr_weight * vdr_loss
            + self.lambda_mae * mae_loss
        )
        parts = {
            "total": float(total.detach().cpu()),
            "soh": float(soh_loss.detach().cpu()),
            "vdr": float(vdr_loss.detach().cpu()),
            "mae": float(mae_loss.detach().cpu()),
        }
        return total, parts


__all__ = ["FixedWeightedLoss"]
