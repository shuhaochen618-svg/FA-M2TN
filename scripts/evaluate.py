"""Evaluate an FA-M2TN checkpoint on the fixed held-out test split."""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPOSITORY_ROOT)

from fa_m2tn.data.dataset_loader import BatteryCycleDataset, split_by_cell
from fa_m2tn.data.masking_engine import MaskedBatteryDataset
from fa_m2tn.evaluation import evaluate_tasks
from fa_m2tn.models.fa_m2tn import FAM2TN
from fa_m2tn.protocol import DATASETS, PROTOCOL


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FA-M2TN")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="./dataset")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--output", default="./outputs/evaluation.json")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--split_seed", type=int, default=PROTOCOL["split_seed"])
    parser.add_argument(
        "--mask_seed", type=int, default=PROTOCOL["test_mask_seed"]
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    configuration = checkpoint["hp"]
    variant = configuration.get("variant", "full")
    model = FAM2TN(
        input_dim=2,
        hidden_dim=int(configuration["hidden_dim"]),
        num_layers=int(configuration["num_layers"]),
        dropout=float(configuration.get("dropout", 0.2)),
        enable_mae=bool(configuration.get("enable_mae", True)),
    ).to(device)
    state = {
        name.replace("module.", ""): value
        for name, value in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    dataset = BatteryCycleDataset(
        data_root=args.data_root,
        datasets=args.datasets,
        seq_len=PROTOCOL["sequence_length"],
        min_cycles=PROTOCOL["minimum_cycles"],
    )
    _, _, test_indices, split_cells = split_by_cell(
        dataset, seed=args.split_seed, return_cell_ids=True
    )
    test_dataset = MaskedBatteryDataset(
        Subset(dataset, test_indices),
        fixed_mask_ratio=PROTOCOL["evaluation_mask_ratio"],
        seed=args.mask_seed,
    )
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    pooled, per_dataset = evaluate_tasks(
        model,
        loader,
        device,
        enable_mae=bool(configuration.get("enable_mae", True)),
    )
    if float(configuration.get("vdr_weight", 0.0)) == 0.0:
        pooled["diagnostic_unsupervised_vdr"] = pooled.pop("vdr")
        for values in per_dataset.values():
            values["diagnostic_unsupervised_vdr"] = values.pop("vdr")

    output = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "variant": variant,
        "configuration": configuration,
        "protocol": {
            "profile_mode": PROTOCOL["profile_mode"],
            "split_strategy": "dataset-stratified cell-level 70/15/15",
            "split_seed": args.split_seed,
            "test_mask_seed": args.mask_seed,
            "test_mask_ratio": PROTOCOL["evaluation_mask_ratio"],
            "test_cells": split_cells["test"],
            "test_samples": len(test_indices),
        },
        "test_metrics": pooled,
        "test_per_dataset": per_dataset,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(
        f"SOH RMSE={pooled['soh']['rmse']:.6f}, "
        f"MAE={pooled['soh']['mae']:.6f}, R2={pooled['soh']['r2']:.6f}"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
