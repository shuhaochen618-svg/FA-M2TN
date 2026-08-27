"""Train PG-M2TN or one exact August 2026 ablation variant."""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPOSITORY_ROOT)

from pg_m2tn.data.dataset_loader import BatteryCycleDataset, split_by_cell
from pg_m2tn.data.masking_engine import MaskedBatteryDataset
from pg_m2tn.evaluation import evaluate_soh, evaluate_tasks
from pg_m2tn.models.loss import FixedWeightedLoss
from pg_m2tn.models.pg_m2tn import PGM2TN, count_parameters
from pg_m2tn.protocol import DATASETS, PROTOCOL, VARIANTS, variant_config


class NumpyEncoder(json.JSONEncoder):
    def default(self, value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return super().default(value)


def save_json(value, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, cls=NumpyEncoder)


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_scheduler(optimizer, epochs, warmup_epochs):
    def multiplier(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        denominator = max(1, epochs - warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(denominator)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the August 2026 PG-M2TN configuration"
    )
    parser.add_argument("--data_root", default="./dataset")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="full")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--epochs", type=int, default=PROTOCOL["epochs"])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--per_gpu_batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=PROTOCOL["training_seed"])
    parser.add_argument("--split_seed", type=int, default=PROTOCOL["split_seed"])
    parser.add_argument(
        "--val_mask_seed", type=int, default=PROTOCOL["validation_mask_seed"]
    )
    parser.add_argument(
        "--test_mask_seed", type=int, default=PROTOCOL["test_mask_seed"]
    )
    parser.add_argument("--patience", type=int, default=PROTOCOL["early_stopping_patience"])
    parser.add_argument("--nondeterministic", action="store_true")
    parser.add_argument("--verbose_epochs", action="store_true")
    return parser.parse_args()


def resolve_batch_protocol(args, configuration):
    replicas = torch.cuda.device_count() if torch.cuda.is_available() else 1
    per_gpu_batch = args.per_gpu_batch_size or configuration["per_gpu_batch_size"]
    global_micro_batch = per_gpu_batch * replicas
    target = PROTOCOL["target_effective_batch_size"]
    if args.grad_accum_steps is None:
        if target % global_micro_batch != 0:
            raise ValueError(
                "The target effective batch size 2048 is not divisible by the "
                f"global micro-batch {global_micro_batch}. Set --grad_accum_steps."
            )
        accumulation_steps = target // global_micro_batch
    else:
        accumulation_steps = args.grad_accum_steps
    if accumulation_steps < 1:
        raise ValueError("--grad_accum_steps must be at least 1")
    return replicas, per_gpu_batch, global_micro_batch, accumulation_steps


def build_loaders(args, global_micro_batch):
    dataset = BatteryCycleDataset(
        data_root=args.data_root,
        datasets=args.datasets,
        seq_len=PROTOCOL["sequence_length"],
        min_cycles=PROTOCOL["minimum_cycles"],
    )
    train_indices, val_indices, test_indices, split_cells = split_by_cell(
        dataset, seed=args.split_seed, return_cell_ids=True
    )
    train_dataset = MaskedBatteryDataset(
        Subset(dataset, train_indices),
        min_ratio=PROTOCOL["training_mask_range"][0],
        max_ratio=PROTOCOL["training_mask_range"][1],
    )
    val_dataset = MaskedBatteryDataset(
        Subset(dataset, val_indices),
        fixed_mask_ratio=PROTOCOL["evaluation_mask_ratio"],
        seed=args.val_mask_seed,
    )
    test_dataset = MaskedBatteryDataset(
        Subset(dataset, test_indices),
        fixed_mask_ratio=PROTOCOL["evaluation_mask_ratio"],
        seed=args.test_mask_seed,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common = {
        "batch_size": global_micro_batch,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    loaders = (
        DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        DataLoader(val_dataset, shuffle=False, **common),
        DataLoader(test_dataset, shuffle=False, **common),
    )
    manifest = {
        "data_root": os.path.abspath(args.data_root),
        "datasets": args.datasets,
        "profile_mode": PROTOCOL["profile_mode"],
        "split_strategy": "dataset-stratified cell-level 70/15/15",
        "split_seed": args.split_seed,
        "train_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "test_samples": len(test_indices),
        "train_cells": len(split_cells["train"]),
        "validation_cells": len(split_cells["val"]),
        "test_cells": len(split_cells["test"]),
        "split_cells": split_cells,
    }
    return loaders, manifest


def build_model(configuration, device):
    return PGM2TN(
        input_dim=2,
        hidden_dim=configuration["hidden_dim"],
        num_layers=configuration["num_layers"],
        dropout=configuration["dropout"],
        enable_mae=configuration["enable_mae"],
    ).to(device)


def train(args):
    configuration = variant_config(args.variant)
    set_seed(args.seed, deterministic=not args.nondeterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replicas, per_gpu_batch, global_micro_batch, accumulation_steps = (
        resolve_batch_protocol(args, configuration)
    )
    effective_batch = global_micro_batch * accumulation_steps
    variant_dir = os.path.join(args.output_dir, args.variant)
    checkpoint_path = os.path.join(variant_dir, "best.pt")
    result_path = os.path.join(variant_dir, "results.json")
    os.makedirs(variant_dir, exist_ok=True)

    print("=" * 80)
    print(f"PG-M2TN August 2026 protocol: {args.variant}")
    print(f"Device={device}, replicas={replicas}, output={variant_dir}")
    print(
        f"Batch={replicas} x {per_gpu_batch} x {accumulation_steps} "
        f"accumulation = {effective_batch}"
    )
    print("Loading charge-only profiles...", flush=True)
    (train_loader, val_loader, test_loader), split_manifest = build_loaders(
        args, global_micro_batch
    )
    split_manifest.update(
        {
            "training_seed": args.seed,
            "validation_mask_seed": args.val_mask_seed,
            "test_mask_seed": args.test_mask_seed,
            "evaluation_mask_ratio": PROTOCOL["evaluation_mask_ratio"],
            "per_gpu_batch_size": per_gpu_batch,
            "replicas": replicas,
            "gradient_accumulation_steps": accumulation_steps,
            "effective_optimizer_batch": effective_batch,
        }
    )
    save_json(split_manifest, os.path.join(variant_dir, "split_manifest.json"))

    raw_model = build_model(configuration, device)
    if configuration["task_only_cudnn_anchor"]:
        for parameter in raw_model.mae_decoder.parameters():
            parameter.requires_grad = False
    model = nn.DataParallel(raw_model) if replicas > 1 else raw_model
    criterion = FixedWeightedLoss(
        soh_weight=configuration["soh_weight"],
        vdr_weight=configuration["vdr_weight"],
        lambda_mae=configuration["lambda_mae"],
        reconstruction_scope="masked",
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=configuration["learning_rate"],
        weight_decay=configuration["weight_decay"],
    )
    scheduler = make_scheduler(
        optimizer, args.epochs, PROTOCOL["warmup_epochs"]
    )

    best_val_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = defaultdict(float)
        batch_count = 0
        micro_batches = len(train_loader)

        for batch_index, batch in enumerate(train_loader):
            inputs = batch["x_masked"].to(device, non_blocking=True)
            reconstruction, soh_prediction, vdr_prediction = model(inputs)
            loss, parts = criterion(
                batch, reconstruction, soh_prediction, vdr_prediction
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss")

            group_offset = batch_index % accumulation_steps
            if group_offset == 0:
                group_size = min(accumulation_steps, micro_batches - batch_index)
            scaled_loss = loss / group_size

            if configuration["task_only_cudnn_anchor"]:
                anchor = (
                    configuration["backward_anchor_weight"]
                    * criterion.reconstruction_loss(batch, reconstruction, device)
                    / group_size
                )
                anchor_gradients = torch.autograd.grad(
                    anchor,
                    trainable_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                (scaled_loss + anchor).backward()
                for parameter, anchor_gradient in zip(
                    trainable_parameters, anchor_gradients
                ):
                    if anchor_gradient is not None:
                        parameter.grad.sub_(anchor_gradient)
            else:
                scaled_loss.backward()

            update_now = (
                group_offset + 1 == group_size
                or batch_index + 1 == micro_batches
            )
            if update_now:
                nn.utils.clip_grad_norm_(
                    trainable_parameters, PROTOCOL["gradient_clip"]
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            for name, value in parts.items():
                running[name] += value
            batch_count += 1

        scheduler.step()
        val_metrics = evaluate_soh(model, val_loader, device)
        average_training = {
            name: value / max(batch_count, 1) for name, value in running.items()
        }
        improved = val_metrics["rmse"] < best_val_rmse
        if improved:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "best_val_rmse": best_val_rmse,
                    "hp": {"variant": args.variant, **configuration},
                    "args": vars(args),
                    "protocol": PROTOCOL,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train": average_training,
                "validation": val_metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if args.verbose_epochs or epoch == 1 or epoch % 10 == 0 or improved:
            marker = " *BEST*" if improved else ""
            print(
                f"Epoch {epoch:3d}/{args.epochs}: "
                f"train={average_training['total']:.6f}, "
                f"val_rmse={val_metrics['rmse']:.6f}, "
                f"val_mae={val_metrics['mae']:.6f}{marker}",
                flush=True,
            )
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    evaluation_model = model
    validation_metrics, validation_per_dataset = evaluate_tasks(
        evaluation_model, val_loader, device, enable_mae=configuration["enable_mae"]
    )
    test_metrics, test_per_dataset = evaluate_tasks(
        evaluation_model, test_loader, device, enable_mae=configuration["enable_mae"]
    )
    if configuration["vdr_weight"] == 0.0:
        validation_metrics["diagnostic_unsupervised_vdr"] = validation_metrics.pop("vdr")
        test_metrics["diagnostic_unsupervised_vdr"] = test_metrics.pop("vdr")
        for values in validation_per_dataset.values():
            values["diagnostic_unsupervised_vdr"] = values.pop("vdr")
        for values in test_per_dataset.values():
            values["diagnostic_unsupervised_vdr"] = values.pop("vdr")

    result = {
        "variant": args.variant,
        "configuration": configuration,
        "protocol": split_manifest,
        "selection_metric": "validation_soh_rmse",
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "validation_metrics": validation_metrics,
        "validation_per_dataset": validation_per_dataset,
        "test_metrics": test_metrics,
        "test_per_dataset": test_per_dataset,
        "parameters": sum(parameter.numel() for parameter in raw_model.parameters()),
        "trainable_parameters": count_parameters(raw_model),
        "checkpoint": checkpoint_path,
        "elapsed_seconds": time.time() - start_time,
        "history": history,
    }
    save_json(result, result_path)
    print("=" * 80)
    print(
        f"Complete: best epoch={best_epoch}, val SOH RMSE={best_val_rmse:.6f}, "
        f"test SOH RMSE={test_metrics['soh']['rmse']:.6f}"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Results: {result_path}")
    return result


if __name__ == "__main__":
    train(parse_args())
