"""Machine-readable constants for the FA-M2TN reference protocol."""

DATASETS = ["CALCE", "HUST", "HNEI", "CALB", "ISU_ILCC"]

PROTOCOL = {
    "profile_mode": "charge_only",
    "sequence_length": 512,
    "minimum_cycles": 50,
    "training_seed": 42,
    "split_seed": 42,
    "validation_mask_seed": 20260820,
    "test_mask_seed": 20260821,
    "evaluation_mask_ratio": 0.50,
    "training_mask_range": [0.10, 0.90],
    "epochs": 150,
    "warmup_epochs": 10,
    "early_stopping_patience": 25,
    "gradient_clip": 1.0,
    "target_effective_batch_size": 2048,
}

COMMON_MODEL = {
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.20,
    "learning_rate": 5e-4,
    "weight_decay": 5e-4,
    "enable_mae": True,
}

VARIANTS = {
    "full": {
        **COMMON_MODEL,
        "soh_weight": 0.50,
        "vdr_weight": 0.50,
        "lambda_mae": 0.10,
        "per_gpu_batch_size": 256,
        "task_only_cudnn_anchor": False,
    },
    "no_mae": {
        **COMMON_MODEL,
        "soh_weight": 0.50,
        "vdr_weight": 0.50,
        "lambda_mae": 0.0,
        "per_gpu_batch_size": 64,
        "task_only_cudnn_anchor": True,
        "backward_anchor_weight": 0.10,
    },
    "no_vdr": {
        **COMMON_MODEL,
        "soh_weight": 0.50,
        "vdr_weight": 0.0,
        "lambda_mae": 0.10,
        "per_gpu_batch_size": 64,
        "task_only_cudnn_anchor": False,
    },
    "soh_only": {
        **COMMON_MODEL,
        "soh_weight": 0.50,
        "vdr_weight": 0.0,
        "lambda_mae": 0.0,
        "per_gpu_batch_size": 64,
        "task_only_cudnn_anchor": True,
        "backward_anchor_weight": 0.10,
    },
}


def variant_config(name):
    """Return a copy so callers cannot mutate the published protocol."""
    return dict(VARIANTS[name])


__all__ = ["DATASETS", "PROTOCOL", "VARIANTS", "variant_config"]
