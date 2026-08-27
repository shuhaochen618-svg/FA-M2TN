"""Charge-only battery dataset used by the August 2026 PG-M2TN runs."""

import glob
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def extract_charge_profile(cycle, target_len):
    """Extract charging samples and interpolate them on normalized capacity."""
    voltage = np.asarray(cycle["voltage_in_V"], dtype=np.float64).reshape(-1)
    current = np.asarray(cycle["current_in_A"], dtype=np.float64).reshape(-1)
    capacity = np.asarray(
        cycle["charge_capacity_in_Ah"], dtype=np.float64
    ).reshape(-1)

    length = min(len(voltage), len(current), len(capacity))
    voltage, current, capacity = (
        voltage[:length],
        current[:length],
        capacity[:length],
    )
    finite = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(capacity)
    voltage, current, capacity = voltage[finite], current[finite], capacity[finite]
    if len(capacity) < 10:
        return None

    capacity_span = float(capacity.max() - capacity.min())
    tolerance = max(capacity_span * 1e-7, 1e-10)
    increasing = np.diff(capacity) > tolerance
    active = np.flatnonzero(increasing)
    if len(active) < 8:
        return None

    start, stop = int(active[0]), int(active[-1] + 2)
    voltage = voltage[start:stop]
    current = current[start:stop]
    capacity = np.maximum.accumulate(capacity[start:stop])
    capacity, unique_indices = np.unique(capacity, return_index=True)
    voltage = voltage[unique_indices]
    current = current[unique_indices]
    if len(capacity) < 8 or capacity[-1] - capacity[0] <= tolerance:
        return None

    normalized_capacity = (capacity - capacity[0]) / (capacity[-1] - capacity[0])
    grid = np.linspace(0.0, 1.0, target_len)
    return (
        np.interp(grid, normalized_capacity, voltage).astype(np.float32),
        np.interp(grid, normalized_capacity, current).astype(np.float32),
    )


class BatteryCycleDataset(Dataset):
    """Build cycle samples from unified cell-level pickle files."""

    def __init__(self, data_root, datasets=None, seq_len=512, min_cycles=50):
        super().__init__()
        self.data_root = os.path.abspath(os.path.expanduser(data_root))
        self.datasets = list(datasets or [
            "CALCE", "HUST", "HNEI", "CALB", "ISU_ILCC"
        ])
        self.seq_len = seq_len
        self.profile_mode = "charge_only"
        self.samples = []
        self.cell_meta = {}

        for dataset_name in self.datasets:
            dataset_path = os.path.join(self.data_root, dataset_name)
            files = sorted(glob.glob(os.path.join(dataset_path, "*.pkl")))
            if not files:
                raise FileNotFoundError(
                    f"No .pkl files found for {dataset_name}: {dataset_path}"
                )
            for path in tqdm(files, desc=f"Loading {dataset_name}", leave=False):
                self._load_cell(path, dataset_name, min_cycles)

        if not self.samples:
            raise RuntimeError(f"No usable charge cycles found in {self.data_root}")
        print(
            f"[BatteryCycleDataset] Loaded {len(self.samples)} samples "
            f"from {len(self.cell_meta)} cells (charge_only)."
        )

    def _load_cell(self, path, dataset_name, min_cycles):
        with open(path, "rb") as handle:
            cell = pickle.load(handle)

        cycles = cell["cycle_data"]
        if len(cycles) < min_cycles:
            return

        raw_cell_id = str(cell["cell_id"])
        normalized_id = raw_cell_id.upper().replace("-", "_")
        normalized_dataset = dataset_name.upper().replace("-", "_")
        cell_id = (
            raw_cell_id
            if normalized_id.startswith(normalized_dataset)
            else f"{dataset_name}_{raw_cell_id}"
        )
        if cell_id in self.cell_meta:
            raise ValueError(f"Duplicate cell ID: {cell_id}")

        nominal_capacity = float(cell.get("nominal_capacity_in_Ah", 1.0))
        self.cell_meta[cell_id] = {
            "dataset": dataset_name,
            "raw_cell_id": raw_cell_id,
            "nominal_capacity": nominal_capacity,
            "cathode": cell.get("cathode_material", "Unknown"),
            "num_cycles": len(cycles),
            "source_file": os.path.abspath(path),
        }

        discharge_capacities = []
        prepared_profiles = []
        cycle_vdr = []
        for cycle in cycles:
            discharge = np.asarray(
                cycle["discharge_capacity_in_Ah"], dtype=np.float64
            )
            discharge_capacities.append(
                np.nanmax(discharge) if len(discharge) else 0.0
            )
            profile = extract_charge_profile(cycle, self.seq_len)
            prepared_profiles.append(profile)
            valid_voltage = profile[0] if profile is not None else np.array([])
            cycle_vdr.append(
                float(
                    np.std(valid_voltage)
                    / (np.mean(np.abs(valid_voltage)) + 1e-6)
                )
                if len(valid_voltage) > 10
                else 0.0
            )

        discharge_capacities = np.asarray(discharge_capacities)
        reference_capacity = np.nanmax(discharge_capacities[:5])
        if reference_capacity <= 0 or not np.isfinite(reference_capacity):
            reference_capacity = nominal_capacity

        reference_vdr = float(np.mean(cycle_vdr[:5]))
        if reference_vdr < 0.01:
            reference_vdr = 0.1

        for cycle_index, profile in enumerate(prepared_profiles):
            if profile is None:
                continue
            voltage, current = profile
            voltage_normalized = (
                np.clip((voltage - 1.5) / (4.5 - 1.5), 0.0, 1.0) * 2.0 - 1.0
            )
            c_rate = current / max(nominal_capacity, 1e-6)
            current_normalized = np.clip(c_rate / 6.0, -1.0, 1.0)
            features = np.stack(
                [voltage_normalized, current_normalized], axis=-1
            )

            soh = float(
                np.clip(
                    discharge_capacities[cycle_index] / reference_capacity,
                    0.0,
                    1.0,
                )
            )
            vdr = float(np.clip(cycle_vdr[cycle_index] / reference_vdr, 0.0, 3.0))
            if not np.isfinite(soh):
                soh = 1.0
            if not np.isfinite(vdr):
                vdr = 0.0

            self.samples.append(
                {
                    "features": torch.from_numpy(features).float(),
                    "soh": torch.tensor(soh, dtype=torch.float32),
                    "vdr": torch.tensor(vdr, dtype=torch.float32),
                    "cycle_idx": cycle_index,
                    "cycle_fraction": float(
                        cycle_index / max(len(cycles) - 1, 1)
                    ),
                    "cell_id": cell_id,
                    "dataset": dataset_name,
                    "profile_mode": "charge_only",
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def split_by_cell(
    dataset,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
    return_cell_ids=False,
):
    """Split cells within each source dataset; cycles never cross partitions."""
    random_state = np.random.RandomState(seed)
    known_datasets = ("CALCE", "HUST", "HNEI", "CALB", "ISU_ILCC")

    def dataset_prefix(cell_id):
        normalized = cell_id.upper().replace("-", "_")
        return next(
            (name for name in known_datasets if normalized.startswith(name)),
            "OTHER",
        )

    groups = defaultdict(list)
    for cell_id in sorted(dataset.cell_meta):
        groups[dataset_prefix(cell_id)].append(cell_id)

    train_cells, val_cells, test_cells = set(), set(), set()
    for dataset_name, cells in groups.items():
        cells.sort()
        random_state.shuffle(cells)
        count = len(cells)
        if count >= 3:
            train_count = max(1, int(count * train_ratio))
            val_count = max(1, int(count * val_ratio))
            if train_count + val_count >= count:
                train_count, val_count = max(1, count - 2), 1
        elif count == 2:
            train_count, val_count = 1, 0
        else:
            train_count, val_count = 1, 0
        train_cells.update(cells[:train_count])
        val_cells.update(cells[train_count:train_count + val_count])
        test_cells.update(cells[train_count + val_count:])
        print(
            f"  [Split] {dataset_name}: {count} cells -> "
            f"test={len(cells[train_count + val_count:])}"
        )

    train_indices, val_indices, test_indices = [], [], []
    for index, sample in enumerate(dataset.samples):
        cell_id = sample["cell_id"]
        if cell_id in train_cells:
            train_indices.append(index)
        elif cell_id in val_cells:
            val_indices.append(index)
        else:
            test_indices.append(index)

    print(
        f"[Split] Train: {len(train_indices)} samples ({len(train_cells)} cells) | "
        f"Val: {len(val_indices)} samples ({len(val_cells)} cells) | "
        f"Test: {len(test_indices)} samples ({len(test_cells)} cells)"
    )
    if return_cell_ids:
        return train_indices, val_indices, test_indices, {
            "train": sorted(train_cells),
            "val": sorted(val_cells),
            "test": sorted(test_cells),
        }
    return train_indices, val_indices, test_indices


__all__ = ["BatteryCycleDataset", "extract_charge_profile", "split_by_cell"]
