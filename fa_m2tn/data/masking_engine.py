"""
FA-M2TN Data Engine - Module 2: Masking Engine
================================================
Simulates extreme non-stationary conditions (NC) by applying continuous
random masking to battery cycle sequences.

Supports:
  - Random continuous masking (block mask)
  - Configurable fixed or random masking ratios [0.1, 0.9]
  - Returns both masked input and original for MAE reconstruction loss
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class MaskedBatteryDataset(Dataset):
    """
    Wraps a BatteryCycleDataset and applies continuous-block masking
    to simulate real-world fragmented charging data.
    """

    def __init__(self, base_dataset, fixed_mask_ratio=None, mask_value=0.0,
                 min_ratio=0.1, max_ratio=0.9, seed=None,
                 mask_pattern='random_block', num_bursts=5,
                 historical_random_start=False):
        """
        Args:
            base_dataset     : Instance of BatteryCycleDataset.
            fixed_mask_ratio : If set, uses this ratio for all samples (for evaluation).
                               If None, samples uniformly from [min_ratio, max_ratio].
            mask_value       : Value to fill masked positions (default 0.0).
            min_ratio        : Minimum masking ratio when sampling randomly.
            max_ratio        : Maximum masking ratio when sampling randomly.
            seed             : Optional integer seed for reproducible masks.
                               Use the same seed across models to ensure fair comparison.
                               None (default) = random start_idx per call (training mode).
            mask_pattern     : random_block, early, middle, late, or packet_bursts.
            num_bursts       : Number of separated blocks for packet_bursts.
            historical_random_start: reproduce the original exclusive upper
                                     bound for a random continuous block.
        """
        self.base_dataset = base_dataset
        self.fixed_mask_ratio = fixed_mask_ratio
        self.mask_value = mask_value
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.seed = seed
        self.mask_pattern = mask_pattern
        self.num_bursts = num_bursts
        self.historical_random_start = historical_random_start
        if fixed_mask_ratio is not None and not 0.0 <= fixed_mask_ratio < 1.0:
            raise ValueError("fixed_mask_ratio must be in [0, 1)")
        if not 0.0 <= min_ratio <= max_ratio < 1.0:
            raise ValueError("Require 0 <= min_ratio <= max_ratio < 1")
        if mask_pattern not in ('random_block', 'early', 'middle', 'late',
                                'packet_bursts'):
            raise ValueError(f"Unsupported mask_pattern: {mask_pattern}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        x_full = data['features'].clone()  # [SeqLen, 2] — (V, I) two-channel
        seq_len = x_full.shape[0]

        # --- Determine masking ratio ---
        rng = np.random.RandomState(self.seed + idx) if self.seed is not None else np.random
        if self.fixed_mask_ratio is not None:
            p = self.fixed_mask_ratio
        else:
            p = rng.uniform(self.min_ratio, self.max_ratio)

        # --- Create the requested missingness pattern ---
        mask_len = int(seq_len * p)
        x_masked = x_full.clone()
        binary_mask = torch.zeros(seq_len, dtype=torch.bool)

        if mask_len > 0 and mask_len < seq_len:
            if self.mask_pattern == 'early':
                binary_mask[:mask_len] = True
            elif self.mask_pattern == 'middle':
                start_idx = (seq_len - mask_len) // 2
                binary_mask[start_idx:start_idx + mask_len] = True
            elif self.mask_pattern == 'late':
                binary_mask[seq_len - mask_len:] = True
            elif self.mask_pattern == 'packet_bursts':
                burst_lengths = np.full(self.num_bursts, mask_len // self.num_bursts,
                                        dtype=int)
                burst_lengths[:mask_len % self.num_bursts] += 1
                window_edges = np.linspace(0, seq_len, self.num_bursts + 1,
                                           dtype=int)
                for burst_idx, burst_len in enumerate(burst_lengths):
                    low = int(window_edges[burst_idx])
                    high = int(window_edges[burst_idx + 1] - burst_len)
                    start_idx = rng.randint(low, high + 1)
                    binary_mask[start_idx:start_idx + burst_len] = True
            else:
                high = seq_len - mask_len
                if not self.historical_random_start:
                    high += 1
                start_idx = rng.randint(0, high)
                binary_mask[start_idx:start_idx + mask_len] = True

            # Packet bursts can overlap slightly; fill any remaining positions
            # deterministically so every pattern has the requested mask ratio.
            remaining = mask_len - int(binary_mask.sum())
            if remaining > 0:
                candidates = torch.where(~binary_mask)[0].numpy()
                chosen = rng.choice(candidates, size=remaining, replace=False)
                binary_mask[torch.from_numpy(np.asarray(chosen))] = True
            x_masked[binary_mask, :] = self.mask_value

        result = {
            'x_full': x_full,           # [SeqLen, 2] Original complete sequence
            'x_masked': x_masked,       # [SeqLen, 2] Masked (fragmented) sequence
            'binary_mask': binary_mask,  # [SeqLen]    True = masked position
            'mask_ratio': float(p),
            'soh': data['soh'],
            'vdr': data['vdr'],
            'cycle_idx': data['cycle_idx'],
            'cycle_fraction': data.get('cycle_fraction', 0.0),
            'cell_id': data['cell_id'],
            'dataset': data.get('dataset', ''),
        }
        if 'V_raw' in data:
            result['V_raw'] = data['V_raw']
            result['Q_raw'] = data['Q_raw']
        return result
