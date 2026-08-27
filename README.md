<div align="center">

<h1>PG-M2TN</h1>

<h3>Physics-Guided Masked Multi-Task Network for Edge-Friendly Battery Health Diagnostics from Stochastically Fragmented Charging Profiles</h3>

<p><strong>Compact SOH estimation, VDR prediction, and masked-profile reconstruction from incomplete charging records</strong></p>

<p><code>Python 3.11</code> &nbsp;&middot;&nbsp; <code>PyTorch 2.5</code> &nbsp;&middot;&nbsp; <code>Charge-only profiles</code> &nbsp;&middot;&nbsp; <code>630,149 parameters</code></p>

<p><a href="#overview">Overview</a> &nbsp;&middot;&nbsp; <a href="#core-contributions">Contributions</a> &nbsp;&middot;&nbsp; <a href="#results">Results</a> &nbsp;&middot;&nbsp; <a href="#data">Data</a> &nbsp;&middot;&nbsp; <a href="#reproduction">Reproduction</a></p>

</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/shuhaochen618-svg/PG-M2TN/main/assets/figure1.png" width="100%" alt="PG-M2TN framework overview">
</p>

## Overview

PG-M2TN estimates battery state of health (SOH) from stochastically fragmented charging profiles. It combines SOH regression, voltage-dispersion-rate (VDR) prediction, and masked-profile reconstruction in a compact multi-task network designed for incomplete field observations and resource-constrained deployment.

Real charging records may be interrupted by charger disconnection, communication loss, or partial data collection. PG-M2TN learns from continuous missing regions instead of assuming that a complete charging curve is always available.

| Item | Setting |
|---|---|
| Input | Fragmented charging voltage and current |
| Sequence length | 512 points |
| Primary task | SOH estimation |
| Auxiliary tasks | VDR prediction and masked reconstruction |
| Datasets | CALCE, HUST, HNEI, CALB, ISU-ILCC |
| Main configuration | PG-M2TN-128 |
| Parameters | 630,149 |
| Data split | Dataset-stratified, cell-level 70/15/15 |

## Core Contributions

1. **Fragment-aware health diagnostics.** Continuous random block masking represents incomplete charging observations and trains the encoder to recover information from missing profile regions.
2. **Physics-related auxiliary supervision.** VDR is derived from the charging profile and introduced as a fixed-weight auxiliary target, adding charge-curve information without requiring electrochemical equation parameters at inference time.
3. **Compact multi-task learning.** One shared encoder supports SOH estimation, VDR prediction, and profile reconstruction with 630,149 parameters.
4. **Transparent training protocol.** Fixed task coefficients, cell-level data separation, deterministic validation and test masking, and validation-only model selection make the experiment directly reproducible.

The fixed training objective is:

```text
L = 0.50 * L_SOH + 0.50 * L_VDR + 0.10 * L_MAE
```

## Results

PG-M2TN was evaluated on a held-out cell-level test partition. The checkpoint was selected using validation SOH RMSE only; the test partition was not used for model selection.

| Configuration | Parameters | Best epoch | Validation RMSE | Test RMSE | Test MAE | Test R2 |
|---|---:|---:|---:|---:|---:|---:|
| **PG-M2TN-128** | **630,149** | **98** | **0.040604** | **0.089500** | **0.046316** | **0.943954** |

### Component Ablations

| Configuration | SOH weight | VDR weight | MAE weight | Best epoch | Test RMSE | Test MAE | Test R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full` | 0.50 | 0.50 | 0.10 | 98 | 0.089500 | 0.046316 | 0.943954 |
| `no_mae` | 0.50 | 0.50 | 0.00 | 83 | 0.110127 | 0.053817 | 0.915143 |
| `no_vdr` | 0.50 | 0.00 | 0.10 | 61 | 0.089429 | 0.049512 | 0.944043 |
| `soh_only` | 0.50 | 0.00 | 0.00 | 74 | 0.072608 | 0.042413 | 0.963113 |

The ablations isolate the role of each PG-M2TN objective. Their effects are dataset-dependent and reveal the trade-off between pooled SOH accuracy and joint SOH-VDR-reconstruction capability.

## Why It Matters

- **Incomplete observations:** battery health can be estimated when continuous charging regions are unavailable.
- **Efficient inference:** the compact fixed architecture provides a practical basis for edge-deployment studies.
- **Interpretable supervision:** VDR connects representation learning with a charge-profile descriptor while remaining separate from inference inputs.
- **Reproducible evaluation:** cycles from the same cell never cross training, validation, and test partitions.

## Repository Structure

```text
PG-M2TN/
|-- pg_m2tn/
|   |-- data/
|   |   |-- dataset_loader.py   # charge extraction and cell-level split
|   |   `-- masking_engine.py   # random and fixed-seed block masking
|   |-- models/
|   |   |-- pg_m2tn.py          # PG-M2TN-128 implementation
|   |   `-- loss.py             # fixed SOH/VDR/MAE objective
|   |-- evaluation.py           # pooled and per-dataset evaluation
|   |-- protocol.py             # experiment constants and ablations
|   `-- utils/metrics.py
|-- scripts/
|   |-- train.py                # main and ablation training
|   |-- evaluate.py             # held-out test evaluation
|   `-- run_ablation.sh         # four-configuration launcher
|-- tests/test_smoke.py
|-- requirements.txt
`-- setup.py
```

> This repository contains only PG-M2TN training, evaluation, and component-ablation code. Datasets, trained checkpoints, manuscript plotting utilities, and comparison-model implementations are not included.

## Installation

```bash
git clone https://github.com/shuhaochen618-svg/PG-M2TN.git
cd PG-M2TN

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m unittest discover -s tests -v
```

The reported environment used PyTorch 2.5.1 with CUDA 12.1, NumPy 2.4.6, and tqdm 4.68.4.

## Data

Place the five unified datasets under one root:

```text
battery_data/
|-- CALCE/*.pkl
|-- HUST/*.pkl
|-- HNEI/*.pkl
|-- CALB/*.pkl
`-- ISU_ILCC/*.pkl
```

Each pickle file represents one cell and follows this structure:

```python
{
    "cell_id": "CS2_35",
    "nominal_capacity_in_Ah": 1.1,
    "cycle_data": [
        {
            "voltage_in_V": numpy_array,
            "current_in_A": numpy_array,
            "charge_capacity_in_Ah": numpy_array,
            "discharge_capacity_in_Ah": numpy_array,
        },
    ],
}
```

The evaluated snapshot contains 753,646 charge-only cycle samples from 274 cells:

| Partition | Cells | Cycle samples |
|---|---:|---:|
| Train | 189 | 542,831 |
| Validation | 39 | 84,413 |
| Test | 46 | 126,402 |

The split uses seed `42`. Per-dataset train/validation/test cell counts are CALCE `9/1/3`, HUST `53/11/13`, HNEI `9/2/3`, CALB `18/4/5`, and ISU-ILCC `100/21/22`.

## Reproduction

### Train PG-M2TN-128

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python scripts/train.py \
    --data_root /path/to/battery_data \
    --variant full \
    --output_dir outputs/pg_m2tn
```

The script uses the reported split, masking seeds, optimizer settings, fixed loss coefficients, and validation-based checkpoint selection by default.

### Run the Ablations

```bash
bash scripts/run_ablation.sh /path/to/battery_data outputs/pg_m2tn
```

This runs `full`, `no_mae`, `no_vdr`, and `soh_only` sequentially.

### Evaluate the Selected Checkpoint

```bash
python scripts/evaluate.py \
    --checkpoint outputs/pg_m2tn/full/best.pt \
    --data_root /path/to/battery_data \
    --output outputs/pg_m2tn/full/evaluation.json
```

Results, per-dataset metrics, checkpoints, and split metadata are written to `outputs/pg_m2tn/<variant>/`.
