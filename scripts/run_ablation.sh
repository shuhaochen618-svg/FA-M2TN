#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/run_ablation.sh /path/to/battery_data [output_dir]"
    exit 2
fi

DATA_ROOT="$1"
OUTPUT_DIR="${2:-outputs/august_2026}"

for variant in full no_mae no_vdr soh_only; do
    python scripts/train.py \
        --data_root "$DATA_ROOT" \
        --variant "$variant" \
        --output_dir "$OUTPUT_DIR"
done
