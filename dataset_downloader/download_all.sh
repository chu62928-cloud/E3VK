#!/usr/bin/env bash
set -euo pipefail

OUTDIR=${1:-data}

python scripts/download_datasets.py --outdir "$OUTDIR" --all
