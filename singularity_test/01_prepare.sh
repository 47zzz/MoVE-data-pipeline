#!/bin/bash
# ============================================================
# 01_prepare.sh (Singularity) - Generate shards and job list
# ============================================================
#
# Usage:
#     ./01_prepare.sh                   # default 200 shards
#     ./01_prepare.sh --shards 100      # custom shard count
#
# Notes:
#     - manifest_local.jsonl must be prepared beforehand
#     - Uses Singularity instead of enroot
#
# ============================================================

# Parameters
NUM_SHARDS=200
MANIFEST_INPUT="manifest_local.jsonl"
OUTPUT_DIR="output"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --shards)
            NUM_SHARDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Singularity settings
HOST_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DOCKER_IMAGE="docker://ghcr.io/47zzz/indextts:latest"
SIF_FILE="${HOST_DIR}/indextts_latest.sif"

# Use local temp dirs to avoid /tmp space issues
export SINGULARITY_TMPDIR="${HOST_DIR}/.singularity_tmp"
export SINGULARITY_CACHEDIR="${HOST_DIR}/.singularity_cache"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

echo "============================================================"
echo "01_prepare.sh (Singularity) - Shards & Jobs Preparation"
echo "============================================================"
echo "Manifest input: $MANIFEST_INPUT"
echo "Number of shards: $NUM_SHARDS"
echo "Output directory: $OUTPUT_DIR"
echo "Host directory: $HOST_DIR"
echo "SINGULARITY_TMPDIR: $SINGULARITY_TMPDIR"
echo "============================================================"

# Phase 1: Check files & container
echo ""
echo "=== [Phase 1] Checking Files & Container ==="

if [ ! -f "$MANIFEST_INPUT" ]; then
    echo "ERROR: Manifest $MANIFEST_INPUT not found!"
    exit 1
else
    echo "Manifest $MANIFEST_INPUT found."
fi

# Pull SIF from GHCR if not present
if [ ! -f "$SIF_FILE" ]; then
    echo "[INFO] SIF file not found, pulling from GHCR..."
    singularity pull "$SIF_FILE" "$DOCKER_IMAGE"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to pull container from $DOCKER_IMAGE"
        exit 1
    fi
    echo "[SUCCESS] Container pulled to $SIF_FILE"
else
    echo "SIF file $SIF_FILE found."
fi

# Phase 2: Generate shards & jobs
echo ""
echo "=== [Phase 2] Generating Shards & Jobs ==="

singularity exec --nv \
    --bind "${HOST_DIR}:/workspace" \
    "$SIF_FILE" \
    bash -c "
        export PYTHONPATH=/workspace
        cd /workspace

        # 1. Split manifest into shards
        echo '[Step 1] Splitting into $NUM_SHARDS shards...'
        rm -rf shards/*
        mkdir -p shards
        /app/.venv/bin/python split_shards.py \
            --manifest $MANIFEST_INPUT \
            --num-shards $NUM_SHARDS \
            --output-dir shards

        # 2. Generate job list
        echo '[Step 2] Generating jobs.jsonl...'
        /app/.venv/bin/python generate_jobs.py \
            --shards-dir shards \
            --output jobs.jsonl

        # 3. Create output directories
        echo '[Step 3] Creating output directories...'
        mkdir -p $OUTPUT_DIR/metadata
        mkdir -p logs
    "

echo ""
echo "============================================================"
echo "Preparation Complete!"
echo "============================================================"
echo ""
echo "Generated files:"
echo "  - shards/manifest_*.jsonl ($NUM_SHARDS files)"
echo "  - jobs.jsonl"
echo ""
echo "Next step:"
echo "  ./02_run_inference.sh 0        # test single job"
echo "  sbatch 02_run_inference.sh     # SLURM batch"
echo "============================================================"
