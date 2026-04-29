#!/bin/bash
# ============================================================
# 02_run_inference.sh (Singularity) - TTS Inference
# ============================================================
#
# Usage:
#     sbatch 02_run_inference.sh       # SLURM batch (200 jobs)
#     ./02_run_inference.sh 0          # run single job manually
#
# Notes:
#     - Run 01_prepare.sh first
#
# ============================================================

#SBATCH --job-name=tts-infer
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-199

# Singularity settings
HOST_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SIF_FILE="${HOST_DIR}/indextts_latest.sif"
OUTPUT_DIR="output"

# Determine job ID: from SLURM or command-line argument
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    JOB_ID=$SLURM_ARRAY_TASK_ID
elif [ -n "$1" ]; then
    JOB_ID="$1"
else
    echo "Usage:"
    echo "  ./02_run_inference.sh <job_id>    # run manually"
    echo "  sbatch 02_run_inference.sh        # SLURM batch"
    exit 1
fi

# Create log directory if not present
mkdir -p "${HOST_DIR}/logs"

echo "============================================================"
echo "TTS Inference (Singularity)"
echo "============================================================"
echo "Job ID: $JOB_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Host directory: $HOST_DIR"
if [ -n "$SLURM_JOB_ID" ]; then
    echo "SLURM_JOB_ID: $SLURM_JOB_ID"
fi
echo "Hostname: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

# Check SIF file
if [ ! -f "$SIF_FILE" ]; then
    echo "ERROR: SIF file $SIF_FILE not found!"
    echo "Please run 01_prepare.sh first to pull the container."
    exit 1
fi

singularity exec --nv \
    --bind "${HOST_DIR}:/workspace" \
    "$SIF_FILE" \
    bash -c "
        # Set environment variables
        export HF_HOME=/workspace/hf_cache
        export PYTHONPATH=/workspace
        export NUMBA_DISABLE_INTEL_SVML=1
        export NUMBA_OPT=0
        export TORCHINDUCTOR_DISABLE=1
        mkdir -p \$HF_HOME

        cd /workspace

        echo '[INFO] Running job $JOB_ID...'

        # Run Python inference
        /app/.venv/bin/python -u run_job.py \
            --jobs jobs.jsonl \
            --job-id $JOB_ID \
            --model-dir checkpoints_writable \
            --output-dir $OUTPUT_DIR \
            --emo-prompts-dir emo_prompts \
            --whisper-model small
    "

echo ""
echo "============================================================"
echo "Job $JOB_ID completed"
echo "============================================================"
