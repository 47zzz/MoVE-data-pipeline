# MoVE Dataset Pipeline

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/pdf/2604.17435)
[![Demo](https://img.shields.io/badge/Demo-Live-4f46e5)](https://47zzz.github.io/MoVE/)
[![Model Code](https://img.shields.io/badge/Code-MoVE--code-181717)](https://github.com/47zzz/MoVE-code)
[![Project](https://img.shields.io/badge/Project-MoVE-181717)](https://github.com/47zzz/MoVE)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-ff9d00)](https://huggingface.co/datasets/47z/MoVE)

Data generation pipeline for **MoVE: Translating Laughter and Tears via Mixture of Vocalization Experts in Speech-to-Speech Translation** (Interspeech 2026, Under Review).

This repository contains the scripts for synthesizing the MoVE bilingual (Chinese ↔ English) expressive speech dataset. The pipeline uses **IndexTTS2** to generate ~1,000 hours of paired audio across 5 emotion categories, with Whisper-based ASR quality filtering.

---

## Related Repositories

| Repository | Description |
|---|---|
| [47zzz/MoVE](https://github.com/47zzz/MoVE) | Project page & demo (GitHub Pages) |
| [47zzz/MoVE-code](https://github.com/47zzz/MoVE-code) | Model training & inference (LoRA / xLoRA on Kimi-Audio) |
| **47zzz/MoVE-data-pipeline** (this repo) | Dataset generation pipeline |
| [datasets/47z/MoVE](https://huggingface.co/datasets/47z/MoVE) | Generated dataset on HuggingFace |

---

## Overview

The pipeline synthesizes paired EN-ZH speech with five emotion categories (happy, sad, angry, laugh, crying) from a text manifest, then filters by ASR quality (WER/CER ≤ 0.5) to produce fine-tuning data for S2ST models.

## Key Components

| File | Description |
|---|---|
| `split_shards.py` | Splits manifest into shards for parallel processing (pair-safe: EN-ZH pairs stay in the same shard) |
| `generate_jobs.py` | Generates `jobs.jsonl` mapping shard files to job IDs |
| `tts_inference.py` | Core TTS inference using IndexTTS2, silence trimming, Whisper ASR verification, and resume support |
| `run_job.py` | Wrapper that executes inference for a specific shard |
| `utils.py` | Audio processing, ASR, WER/CER, and quality-filter utilities |
| `01_prepare.sh` | Pulls container from GHCR, generates shards and jobs |
| `02_run_inference.sh` | SLURM batch script for parallel GPU inference |
| `singularity_test/` | Alternative scripts using Singularity instead of enroot |

## Container

Uses **enroot** (or **Singularity**) with the Docker image from GHCR:

- **Image**: `ghcr.io/47zzz/indextts:latest`
- **Enroot squash**: `indextts_latest.sqsh` (auto-generated on first run)
- **Singularity SIF**: `indextts_latest.sif` (auto-generated on first run)

## Prerequisites

Before running the pipeline you need to prepare:

1. **`manifest_local.jsonl`** — input text manifest, one entry per line:
   ```jsonl
   {"id": "happy_sid123_en", "text": "...", "emotion": "happy", "lang": "en", "gigaspeech_sid": "sid123", "spk_audio_prompt": "happy/ref.wav", "emo_audio_prompt": "happy/emo.wav", "dataset": "gigaspeech"}
   ```
2. **`emo_prompts/`** — directory of emotion prompt audio files referenced by the manifest
3. **`checkpoints_writable/`** — IndexTTS2 model weights (`config.yaml` + checkpoints)

## Workflow

### 1. Prepare shards and jobs

```bash
./01_prepare.sh                   # default 200 shards
./01_prepare.sh --shards 400      # custom shard count
```

### 2. Run inference

```bash
./02_run_inference.sh 0           # test single job locally
sbatch 02_run_inference.sh        # SLURM array (200 jobs)
```

## Output Structure

```
output/
├── en/
│   ├── happy/
│   ├── sad/
│   ├── angry/
│   ├── laugh/
│   └── crying/
├── zh/
│   ├── happy/
│   ├── sad/
│   ├── angry/
│   ├── laugh/
│   └── crying/
└── metadata/
    ├── entries_shard_000.jsonl    # per-entry detail (all, including filtered)
    └── metadata_shard_000.jsonl  # S2ST fine-tune format (quality-filtered pairs only)
```

## Citation

```bibtex
@article{chen2026move,
  title         = {MoVE: Translating Laughter and Tears via Mixture
                   of Vocalization Experts in Speech-to-Speech Translation},
  author        = {Chen, Szu-Chi and Tsai, I-Ning and Lin, Yi-Cheng and
                   Huang, Sung-Feng and Lee, Hung-yi},
  journal       = {arXiv preprint arXiv:2604.17435},
  year          = {2026},
  eprint        = {2604.17435},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2604.17435}
}
```
