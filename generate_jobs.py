import os
import json
import argparse
from pathlib import Path

DEFAULT_SHARDS_DIR = "shards"
DEFAULT_OUTPUT = "jobs.jsonl"


def generate_jobs(shards_dir: str, output_path: str):
    """Generate jobs.jsonl from shard files."""
    print("=" * 60)
    print("Generate jobs.jsonl")
    print("=" * 60)

    shards_path = Path(shards_dir)
    if not shards_path.exists():
        print(f"[ERROR] Shards directory not found: {shards_dir}")
        return

    shard_files = sorted(shards_path.glob("manifest_*.jsonl"))

    if not shard_files:
        print(f"[ERROR] No manifest_*.jsonl files found in {shards_dir}")
        return

    print(f"[INFO] Found {len(shard_files)} shard files")

    jobs = []
    for shard_file in shard_files:
        # Parse job_id from filename: manifest_000.jsonl -> job_id = 0
        stem = shard_file.stem
        job_id = int(stem.split('_')[1])

        # Relative path
        manifest_relative = os.path.join(shards_dir, shard_file.name)

        jobs.append({
            "job_id": job_id,
            "task_type": "tts",
            "manifest": manifest_relative
        })

    jobs.sort(key=lambda x: x['job_id'])

    print(f"[INFO] Writing {len(jobs)} jobs to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + '\n')

    print(f"[SUCCESS] Generated {len(jobs)} jobs")
    print(f"[SUCCESS] Saved to {output_path}")

    print("\nFirst 3 jobs:")
    for job in jobs[:3]:
        print(f"  Job {job['job_id']}: {job['manifest']}")

    if len(jobs) > 6:
        print("  ...")
        print("\nLast 3 jobs:")
        for job in jobs[-3:]:
            print(f"  Job {job['job_id']}: {job['manifest']}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate jobs.jsonl from shards")
    parser.add_argument("--shards-dir", type=str, default=DEFAULT_SHARDS_DIR,
                        help=f"Shards directory (default: {DEFAULT_SHARDS_DIR})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output jobs.jsonl path (default: {DEFAULT_OUTPUT})")

    args = parser.parse_args()
    generate_jobs(args.shards_dir, args.output)


if __name__ == "__main__":
    main()
