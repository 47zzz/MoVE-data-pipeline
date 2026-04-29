import os
import json
import argparse
from pathlib import Path

DEFAULT_MANIFEST = "manifest_local.jsonl"
DEFAULT_OUTPUT_DIR = "shards"
DEFAULT_NUM_SHARDS = 200


def split_manifest(manifest_path: str, output_dir: str, num_shards: int):
    print("=" * 60)
    print("Split Manifest into Shards (Pair-Safe Mode)")
    print("=" * 60)

    # Read all entries
    print(f"[INFO] Reading {manifest_path}...")
    entries = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line))

    total_entries = len(entries)
    print(f"[INFO] Total entries: {total_entries}")

    if total_entries % 2 != 0:
        print("[WARNING] Total entries is odd — last entry may not have a pair.")

    # Group consecutive entries into EN-ZH pairs
    pairs = []
    for i in range(0, total_entries, 2):
        if i + 1 < total_entries:
            pair = [entries[i], entries[i + 1]]
            # Validate pair alignment
            id_1 = entries[i]['id'].rsplit('_', 1)[0]
            id_2 = entries[i + 1]['id'].rsplit('_', 1)[0]
            if id_1 != id_2:
                print(f"[WARNING] Pair mismatch at index {i}: {entries[i]['id']} vs {entries[i+1]['id']}")
            pairs.append(pair)
        else:
            pairs.append([entries[i]])

    total_pairs = len(pairs)
    print(f"[INFO] Total pairs: {total_pairs}")

    # Compute pairs per shard
    pairs_per_shard = total_pairs // num_shards
    remainder = total_pairs % num_shards

    print(f"[INFO] Splitting into {num_shards} shards")
    print(f"[INFO] Pairs per shard: ~{pairs_per_shard}")

    os.makedirs(output_dir, exist_ok=True)

    # Split and write shards
    start_idx = 0
    shard_info = []

    for shard_id in range(num_shards):
        current_shard_pairs_count = pairs_per_shard + (1 if shard_id < remainder else 0)
        end_idx = start_idx + current_shard_pairs_count

        shard_pairs = pairs[start_idx:end_idx]
        shard_entries = [entry for pair in shard_pairs for entry in pair]

        shard_filename = f"manifest_{shard_id:03d}.jsonl"
        shard_path = os.path.join(output_dir, shard_filename)

        with open(shard_path, 'w', encoding='utf-8') as f:
            for entry in shard_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        shard_info.append({
            'shard_id': shard_id,
            'filename': shard_filename,
            'entries': len(shard_entries),
            'pairs': len(shard_pairs)
        })

        start_idx = end_idx

    print(f"\n[SUCCESS] Created {num_shards} shards in {output_dir}/")
    print("\nShard distribution:")
    print(f"  Min pairs per shard: {min(s['pairs'] for s in shard_info)}")
    print(f"  Max pairs per shard: {max(s['pairs'] for s in shard_info)}")
    print(f"  Min entries per shard: {min(s['entries'] for s in shard_info)}")
    print(f"  Max entries per shard: {max(s['entries'] for s in shard_info)}")
    print("=" * 60)
    return shard_info


def main():
    parser = argparse.ArgumentParser(description="Split manifest into shards")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST,
                        help=f"Input manifest path (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS,
                        help=f"Number of shards (default: {DEFAULT_NUM_SHARDS})")

    args = parser.parse_args()
    split_manifest(args.manifest, args.output_dir, args.num_shards)


if __name__ == "__main__":
    main()
