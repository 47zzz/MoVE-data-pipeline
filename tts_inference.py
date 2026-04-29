import os
import json
import whisper
from tqdm import tqdm

from utils import (
    trim_silence,
    get_audio_duration,
    run_whisper_asr,
    compute_wer,
    normalize_text_en,
    normalize_text_zh,
    check_filter_criteria
)


def run_tts_inference(
    manifest_path: str,
    job_id: int,
    model_dir: str,
    base_dir: str = ".",
    emo_prompts_dir: str = "emo_prompts",
    output_dir: str = "output",
    whisper_model_name: str = "small"
):
    """
    Run TTS inference for one shard.

    Args:
        manifest_path: path to the manifest shard (.jsonl)
        job_id: job index used for naming output metadata files
        model_dir: directory containing IndexTTS2 config and weights
        base_dir: working directory (default: current directory)
        emo_prompts_dir: directory containing emotion prompt audio files
        output_dir: root directory for generated audio and metadata
        whisper_model_name: Whisper model size (tiny / base / small / medium / large)
    """
    print("=" * 60)
    print(f"TTS Inference - Job {job_id}")
    print("=" * 60)

    # Load manifest
    print(f"[INFO] Loading manifest: {manifest_path}")
    entries = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line))
    print(f"[INFO] Loaded {len(entries)} entries")

    # Initialize IndexTTS2
    print("[INFO] Initializing IndexTTS2...")
    from indextts.infer_v2 import IndexTTS2
    cfg_path = os.path.join(model_dir, "config.yaml")
    tts = IndexTTS2(cfg_path=cfg_path, model_dir=model_dir, use_fp16=False)
    print("[INFO] IndexTTS2 initialized")

    # Initialize Whisper
    print(f"[INFO] Initializing Whisper ({whisper_model_name})...")
    whisper_model = whisper.load_model(whisper_model_name)
    print("[INFO] Whisper initialized")

    # Create output directories
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    # Metadata file paths
    entries_metadata_path = os.path.join(metadata_dir, f"entries_shard_{job_id:03d}.jsonl")
    finetune_metadata_path = os.path.join(metadata_dir, f"metadata_shard_{job_id:03d}.jsonl")

    # Load existing entry IDs for resume support
    existing_entry_ids = set()
    existing_finetune_pairs = set()

    if os.path.exists(entries_metadata_path):
        print(f"[INFO] Loading existing entries from {entries_metadata_path}...")
        with open(entries_metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    existing_entry_ids.add(entry['id'])
                except Exception:
                    pass
        print(f"[INFO] Found {len(existing_entry_ids)} existing entries")

    if os.path.exists(finetune_metadata_path):
        print(f"[INFO] Loading existing finetune pairs from {finetune_metadata_path}...")
        with open(finetune_metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # output_path format: en/happy/xxx.wav
                    if 'conversation' in data and len(data['conversation']) >= 2:
                        audio_path = data['conversation'][1].get('content', '')
                        if audio_path:
                            parts = audio_path.split('/')
                            if len(parts) >= 3:
                                emotion = parts[1]
                                entry_id = parts[2].replace('.wav', '')
                                # entry_id format: {emotion}_{gigaspeech_sid}_{lang}
                                id_parts = entry_id.rsplit('_', 1)
                                if len(id_parts) >= 1:
                                    base_id = id_parts[0]
                                    existing_finetune_pairs.add(base_id)
                except Exception:
                    pass
        print(f"[INFO] Found {len(existing_finetune_pairs)} existing finetune pairs")

    # Open metadata files in append mode
    entries_file = open(entries_metadata_path, 'a', encoding='utf-8')
    finetune_file = open(finetune_metadata_path, 'a', encoding='utf-8')

    # Buffer for pairing EN and ZH entries
    pair_buffer = {}

    total_generated = 0
    total_skipped = 0
    total_resumed = 0
    total_failed_filter = 0

    print("[INFO] Starting TTS generation...")

    for entry in tqdm(entries, desc=f"Job {job_id}"):
        entry_id = entry['id']
        text = entry['text']
        emotion = entry['emotion']
        lang = entry['lang']
        dataset = entry.get('dataset', '')
        gigaspeech_sid = entry.get('gigaspeech_sid', '')

        # Get prompt paths
        spk_audio_relative = entry['spk_audio_prompt']
        emo_audio_relative = entry['emo_audio_prompt']
        spk_audio_path = os.path.join(base_dir, emo_prompts_dir, spk_audio_relative)
        emo_audio_path = os.path.join(base_dir, emo_prompts_dir, emo_audio_relative)

        # Determine output path
        output_path = os.path.join(output_dir, lang, emotion, f"{entry_id}.wav")

        # Skip if already recorded in metadata (full resume)
        if entry_id in existing_entry_ids:
            total_resumed += 1
            continue

        # Skip if audio file already exists
        if os.path.exists(output_path):
            total_skipped += 1
            continue

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Run TTS
        try:
            if emotion in ['happy', 'sad', 'angry']:
                # Standard emotions: single prompt (speaker = emotion prompt)
                tts.infer(
                    spk_audio_prompt=spk_audio_path,
                    text=text,
                    output_path=output_path,
                    verbose=False
                )
            else:
                # Special emotions (laugh, crying): dual prompt
                tts.infer(
                    spk_audio_prompt=spk_audio_path,
                    text=text,
                    output_path=output_path,
                    emo_audio_prompt=emo_audio_path,
                    verbose=False
                )
        except Exception as e:
            print(f"[WARNING] TTS failed for {entry_id}: {e}")
            total_skipped += 1
            continue

        # Trim silence (hard gate)
        if not trim_silence(output_path, top_db=30, min_duration=0.5):
            print(f"[WARNING] Audio too short after trim, discarding: {entry_id}")
            if os.path.exists(output_path):
                os.remove(output_path)
            total_skipped += 1
            continue

        # Compute duration
        duration = get_audio_duration(output_path)

        # ASR transcription
        asr_text = run_whisper_asr(whisper_model, output_path, language=lang)

        # Normalize text
        if lang == "en":
            ref_normalized = normalize_text_en(text)
            hyp_normalized = normalize_text_en(asr_text)
        else:
            ref_normalized = normalize_text_zh(text)
            hyp_normalized = normalize_text_zh(asr_text)

        # Compute error rate
        error_rate = compute_wer(ref_normalized, hyp_normalized, lang)

        # Build entry metadata
        entry_metadata = {
            "id": entry_id,
            "emotion": emotion,
            "lang": lang,
            "text": text,
            "asr_text": asr_text,
            "ref_normalized": ref_normalized,
            "hyp_normalized": hyp_normalized,
            "error_rate": round(error_rate, 4),
            "gigaspeech_sid": gigaspeech_sid,
            "spk_audio_prompt": spk_audio_relative,
            "emo_audio_prompt": emo_audio_relative,
            "output_path": os.path.relpath(output_path, output_dir),
            "duration": round(duration, 2)
        }

        # Write entry regardless of filter result
        entries_file.write(json.dumps(entry_metadata, ensure_ascii=False) + '\n')
        entries_file.flush()

        # Add to pair buffer
        pair_key = (gigaspeech_sid, emotion)
        if pair_key not in pair_buffer:
            pair_buffer[pair_key] = {}
        pair_buffer[pair_key][lang] = entry_metadata

        # When both EN and ZH are ready, apply filter and emit finetune metadata
        if 'en' in pair_buffer[pair_key] and 'zh' in pair_buffer[pair_key]:
            en_data = pair_buffer[pair_key]['en']
            zh_data = pair_buffer[pair_key]['zh']

            pair_base_id = f"{emotion}_{gigaspeech_sid}"

            # Skip if pair already exists in finetune metadata
            if pair_base_id in existing_finetune_pairs:
                del pair_buffer[pair_key]
                continue

            # Apply quality filter
            if check_filter_criteria(en_data, zh_data, error_threshold=0.5):
                # EN -> ZH
                finetune_en_to_zh = {
                    "task_type": "s-s",
                    "conversation": [
                        {
                            "role": "user",
                            "message_type": "text",
                            "content": "Translate the given English speech into Chinese while preserving its expressiveness."
                        },
                        {
                            "role": "user",
                            "message_type": "audio",
                            "content": en_data['output_path']
                        },
                        {
                            "role": "assistant",
                            "message_type": "audio-text",
                            "content": [zh_data['output_path'], zh_data['text']]
                        }
                    ]
                }
                finetune_file.write(json.dumps(finetune_en_to_zh, ensure_ascii=False) + '\n')

                # ZH -> EN
                finetune_zh_to_en = {
                    "task_type": "s-s",
                    "conversation": [
                        {
                            "role": "user",
                            "message_type": "text",
                            "content": "Translate the given Chinese speech into English while preserving its expressiveness."
                        },
                        {
                            "role": "user",
                            "message_type": "audio",
                            "content": zh_data['output_path']
                        },
                        {
                            "role": "assistant",
                            "message_type": "audio-text",
                            "content": [en_data['output_path'], en_data['text']]
                        }
                    ]
                }
                finetune_file.write(json.dumps(finetune_zh_to_en, ensure_ascii=False) + '\n')
                finetune_file.flush()
            else:
                total_failed_filter += 1

            del pair_buffer[pair_key]

        total_generated += 1

    entries_file.close()
    finetune_file.close()

    print(f"\n[SUCCESS] Job {job_id} completed")
    print(f"  Generated: {total_generated} files")
    print(f"  Resumed (skipped from metadata): {total_resumed} files")
    print(f"  Skipped (audio exists): {total_skipped} files")
    print(f"  Failed filter: {total_failed_filter} pairs")
    print(f"  Entries metadata: {entries_metadata_path}")
    print(f"  Finetune metadata: {finetune_metadata_path}")
    print("=" * 60)
