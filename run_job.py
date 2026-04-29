import os
import sys
import json
import argparse


def load_job_config(jobs_path: str, job_id: int):
    """Load job config for the given job_id from jobs.jsonl."""
    with open(jobs_path, 'r', encoding='utf-8') as f:
        for line in f:
            job = json.loads(line)
            if job['job_id'] == job_id:
                return job
    raise ValueError(f"Job ID {job_id} not found in {jobs_path}")


def main():
    parser = argparse.ArgumentParser(description="Run a single job from jobs.jsonl")
    parser.add_argument("--jobs", type=str, required=True,
                        help="Path to jobs.jsonl")
    parser.add_argument("--job-id", type=int, required=True,
                        help="Job ID to execute")
    parser.add_argument("--base-dir", type=str, default=".",
                        help="Base directory (default: current directory)")
    parser.add_argument("--emo-prompts-dir", type=str, default="emo_prompts",
                        help="Emotion prompts directory (default: emo_prompts)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--model-dir", type=str, default="checkpoints_writable",
                        help="IndexTTS2 model directory")
    parser.add_argument("--whisper-model", type=str, default="small",
                        help="Whisper model size (default: small)")

    args = parser.parse_args()

    print(f"[INFO] Loading job {args.job_id} from {args.jobs}")
    job_config = load_job_config(args.jobs, args.job_id)
    print(f"[INFO] Job config: {json.dumps(job_config, ensure_ascii=False)}")

    task_type = job_config.get('task_type', 'tts')

    if task_type == 'tts':
        from tts_inference import run_tts_inference
        run_tts_inference(
            manifest_path=job_config['manifest'],
            job_id=args.job_id,
            model_dir=args.model_dir,
            base_dir=args.base_dir,
            emo_prompts_dir=args.emo_prompts_dir,
            output_dir=args.output_dir,
            whisper_model_name=args.whisper_model
        )
    else:
        print(f"[ERROR] Unknown task type: {task_type}")
        sys.exit(1)


if __name__ == "__main__":
    main()
