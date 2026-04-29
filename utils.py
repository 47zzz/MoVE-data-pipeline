import re
import librosa
import soundfile as sf


def trim_silence(audio_path: str, top_db: float = 30, min_duration: float = 0.5) -> bool:
    """Trim leading/trailing silence in-place. Returns False if result is shorter than min_duration."""
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        if len(yt) / sr < min_duration:
            return False
        sf.write(audio_path, yt, sr)
        return True
    except Exception as e:
        print(f"[WARNING] trim_silence failed for {audio_path}: {e}")
        return False


def get_audio_duration(audio_path: str) -> float:
    """Return duration of an audio file in seconds."""
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        return len(y) / sr
    except Exception:
        return 0.0


def run_whisper_asr(model, audio_path: str, language: str = "en") -> str:
    """Transcribe audio with a Whisper model and return the transcript string."""
    try:
        result = model.transcribe(audio_path, language=language)
        return result.get("text", "").strip()
    except Exception as e:
        print(f"[WARNING] Whisper ASR failed for {audio_path}: {e}")
        return ""


def _expand_numbers_en(text: str) -> str:
    """Convert digit sequences to English words (best-effort; skipped if num2words not installed)."""
    try:
        import num2words
        def replace(m):
            return num2words.num2words(int(m.group()), lang="en")
        return re.sub(r"\b\d+\b", replace, text)
    except ImportError:
        return text


def normalize_text_en(text: str) -> str:
    """Lowercase, expand digits to words, and strip punctuation for English WER computation."""
    text = text.lower()
    text = _expand_numbers_en(text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text_zh(text: str) -> str:
    """Remove whitespace and non-CJK/alphanumeric characters for Chinese CER computation."""
    text = re.sub(r"\s+", "", text)
    # Keep CJK unified ideographs, CJK extension A, CJK compatibility ideographs, and ASCII alphanumerics
    text = re.sub(r"[^一-鿿㐀-䶿豈-﫿0-9a-zA-Z]", "", text)
    return text


def compute_wer(ref: str, hyp: str, lang: str = "en") -> float:
    """
    Compute WER (word-level) for English or CER (character-level) for Chinese.
    Returns Levenshtein distance divided by reference length; 0.0 for empty refs.
    """
    if lang == "zh":
        ref_tokens = list(ref)
        hyp_tokens = list(hyp)
    else:
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()

    n, m = len(ref_tokens), len(hyp_tokens)
    if n == 0:
        return 0.0 if m == 0 else 1.0

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr

    return prev[m] / n


def check_filter_criteria(en_data: dict, zh_data: dict, error_threshold: float = 0.5) -> bool:
    """Return True if both EN and ZH utterances are below the WER/CER threshold."""
    return (
        en_data.get("error_rate", 1.0) <= error_threshold
        and zh_data.get("error_rate", 1.0) <= error_threshold
    )
