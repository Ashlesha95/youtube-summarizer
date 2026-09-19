import os
import time
import logging
import tempfile
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
import yt_dlp

# faster-whisper instead of openai-whisper: same models, ~4-8x faster on CPU,
# lower memory, no re-encode required before feeding it audio.
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("transcript_utils")

_whisper_model = None
_whisper_model_size = None


def _get_whisper_model(size="base", device="auto", compute_type="auto"):
    """
    Cached model loader. Cache is per-process, so this only helps if
    fetch_transcript() is called multiple times in the same running process
    (e.g. a server, a notebook, a loop) - not across separate script runs.

    device="auto" will use GPU if available, else CPU.
    compute_type="auto" picks the best precision for the device
    (int8 on CPU is much faster than float32 with minimal quality loss).
    """
    global _whisper_model, _whisper_model_size
    if _whisper_model is None or _whisper_model_size != size:
        t0 = time.time()
        log.info(f"Loading whisper model '{size}' (device={device}, compute_type={compute_type})...")
        _whisper_model = WhisperModel(size, device=device, compute_type=compute_type)
        _whisper_model_size = size
        log.info(f"Model loaded in {time.time() - t0:.1f}s")
    return _whisper_model


def extract_youtube_id(url):
    if urlparse(url).netloc == "www.youtube.com":
        return parse_qs(urlparse(url).query).get('v')[0]
    else:
        return urlparse(url).path.split('/')[-1]


def _get_transcript_snippets(video_id, preferred_languages=None, translate_to_english=True):
    """
    Fetch transcript snippets, robust to regional codes (en-US vs en).
    If no English transcript exists but a translatable one does, translate it to English.
    Raises TranscriptsDisabled / NoTranscriptFound / VideoUnavailable on failure.
    """
    if preferred_languages is None:
        preferred_languages = ['en', 'en-US', 'en-GB']

    ytt_api = YouTubeTranscriptApi()

    # Fast path: exact language match
    try:
        return ytt_api.fetch(video_id, languages=preferred_languages)
    except (TranscriptsDisabled, VideoUnavailable):
        raise
    except Exception:
        pass

    transcript_list = ytt_api.list(video_id)

    # Try to find an English (or near-English) transcript directly
    try:
        transcript = transcript_list.find_transcript(preferred_languages)
        return transcript.fetch()
    except Exception:
        pass

    # No English transcript found - grab whatever exists and translate it
    available = list(transcript_list)
    if not available:
        raise NoTranscriptFound(video_id, preferred_languages, transcript_list)

    # Prefer a manually created transcript over an auto-generated one
    manual = [t for t in available if not t.is_generated]
    transcript = manual[0] if manual else available[0]

    if translate_to_english and transcript.is_translatable:
        transcript = transcript.translate('en')

    return transcript.fetch()


def _download_audio(url, output_dir):
    """
    Downloads the best audio-only track, in its native format (webm/m4a/etc).
    Deliberately does NOT re-encode to mp3 - faster-whisper (via ffmpeg under
    the hood at decode time) can consume the native format directly, so the
    extra transcode step is just wasted time and a wasted ffmpeg pass.
    """
    t0 = time.time()
    output_template = os.path.join(output_dir, "audio.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'js_runtimes': {'deno': {}},  # dict format required by newer yt-dlp
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = ydl.prepare_filename(info)

    log.info(f"Audio downloaded in {time.time() - t0:.1f}s -> {audio_path}")
    return audio_path


def _transcribe_with_whisper(
        url,
        model_size="base",
        translate_to_english=True,
        device="auto",
        compute_type="auto",
        detect_language_first=True,
):
    """
    Fallback: download audio and transcribe with faster-whisper.

    If translate_to_english=True, Whisper translates non-English speech
    directly to English text (task="translate"). Translation is noticeably
    slower than plain transcription, so if detect_language_first=True we
    skip it entirely when the source audio is already English.

    Returns list of {"text", "start", "duration", "end"} dicts.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = _download_audio(url, tmp_dir)
        model = _get_whisper_model(model_size, device=device, compute_type=compute_type)

        task = "transcribe"
        if translate_to_english:
            if detect_language_first:
                # Cheap language-id pass on a short audio window, not a full decode
                t0 = time.time()
                _, info = model.transcribe(audio_path, task="transcribe", vad_filter=False)
                # NOTE: model.transcribe returns a generator for segments; touching
                # info doesn't consume it, but we don't iterate segments here.
                detected_lang = info.language
                log.info(f"Detected language '{detected_lang}' (confidence {info.language_probability:.2f}) in {time.time() - t0:.1f}s")
                task = "transcribe" if detected_lang == "en" else "translate"
            else:
                task = "translate"

        t0 = time.time()
        log.info(f"Running whisper task='{task}'...")
        segments, info = model.transcribe(audio_path, task=task, vad_filter=True)

        snippets = []
        for seg in segments:
            snippets.append({
                "text": seg.text.strip(),
                "start": seg.start,
                "duration": seg.end - seg.start,
                "end": seg.end,
            })
        log.info(f"Transcription finished in {time.time() - t0:.1f}s ({len(snippets)} segments)")
        return snippets


def fetch_transcript(
        url,
        languages=None,
        whisper_model_size="base",
        translate_to_english=True,
        device="auto",
        compute_type="auto",
):
    video_id = extract_youtube_id(url)

    try:
        t0 = time.time()
        fetched_transcript = _get_transcript_snippets(video_id, languages, translate_to_english)
        log.info(f"Fetched existing captions via API in {time.time() - t0:.1f}s")
        text = [snippet.text for snippet in fetched_transcript]
        return " ".join(text)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        log.info(f"No usable captions via API ({type(e).__name__}), falling back to Whisper")
        snippets = _transcribe_with_whisper(
            url, whisper_model_size, translate_to_english,
            device=device, compute_type=compute_type,
        )
        return " ".join(s["text"] for s in snippets)


def fetch_timestamped_transcript(
        url,
        languages=None,
        whisper_model_size="base",
        translate_to_english=True,
        device="auto",
        compute_type="auto",
):
    video_id = extract_youtube_id(url)

    try:
        t0 = time.time()
        fetched_transcript = _get_transcript_snippets(video_id, languages, translate_to_english)
        log.info(f"Fetched existing captions via API in {time.time() - t0:.1f}s")
        transcript = []
        for snippet in fetched_transcript:
            transcript.append({
                "text": snippet.text,
                "start": snippet.start,
                "duration": snippet.duration,
                "end": snippet.start + snippet.duration
            })
        return transcript
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        log.info(f"No usable captions via API ({type(e).__name__}), falling back to Whisper")
        return _transcribe_with_whisper(
            url, whisper_model_size, translate_to_english,
            device=device, compute_type=compute_type,
        )

