import logging
import os
import time
import uuid

import requests
from requests.exceptions import ChunkedEncodingError, ReadTimeout
from fal_client import SyncClient

from config import FAL_KEY, FAL_ELEVENLABS_MODEL, ELEVENLABS_VOICE_ID, OUTPUT_DIR

log = logging.getLogger(__name__)

_fal = SyncClient(key=FAL_KEY)


def _download_audio(url: str, ext: str = "mp3", max_attempts: int = 5) -> str:
    """Download audio from URL to OUTPUT_DIR, return local path."""
    filename = f"audio_{uuid.uuid4().hex[:8]}.{ext}"
    filepath = os.path.join(OUTPUT_DIR, filename)
    last_exc = None
    for attempt in range(max_attempts):
        delay = 2 ** attempt
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 502:
                log.warning("audio download attempt %d/%d: 502, retrying in %ds", attempt + 1, max_attempts, delay)
                time.sleep(delay)
                continue
            r.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(r.content)
            log.info("Saved audio: %s (%d bytes)", filepath, len(r.content))
            return filepath
        except ReadTimeout as exc:
            last_exc = exc
            log.warning("audio download attempt %d/%d: ReadTimeout, retrying in %ds", attempt + 1, max_attempts, delay)
            time.sleep(delay)
        except ChunkedEncodingError as exc:
            last_exc = exc
            log.warning("audio download attempt %d/%d: ChunkedEncodingError, retrying in %ds", attempt + 1, max_attempts, delay)
            time.sleep(delay)
    raise RuntimeError(f"Audio download failed after {max_attempts} attempts: {last_exc}")


def generate_audio_fal(text: str, voice_id: str = ELEVENLABS_VOICE_ID) -> str:
    """
    Generate TTS audio via fal-ai/elevenlabs/tts/eleven-v3.
    Returns local audio file path.
    """
    log.info("Generating ElevenLabs audio via fal.ai (model=%s, voice=%s)", FAL_ELEVENLABS_MODEL, voice_id)
    result = _fal.subscribe(FAL_ELEVENLABS_MODEL, arguments={
        "text": text,
        "voice": voice_id,
    })
    log.debug("fal ElevenLabs raw response: %s", result)

    audio = result.get("audio", {})
    audio_url = audio.get("url") if isinstance(audio, dict) else result.get("audio_url", "")
    if not audio_url:
        raise RuntimeError(f"fal.ai ElevenLabs returned no audio URL. Response: {result}")

    content_type = audio.get("content_type", "audio/mpeg") if isinstance(audio, dict) else "audio/mpeg"
    ext = "mp3" if "mp3" in content_type or "mpeg" in content_type else "wav"
    return _download_audio(audio_url, ext=ext)
