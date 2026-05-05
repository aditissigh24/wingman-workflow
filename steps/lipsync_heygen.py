import logging
import os
import uuid

import requests
from fal_client import SyncClient

from config import FAL_KEY, FAL_LIPSYNC_MODEL, OUTPUT_DIR

log = logging.getLogger(__name__)

_fal = SyncClient(key=FAL_KEY)


def _download_video(video_url: str) -> str:
    """Download lipsynced video from URL, return local path."""
    filename = f"lipsynced_{uuid.uuid4().hex[:8]}.mp4"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with requests.get(video_url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    log.info("Saved lipsynced video: %s", filepath)
    return filepath


def lipsync_heygen(video_path: str, audio_path: str) -> str:
    """
    Upload video + audio to fal.ai CDN and run HeyGen precision lipsync.
    HeyGen replaces the video's existing audio track internally.
    Returns local path to final lipsynced .mp4.
    """
    log.info("Uploading video to fal.ai CDN: %s", video_path)
    video_url = _fal.upload_file(video_path)
    log.info("Uploading audio to fal.ai CDN: %s", audio_path)
    audio_url = _fal.upload_file(audio_path)

    log.info("Submitting HeyGen lipsync job (model=%s)…", FAL_LIPSYNC_MODEL)
    result = _fal.subscribe(FAL_LIPSYNC_MODEL, arguments={
        "video_url": video_url,
        "audio_url": audio_url,
    })
    log.debug("fal HeyGen lipsync raw response: %s", result)

    video = result.get("video", {})
    output_url = video.get("url") if isinstance(video, dict) else result.get("video_url", "")
    if not output_url:
        raise RuntimeError(f"fal.ai HeyGen lipsync returned no video URL. Response: {result}")

    return _download_video(output_url)
