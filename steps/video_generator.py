import base64
import io
import logging
import os
import subprocess
import time
import uuid
from typing import Callable, Optional

import jwt
import requests
from PIL import Image

from config import (
    KLING_ACCESS_KEY,
    KLING_BASE_URL,
    KLING_MODEL,
    KLING_SECRET_KEY,
    OUTPUT_DIR,
    SEGMENT_DURATION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("video_generator")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _make_kling_token() -> str:
    """Generate a short-lived JWT for the Kling API (HS256, 30-min TTL)."""
    now = int(time.time())
    payload = {
        "iss": KLING_ACCESS_KEY,
        "iat": now,
        "nbf": now - 5,   # 5-second grace for clock skew — required by Kling
        "exp": now + 1800,
    }
    return jwt.encode(payload, KLING_SECRET_KEY, algorithm="HS256")


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_make_kling_token()}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _image_to_base64_jpeg(image_path: str) -> str:
    """Convert any image file to a base64-encoded JPEG string."""
    pil_image = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Kling image-to-video: submit + poll
# ---------------------------------------------------------------------------

def _submit_image2video(
    image_b64: str,
    prompt: str,
    duration: int,
    max_retries: int = 8,
) -> str:
    """Submit an image-to-video task to Kling. Returns the task_id.

    Retries with exponential backoff on 429 (rate limit) responses.
    """
    url = f"{KLING_BASE_URL}/v1/videos/image2video"
    payload = {
        "model_name": KLING_MODEL,
        "prompt": prompt,
        "negative_prompt": "blurry, distorted, low quality, jump cut, duplicate person, extra limbs",
        "image": image_b64,
        "duration": str(duration),
        "mode": "pro",
        "aspect_ratio": "16:9",
    }
    wait = 30  # initial backoff in seconds — Kling rate-limit windows are typically 30–60s
    logger.info("Submitting task to Kling (model=%s, duration=%ss)", KLING_MODEL, duration)
    for attempt in range(max_retries):
        logger.info("  Submit attempt %d/%d …", attempt + 1, max_retries)
        resp = requests.post(url, headers=_headers(), json=payload, timeout=60)
        if resp.status_code == 429:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            logger.warning(
                "  Rate-limited (429). Response body: %s | Waiting %ds before retry …",
                body, wait,
            )
            if attempt < max_retries - 1:
                time.sleep(wait)
                wait = min(wait * 2, 120)  # cap at 2 minutes
                continue
            resp.raise_for_status()  # raise after all retries exhausted
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Kling submit error: {data.get('message', data)}")
        task_id = data["data"]["task_id"]
        logger.info("  Task submitted successfully → task_id=%s", task_id)
        return task_id


def _poll_task(
    task_id: str,
    timeout: int = 300,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> str:
    """Poll until task succeeds. Returns the video URL.

    on_progress(elapsed_secs, status_str) is called after every poll tick
    so callers (e.g. Streamlit) can update their UI with live progress.
    """
    url = f"{KLING_BASE_URL}/v1/videos/image2video/{task_id}"
    poll_interval = 10
    t_start = time.time()
    elapsed = 0

    logger.info("Polling task %s (timeout=%ds, interval=%ds) …", task_id, timeout, poll_interval)

    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed = int(time.time() - t_start)

        resp = requests.get(url, headers=_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"Kling poll error: {data.get('message', data)}")

        task = data["data"]
        status = task.get("task_status", "")
        logger.info("  [%3ds] task_id=%s  status=%s", elapsed, task_id, status)

        if on_progress:
            on_progress(elapsed, status)

        if status == "succeed":
            videos = task.get("task_result", {}).get("videos", [])
            if not videos:
                raise RuntimeError("Kling task succeeded but returned no videos.")
            logger.info("  Task completed successfully in %ds.", elapsed)
            return videos[0]["url"]

        if status == "failed":
            reason = task.get("task_status_msg", "unknown reason")
            raise RuntimeError(f"Kling video generation failed: {reason}")

    raise TimeoutError(f"Kling video generation timed out after {timeout}s (task {task_id})")


def _download_video(video_url: str, filename: str) -> str:
    """Download a video from a URL and save it to OUTPUT_DIR."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with requests.get(video_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return filepath


# ---------------------------------------------------------------------------
# Last-frame extraction
# ---------------------------------------------------------------------------

def extract_last_frame(video_path: str) -> str:
    """
    Extract the very last frame of a video as a JPEG.
    Returns the path to the saved frame image.
    Uses imageio-ffmpeg's bundled binary — no system ffmpeg needed.
    """
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    frame_filename = f"frame_{uuid.uuid4().hex[:8]}.jpg"
    frame_path = os.path.join(OUTPUT_DIR, frame_filename)

    # Two-pass: first get duration, then seek to last frame.
    # ffmpeg can seek to the last frame using sseof=-0.1 (0.1 s before end).
    cmd = [
        ffmpeg_exe,
        "-y",
        "-sseof", "-0.1",          # seek 0.1 s before end-of-file
        "-i", video_path,
        "-vframes", "1",           # grab exactly one frame
        "-q:v", "2",               # high JPEG quality
        "-vf", "scale=iw:ih",      # keep original resolution
        frame_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to extract last frame (exit {result.returncode}):\n{result.stderr}"
        )
    return frame_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_video_segment(
    image_path: str,
    scene_description: str,
    shot_description: str,
    segment_number: int,
    continuation_note: str = "",
    duration: int = SEGMENT_DURATION,
    timeout: int = 300,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> str:
    """
    Generate a single video segment using Kling image-to-video.

    image_path        — reference image: portrait for segment 1, last frame for subsequent segments
    scene_description — environment / background instructions (from script generator)
    shot_description  — character motion / action instructions (from script generator)
    segment_number    — 1-based index, used for logging and filenames
    continuation_note — context hint appended to the prompt for narrative continuity
    duration          — clip length in seconds (5 or 10)
    timeout           — polling timeout in seconds
    on_progress       — optional callback(elapsed_secs: int, status: str) for live UI updates

    Returns the local file path to the saved .mp4 segment.
    """
    t_start = time.time()
    logger.info("━━━ Segment %d: starting generation (duration=%ds) ━━━", segment_number, duration)

    # Build the Kling prompt: scene + shot + continuity hint
    prompt_parts = [scene_description.strip(), shot_description.strip()]
    if continuation_note:
        prompt_parts.append(continuation_note.strip())
    prompt = " ".join(p for p in prompt_parts if p)
    logger.info("  Prompt: %s", prompt[:120] + ("…" if len(prompt) > 120 else ""))

    logger.info("  Encoding reference image …")
    image_b64 = _image_to_base64_jpeg(image_path)

    task_id = _submit_image2video(image_b64, prompt, duration)

    t_submit = int(time.time() - t_start)
    logger.info("  Submit phase done in %ds. Starting poll …", t_submit)

    video_url = _poll_task(task_id, timeout=timeout, on_progress=on_progress)

    t_poll = int(time.time() - t_start)
    logger.info("  Poll phase done in %ds (total so far). Downloading video …", t_poll)

    filename = f"segment_{segment_number:02d}_{uuid.uuid4().hex[:6]}.mp4"
    local_path = _download_video(video_url, filename)

    total = int(time.time() - t_start)
    logger.info("━━━ Segment %d: complete in %ds ━━━", segment_number, total)
    return local_path
