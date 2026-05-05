"""
Seedance video segment generator via fal.ai (bytedance/seedance-2.0/fast/image-to-video).

LLM agents (prompt critique + rewrite) use OpenRouter via utils.llm_client.
Video generation uses fal_client — no Google AI SDK required.

Pattern:
  - Segment 1: image-to-video from the character portrait
  - Subsequent segments: last frame of previous clip → next segment (chained)
"""

import logging
import os
import subprocess
import time
import uuid
from typing import Callable, Optional

import requests
from fal_client import SyncClient

from config import FAL_KEY, FAL_VIDEO_MODEL, OUTPUT_DIR, SEGMENT_DURATION
from utils import llm_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seedance_generator")

_fal = SyncClient(key=FAL_KEY)


class AudioFilteredError(RuntimeError):
    """Raised when the model rejects a prompt due to audio/dialogue content filtering."""


# ---------------------------------------------------------------------------
# Prompt critique + rewrite agents (pre-flight content safety check)
# ---------------------------------------------------------------------------

_CHECKLIST = """
You are a video generation content compliance checker. Your ONLY job is to identify whether a video generation prompt will be rejected by safety filters.

Evaluate the prompt against EXACTLY these rules:
1. No real celebrity, public figure, actor, politician, athlete, or influencer names or likenesses
2. No brand names, trademarks, or copyrighted IP references
3. Dialogue must be emotionally safe — no explicit language, no sexual content, no graphic violence, no self-harm references
4. No political statements or hate speech
5. Hinglish (Hindi + English mix) is fully allowed and must NOT be flagged
6. Emotional intensity, romance, heartbreak, flirting are all allowed

Return ONLY a JSON object:
{"passes": true, "issues": []}
OR
{"passes": false, "issues": ["Exact issue 1 with the specific offending text", "Exact issue 2..."]}

Do NOT rewrite anything. Do NOT suggest fixes. Only identify problems.
""".strip()

_REWRITER = """
You are a video prompt rewriter. You receive a prompt and a list of specific issues identified by a compliance checker.

Your ONLY job is to fix exactly the listed issues while preserving:
- The overall scene, shot framing, and visual intent
- All spoken dialogue (meaning, emotion, Hinglish language)
- The character's voice, tone, and cultural specificity
- Creative energy — do NOT sanitize beyond what the issues require

Rules:
- Replace real person names with vivid fictional descriptors (e.g. "a legendary cricketer" not "Virat Kohli")
- Replace brand names with generic equivalents (e.g. "a luxury SUV" not "Audi Q7")
- If dialogue has an explicit word, rephrase just that word/phrase while keeping the emotional beat
- Do NOT add disclaimers, do NOT over-sanitize, do NOT change what isn't broken

Return ONLY the rewritten prompt text. No preamble, no explanation.
""".strip()


def _critique_prompt(prompt: str) -> dict:
    """
    Agent 1 — Critique only (Claude via OpenRouter).
    Returns {"passes": bool, "issues": [str, ...]}.
    """
    import json
    try:
        text = llm_client.complete(system=_CHECKLIST, user=prompt, model="claude", max_tokens=512)
        result = json.loads(text)
        return {"passes": bool(result.get("passes", True)), "issues": result.get("issues", [])}
    except Exception:
        logger.warning("Critique agent returned unparseable response — treating as passed")
        return {"passes": True, "issues": []}


def _rewrite_prompt(prompt: str, issues: list) -> str:
    """
    Agent 2 — Rewrite only (Gemini via OpenRouter).
    Fixes exactly the listed issues while preserving creative intent.
    """
    issues_block = "\n".join(f"- {issue}" for issue in issues)
    user_content = f"ORIGINAL PROMPT:\n{prompt}\n\nISSUES TO FIX:\n{issues_block}"
    return llm_client.complete(system=_REWRITER, user=user_content, model="gemini", max_tokens=1024)


# Keep old names as aliases so any existing imports of the veo3 names still work
_critique_veo3_prompt = _critique_prompt
_rewrite_veo3_prompt = _rewrite_prompt


# ---------------------------------------------------------------------------
# Last-frame extraction (ffmpeg — backend agnostic)
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

    cmd = [
        ffmpeg_exe,
        "-y",
        "-sseof", "-1.5",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        "-vf", "scale=iw:ih",
        frame_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to extract last frame (exit {result.returncode}):\n{result.stderr}"
        )
    return frame_path


# ---------------------------------------------------------------------------
# fal.ai video generation helpers
# ---------------------------------------------------------------------------

def _upload_image(image_path: str) -> str:
    """Upload a local image to fal.ai CDN and return the public URL."""
    logger.info("  Uploading image to fal.ai CDN: %s", image_path)
    url = _fal.upload_file(image_path)
    logger.info("  Uploaded → %s", url)
    return url


def _download_video(video_url: str, filename: str) -> str:
    """Download a video from a URL and save it to OUTPUT_DIR."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with requests.get(video_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return filepath


def _is_image_policy_violation(exc: Exception) -> bool:
    """Return True if the fal.ai error is a partner image validation rejection."""
    err = str(exc)
    return "partner_validation_failed" in err or "likenesses of real people" in err


def _subscribe(arguments: dict, on_queue_update) -> dict:
    return _fal.subscribe(
        FAL_VIDEO_MODEL,
        arguments=arguments,
        with_logs=True,
        on_queue_update=on_queue_update,
    )


def _generate_video_fal(
    image_path: str,
    prompt: str,
    duration_secs: int = 5,
    aspect_ratio: str = "16:9",
    avatar_description: str = "",
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> str:
    """
    Submit a Seedance image-to-video job via fal.ai and return the local .mp4 path.
    If the image is rejected by fal.ai's partner validation (common for AI-generated
    portraits that resemble real people), retries as text-to-video with the character's
    avatar_description prepended to the prompt so the output still matches the character.
    fal_client.subscribe blocks until complete — no manual polling needed.
    """
    image_url = _upload_image(image_path)

    t_start = time.time()

    def _on_queue_update(update):
        elapsed = int(time.time() - t_start)
        if hasattr(update, "logs"):
            for log in update.logs:
                msg = log.get("message", "")
                if msg:
                    logger.info("  [fal/seedance %3ds] %s", elapsed, msg)
        if on_progress:
            on_progress(elapsed, "processing")

    base_args = {
        "prompt": prompt,
        "duration": duration_secs,
        "aspect_ratio": aspect_ratio,
    }

    logger.info("Submitting Seedance job via fal.ai (model=%s, duration=%ds)…", FAL_VIDEO_MODEL, duration_secs)
    try:
        result = _subscribe({**base_args, "image_url": image_url}, _on_queue_update)
    except Exception as exc:
        if _is_image_policy_violation(exc):
            # Enrich the text-only fallback with the character's physical description
            # so the video still generates a character matching the portrait.
            fallback_prompt = prompt
            if avatar_description:
                fallback_prompt = f"Character appearance: {avatar_description}. {prompt}"
            logger.warning(
                "Image rejected by fal.ai partner validation — retrying as text-to-video. "
                "Avatar description %s to prompt.",
                "prepended" if avatar_description else "not available; falling back as-is",
            )
            result = _subscribe({**base_args, "prompt": fallback_prompt}, _on_queue_update)
        else:
            raise

    video_url = result["video"]["url"]
    elapsed = int(time.time() - t_start)
    logger.info("  Seedance complete in %ds. Downloading…", elapsed)
    if on_progress:
        on_progress(elapsed, "done")

    filename = f"seedance_{uuid.uuid4().hex[:8]}.mp4"
    return _download_video(video_url, filename)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_video_segment(
    image_path: str,
    scene_description: str,
    shot_description: str,
    segment_number: int,
    dialogue: str = "",
    continuation_note: str = "",
    duration: int = SEGMENT_DURATION,
    timeout: int = 600,
    avatar_description: str = "",
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> str:
    """
    Generate a single video segment using Seedance via fal.ai.

    image_path          — reference image (portrait for seg 1, last frame for rest)
    scene_description   — environment / background instructions
    shot_description    — character motion / camera framing
    segment_number      — 1-based index for logging and filenames
    dialogue            — spoken words embedded in the prompt
    continuation_note   — visual continuity hint from previous segment
    duration            — target clip length in seconds
    avatar_description  — character physical description; used to enrich the text-to-video
                          fallback if the reference image is rejected by fal.ai validation
    on_progress         — optional callback(elapsed_secs: int, status: str)

    Returns the local .mp4 file path.
    """
    t_start = time.time()
    logger.info("━━━ Segment %d: starting Seedance/fal generation (duration=%ds) ━━━", segment_number, duration)

    # Build prompt: scene + shot + dialogue + continuity
    prompt_parts = [scene_description.strip(), shot_description.strip()]
    if dialogue:
        prompt_parts.append(f'The character speaks: "{dialogue.strip()}"')
    if continuation_note:
        prompt_parts.append(continuation_note.strip())
    prompt = " ".join(p for p in prompt_parts if p)
    logger.info("  Raw prompt: %s", prompt[:120] + ("…" if len(prompt) > 120 else ""))

    # --- Critique → Rewrite loop (max 3 attempts) ---
    _MAX_CRITIQUE_TRIES = 3
    _silent_fallback = " ".join([scene_description.strip(), shot_description.strip()])
    for _attempt in range(_MAX_CRITIQUE_TRIES):
        _critique = _critique_prompt(prompt)
        if _critique["passes"]:
            logger.info("  Prompt passed compliance check (attempt %d)", _attempt + 1)
            break
        logger.warning("  Attempt %d — compliance issues: %s", _attempt + 1, _critique["issues"])
        prompt = _rewrite_prompt(prompt, _critique["issues"])
        logger.info("  Rewritten: %s", prompt[:120] + ("…" if len(prompt) > 120 else ""))
    else:
        logger.warning("  All rewrite attempts exhausted — falling back to silent prompt")
        prompt = _silent_fallback

    logger.info("  Final prompt: %s", prompt[:120] + ("…" if len(prompt) > 120 else ""))

    tmp_path = _generate_video_fal(
        image_path=image_path,
        prompt=prompt,
        duration_secs=duration,
        avatar_description=avatar_description,
        on_progress=on_progress,
    )

    # Rename to segment-numbered filename
    final_filename = f"segment_{segment_number:02d}_{uuid.uuid4().hex[:6]}.mp4"
    final_path = os.path.join(OUTPUT_DIR, final_filename)
    os.replace(tmp_path, final_path)

    total = int(time.time() - t_start)
    logger.info("━━━ Segment %d: complete in %ds ━━━", segment_number, total)
    return final_path
