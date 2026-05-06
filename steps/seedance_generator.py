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
You are a video generation prompt compliance checker for Google Veo3. Your ONLY job is to identify issues that will cause Veo3 to reject the prompt or return no output.

Evaluate the prompt against EXACTLY these rules:

CONTENT SAFETY (causes safety filter rejection):
1. No real celebrity, public figure, actor, politician, athlete, or influencer names or likenesses
2. No brand names, trademarks, or copyrighted IP references
3. Dialogue must be emotionally safe — no explicit language, no sexual content, no graphic violence, no self-harm references
4. No political statements or hate speech
5. Hinglish (Hindi + English mix) is fully allowed and must NOT be flagged
6. Emotional intensity, romance, heartbreak, flirting are all allowed

VEO3 PROMPT QUALITY (causes no_media_generated rejection):
7. No duplicate or near-identical instruction blocks — if the same instruction appears more than once, flag the duplicates
8. No contradictory visual instructions in the same prompt (e.g., "she smiles" and "she looks devastated" together)
9. Dialogue must be natural spoken language — no code, symbols, or unpronounceable strings
10. Prompt must not simultaneously demand conflicting camera angles or lighting states
11. Character appearance instructions must not contradict normal human anatomy or physics

Return ONLY a JSON object:
{"passes": true, "issues": []}
OR
{"passes": false, "issues": ["Exact issue 1 with the specific offending text", "Exact issue 2..."]}

Do NOT rewrite anything. Do NOT suggest fixes. Only identify problems.
""".strip()

_REWRITER = """
You are a video prompt rewriter for Google Veo3. You receive a prompt and a list of specific issues identified by a compliance checker.

Your ONLY job is to fix exactly the listed issues while preserving:
- The overall scene, shot framing, and visual intent
- All spoken dialogue (meaning, emotion, Hinglish language)
- The character's voice, tone, and cultural specificity
- Creative energy — do NOT sanitize beyond what the issues require

Rules:
- Replace real person names with vivid fictional descriptors (e.g. "a legendary cricketer" not "Virat Kohli")
- Replace brand names with generic equivalents (e.g. "a luxury SUV" not "Audi Q7")
- If dialogue has an explicit word, rephrase just that word/phrase while keeping the emotional beat
- Remove duplicate instruction blocks — keep only one copy of repeated text
- If instructions contradict each other, keep the one that best serves the scene's emotional intent
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
    aspect_ratio: str = "9:16",
    avatar_description: str = "",
    portrait_path: str = "",
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> str:
    """
    Submit a Veo3 image-to-video job via fal.ai and return the local .mp4 path.

    image_path    — reference image for this segment (portrait for seg 1, last frame for rest)
    portrait_path — original character portrait; uploaded fresh every segment and used as
                    the fallback reference if the primary image is rejected by fal.ai validation
    """
    image_url = _upload_image(image_path)

    # Always upload the original portrait fresh so we have a clean face-reference URL.
    # Used as the image_url in the text-to-video fallback if the primary image is rejected.
    portrait_url = _upload_image(portrait_path) if portrait_path else image_url
    logger.info("  Portrait uploaded → %s", portrait_url)

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

    # Veo3 API only accepts '4s', '6s', or '8s' — snap to nearest valid, rounding up on ties
    # so generated video is always long enough to cover the audio
    _VALID_DURATIONS = [4, 6, 8]
    snapped = min(_VALID_DURATIONS, key=lambda d: (abs(d - duration_secs), -d))
    duration_str = f"{snapped}s"

    base_args = {
        "prompt": prompt,
        "duration": duration_str,
        "aspect_ratio": aspect_ratio,
    }

    logger.info("Submitting video job via fal.ai (model=%s, requested=%ds, api=%s)…", FAL_VIDEO_MODEL, duration_secs, duration_str)
    try:
        result = _subscribe({**base_args, "image_url": image_url}, _on_queue_update)
    except Exception as exc:
        if _is_image_policy_violation(exc):
            # Primary image was rejected — retry with the original portrait (fresh upload)
            # so the fallback still has a face reference rather than going fully text-only.
            fallback_prompt = prompt
            if avatar_description:
                fallback_prompt = f"Character appearance: {avatar_description}. {prompt}"
            logger.warning(
                "Image rejected by fal.ai partner validation — retrying with portrait as reference. "
                "Avatar description %s to prompt.",
                "prepended" if avatar_description else "not available; falling back as-is",
            )
            result = _subscribe({**base_args, "prompt": fallback_prompt, "image_url": portrait_url}, _on_queue_update)
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
    is_last_segment: bool = False,
    portrait_path: str = "",
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
    is_last_segment     — when False, instructs model to end on a stable face shot for chaining
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

    # These constants are checked against the assembled text so they are never duplicated
    # (guards against stale session-state prompts that already contain them)
    _FACE_INSTR = "Maintain consistent character face, skin tone, hair, and facial features matching the reference image exactly."
    _TRANS_INSTR = (
        "End on a clean, stable, centered composition with the character's face clearly visible "
        "and forward-facing, so the final frame is suitable for seamless continuation into the next clip."
    )
    _assembled_so_far = " ".join(p for p in prompt_parts if p)
    if _FACE_INSTR not in _assembled_so_far:
        prompt_parts.append(_FACE_INSTR)
    if not is_last_segment and _TRANS_INSTR not in _assembled_so_far:
        prompt_parts.append(_TRANS_INSTR)

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
        portrait_path=portrait_path,
        on_progress=on_progress,
    )

    # Rename to segment-numbered filename
    final_filename = f"segment_{segment_number:02d}_{uuid.uuid4().hex[:6]}.mp4"
    final_path = os.path.join(OUTPUT_DIR, final_filename)
    os.replace(tmp_path, final_path)

    total = int(time.time() - t_start)
    logger.info("━━━ Segment %d: complete in %ds ━━━", segment_number, total)
    return final_path
