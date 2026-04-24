"""
Veo3 video segment generator — replaces Kling.
Uses Google AI SDK: client.models.generate_videos() with veo-3.0-generate-preview.

Pattern:
  - Segment 1: image-to-video (character portrait as first frame)
  - Subsequent segments: last frame of previous clip → next segment (chained)
  - Built-in polling loop; downloads and saves each .mp4 to OUTPUT_DIR
"""

import io
import logging
import os
import subprocess
import time
import uuid
from typing import Callable, Optional

import anthropic
from google import genai
from google.genai import types
from PIL import Image

from config import GOOGLE_API_KEY, OUTPUT_DIR, VEO3_MODEL, SEGMENT_DURATION, GEMINI_TEXT_MODEL, ANTHROPIC_API_KEY, CLAUDE_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("veo3_generator")


class AudioFilteredError(RuntimeError):
    """Raised when Veo3 rejects a prompt due to audio/dialogue content filtering."""


client = genai.Client(api_key=GOOGLE_API_KEY)
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ---------------------------------------------------------------------------
# Prompt critique + rewrite agents (pre-flight before Veo3 submission)
# ---------------------------------------------------------------------------

_VEO3_CHECKLIST = """
You are a Veo3 content compliance checker. Your ONLY job is to identify whether a video generation prompt will be rejected by Veo3's RAI (Responsible AI) safety filters.

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

_VEO3_REWRITER = """
You are a Veo3 video prompt rewriter. You receive a prompt and a list of specific issues identified by a compliance checker.

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


def _critique_veo3_prompt(prompt: str) -> dict:
    """
    Agent 1 — Critique only (Claude).
    Evaluates prompt against Veo3 RAI checklist.
    Returns {"passes": bool, "issues": [str, ...]}.
    Never rewrites anything.
    """
    import json
    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        system=_VEO3_CHECKLIST,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        result = json.loads(response.content[0].text.strip())
        return {"passes": bool(result.get("passes", True)), "issues": result.get("issues", [])}
    except Exception:
        logger.warning("Critique agent returned unparseable response — treating as passed")
        return {"passes": True, "issues": []}


def _rewrite_veo3_prompt(prompt: str, issues: list) -> str:
    """
    Agent 2 — Rewrite only.
    Receives the original prompt + the critique's issue list.
    Fixes exactly those issues while preserving creative intent.
    Never evaluates or judges — only rewrites.
    """
    issues_block = "\n".join(f"- {issue}" for issue in issues)
    contents = f"ORIGINAL PROMPT:\n{prompt}\n\nISSUES TO FIX:\n{issues_block}"
    response = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=_VEO3_REWRITER,
            max_output_tokens=1024,
        ),
    )
    return response.text.strip()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_image_bytes(image_path: str) -> bytes:
    """Load any image file, convert to JPEG bytes."""
    pil_image = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Last-frame extraction (reused from old pipeline)
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
# Veo3 generation: submit (image-to-video) + poll
# ---------------------------------------------------------------------------

def _generate_video_from_image(
    image_bytes: bytes,
    prompt: str,
    duration_secs: int = 8,
    aspect_ratio: str = "9:16",
    timeout: int = 600,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> str:
    """
    Submit a Veo3 image-to-video job and poll until done.
    Returns the local .mp4 file path.

    duration_secs: desired clip length (Veo3 supports 5 or 8 seconds natively)
    """
    logger.info("Submitting Veo3 image-to-video (model=%s, ~%ds clip)…", VEO3_MODEL, duration_secs)

    operation = client.models.generate_videos(
        model=VEO3_MODEL,
        prompt=prompt,
        image=types.Image(
            image_bytes=image_bytes,
            mime_type="image/jpeg",
        ),
        config=types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            number_of_videos=1,
            duration_seconds=int(8),
            person_generation="allow_adult",
        ),
    )

    logger.info("  Operation name: %s — polling…", operation.name)
    t_start = time.time()
    poll_interval = 10

    while not operation.done:
        time.sleep(poll_interval)
        operation = client.operations.get(operation)
        elapsed = int(time.time() - t_start)
        status = "processing" if not operation.done else "done"
        logger.info("  [%3ds] status=%s", elapsed, status)
        if on_progress:
            on_progress(elapsed, status)
        if elapsed > timeout:
            raise TimeoutError(f"Veo3 timed out after {timeout}s")

    logger.info(operation.response)

    if operation.response is None or not operation.response.generated_videos:
        logger.error(operation.response)
        raise RuntimeError(f"Veo3 returned no videos in response: {operation.response}")

    return operation.response.generated_videos[0].video


def _save_video(video_obj, filename: str) -> str:
    """Download and save a Veo3 video object to OUTPUT_DIR using the SDK client."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    client.files.download(file=video_obj)
    video_obj.save(filepath)
    return filepath


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
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> str:
    """
    Generate a single video segment using Veo3 image-to-video.

    image_path        — reference image: portrait for segment 1, last frame for subsequent
    scene_description — environment / background instructions
    shot_description  — character motion / action instructions
    segment_number    — 1-based index, used for logging and filenames
    continuation_note — context hint for narrative continuity
    duration          — target clip length in seconds (Veo3 native: 5 or 8)
    timeout           — polling timeout in seconds
    on_progress       — optional callback(elapsed_secs: int, status: str)

    Returns the local file path to the saved .mp4 segment.
    """
    t_start = time.time()
    logger.info("━━━ Segment %d: starting Veo3 generation (duration=%ds) ━━━", segment_number, duration)

    # Veo3 only supports 5 or 8 second clips natively — round to nearest
    veo3_duration = 8 if duration >= 7 else 5

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
        _critique = _critique_veo3_prompt(prompt)
        if _critique["passes"]:
            logger.info("  Prompt passed compliance check (attempt %d)", _attempt + 1)
            break
        logger.warning(
            "  Attempt %d — compliance issues: %s",
            _attempt + 1, _critique["issues"],
        )
        prompt = _rewrite_veo3_prompt(prompt, _critique["issues"])
        logger.info("  Rewritten prompt: %s", prompt[:120] + ("…" if len(prompt) > 120 else ""))
    else:
        logger.warning(
            "  All %d rewrite attempts exhausted — falling back to silent (no-dialogue) prompt",
            _MAX_CRITIQUE_TRIES,
        )
        prompt = _silent_fallback

    logger.info("  Final prompt: %s", prompt[:120] + ("…" if len(prompt) > 120 else ""))
    logger.info("  Loading reference image…")
    image_bytes = _load_image_bytes(image_path)

    video_obj = _generate_video_from_image(
        image_bytes=image_bytes,
        prompt=prompt,
        duration_secs=veo3_duration,
        timeout=timeout,
        on_progress=on_progress,
    )

    filename = f"segment_{segment_number:02d}_{uuid.uuid4().hex[:6]}.mp4"
    local_path = _save_video(video_obj, filename)

    total = int(time.time() - t_start)
    logger.info("━━━ Segment %d: complete in %ds ━━━", segment_number, total)
    return local_path
