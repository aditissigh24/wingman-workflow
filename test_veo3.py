"""
Standalone test script for Veo3 video generation.

Usage:
    python test_veo3.py                          # uses auto-generated test image
    python test_veo3.py --image path/to/img.jpg  # uses your own image
    python test_veo3.py --duration 5             # 5s clip (default) or 8
    python test_veo3.py --keep                   # keep output file after test

Checks:
  1. GOOGLE_API_KEY is set
  2. Veo3 API accepts the request
  3. Polling completes within timeout
  4. Video is saved to output/ with correct file size (> 10 KB)
  5. Video is a valid MP4 (checks file header bytes)
"""

import argparse
import io
import logging
import os
import sys
import time

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_veo3")

# ---------------------------------------------------------------------------
# Resolve project root so imports work regardless of CWD
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_image() -> str:
    """Create a simple gradient JPEG in the output dir and return its path."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        log.error("Pillow is not installed. Run: pip install Pillow")
        sys.exit(1)

    from config import OUTPUT_DIR

    img = Image.new("RGB", (1280, 720), color=(30, 60, 120))
    draw = ImageDraw.Draw(img)
    # simple gradient-ish overlay
    for y in range(720):
        shade = int(80 + (y / 720) * 120)
        draw.line([(0, y), (1280, y)], fill=(shade, shade // 2, 200 - shade // 2))
    draw.text((80, 320), "Veo3 Test Frame", fill=(255, 255, 255))

    path = os.path.join(OUTPUT_DIR, "veo3_test_input.jpg")
    img.save(path, format="JPEG", quality=90)
    log.info("Created test image: %s", path)
    return path


def _check_mp4(path: str) -> bool:
    """Return True if the file starts with a valid MP4/ftyp box."""
    try:
        with open(path, "rb") as f:
            header = f.read(12)
        # MP4 files have 'ftyp' at bytes 4-8
        return header[4:8] == b"ftyp"
    except Exception:
        return False


def _fmt_size(path: str) -> str:
    size = os.path.getsize(path)
    if size >= 1_048_576:
        return f"{size / 1_048_576:.1f} MB"
    return f"{size / 1024:.1f} KB"


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_test(image_path: str, duration: int, keep: bool) -> bool:
    log.info("=" * 60)
    log.info("Veo3 Video Generation Test")
    log.info("=" * 60)

    # --- 1. Config check ---
    try:
        from config import GOOGLE_API_KEY, VEO3_MODEL, OUTPUT_DIR
    except ImportError as e:
        log.error("Cannot import config: %s", e)
        return False

    if not GOOGLE_API_KEY:
        log.error("GOOGLE_API_KEY is not set in .env / environment.")
        return False
    log.info("✓ GOOGLE_API_KEY found (len=%d)", len(GOOGLE_API_KEY))
    log.info("  Model : %s", VEO3_MODEL)
    log.info("  Output: %s", OUTPUT_DIR)

    # --- 2. Image ---
    if image_path:
        if not os.path.isfile(image_path):
            log.error("Image not found: %s", image_path)
            return False
        log.info("✓ Using provided image: %s", image_path)
    else:
        log.info("No image provided — generating a test image…")
        image_path = _make_test_image()

    # --- 3. Generate ---
    log.info("-" * 60)
    log.info("Submitting generation request (duration=%ds)…", duration)

    try:
        from steps.seedance_generator import generate_video_segment
    except ImportError as e:
        log.error("Cannot import seedance_generator: %s", e)
        return False

    progress_log: list[tuple[int, str]] = []

    def on_progress(elapsed: int, status: str) -> None:
        progress_log.append((elapsed, status))

    t_start = time.time()
    try:
        output_path = generate_video_segment(
            image_path=image_path,
            scene_description="A cinematic outdoor scene with soft golden hour lighting.",
            shot_description="The camera slowly pushes in toward the horizon.",
            segment_number=1,
            continuation_note="",
            duration=duration,
            timeout=600,
            on_progress=on_progress,
        )
    except TimeoutError:
        log.error("✗ Generation timed out after 600s.")
        return False
    except Exception as e:
        log.error("✗ Generation failed: %s", e)
        return False

    elapsed = int(time.time() - t_start)
    log.info("Generation completed in %ds", elapsed)

    # --- 4. Validate output ---
    log.info("-" * 60)

    if not os.path.isfile(output_path):
        log.error("✗ Output file not found: %s", output_path)
        return False
    log.info("✓ File exists: %s", output_path)

    size = os.path.getsize(output_path)
    if size < 10_240:  # less than 10 KB is definitely wrong
        log.error("✗ File too small (%s) — likely corrupt or empty.", _fmt_size(output_path))
        return False
    log.info("✓ File size: %s", _fmt_size(output_path))

    if _check_mp4(output_path):
        log.info("✓ Valid MP4 header detected.")
    else:
        log.warning("⚠ File header does not look like MP4 — may be corrupt.")

    # --- 5. Summary ---
    log.info("=" * 60)
    log.info("TEST PASSED — video saved to: %s", output_path)
    log.info("Total wall-clock time: %ds", elapsed)
    if progress_log:
        log.info("Poll steps: %d", len(progress_log))
    log.info("=" * 60)

    if not keep:
        try:
            os.remove(output_path)
            log.info("(Output file removed; use --keep to retain it)")
        except OSError:
            pass

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Test Veo3 video generation end-to-end.")
    parser.add_argument(
        "--image", "-i",
        default="",
        help="Path to a reference image (JPEG/PNG). If omitted, a test image is auto-generated.",
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        choices=[5, 8],
        default=5,
        help="Clip duration in seconds: 5 (default) or 8.",
    )
    parser.add_argument(
        "--keep", "-k",
        action="store_true",
        help="Keep the generated video file after the test.",
    )
    args = parser.parse_args()

    success = run_test(
        image_path=args.image,
        duration=args.duration,
        keep=args.keep,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
