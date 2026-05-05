import logging
import os
import time
import uuid
import requests
from requests.exceptions import ChunkedEncodingError, ReadTimeout
from fal_client import SyncClient
from config import FAL_KEY, FAL_IMAGE_MODEL, OUTPUT_DIR

log = logging.getLogger(__name__)

_fal = SyncClient(key=FAL_KEY)


def _download_with_retry(url: str, max_attempts: int = 5) -> bytes:
    last_exc = None
    for attempt in range(max_attempts):
        delay = 2 ** attempt
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 502:
                log.warning("fal download attempt %d/%d: 502, retrying in %ds", attempt + 1, max_attempts, delay)
                time.sleep(delay)
                continue
            r.raise_for_status()
            return r.content
        except ReadTimeout as exc:
            last_exc = exc
            log.warning("fal download attempt %d/%d: ReadTimeout, retrying in %ds", attempt + 1, max_attempts, delay)
            time.sleep(delay)
        except ChunkedEncodingError as exc:
            last_exc = exc
            log.warning("fal download attempt %d/%d: IncompleteRead — %s, retrying in %ds", attempt + 1, max_attempts, exc, delay)
            time.sleep(delay)
    raise RuntimeError(f"fal image download failed after {max_attempts} attempts: {last_exc}")


def _fal_generate(prompt: str, image_url: str = None) -> str:
    """Single fal call. Returns the image URL from the result."""
    args = {
        "prompt": prompt,
        "image_size": "portrait_4_3",
        "num_images": 1,
        "output_format": "jpeg",
        "safety_tolerance": 3,
    }
    if image_url:
        args["image_url"] = image_url

    result = _fal.subscribe(FAL_IMAGE_MODEL, arguments=args)
    log.debug("fal raw response: %s", result)

    images = result.get("images", [])
    if not images:
        raise RuntimeError("fal.ai returned no images.")
    return images[0]["url"]


def generate_image(enhanced_prompt: str) -> list[str]:
    """
    Generate two portraits. First is text-only; second uses the first as image input
    so fal produces the same face and features with a different composition.
    Returns two local file paths.
    """
    log.info("Generating portrait 1 (text-only)")
    url1 = _fal_generate(enhanced_prompt)
    log.info("Portrait 1 URL: %s", url1)

    log.info("Generating portrait 2 (using portrait 1 as image input)")
    url2 = _fal_generate(enhanced_prompt, image_url=url1)
    log.info("Portrait 2 URL: %s", url2)

    filepaths = []
    for url in (url1, url2):
        content = _download_with_retry(url)
        filename = f"image_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(content)
        log.info("Saved %s (%d bytes)", filepath, len(content))
        filepaths.append(filepath)

    return filepaths


def generate_images(enhanced_prompt: str) -> str:
    """Convenience wrapper — returns the first image path."""
    return generate_image(enhanced_prompt)[0]
