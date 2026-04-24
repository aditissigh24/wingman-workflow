import os
import uuid
import logging
from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_BUCKET

logger = logging.getLogger("supabase_client")

_client = None


def _get_client():
    global _client
    if _client is None:
        from supabase import create_client
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _client


def upload_file(local_path: str, remote_key: str, content_type: str) -> str:
    """
    Upload a local file to Supabase Storage.
    Returns the public URL for the uploaded file.
    """
    sb = _get_client()
    with open(local_path, "rb") as f:
        data = f.read()

    logger.info("Uploading %s → %s/%s", local_path, SUPABASE_BUCKET, remote_key)
    sb.storage.from_(SUPABASE_BUCKET).upload(
        path=remote_key,
        file=data,
        file_options={"content-type": content_type, "upsert": "true"},
    )

    public_url = sb.storage.from_(SUPABASE_BUCKET).get_public_url(remote_key)
    logger.info("  Public URL: %s", public_url)
    return public_url


def upload_image(local_path: str, prefix: str = "images") -> str:
    """Upload a JPEG/PNG image to Supabase. Returns public URL."""
    ext = os.path.splitext(local_path)[1].lower() or ".jpg"
    remote_key = f"{prefix}/{uuid.uuid4().hex}{ext}"
    content_type = "image/png" if ext == ".png" else "image/jpeg"
    return upload_file(local_path, remote_key, content_type)


def upload_audio(local_path: str, prefix: str = "audio") -> str:
    """Upload an MP3/WAV audio file to Supabase. Returns public URL."""
    ext = os.path.splitext(local_path)[1].lower() or ".mp3"
    remote_key = f"{prefix}/{uuid.uuid4().hex}{ext}"
    content_type = "audio/wav" if ext == ".wav" else "audio/mpeg"
    return upload_file(local_path, remote_key, content_type)


def upload_video(local_path: str, prefix: str = "videos") -> str:
    """Upload an MP4 video to Supabase. Returns public URL."""
    remote_key = f"{prefix}/{uuid.uuid4().hex}.mp4"
    return upload_file(local_path, remote_key, "video/mp4")
