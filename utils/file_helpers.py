import os
import uuid
import requests


def upload_to_tmpfiles(filepath: str) -> str:
    """
    Upload a file to litterbox.catbox.moe and return a temporary public download URL.
    Files expire after 72 hours — more than sufficient for the lip-sync step.
    """
    with open(filepath, "rb") as f:
        response = requests.post(
            "https://litterbox.catbox.moe/resources/internals/api.php",
            data={"reqtype": "fileupload", "time": "72h"},
            files={"fileToUpload": (os.path.basename(filepath), f)},
            timeout=120,
        )
    if not response.ok:
        raise RuntimeError(
            f"litterbox upload failed ({response.status_code}): {response.text[:300]}"
        )
    url = response.text.strip()
    if not url.startswith("https://"):
        raise RuntimeError(f"litterbox returned unexpected response: {url[:300]}")
    return url


def generate_output_path(prefix: str, extension: str) -> str:
    """Generate a unique output file path."""
    from config import OUTPUT_DIR
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.{extension}"
    return os.path.join(OUTPUT_DIR, filename)
