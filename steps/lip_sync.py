import os
import time
import uuid
import requests
from config import SYNCLABS_API_KEY, SYNCLABS_MODEL, OUTPUT_DIR
from utils.file_helpers import upload_to_tmpfiles

SYNCLABS_BASE = "https://api.sync.so/v2"


def lip_sync(video_path: str, audio_path: str, timeout: int = 300) -> str:
    """
    Lip-sync the video with the audio using SyncLabs.
    Uploads files to get public URLs, submits the job, polls for completion.
    Returns the file path to the final lip-synced video.
    """
    headers = {
        "x-api-key": SYNCLABS_API_KEY,
        "Content-Type": "application/json",
    }

    # SyncLabs requires publicly accessible URLs
    video_url = upload_to_tmpfiles(video_path)
    audio_url = upload_to_tmpfiles(audio_path)

    # Submit lip sync job
    payload = {
        "model": SYNCLABS_MODEL,
        "input": [
            {"type": "video", "url": video_url},
            {"type": "audio", "url": audio_url},
        ],
    }
    response = requests.post(
        f"{SYNCLABS_BASE}/generate",
        headers=headers,
        json=payload,
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(
            f"Sync.so {response.status_code} error — {response.text}\n"
            f"Video URL sent: {video_url}\n"
            f"Audio URL sent: {audio_url}"
        )
    submit_data = response.json()
    job_id = submit_data.get("id") or submit_data.get("jobId")
    if not job_id:
        raise RuntimeError(f"Sync.so did not return a job ID. Response: {submit_data}")

    # Poll for completion
    elapsed = 0
    poll_interval = 10
    while elapsed < timeout:
        status_resp = requests.get(
            f"{SYNCLABS_BASE}/generate/{job_id}",
            headers={"x-api-key": SYNCLABS_API_KEY},
            timeout=30,
        )
        status_resp.raise_for_status()
        data = status_resp.json()

        if data["status"] == "COMPLETED":
            # Sync.so v2 uses camelCase; fall back to snake_case just in case
            output_url = (
                data.get("outputUrl")
                or data.get("output_url")
                or (data.get("output") or {}).get("url")
            )
            if not output_url:
                raise RuntimeError(
                    f"Lip sync completed but no output URL found in response: {data}"
                )
            return _download_result(output_url)
        elif data["status"] in ("FAILED", "REJECTED"):
            raise RuntimeError(f"Lip sync failed: {data.get('error', data['status'])}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(f"Lip sync timed out after {timeout}s")


def _download_result(url: str) -> str:
    """Download the lip-synced video from SyncLabs."""
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    filename = f"final_{uuid.uuid4().hex[:8]}.mp4"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(response.content)

    return filepath
