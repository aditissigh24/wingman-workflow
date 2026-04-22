import os
import subprocess
import uuid
import imageio_ffmpeg
from config import OUTPUT_DIR


def concatenate_segments(segment_paths: list) -> str:
    """
    Concatenate a list of video segment files into a single MP4 using the
    ffmpeg concat demuxer. No re-encoding — video streams are copied directly.

    Returns the file path to the concatenated video.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    concat_list_path = os.path.join(OUTPUT_DIR, f"concat_{uuid.uuid4().hex[:8]}.txt")
    output_filename = f"combined_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Write the ffmpeg concat manifest — paths must use forward slashes
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for seg_path in segment_paths:
            safe_path = seg_path.replace("\\", "/")
            f.write(f"file '{safe_path}'\n")

    cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Clean up the manifest file
    try:
        os.remove(concat_list_path)
    except OSError:
        pass

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg concat failed (exit {result.returncode}):\n{result.stderr}"
        )

    return output_path


def merge_audio_video(video_path: str, audio_path: str) -> str:
    """
    Mux an audio file into a video file without re-encoding the video stream.
    The output is trimmed to the shorter of the two streams (-shortest).
    Returns the file path to the merged MP4.

    Uses the ffmpeg binary bundled with imageio-ffmpeg — no system install needed.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    filename = f"final_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)

    cmd = [
        ffmpeg_exe,
        "-y",                 # overwrite output without prompting
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",       # copy video stream without re-encoding
        "-c:a", "aac",        # encode audio to AAC for MP4 compatibility
        "-shortest",          # trim to the shorter stream
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg merge failed (exit {result.returncode}):\n{result.stderr}"
        )

    return output_path
