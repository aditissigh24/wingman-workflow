"""
Video generation dispatcher — routes to Seedance via fal.ai.

The video model is set by FAL_VIDEO_MODEL in .env (default: bytedance/seedance-2.0/fast/image-to-video).
"""

from typing import Callable, Optional
from config import SEGMENT_DURATION

from steps.seedance_generator import generate_video_segment as _seedance
from steps.seedance_generator import extract_last_frame


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
    **_,
) -> str:
    """
    Generate a single video segment using Seedance via fal.ai.
    Returns the local .mp4 file path.
    """
    return _seedance(
        image_path=image_path,
        scene_description=scene_description,
        shot_description=shot_description,
        segment_number=segment_number,
        dialogue=dialogue,
        continuation_note=continuation_note,
        duration=duration,
        timeout=timeout,
        avatar_description=avatar_description,
        on_progress=on_progress,
    )
