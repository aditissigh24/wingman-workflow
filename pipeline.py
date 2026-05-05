from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from utils.polling import PipelineError
from steps.prompt_enhancer import enhance_image_prompt, write_video_script, enhance_audio_prompt
from steps.image_generator import generate_image
from steps.video_dispatcher import generate_video_segment as generate_video
from steps.audio_generator import generate_audio
from steps.lip_sync import lip_sync


class Pipeline:
    def __init__(self, user_prompt: str, progress_callback: Optional[Callable] = None):
        self.user_prompt = user_prompt
        self.progress = progress_callback or (lambda step, pct, msg: None)
        self.artifacts: dict = {}

    def run(self) -> str:
        """
        Execute the full pipeline. Returns path to the final video.
        Raises PipelineError with step name on failure.
        """
        try:
            # Step 1: Enhance image prompt
            self.progress("enhance_prompt", 5, "Enhancing image prompt with Claude...")
            self.artifacts["image_prompt"] = enhance_image_prompt(self.user_prompt)

            # Step 2: Generate image
            self.progress("image_gen", 15, "Generating image with Gemini...")
            self.artifacts["image_path"] = generate_image(self.artifacts["image_prompt"])

            # Step 3: Write video script + voiceover
            self.progress("script", 30, "Writing video script with Claude...")
            script = write_video_script(self.user_prompt, self.artifacts["image_prompt"])
            self.artifacts["scene_description"] = script["scene_description"]
            self.artifacts["voiceover_script"] = script["voiceover_script"]

            # Step 3b: Enhance voiceover for TTS
            self.progress("audio_enhance", 38, "Enhancing voiceover script...")
            self.artifacts["enhanced_voiceover"] = enhance_audio_prompt(
                self.artifacts["voiceover_script"]
            )

            # Steps 4 & 5: Video generation + Audio generation IN PARALLEL
            self.progress("parallel_gen", 45, "Generating video and audio in parallel...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                video_future = executor.submit(
                    generate_video,
                    self.artifacts["image_path"],
                    self.artifacts["scene_description"],
                )
                audio_future = executor.submit(
                    generate_audio,
                    self.artifacts["enhanced_voiceover"],
                )
                # Audio finishes first typically, but we need both
                self.artifacts["audio_path"] = audio_future.result()
                self.artifacts["video_path"] = video_future.result()

            # Step 6: Lip sync
            self.progress("lip_sync", 80, "Lip-syncing video with audio via SyncLabs...")
            self.artifacts["final_path"] = lip_sync(
                self.artifacts["video_path"],
                self.artifacts["audio_path"],
            )

            self.progress("done", 100, "Pipeline complete!")
            return self.artifacts["final_path"]

        except PipelineError:
            raise
        except TimeoutError as e:
            raise PipelineError("timeout", str(e), self.artifacts)
        except Exception as e:
            # Determine which step failed based on what artifacts exist
            step = self._detect_failed_step()
            raise PipelineError(step, str(e), self.artifacts)

    def _detect_failed_step(self) -> str:
        """Infer which step failed based on which artifacts are present."""
        if "image_prompt" not in self.artifacts:
            return "enhance_prompt"
        if "image_path" not in self.artifacts:
            return "image_gen"
        if "scene_description" not in self.artifacts:
            return "script"
        if "video_path" not in self.artifacts or "audio_path" not in self.artifacts:
            return "parallel_gen"
        return "lip_sync"
