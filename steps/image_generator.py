import os
import uuid
from google import genai
from google.genai import types
from config import GOOGLE_API_KEY, GEMINI_IMAGE_MODEL, OUTPUT_DIR


client = genai.Client(api_key=GOOGLE_API_KEY)


def generate_image(enhanced_prompt: str) -> str:
    """
    Generate an image from the enhanced prompt using Gemini.
    Returns the file path to the saved image.
    """
    response = client.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=enhanced_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="16:9"),
        ),
    )

    # Find the image part in the response
    candidates = response.candidates or []
    if not candidates:
        raise RuntimeError(
            "Gemini returned no candidates. The prompt was likely blocked by content safety filters. "
            "Try rephrasing your prompt."
        )

    content = candidates[0].content
    if content is None or not content.parts:
        finish_reason = getattr(candidates[0], "finish_reason", "unknown")
        raise RuntimeError(
            f"Gemini returned a candidate with no image content (finish_reason={finish_reason}). "
            "The prompt may have been blocked by content safety filters. Try rephrasing your prompt."
        )

    for part in content.parts:
        if part.inline_data is not None:
            image = part.as_image()
            filename = f"image_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            image.save(filepath)
            return filepath

    raise RuntimeError(
        "Gemini returned a response but with no image data inside it. "
        "The prompt may have been blocked by content safety filters. Try rephrasing your prompt."
    )
