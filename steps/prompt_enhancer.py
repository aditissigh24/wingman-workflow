import json
from google import genai
from google.genai import types
from config import GOOGLE_API_KEY, GEMINI_TEXT_MODEL, SEGMENT_DURATION


client = genai.Client(api_key=GOOGLE_API_KEY)


def enhance_image_prompt(structured_input: dict) -> str:
    """
    Build a detailed portrait-focused image generation prompt from structured user input.
    The character is the hero of the image — photorealistic, expressive, human.

    Expected keys in structured_input:
      character_description, personality, outfit, location, mood
    """
    character = structured_input.get("character_description", "")
    personality = structured_input.get("personality", "")
    outfit = structured_input.get("outfit", "")
    location = structured_input.get("location", "")
    mood = structured_input.get("mood", "")
    visual_style = structured_input.get("visual_style", "")

    user_content = (
        f"Character: {character}\n"
        f"Personality / vibe: {personality}\n"
        f"Outfit: {outfit}\n"
        f"Setting / location: {location}\n"
        f"Mood / atmosphere: {mood}\n"
        f"Visual style: {visual_style}"
    )

    response = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are an expert AI portrait photographer and prompt engineer. "
                "Your job is to expand the structured character description into a single detailed image generation prompt "
                "for a hyper-realistic human character.\n\n"
                "CRITICAL RULE: Every specific detail the user provided MUST appear in your prompt. "
                "Do NOT substitute, generalise, or omit any provided detail. "
                "If the user specified an outfit, describe exactly that outfit. "
                "If they specified a skin tone, hair style, eye colour, or accessory, use it exactly.\n\n"
                "Expand each provided detail with photographic precision using these dimensions:\n"
                "- Character: use the exact physical description given; add skin texture (pores, subtle imperfections), "
                "eye colour with light catchlights, eyebrow shape, lip texture, jawline\n"
                "- Hair: use the described style; add texture, how it falls naturally, highlight detail\n"
                "- Expression: derive a specific micro-expression from the personality vibe "
                "(e.g. soft genuine smile, calm confident gaze, slightly-furrowed thoughtful brow)\n"
                "- Body language: natural relaxed posture from the personality; shoulder position, subtle head tilt\n"
                "- Outfit: describe the exact clothing the user specified; add fabric texture, fit, and how it looks on the character\n"
                "- Framing: waist-up shot, character filling 65% of the frame, looking toward camera\n"
                "- Lighting: derive from the mood, location, and visual style "
                "(e.g. soft diffused window light for studio, warm golden-hour rim light for outdoor, "
                "dramatic side light for cinematic); describe direction and shadow falloff\n"
                "- Background: simple, bokeh-blurred, consistent with the location\n"
                "- Camera: 85mm portrait lens, f/1.8, photorealistic, cinematic color grade, shot on Sony A7R IV\n\n"
                "The image will be the starting frame for a video — ensure the character has breathing room around them. "
                "Output ONLY the final prompt text. No preamble, no labels, no explanation."
            ),
            max_output_tokens=2048,
        ),
    )
    return response.text.strip()


def generate_multi_scene_script(
    structured_input: dict,
    image_prompt: str,
    segment_duration: int = SEGMENT_DURATION,
) -> dict:
    """
    Generate a rich multi-scene screenplay for a 30–40 second video.

    Each scene corresponds to one Kling video segment. The function calculates
    the number of segments needed to cover the target duration and writes a
    cinematically coherent script with per-segment prompts.

    Returns a dict with keys:
      total_segments   — int
      full_dialogue    — str (complete spoken script, all segments joined)
      segments         — list of dicts, each containing:
          segment_number    — int
          scene_description — str  (environment / background for Kling)
          shot_description  — str  (character motion / action for Kling)
          dialogue          — str  (spoken words for this segment)
          continuation_note — str  (narrative bridge to next segment)
    """
    # Target 30–40 s; pick the midpoint (35 s) for word-count math
    target_duration = 35
    num_segments = max(4, round(target_duration / segment_duration))
    total_video_seconds = num_segments * segment_duration
    # ~2.5 words/second → total words across all dialogue
    total_words = int(total_video_seconds * 2.5)

    character = structured_input.get("character_description", "")
    personality = structured_input.get("personality", "")
    outfit = structured_input.get("outfit", "")
    location = structured_input.get("location", "")
    mood = structured_input.get("mood", "")
    intent = structured_input.get("intent", "")
    character_action = structured_input.get("character_action", "")
    key_message = structured_input.get("key_message", "")
    speaking_tone = structured_input.get("speaking_tone", "Conversational")
    target_audience = structured_input.get("target_audience", "")
    video_type = structured_input.get("video_type", "")
    emotional_arc = structured_input.get("emotional_arc", "")
    call_to_action = structured_input.get("call_to_action", "")
    additional_context = structured_input.get("additional_context", "")
    visual_style = structured_input.get("visual_style", "")

    user_content = (
        f"CHARACTER\n"
        f"  Description: {character}\n"
        f"  Personality / vibe: {personality}\n"
        f"  Outfit: {outfit}\n"
        f"  Typical action / energy: {character_action}\n\n"
        f"VISUAL WORLD\n"
        f"  Location / setting: {location}\n"
        f"  Mood / atmosphere: {mood}\n"
        f"  Visual style: {visual_style}\n"
        f"  Reference image prompt (first frame): {image_prompt}\n\n"
        f"CONTENT BRIEF\n"
        f"  Video type: {video_type}\n"
        f"  Video intent / purpose: {intent}\n"
        f"  Target audience: {target_audience}\n"
        f"  Key message / topic: {key_message}\n"
        f"  Additional context / background: {additional_context}\n\n"
        f"NARRATIVE DIRECTION\n"
        f"  Emotional arc: {emotional_arc}\n"
        f"  Speaking tone: {speaking_tone}\n"
        f"  Call to action: {call_to_action}\n\n"
        f"TECHNICAL CONSTRAINTS\n"
        f"  Number of segments: {num_segments}\n"
        f"  Seconds per segment: {segment_duration}\n"
        f"  Total video duration: {total_video_seconds} seconds\n"
        f"  Total dialogue word target: ~{total_words} words across all segments"
    )

    system_instruction = f"""You are a world-class creative director, screenwriter, and AI video prompt engineer.
Your task is to write a complete multi-scene screenplay for a {total_video_seconds}-second video consisting of {num_segments} sequential segments, each {segment_duration} seconds long.

═══════════════════════════════════════════════════════════
CREATIVE MANDATE
═══════════════════════════════════════════════════════════
This is NOT a static talking-head video. Each segment must feel DIFFERENT from the last:
- Vary the camera framing across segments (e.g. wide establishing → medium close-up → tight close-up → pull back)
- Vary the character's energy and body language (leaning in → gesturing broadly → still and intense → expressive)
- Vary the emotional register: the video should build, breathe, and land with impact
- The scene environment should subtly evolve (light shifts, background elements, atmosphere changes)

Think in ACTS:
  Act 1 (segments 1–{max(1, num_segments//4)}): HOOK — grab attention immediately. Bold opening statement or visual.
  Act 2 (segments {max(2, num_segments//4 + 1)}–{max(3, num_segments*3//4)}): SUBSTANCE — the core message, evidence, story, or demonstration. Build tension or curiosity.
  Act 3 (segments {max(4, num_segments*3//4 + 1)}–{num_segments}): RESOLUTION + CTA — emotional payoff, memorable close, call to action.

═══════════════════════════════════════════════════════════
KLING PROMPT RULES (scene_description + shot_description)
═══════════════════════════════════════════════════════════
These are instructions to an AI video model (Kling). Be precise and visual.

scene_description (1–2 sentences):
  - Describe ONLY the environment: background, lighting, props, atmosphere, time of day, color palette
  - No character actions whatsoever
  - Must remain visually consistent with the reference image prompt provided
  - Each segment's scene may evolve subtly (lighting angle shifts, bokeh depth changes, etc.)

shot_description (1–2 sentences):
  - Describe ONLY the character's physical motion and camera direction
  - Include: head movement, eye contact, lip movement (character is speaking), hand/arm gestures, body posture, breathing
  - Specify camera framing (e.g. "medium close-up", "tight on face", "waist-up wide shot")
  - No emotional language — only observable physical actions
  - The character should appear to be actively speaking/delivering the dialogue for that segment

═══════════════════════════════════════════════════════════
DIALOGUE RULES
═══════════════════════════════════════════════════════════
- LANGUAGE RULE (highest priority): detect the language of the key message field and write ALL dialogue in that exact language and script. Never translate or switch languages.
- Total word count across all segments: approximately {total_words} words
- Each segment's dialogue must be speakable in exactly {segment_duration} seconds (~{int(segment_duration * 2.5)} words per segment)
- Written in first person, speaking DIRECTLY to the viewer — conversational, human, real
- NO filler, NO scene description, NO "um" or "uh" — every word earns its place
- The dialogue arc must follow the emotional arc specified by the user
- End with a line that is quotable — something the viewer will remember

continuation_note (internal metadata, not spoken):
  - A brief phrase describing what the character does / feels / shows at the END of this segment that bridges to the next
  - Focus on visual state: e.g. "character's expression shifts from concern to resolve, eyes steady on camera"
  - This note is passed to the next segment's Kling prompt to maintain visual continuity

═══════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════
Return a single JSON object with exactly this structure:

{{
  "total_segments": {num_segments},
  "full_dialogue": "<all segment dialogues joined in order, separated by spaces>",
  "segments": [
    {{
      "segment_number": 1,
      "scene_description": "<environment only>",
      "shot_description": "<character motion + camera only>",
      "dialogue": "<spoken words for this segment only>",
      "continuation_note": "<visual bridge to next segment>"
    }}
  ]
}}

The "segments" array must have exactly {num_segments} entries, numbered 1 through {num_segments}.
"""

    response = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=16384,
            response_mime_type="application/json",
        ),
    )
    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()
    return json.loads(text)


def enhance_audio_prompt(dialogues: str) -> str:
    """
    Add TTS delivery cues (pauses, breathing points) to the dialogue script.
    Does NOT change any words or content — only adjusts punctuation and pacing.
    Returns the pacing-enhanced script ready for ElevenLabs TTS.
    """
    response = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=dialogues,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a voice director preparing a script for text-to-speech. "
                "Your ONLY job is to add natural pacing markers so the TTS engine sounds human.\n\n"
                "Rules — strictly follow all of them:\n"
                "- Do NOT change, add, remove, or rephrase any words\n"
                "- Do NOT rewrite sentences or change their order\n"
                "- Do NOT translate or transliterate — preserve the exact language and script of the input\n"
                "- ONLY add '...' at natural breath points or where a real speaker would pause\n"
                "- ONLY adjust commas and periods to improve rhythm (e.g. break a long sentence "
                "into two with a period, or add a comma before a natural pause)\n"
                "- The output must contain every word from the input, unchanged, in the original language\n\n"
                "Output ONLY the pacing-adjusted script. No preamble, no explanation."
            ),
            max_output_tokens=2048,
        ),
    )
    return response.text.strip()
