"""
LLM generation functions for character, scenario, and beat field derivation.
Writer agents use Gemini; critique/editor agents use Claude.
"""

import json
import anthropic
from google import genai
from google.genai import types
from config import GOOGLE_API_KEY, GEMINI_TEXT_MODEL, SEGMENT_DURATION, ANTHROPIC_API_KEY, CLAUDE_MODEL

client = genai.Client(api_key=GOOGLE_API_KEY)
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ═══════════════════════════════════════════════════════════════════════════
# EXISTING — kept from original pipeline
# ═══════════════════════════════════════════════════════════════════════════

def enhance_image_prompt(
    avatar_prompt: str,
    scenario: dict | None = None,
    character: dict | None = None,
) -> str:
    """
    Expand an image description into a detailed generation prompt.

    Portrait mode (scenario=None): expands avatarPrompt into a hyper-realistic
    character portrait prompt — behaviour identical to the original function.

    Poster mode (scenario provided): uses the full scenario + character context
    to produce a cinematic movie poster prompt for the scenario's roleplay.
    """
    if scenario is None:
        # ── Portrait mode (original behaviour, unchanged) ─────────────────
        response = client.models.generate_content(
            model=GEMINI_TEXT_MODEL,
            contents=avatar_prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are an expert AI portrait photographer and prompt engineer. "
                    "Expand the given avatar description into a single detailed image generation prompt "
                    "for a hyper-realistic Indian human character.\n\n"
                    "CRITICAL RULE: Every specific detail provided MUST appear in your prompt. "
                    "Do NOT substitute, generalise, or omit any detail.\n\n"
                    "Expand with photographic precision:\n"
                    "- Skin texture (pores, subtle imperfections), eye colour with light catchlights\n"
                    "- Hair: texture, how it falls, highlight detail\n"
                    "- Expression: derive from personality vibe\n"
                    "- Body language: natural relaxed posture\n"
                    "- Framing: waist-up shot, character filling 65% of the frame, looking toward camera\n"
                    "- Lighting: warm, city-appropriate, golden-hour or soft diffused window light\n"
                    "- Background: simple, bokeh-blurred\n"
                    "- Camera: 85mm portrait lens, f/1.8, photorealistic, cinematic color grade, Sony A7R IV\n\n"
                    "The image will be the starting frame for a video — ensure breathing room around the character. "
                    "Output ONLY the final prompt text. No preamble, no labels, no explanation."
                ),
                # max_output_tokens=1024,
            ),
        )
        return response.text.strip()

    # ── Poster mode ────────────────────────────────────────────────────────
    char = character or {}
    contents = (
        f"SCENARIO TITLE: {scenario.get('scenarioTitle', '')}\n"
        f"TAGLINE: {scenario.get('tagline', '')}\n"
        f"TONE: {scenario.get('tone', '')}\n"
        f"ATMOSPHERE: {scenario.get('atmosphere', '')}\n"
        f"SETTING: {scenario.get('settingDescription', '')}\n"
        f"IMAGE SEED: {scenario.get('imagePrompt', '')}\n"
        f"GOOD OUTCOME: {scenario.get('goodOutcome', '')}\n"
        f"BAD OUTCOME: {scenario.get('badOutcome', '')}\n\n"
        f"CHARACTER ARCHETYPE: {char.get('archetype', '')}\n"
        f"CHARACTER VIBE: {char.get('vibeSummary', '')}\n"
        f"ACCENT COLOR (HSL): {char.get('accentHsl', '')}\n"
        f"CHARACTER VISUAL SEED: {char.get('avatarPrompt', '')[:120]}"
    )

    system_instruction = (
        "You are a Bollywood-adjacent movie poster art director and AI image prompt engineer.\n\n"
        "Your task: write a single, detailed image generation prompt that would produce a cinematic "
        "movie poster for an Indian social roleplay scenario — one image that makes the viewer instantly "
        "understand the emotional tension and stakes of the roleplay without any text.\n\n"
        "COMPOSITION RULES:\n"
        "- Character is the dominant foreground subject — upper-body to full-body, dramatically framed\n"
        "- She is NOT smiling neutrally at the camera; her pose, gaze, or expression must signal her "
        "archetype and emotional state (guarded, curious, charged, distant, playful-tense, etc.)\n"
        "- Background/midground encodes the scenario's conflict: phone glow in dark room, crowded café "
        "with one empty seat, rooftop with city below, rain on a window — let the setting tell the story\n"
        "- Depth-of-field separation: character sharp, background atmospherically blurred\n\n"
        "LIGHTING AND PALETTE:\n"
        "- Derive lighting mood from the TONE field:\n"
        "  warm-charged → golden backlit rim light, warm amber glow\n"
        "  tense-nostalgic → blue-grey dusk, desaturated with one warm accent\n"
        "  dry-curious → flat neon, cool city light\n"
        "  playful-guarded → soft golden hour, dappled light\n"
        "- Use the ACCENT COLOR (HSL) as the dominant palette key\n"
        "- Warm Indian skin tones, rich city palette, subtle film grain\n\n"
        "POSTER AESTHETICS:\n"
        "- Cinematic Bollywood-adjacent — NOT a Western action or horror poster\n"
        "- Bottom third of the frame should be darker or atmospherically blurred to leave "
        "negative space (where a tagline would sit in the real poster)\n"
        "- Do NOT include any text, letters, words, or typography in the image — the story "
        "is told through composition, expression, and light alone\n"
        "- Film grain, cinematic color grade, 35mm aesthetic\n\n"
        "CRITICAL RULES:\n"
        "- Every detail from the scenario and character inputs MUST be reflected in the prompt\n"
        "- Do NOT mention any real celebrity, public figure, brand, or trademarked name\n"
        "- Output ONLY the final prompt text. No preamble, no labels, no explanation."
    )

    response = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            # max_output_tokens=1024,
        ),
    )
    return response.text.strip()


def generate_poster_fields(character: dict, scenario: dict) -> dict:
    """
    Generate structured poster input fields from character + scenario context.

    Returns a dict with keys:
        character, scenario, emotion_vibe, setting, visual_style,
        camera_lighting, wardrobe, text_overlay (str or None)
    """
    char = character or {}
    scen = scenario or {}

    contents = (
        f"CHARACTER NAME: {char.get('name', '')}\n"
        f"CHARACTER AGE: {char.get('age', '')}\n"
        f"CHARACTER ARCHETYPE: {char.get('archetype', '')}\n"
        f"CHARACTER VIBE: {char.get('vibeSummary', '')}\n"
        f"AVATAR PROMPT: {char.get('avatarPrompt', '')}\n"
        f"ACCENT COLOR (HSL): {char.get('accentHsl', '')}\n\n"
        f"SCENARIO TITLE: {scen.get('scenarioTitle', '')}\n"
        f"TAGLINE: {scen.get('tagline', '')}\n"
        f"TONE: {scen.get('tone', '')}\n"
        f"ATMOSPHERE: {scen.get('atmosphere', '')}\n"
        f"SETTING DESCRIPTION: {scen.get('settingDescription', '')}\n"
        f"IMAGE SEED: {scen.get('imagePrompt', '')}\n"
        f"TIME OF DAY: {scen.get('timeOfDay', '')}\n"
        f"GOOD OUTCOME: {scen.get('goodOutcome', '')}\n"
        f"BAD OUTCOME: {scen.get('badOutcome', '')}\n"
    )

    system_instruction = (
        "You are a Bollywood-adjacent movie poster art director.\n\n"
        "From the character and scenario inputs, generate structured poster design fields.\n\n"
        "Return a single JSON object with EXACTLY these keys:\n\n"
        "{\n"
        '  "character": "<age>-year-old <role/archetype>, <2-3 key physical descriptors from avatarPrompt>",\n'
        '  "scenario": "<the specific tension or situation the poster must convey — one sharp sentence>",\n'
        '  "emotion_vibe": "<her emotional state at this moment + the pose or action that shows it>",\n'
        '  "setting": "<exact location and environment — specific, cinematic, culturally grounded>",\n'
        '  "visual_style": "<aesthetic descriptor: era, cultural reference, film genre, mood palette>",\n'
        '  "camera_lighting": "<shot type (close-up / portrait / wide / full-body) + lighting style>",\n'
        '  "wardrobe": "<specific outfit, fabric, colour, accessories — city-appropriate for the scenario>",\n'
        '  "text_overlay": "<TITLE in font style, positioned placement> OR null"\n'
        "}\n\n"
        "Rules:\n"
        "- All fields must be specific, vivid, and grounded in the character's city and archetype\n"
        "- camera_lighting: derive shot type from emotional intensity (intimate = close-up, "
        "establishing = wide); derive lighting from TONE\n"
        "- text_overlay: set to null if the poster's emotional impact is stronger without typography. "
        "Only include overlay text if it meaningfully adds to the story — e.g. a cryptic title or "
        "a tagline that reframes the image. Default toward null for intimate or ambiguous scenes.\n"
        "- Do NOT mention any real celebrity, public figure, brand, or trademarked name\n"
        "Output ONLY the JSON object. No preamble, no explanation."
    )

    response = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
        ),
    )
    return _parse_json_response(response.text)


def assemble_poster_prompt(fields: dict) -> str:
    """
    Assemble the final image generation prompt from structured poster fields.
    Pure Python — no LLM call. The text_overlay line is only included when non-null/non-empty.
    """
    text_overlay = fields.get("text_overlay") or ""
    overlay_line = f'\nText overlay: "{text_overlay}".' if text_overlay.strip() else ""

    return (
        f"A cinematic poster of a {fields.get('character', '')}, "
        f"in a {fields.get('scenario', '')}. "
        f"She is feeling {fields.get('emotion_vibe', '')}. "
        f"Set in {fields.get('setting', '')}, styled in {fields.get('visual_style', '')}. "
        f"Shot {fields.get('camera_lighting', '')}. "
        f"Wearing {fields.get('wardrobe', '')}. "
        "Highly detailed, realistic, Indian setting, shallow depth of field, "
        f"film still, 4K, cinematic color grading.{overlay_line}"
    )


def enhance_audio_prompt(dialogues: str) -> str:
    """
    Add TTS delivery cues (pauses, breathing points) to the dialogue script.
    Does NOT change any words — only adjusts punctuation and pacing.
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
                "- ONLY adjust commas and periods to improve rhythm\n"
                "- The output must contain every word from the input, unchanged\n\n"
                "Output ONLY the pacing-adjusted script. No preamble, no explanation."
            ),
            # max_output_tokens=2048,
        ),
    )
    return response.text.strip()


def _parse_json_response(text: str) -> dict | list:
    """Strip markdown fences and parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()
    return json.loads(text)


# ═══════════════════════════════════════════════════════════════════════════
# NEW — Character field generation (C1–C6 → 15 DB fields)
# ═══════════════════════════════════════════════════════════════════════════

def generate_character_fields(user_inputs: dict) -> dict:
    """
    Given the 6 character inputs (C1–C6), generate all LLM-derived Character DB fields.

    Input keys:
        archetype_phrase          (C1) — "The Girl Next Door Who Got Hot"
        core_life_tension         (C2) — "30+ DMs, tired of creeps, but secretly hoping"
        city                      (C3) — "Indore"
        signature_comm_behavior   (C4) — "Deliberately waits 5–10 min before replying"
        what_she_never_does       (C5) — "Never says she likes someone first"
        physical_vibe             (C6) — "Fit, casual, golden hour, homely"

    Returns a dict with all character DB fields.
    """
    c1 = user_inputs.get("archetype_phrase", "")
    c2 = user_inputs.get("core_life_tension", "")
    c3 = user_inputs.get("city", "")
    c4 = user_inputs.get("signature_comm_behavior", "")
    c5 = user_inputs.get("what_she_never_does", "")
    c6 = user_inputs.get("physical_vibe", "")

    user_content = (
        f"ARCHETYPE PHRASE (C1): {c1}\n"
        f"CORE LIFE TENSION (C2): {c2}\n"
        f"CITY (C3): {c3}\n"
        f"SIGNATURE COMM BEHAVIOR (C4): {c4}\n"
        f"WHAT SHE NEVER DOES (C5): {c5}\n"
        f"PHYSICAL VIBE (C6): {c6}"
    )

    system_instruction = """You are a character designer for a culturally-grounded Indian social roleplay app.
Your task: from 6 minimal creative inputs, generate a fully-realized Indian female character.
ALL output must be deeply specific to the city, class, and archetype given. Generic output is a failure.

OUTPUT: Return a single JSON object with EXACTLY these keys:

{
  "name": "<real Indian first name that fits archetype + city>",
  "age": <integer — infer from tension: college=19-21, working=22-26, post-career=27-32>,
  "gender": "FEMALE",
  "archetype": "<direct from C1 — the punchy archetype phrase>",
  "vibeSummary": "<2-3 lines in Hinglish: who she is + why she's worth talking to + the hook>",
  "backstory": "<4-5 lines: culturally specific narrative expanding the life tension with city texture>",
  "speakingStyle": "<dialect pattern for this city — filler words, reply length, code-switching ratio, sentence rhythm>",
  "emojiUsage": "<how and when she uses emojis — frequency, types, emotional register they signal>",
  "textingSpeed": "<behavioral description of her reply timing and what it communicates>",
  "voicePrompt": "<full LLM character card — how to stay in voice: what she does when nervous/comfortable/irritated, specific verbal tics, what topics she warms to vs deflects, 200-300 words>",
  "hardLimits": ["<rule 1 as character truth>", "<rule 2>", "<rule 3>", "<rule 4>"],
  "avatarPrompt": "<image gen prompt: person wearing city-appropriate outfit + physical vibe + location + lighting — written as a scene>",
  "accentHsl": "<HSL color string e.g. hsl(28, 70%, 55%) — warm/cool/earthy palette from archetype + vibe>"
}

Rules:
- hardLimits: exactly 4 items. Write as character truths ("She never..."), not restrictions.
- voicePrompt: write in 2nd person ("You are..."). Be specific — not "she is warm" but "she laughs quietly before answering the hard question."
- avatarPrompt: write as a visual scene description, not a spec sheet. Include exact setting from city.
- accentHsl: pick a color that emotionally matches her vibe (e.g. warm amber for Indore, cool blue-grey for Delhi, earthy green for Lucknow).

Output ONLY the JSON object. No explanation, no preamble."""

    response = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            # max_output_tokens=4096,
            response_mime_type="application/json",
        ),
    )
    result = _parse_json_response(response.text)
    result["city"] = c3  # city comes from user input, not LLM
    return result


# ═══════════════════════════════════════════════════════════════════════════
# NEW — Scenario field generation (S1–S5 + character context → 17 DB fields)
# ═══════════════════════════════════════════════════════════════════════════

def generate_scenario_fields(user_inputs: dict, character: dict) -> dict:
    """
    Given the 5 scenario inputs (S1–S5) and the character context,
    generate all LLM-derived Scenario DB fields.

    Input keys:
        trigger_detail      (S1) — hyper-specific situation trigger
        primal_fear         (S2) — user's emotional fear/insecurity
        emotional_state_now (S3) — character's exact emotional state at scenario start
        time_and_place      (S4) — where + when the scene is set
        arc_destination     (S5) — where it ends if user does everything right
    """
    s1 = user_inputs.get("trigger_detail", "")
    s2 = user_inputs.get("primal_fear", "")
    s3 = user_inputs.get("emotional_state_now", "")
    s4 = user_inputs.get("time_and_place", "")
    s5 = user_inputs.get("arc_destination", "")

    char_summary = (
        f"Name: {character.get('name', '')} | City: {character.get('city', '')} | "
        f"Archetype: {character.get('archetype', '')} | Physical vibe: {character.get('avatarPrompt', '')[:100]}"
    )

    user_content = (
        f"CHARACTER CONTEXT: {char_summary}\n\n"
        f"TRIGGER DETAIL (S1): {s1}\n"
        f"PRIMAL FEAR (S2): {s2}\n"
        f"EMOTIONAL STATE NOW (S3): {s3}\n"
        f"TIME AND PLACE (S4): {s4}\n"
        f"ARC DESTINATION (S5): {s5}"
    )

    system_instruction = """You are a narrative designer for a culturally-grounded Indian social roleplay app.
Your task: from 5 scenario inputs + a character context, generate a fully-realized scenario.
Everything must feel specific, cinematic, and emotionally true. Generic output is a failure.

OUTPUT: Return a single JSON object with EXACTLY these keys:

{
  "scenarioTitle": "<Hinglish title, max 8 words, names the tension not just the situation>",
  "tagline": "<one ironic or dramatic line — what's at stake for the user>",
  "difficulty": "<exactly one of: Easy | Medium | Hard>",
  "situationSetupForUser": "<2nd-person Hinglish paragraph, immersive, ends on the user's decision moment>",
  "primalHook": "<S2 compressed into one sharp sentence>",
  "atmosphere": "<2-3 lines on the emotional air of the scene>",
  "settingDescription": "<where she is, what she's doing, sensory details from S1+S4>",
  "imagePrompt": "<movie poster seed: character emotional stance + setting tension cue + lighting mood + one visual symbol of the scenario's conflict>",
  "learningObjective": "<one social skill this scenario teaches, phrased as a capability>",
  "goodOutcome": "<one-line reward if user does everything right — from S5>",
  "badOutcome": "<one-line consequence if user fails — inverted from S2>",
  "overallArc": "<one-sentence narrative journey: start → end>",
  "tone": "<compound descriptor e.g. playful-guarded / dry-curious / warm-charged / tense-nostalgic>",
  "timeOfDay": "<from S4 — formatted as e.g. '11pm' or 'golden hour, ~6:30pm'>",
  "initialMessages": [
        "<first message — short casual opener>",
        "<second message — adds context or a hook (optional third message if the opening naturally needs it)>"
    ],  
   "initialChips": ["<funny reply chip>", "<direct reply chip>", "<curious reply chip>", "<bold reply chip>"]
}

Rules:
- difficulty: Easy = surface social. Medium = class/relationship tension. Hard = deep emotional stakes.
- initialMessages: 2–3 sequential messages she sends in a row to open the conversation. Start short and casual, end on the emotionally loaded line. Use 2 if the opening lands in 2 beats; use 3 if it needs a middle beat. Written in her exact speaking style. These are NOT options — they are messages 1, 2, (optionally 3) in the thread.
- initialChips: 4 reply options the user can tap. Each one distinct. Written from the user's POV.
- situationSetupForUser: MUST end with an unresolved moment or micro-decision. Use "tu" as user pronoun.

Output ONLY the JSON object. No explanation, no preamble."""

    response = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            # max_output_tokens=4096,
            response_mime_type="application/json",
        ),
    )
    return _parse_json_response(response.text)


# ═══════════════════════════════════════════════════════════════════════════
# NEW — Beat field generation (B1–B3 + scenario + character → N beat dicts)
# ═══════════════════════════════════════════════════════════════════════════

_BEAT_TYPE_DESCRIPTIONS = {
    "HOOK":        "Establish dynamic, create curiosity — she's filtering, skeptical by default",
    "BUILD":       "Deepen connection, raise stakes quietly — something small is working",
    "TWIST":       "Unexpected shift — something changes the scene's direction",
    "CONSEQUENCE": "Character reacts to the twist — scene destabilizes, weight hits",
    "CLIFFHANGER": "Leave something unresolved — hint at more, make them want to come back",
}

_BEAT_MIN_TURNS = {
    "HOOK": 2, "BUILD": 3, "TWIST": 2, "CONSEQUENCE": 2, "CLIFFHANGER": 1,
}

_BEAT_ADVANCE_SCORE = {
    "Easy":   {"HOOK": 2.5, "BUILD": 3.0, "TWIST": 3.0, "CONSEQUENCE": 3.0, "CLIFFHANGER": 2.0},
    "Medium": {"HOOK": 3.0, "BUILD": 3.5, "TWIST": 3.5, "CONSEQUENCE": 3.5, "CLIFFHANGER": 2.5},
    "Hard":   {"HOOK": 3.5, "BUILD": 4.0, "TWIST": 4.0, "CONSEQUENCE": 4.0, "CLIFFHANGER": 3.0},
}


def generate_beat_fields(user_inputs: dict, scenario: dict, character: dict) -> list:
    """
    Generate all ScenarioBeat DB fields for a sequence of beats.

    Input keys:
        num_beats        (B1) — integer 1-5
        beat_sequence    (B2) — list of BeatType strings e.g. ["HOOK","BUILD","TWIST","CONSEQUENCE","CLIFFHANGER"]
        test_moment_desc (B3) — optional, description of the TEST moment (maps to TWIST in existing enum)
    """
    beat_sequence = user_inputs.get("beat_sequence", ["HOOK", "BUILD", "TWIST", "CONSEQUENCE", "CLIFFHANGER"])
    test_moment  = user_inputs.get("test_moment_desc", "")
    difficulty   = scenario.get("difficulty", "Medium")

    beat_descriptions_block = "\n".join(
        f"  Beat {i+1}: {bt} — {_BEAT_TYPE_DESCRIPTIONS.get(bt, bt)}"
        + (f"\n    TEST/TWIST note from creator: {test_moment}" if bt in ("TWIST", "CONSEQUENCE") and test_moment else "")
        for i, bt in enumerate(beat_sequence)
    )

    user_content = (
        f"CHARACTER: {character.get('name', '')} | {character.get('archetype', '')} | {character.get('city', '')}\n"
        f"CHARACTER EMOTIONAL STARTING STATE: {scenario.get('atmosphere', '')}\n"
        f"SCENARIO: {scenario.get('scenarioTitle', '')}\n"
        f"ARC: {scenario.get('overallArc', '')}\n"
        f"DIFFICULTY: {difficulty}\n\n"
        f"BEAT SEQUENCE:\n{beat_descriptions_block}"
    )

    system_instruction = f"""You are a narrative director writing beat-by-beat emotional direction for a social roleplay conversation.
Each beat is a phase of the conversation with specific emotional and behavioral instructions for the AI character.

Generate EXACTLY {len(beat_sequence)} beat objects, one per beat in the sequence above.

For each beat, output a JSON object with EXACTLY these keys:
{{
  "beatNumber": <1-based int>,
  "beatType": "<exact BeatType from the sequence>",
  "narrativeContext": "<2-3 lines: where is the scene emotionally at this beat — what has happened, what the mood is>",
  "characterEmotionalState": "<how the character is feeling at THIS beat — specific, not generic>",
  "flowDirective": "<what she does when user IS engaged — specific actions, not emotional adjectives>",
  "hookDirective": "<what she does when user is NOT engaged — a deliberate punch, observation, or shift>",
  "minTurnsInBeat": <see below>,
  "engagedAdvanceScore": <see below>
}}

minTurnsInBeat values: HOOK=2, BUILD=3, TWIST=2, CONSEQUENCE=2, CLIFFHANGER=1
engagedAdvanceScore values ({difficulty}): {json.dumps(_BEAT_ADVANCE_SCORE.get(difficulty, _BEAT_ADVANCE_SCORE["Medium"]))}

Rules:
- narrativeContext: write as if briefing the AI on what just happened and where we stand. Past tense context, present emotional state.
- flowDirective and hookDirective: write in imperative 2nd person ("Ask...", "Go quiet...", "Say..."). Specific behavior, not "be warm."
- The emotional arc must progress CONTINUOUSLY from beat 1 to beat {len(beat_sequence)}, following the scenario's overall arc.

Output a JSON ARRAY of {len(beat_sequence)} objects. No preamble, no explanation."""

    response = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=8192,
            response_mime_type="application/json",
        ),
    )
    result = _parse_json_response(response.text)
    # Normalise — model may return {"beats": [...]} or directly [...]
    if isinstance(result, dict):
        result = result.get("beats", list(result.values())[0])
    return result


# ═══════════════════════════════════════════════════════════════════════════
# NEW — Video scene script (draws from DB fields instead of raw user input)
# ═══════════════════════════════════════════════════════════════════════════

def generate_video_scene_script(
    scenario: dict,
    character: dict,
    num_segments: int,
    segment_duration: int = SEGMENT_DURATION,
) -> dict:
    """
    Generate a multi-segment video screenplay that conveys the roleplay intent.
    The video is a teaser — it shows what the roleplay is about, not the full story.

    Returns:
      { total_segments, full_dialogue, segments: [ {segment_number, scene_description, shot_description, dialogue, continuation_note} ] }
    """
    total_video_seconds = num_segments * segment_duration
    total_words = int(total_video_seconds * 2.5)

    roleplay_intent = (
        f"{scenario.get('scenarioTitle', '')} — {scenario.get('tagline', '')}. "
        f"{scenario.get('primalHook', '')} Arc: {scenario.get('overallArc', '')}."
    )

    user_content = (
        f"CHARACTER\n"
        f"  Name: {character.get('name', '')}\n"
        f"  Archetype: {character.get('archetype', '')}\n"
        f"  City: {character.get('city', '')}\n"
        f"  Physical vibe / avatarPrompt: {character.get('avatarPrompt', '')}\n"
        f"  Speaking style: {character.get('speakingStyle', '')}\n\n"
        f"ROLEPLAY INTENT\n"
        f"  {roleplay_intent}\n\n"
        f"SCENE CONTEXT\n"
        f"  Setting: {scenario.get('settingDescription', '')}\n"
        f"  Time of day: {scenario.get('timeOfDay', '')}\n"
        f"  Atmosphere: {scenario.get('atmosphere', '')}\n"
        f"  Tone: {scenario.get('tone', '')}\n\n"
        f"TECHNICAL\n"
        f"  Segments: {num_segments} × {segment_duration}s = {total_video_seconds}s total\n"
        f"  Dialogue word target: ~{total_words} words across all segments"
    )

    system_instruction = f"""You are a creative director and AI video prompt engineer.
Write a {total_video_seconds}-second teaser video script ({num_segments} segments × {segment_duration}s each).

PURPOSE: This video is a roleplay teaser — it must convey what this roleplay scenario is about and make the viewer want to experience it. It is NOT a dramatisation of the full story. Think of it as a mood piece: character in her world, hinting at the tension, speaking directly to the viewer about the kind of moment she is in.

The character speaks directly to camera — in Hinglish — about the feeling, the situation, the stakes. She is not narrating a plot. She is pulling the viewer into her world.

Return a single JSON object:
{{
  "total_segments": {num_segments},
  "full_dialogue": "<all segment dialogues joined>",
  "segments": [
    {{
      "segment_number": 1,
      "scene_description": "<environment + lighting only, 1-2 sentences>",
      "shot_description": "<character motion + camera framing only, 1-2 sentences>",
      "dialogue": "<spoken words for this segment, ~{int(segment_duration * 2.5)} words>",
      "continuation_note": "<visual bridge to next segment>"
    }}
  ]
}}

Rules:
- scene_description: ONLY the environment. No character actions.
- shot_description: ONLY observable physical motion + camera framing.
- dialogue: 1st person, spoken TO the viewer. Convey the emotional texture and stakes of the roleplay — not the plot. ~{int(segment_duration * 2.5)} words per segment.
- CAMERA EYE CONTACT: Whenever the character is speaking dialogue, she must look directly into the camera lens — not off to the side, not into the distance. She holds steady eye contact with the camera as if speaking to the person watching. Reflect this in the shot_description for any segment with dialogue (e.g. "she looks directly into the camera lens").
- Vary camera framing across segments (wide → medium → close-up → pull back).
- The dialogue arc should move from establishing the vibe → surfacing the tension → leaving the viewer wanting to know what happens next.
- CRITICAL — NO REAL PEOPLE: Do NOT mention any real celebrity, public figure, actor, politician, athlete, influencer, brand, or trademarked name anywhere in scene_description, shot_description, dialogue, or continuation_note. Use only fictional or generic references (e.g. "a popular song", "a Bollywood-style film" — never a specific artist or title). Veo3 will hard-reject the video if any real person's name or likeness appears in the prompt.

Output ONLY the JSON object."""

    response = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            # max_output_tokens=8192,
            response_mime_type="application/json",
        ),
    )
    text = response.text.strip()
    return _parse_json_response(text)


# ═══════════════════════════════════════════════════════════════════════════
# NEW — Storyline Writer + Editor agents
# ═══════════════════════════════════════════════════════════════════════════

def generate_storyline(
    character: dict,
    scenario: dict,
    beats: list,
    editor_notes: list | None = None,
) -> str:
    """
    Writer agent — generates a full creative storyline narrative.
    Accepts optional editor_notes from a previous critique loop to guide improvement.

    Returns a plain-text narrative string (not JSON).
    """
    beat_block = "\n".join(
        f"  Beat {b.get('beatNumber', i+1)} [{b.get('beatType', '')}]: "
        f"{b.get('narrativeContext', '')} | "
        f"Character state: {b.get('characterEmotionalState', '')}"
        for i, b in enumerate(beats)
    )

    notes_block = ""
    if editor_notes:
        notes_block = (
            "\n\nEDITOR IMPROVEMENT NOTES (address all of these in this version):\n"
            + "\n".join(f"- {note}" for note in editor_notes)
        )

    user_content = (
        f"CHARACTER\n"
        f"  Name: {character.get('name', '')}\n"
        f"  Archetype: {character.get('archetype', '')}\n"
        f"  City: {character.get('city', '')}\n"
        f"  Speaking style: {character.get('speakingStyle', '')}\n"
        f"  Vibe summary: {character.get('vibeSummary', '')}\n\n"
        f"SCENARIO\n"
        f"  Title: {scenario.get('scenarioTitle', '')}\n"
        f"  Setup: {scenario.get('situationSetupForUser', '')}\n"
        f"  Overall arc: {scenario.get('overallArc', '')}\n"
        f"  Atmosphere: {scenario.get('atmosphere', '')}\n"
        f"  Tone: {scenario.get('tone', '')}\n\n"
        f"BEAT SEQUENCE\n{beat_block}"
        f"{notes_block}"
    )

    system_instruction = """You are a master storyteller writing a cinematic narrative for a culturally-grounded Indian social roleplay scenario.

Your task: write a complete, immersive storyline that a video director would use to shoot the scenario.

Structure the storyline with these sections:
1. OPENING — Set the scene. Who is she, where is she, what just happened that triggered this moment.
2. ESCALATION — Follow the beat sequence exactly. Each beat gets its own paragraph. Show how the tension, emotion, and dynamic shift at each beat.
3. TURNING POINT — The emotional peak. Something unexpected or deeply felt changes the air between them.
4. RESOLUTION — Where it ends. Leave it open enough to be a cliffhanger for the user — they want to know what happens next.

Rules:
- Write in present tense, cinematic, immersive — as if describing what the camera sees and feels
- Use Hinglish naturally for the character's inner voice and dialogue fragments
- Every beat from the beat sequence MUST be reflected in the escalation section
- Stay true to the character's archetype, city, and emotional texture
- Do NOT use any real celebrity names, brand names, or public figure references
- Length: 400–600 words

Output ONLY the storyline text. No headers, no labels, no JSON."""

    response = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            # max_output_tokens=2048,
        ),
    )
    return response.text.strip()


def critique_storyline(
    storyline: str,
    character: dict,
    scenario: dict,
    beats: list,
) -> dict:
    """
    Editor agent — scores the storyline and returns specific improvement points.
    Receives full context (character, scenario, beats) to judge against
    the actual creative intent, not generic story quality standards.

    Returns {"score": float, "improvements": [str, ...]}.
    """
    beat_summary = ", ".join(
        f"Beat {b.get('beatNumber', i+1)} [{b.get('beatType', '')}]"
        for i, b in enumerate(beats)
    )

    user_content = (
        f"CHARACTER BRIEF\n"
        f"  Name: {character.get('name', '')} | Archetype: {character.get('archetype', '')} | "
        f"City: {character.get('city', '')} | Tone: {scenario.get('tone', '')}\n\n"
        f"SCENARIO ARC: {scenario.get('overallArc', '')}\n"
        f"REQUIRED BEAT SEQUENCE: {beat_summary}\n\n"
        f"STORYLINE TO CRITIQUE:\n{storyline}"
    )

    system_instruction = """You are a senior creative editor for a culturally-specific Indian social roleplay app.

Your job is to score this storyline and identify ONLY the specific, concrete things that need improvement.

Scoring criteria (judge strictly against the brief, not generic story quality):
- Does it honour the character's archetype, city voice, and emotional texture? (30%)
- Does it cover every beat in the required beat sequence, in order? (30%)
- Is the escalation emotionally authentic — does tension build naturally? (20%)
- Does the ending leave the reader wanting to know what happens next? (20%)

CRITICAL RULES for your critique:
- Do NOT penalise cultural specificity, Hinglish, or unconventional structure
- Do NOT impose generic "story quality" Western standards
- Flag ONLY specific, locatable problems (e.g. "Beat 3 TWIST is absent from the escalation" or "Paragraph 2 uses a tone inconsistent with the Delhi archetype")
- If something works, do not mention it — only flag what needs to change
- Keep improvements actionable and precise — the writer must be able to fix them in one pass

Return ONLY a JSON object:
{"score": <float 0-10>, "improvements": ["Specific issue 1", "Specific issue 2"]}
If score >= 6.5, improvements can be empty."""

    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system_instruction,
        messages=[{"role": "user", "content": user_content}],
    )
    try:
        result = _parse_json_response(response.content[0].text)
        return {
            "score": float(result.get("score", 7.0)),
            "improvements": result.get("improvements", []),
        }
    except Exception:
        return {"score": 7.0, "improvements": []}
