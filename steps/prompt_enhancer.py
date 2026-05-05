"""
LLM generation functions for character, scenario, and beat field derivation.
All LLM calls go through utils.llm_client (OpenRouter).
Writer/prompt agents use Gemini; critique/editor agents use Claude.
"""

import json
from utils import llm_client
from config import SEGMENT_DURATION


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
    character portrait prompt behaviour identical to the original function.

    Poster mode (scenario provided): uses the full scenario + character context
    to produce a cinematic movie poster prompt for the scenario's roleplay.
    """
    if scenario is None:
        system = (
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
                    "- Camera: -Camera: shot on iPhone 14, slightly soft lens, natural colors," 
                    "no professional lighting setup, candid feel, slight digital noise\n\n"
                    "The image will be the starting frame for a video — ensure breathing room around the character. "
                    "Output ONLY the final prompt text. No preamble, no labels, no explanation."
                )
        return llm_client.complete(system=system, user=avatar_prompt, model="gemini")

    char = character or {}
    user_content = (
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

    system = (
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
    return llm_client.complete(system=system, user=user_content, model="gemini")


def generate_poster_fields(character: dict, scenario: dict) -> dict:
    """
    Generate structured poster input fields from character + scenario context.
    """
    char = character or {}
    scen = scenario or {}

    user_content = (
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

    system = (
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
    result = llm_client.complete(system=system, user=user_content, model="gemini")
    return _parse_json_response(result)


def assemble_poster_prompt(fields: dict) -> str:
    """
    Assemble the final image generation prompt from structured poster fields.
    Pure Python — no LLM call.
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
    system = (
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
    )
    return llm_client.complete(system=system, user=dialogues, model="gemini")


# ═══════════════════════════════════════════════════════════════════════════
# NEW — Character field generation (C1–C6 → 15 DB fields)
# ═══════════════════════════════════════════════════════════════════════════

def generate_character_fields(user_inputs: dict) -> dict:
    """
    Given the 6 character inputs (C1–C6), generate all LLM-derived Character DB fields.
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

    system = """You are a character designer for a culturally-grounded Indian social roleplay app.
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
  "voicePrompt": "<full LLM character card written entirely in 2nd person — start with 'You are [Name]...' then describe: what you do when nervous/comfortable/irritated, your specific verbal tics, topics you warm to vs deflect. Write as if instructing the AI to BE this character. Never use she/her. 200-300 words>",
  "hardLimits": ["<rule 1 as character truth>", "<rule 2>", "<rule 3>", "<rule 4>"],
  "avatarPrompt": "<image gen prompt: person wearing city-appropriate outfit + physical vibe + location + lighting — written as a scene>",
  "accentHsl": "<HSL color string e.g. hsl(28, 70%, 55%) — warm/cool/earthy palette from archetype + vibe>"
}

Rules:
- hardLimits: exactly 4 items. Write as character truths ("She never..."), not restrictions.
- voicePrompt: write entirely in 2nd person. Start with "You are [Name]..." — NEVER use she/her anywhere in this field. Every sentence must address the AI directly: "You laugh quietly before answering the hard question", not "she laughs quietly."
- avatarPrompt: write as a visual scene description, not a spec sheet. Include exact setting from city.
- accentHsl: pick a color that emotionally matches her vibe (e.g. warm amber for Indore, cool blue-grey for Delhi, earthy green for Lucknow).

Output ONLY the JSON object. No explanation, no preamble."""

    result = llm_client.complete(system=system, user=user_content, model="gemini", max_tokens=4096)
    parsed = _parse_json_response(result)
    parsed["city"] = c3
    return parsed


# ═══════════════════════════════════════════════════════════════════════════
# NEW — Scenario field generation (S1–S5 + character context → 17 DB fields)
# ═══════════════════════════════════════════════════════════════════════════

def generate_scenario_fields(user_inputs: dict, character: dict) -> dict:
    """
    Given the 5 scenario inputs (S1–S5) and the character context,
    generate all LLM-derived Scenario DB fields.
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

    system = """You are a narrative designer for a culturally-grounded Indian social roleplay app.
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
  ""initialMessages": [
    "<Character's first line of dialogue — what SHE says to open the scene. Written in first person from her POV. Natural, in-character, not instructional.>",
    "<Her follow-up line — continues her thought, reveals her mood or something about the situation. Still her voice, not a prompt to the user.>"
],
   "initialChips": ["<funny reply chip>", "<direct reply chip>", "<curious reply chip>", "<bold reply chip>"]
}

Rules:
- difficulty: Easy = surface social. Medium = class/relationship tension. Hard = deep emotional stakes.
- initialMessages: 2–3 sequential messages she (the character AI will play) sends in a row to open the conversation.
- initialChips: 4 reply options the user can tap. Each one distinct. Written from the user's POV.
- situationSetupForUser: MUST end with an unresolved moment or micro-decision. Use "tu" as user pronoun.

Output ONLY the JSON object. No explanation, no preamble."""

    result = llm_client.complete(system=system, user=user_content, model="gemini", max_tokens=4096)
    return _parse_json_response(result)


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


def generate_beat_sequence(scenario: dict, character: dict, test_moment_hint: str = "") -> dict:
    """
    First LLM pass: designs the optimal beat sequence for a scenario.
    Returns {"num_beats": int, "beat_sequence": [str], "test_moment_desc": str, "reasoning": str}
    """
    hint_line = f"\nCREATOR HINT (twist/test moment to include): {test_moment_hint}" if test_moment_hint.strip() else ""

    user_content = (
        f"CHARACTER: {character.get('name', '')} | {character.get('archetype', '')} | {character.get('city', '')}\n"
        f"SCENARIO TITLE: {scenario.get('scenarioTitle', '')}\n"
        f"OVERALL ARC: {scenario.get('overallArc', '')}\n"
        f"DIFFICULTY: {scenario.get('difficulty', 'Medium')}\n"
        f"TONE: {scenario.get('tone', '')}\n"
        f"ATMOSPHERE: {scenario.get('atmosphere', '')}\n"
        f"GOOD OUTCOME: {scenario.get('goodOutcome', '')}\n"
        f"BAD OUTCOME: {scenario.get('badOutcome', '')}"
        f"{hint_line}"
    )

    beat_catalog = "\n".join(
        f"  {bt}: {desc}" for bt, desc in _BEAT_TYPE_DESCRIPTIONS.items()
    )

    system = f"""You are a narrative architect designing the beat structure for an Indian social roleplay scenario.

Available beat types:
{beat_catalog}

Your task: choose the OPTIMAL sequence of beats (2–5 beats) that will create the most emotionally resonant arc for this specific scenario. Not every scenario needs all 5 beats. A tight 3-beat sequence can outperform a padded 5-beat one.

Criteria for a good sequence:
- Matches the scenario's difficulty and emotional tone
- Each beat serves a distinct narrative purpose — no filler
- The arc has a clear beginning tension, middle escalation, and unresolved end
- TWIST and CONSEQUENCE should only appear if the scenario has a genuine reversal moment
- CLIFFHANGER is almost always the right final beat for a roleplay scenario

Return a single JSON object:
{{
  "num_beats": <integer 2-5>,
  "beat_sequence": ["<BeatType>", ...],
  "test_moment_desc": "<if a TWIST beat is included: exactly what the character does to test the user at that moment. Empty string if no TWIST.>",
  "reasoning": "<2-3 sentences: why this sequence fits this specific scenario's arc and difficulty>"
}}

Output ONLY the JSON object. No preamble, no explanation."""

    result = llm_client.complete(system=system, user=user_content, model="gemini", max_tokens=1024)
    return _parse_json_response(result)


def generate_beat_fields(beat_sequence: list, scenario: dict, character: dict, test_moment_desc: str = "") -> list:
    """
    Second LLM pass: generates all ScenarioBeat DB fields for an approved beat sequence.
    The LLM owns minTurnsInBeat and engagedAdvanceScore — no hardcoded lookup tables.
    """
    difficulty = scenario.get("difficulty", "Medium")

    beat_descriptions_block = "\n".join(
        f"  Beat {i+1}: {bt} — {_BEAT_TYPE_DESCRIPTIONS.get(bt, bt)}"
        + (f"\n    TEST/TWIST note: {test_moment_desc}" if bt in ("TWIST", "CONSEQUENCE") and test_moment_desc else "")
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

    system = f"""You are a narrative director writing beat-by-beat emotional direction for a social roleplay conversation.
Each beat is a phase of the conversation with specific emotional and behavioral instructions for the AI character.

Generate EXACTLY {len(beat_sequence)} beat objects, one per beat in the sequence above.

For each beat, output a JSON object with EXACTLY these keys:
{{
  "beatNumber": <1-based int>,
  "beatType": "<exact BeatType from the sequence>",
  "narrativeContext": "<2-3 lines in 2nd person: where YOU are emotionally at this beat — what has just happened, what the mood is. Start with 'You have...' or 'You are...'. Never use the character name or she/her.>",
  "characterEmotionalState": "<how YOU are feeling at THIS beat — in 2nd person. Never use she/her.>",
  "flowDirective": "<what YOU do when user IS engaged — specific actions in imperative 2nd person, e.g. 'Ask...', 'Say...'. Not emotional adjectives.>",
  "hookDirective": "<what YOU do when user is NOT engaged — a deliberate punch, observation, or shift. Imperative 2nd person.>",
  "minTurnsInBeat": <integer 1-5 — how many turns the user must spend before this beat can advance. Judge by narrative weight: quick beats (HOOK, CLIFFHANGER) = 1-2; substantial beats (BUILD, TWIST, CONSEQUENCE) = 2-4. Higher = more time for the beat to land.>,
  "engagedAdvanceScore": <float 2.0-5.0 — engagement score threshold to advance to the next beat. Scale to difficulty ({difficulty}): Easy=2.5-3.5, Medium=3.0-4.0, Hard=3.5-4.5. Set higher for emotionally pivotal beats like TWIST and CONSEQUENCE.>
}}

Rules:
- narrativeContext: 2nd person, "You have just...", "You are now...". NEVER use the character name or she/her.
- characterEmotionalState: 2nd person — "You are feeling...". NEVER use she/her.
- flowDirective and hookDirective: imperative 2nd person ("Ask...", "Go quiet...", "Say..."). Specific behavior, not "be warm."
- minTurnsInBeat and engagedAdvanceScore: reason from the beat's narrative weight and the scenario difficulty — do NOT use generic defaults.
- The emotional arc must progress CONTINUOUSLY from beat 1 to beat {len(beat_sequence)}.

Output a JSON ARRAY of {len(beat_sequence)} objects. No preamble, no explanation."""

    result = llm_client.complete(system=system, user=user_content, model="gemini", max_tokens=8192)
    parsed = _parse_json_response(result)
    if isinstance(parsed, dict):
        parsed = parsed.get("beats", list(parsed.values())[0])
    max_turns = len(parsed)
    for beat in parsed:
        beat["maxTurnsInBeat"] = max_turns
    return parsed


# ═══════════════════════════════════════════════════════════════════════════
# NEW — Video scene script (draws from DB fields instead of raw user input)
# ═══════════════════════════════════════════════════════════════════════════

def generate_video_scene_script(
    scenario: dict,
    character: dict,
    num_segments: int,
    segment_duration: int = SEGMENT_DURATION,
    user_name: str = "",
) -> dict:
    """
    Generate a multi-segment video screenplay that conveys the roleplay intent.
    The character introduces herself in segment 1 and speaks directly to the viewer.
    """
    total_video_seconds = num_segments * segment_duration
    words_per_segment = int(segment_duration * 2.5)

    roleplay_intent = (
        f"{scenario.get('scenarioTitle', '')} — {scenario.get('tagline', '')}. "
        f"{scenario.get('primalHook', '')} Arc: {scenario.get('overallArc', '')}."
    )

    viewer_ref = user_name.strip() if user_name.strip() else "the viewer"
    viewer_address = f'"{user_name.strip()}"' if user_name.strip() else "the viewer by feel (no name needed)"

    user_content = (
        f"CHARACTER\n"
        f"  Name: {character.get('name', '')}\n"
        f"  Archetype: {character.get('archetype', '')}\n"
        f"  City: {character.get('city', '')}\n"
        f"  Physical vibe / avatarPrompt: {character.get('avatarPrompt', '')}\n"
        f"  Speaking style: {character.get('speakingStyle', '')}\n"
        f"  Backstory: {character.get('backstory', '')}\n\n"
        f"ROLEPLAY INTENT\n"
        f"  {roleplay_intent}\n\n"
        f"SCENE CONTEXT\n"
        f"  Setting: {scenario.get('settingDescription', '')}\n"
        f"  Time of day: {scenario.get('timeOfDay', '')}\n"
        f"  Atmosphere: {scenario.get('atmosphere', '')}\n"
        f"  Tone: {scenario.get('tone', '')}\n\n"
        f"TECHNICAL\n"
        f"  Segments: {num_segments} × {segment_duration}s = {total_video_seconds}s total\n"
        f"  Dialogue word target: ~{words_per_segment} words per segment"
    )

    system = f"""You are a creative director and AI video prompt engineer.
Write a {total_video_seconds}-second teaser video script ({num_segments} segments × {segment_duration}s each).

PURPOSE: This video introduces the AI character to a real person who will roleplay with her. The character speaks DIRECTLY to camera — pulling them into her world, making them feel seen, and ending on a hook that makes them want to talk back.

This is NOT a plot summary. It is the character being alive on screen — her personality, her world, her feelings — all directed at the viewer as if they are already in a conversation.

SEGMENT 1 — INTRODUCTION (mandatory):
The character must open with a warm, natural Hinglish self-introduction. She says her name, gives a quick glimpse of who she is and her world, and references how she and the viewer are connected (draw from the scenario context). She addresses the viewer as "aap". Keep it conversational and genuine — like the first message from someone who is genuinely excited to meet you. Example style (not to copy): "Hi, main [Name] hun... [something about herself and the connection]..."

SEGMENTS 2 to {num_segments - 1} — CHARACTER'S WORLD:
Each segment reveals a different facet of the character — her life, her feelings, what she cares about, what makes her laugh or worry. She keeps talking TO the viewer, second-person ("tum", "tumhara", "aapko"). She is not narrating a story — she is sharing herself.

SEGMENT {num_segments} — ENGAGEMENT HOOK (mandatory):
The final segment must end with an open question or emotional pull that directly invites the viewer to respond. She leans in, holds eye contact, and asks something genuine — something only THIS viewer can answer. Make it feel personal and slightly vulnerable.

Return a single JSON object:
{{
  "total_segments": {num_segments},
  "full_dialogue": "<all segment dialogues joined>",
  "segments": [
    {{
      "segment_number": 1,
      "scene_description": "<environment + lighting only, 1-2 sentences>",
      "shot_description": "<character motion + camera framing only, 1-2 sentences>",
      "objects": ["<2-4 concrete visible props that ground the scene, in Hinglish or English>"],
      "dialogue": "<spoken words for this segment, ~{words_per_segment} words, in Hinglish>",
      "continuation_note": "<visual bridge to next segment>"
    }}
  ]
}}

Rules:
- scene_description: ONLY the environment and lighting. No character actions.
- shot_description: ONLY observable physical motion + camera framing. For any segment with dialogue, include "she looks directly into the camera lens" — she must hold steady eye contact as if talking to the person watching.
- objects: 2–4 specific physical props visible in the frame. Make them evocative and culturally specific — e.g. "chai ki pyali", "khidki se baarish", "purani diary", "phone screen glow". These ground the scene visually and help the video model render a coherent world.
- dialogue: Hinglish (Hindi in Roman script mixed with English). 1st person. Spoken TO camera. ~{words_per_segment} words. Creative, warm, specific — not generic. Reveal the character through what she says.
- Vary camera framing across segments: wide → medium → close-up → pull back.
- CRITICAL — NO REAL PEOPLE: Do NOT mention any real celebrity, public figure, actor, politician, athlete, influencer, brand, or trademarked name anywhere.

Output ONLY the JSON object."""

    result = llm_client.complete(system=system, user=user_content, model="gemini", max_tokens=4096)
    return _parse_json_response(result.strip())


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

    system = """You are a master storyteller writing a cinematic narrative for a culturally-grounded Indian social roleplay scenario.

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

    return llm_client.complete(system=system, user=user_content, model="gemini", max_tokens=2048)


def critique_storyline(
    storyline: str,
    character: dict,
    scenario: dict,
    beats: list,
) -> dict:
    """
    Editor agent — scores the storyline and returns specific improvement points.
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

    system = """You are a senior creative editor for a culturally-specific Indian social roleplay app.

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

    try:
        result = llm_client.complete(system=system, user=user_content, model="claude", max_tokens=1024)
        parsed = _parse_json_response(result)
        return {
            "score": float(parsed.get("score", 7.0)),
            "improvements": parsed.get("improvements", []),
        }
    except Exception:
        return {"score": 7.0, "improvements": []}



def generate_audio_script(segments: list, character: dict) -> str:
    """
    Synthesize a polished, continuous voiceover script from screenplay segments.
    Returns plain text ready for ElevenLabs TTS.
    """
    
    dialogues = []
    for i, seg in enumerate(segments, 1):
        d = seg.get("dialogue", "").strip()
        if d:
            dialogues.append(f"[Segment {i}] {d}")

    combined_dialogue = "\n".join(dialogues)
    voice_style = character.get("voicePrompt", "") or character.get("voice_prompt", "")
    speaking_style = character.get("speakingStyle", "") or character.get("speaking_style", "")

    system = f"""You are a voice director preparing Indian short-form dialogue for ElevenLabs TTS.

Your task: take plain dialogue and enrich it with ElevenLabs expression tags that match the emotional subtext of each line — so the voice sounds alive, not flat.

VALID EXPRESSION TAGS (use ONLY these, exactly as written):
EXPRESSION TAGS FORMAT:
- Use square brackets with descriptive emotional/delivery directions
- Examples: [sighs], [nervous], [whispers], [laughing], [voice breaking], 
  [softly], [with disbelief], [picking up pace], [trailing off]
- Be descriptive — [with a tired laugh] works better than just [laughs]
- Tags are voice-dependent: keep them natural, not theatrical

HOW TO PLACE TAGS:
- Insert immediately before the word/phrase they emotionally color
- Example: "Main [sighs] kabhi driver banne ka socha hi nahi tha..."
- Example: "[nervous] Log kya sochenge, mera parivaar..."
- Use 1 tag per emotional beat — don't stack multiple tags
- Not every sentence needs a tag — silence is also expression

WHEN TO USE WHICH TAG:
- [sighs] → resignation, wistfulness, exhaling a hard truth
- [laughs] / [chuckles] → self-deprecating humor, nervous lightness
- [nervous] → vulnerability, second-guessing, fear of judgment  
- [whispers] → intimate confession, something just between you two
- [gasps] → realization, surprise landing mid-sentence
- [excited] → a spark of hope breaking through the doubt

PACING:
- Keep all existing commas, ellipses (...), em dashes — they're intentional
- Do NOT rewrite or paraphrase any line
- Preserve Hinglish exactly as written

Output ONLY the enriched script — no labels, no explanation."""

    user = (
        f"CHARACTER VOICE STYLE: {voice_style}\n"
        f"SPEAKING STYLE: {speaking_style}\n\n"
        f"SCREENPLAY DIALOGUE:\n{combined_dialogue}\n\n"
        "Write the continuous voiceover script:"
    )

    return llm_client.complete(system=system, user=user, model="claude", max_tokens=1024).strip()
