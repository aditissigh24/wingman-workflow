"""
Standalone paywall video generator.

Reads a raw marketing script, uses Gemini to break it into Veo3-friendly beats,
auto-generates an opening reference image, produces N chained 8-second Veo3 segments,
and stitches them into a single MP4.

Usage:
    python generate_paywall_video.py                        # uses built-in paywall script, auto-detects segments
    python generate_paywall_video.py --script-file my.txt   # override script from file
    python generate_paywall_video.py --segments 4           # generate 4 × 8s = 32s video
    python generate_paywall_video.py --image hero.jpg       # skip auto-gen, use this opening frame
    python generate_paywall_video.py --keep-segments        # retain intermediate .mp4s after stitching
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import uuid

# ---------------------------------------------------------------------------
# Path setup — allow running from any CWD
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("paywall_video")

# ---------------------------------------------------------------------------
# Default paywall script (baked in — override with --script-file)
# ---------------------------------------------------------------------------
PAYWALL_SCRIPT = """
Hey! ... मैं हूँ Srishti.
... और एक बात पूछूं?
वो लड़की... जो आपके दिमाग में बसी है...
उसके सामने जाके... सही बात बोल पाते हो?
या बाद में सोचते रह जाते हो...
"यार... ये बोलना चाहिए था।"
...
Wingman लाया है roleplaying stories...
but असली situations पर based.
First date की awkward silence...
उसका ex वाला सवाल...
group में उसको impress करना...
breakup के बाद वापस connect करना...
हर मुश्किल moment का practice...
बिना real life में embarrass हुए।
आप choose करते हो क्या बोलना है...
और देखते हो... कैसे react करती है।
हर choice... एक नया ending।
...
तो सोचना क्या?
अभी सिर्फ ₹2 में trial शुरू करो...
unlimited stories unlock करो...
और अगली बार जब वो सामने हो...
freeze मत होना।
नीचे Start Trial button दिख रहा है...
click करो... और आज से practice शुरू।
""".strip()

# ---------------------------------------------------------------------------
# Script parsing helpers — derive structure directly from the raw script so
# Gemini cannot silently merge or drop beats.
# ---------------------------------------------------------------------------

# Match patterns like [0–3 sec:, [3-10 sec:, [16–21 sec: (en-dash or hyphen, any spacing)
_BEAT_TIME_MARKER = re.compile(r"\[\s*\d+\s*[\-\u2013\u2014]\s*\d+\s*sec\s*:", re.IGNORECASE)
# Same prefix but capture the entire bracketed segment so we can pull the description out
_BEAT_BRACKET_FULL = re.compile(
    r"\[\s*\d+\s*[\-\u2013\u2014]\s*\d+\s*sec\s*:[^\]]+\]",
    re.IGNORECASE,
)


def count_beats_in_script(raw_script: str) -> int:
    """
    Count the number of explicit time-bracketed beats in the raw script.
    Looks for patterns like [0–3 sec:, [3-10 sec:, [16–21 sec: etc.
    Falls back to 4 (safe default for the built-in paywall script) if none found.
    """
    count = len(_BEAT_TIME_MARKER.findall(raw_script))
    if count < 2:
        log.warning(
            "Could not detect beat count from script time markers — defaulting to 4 segments."
        )
        return 4
    log.info("Detected %d beats from script time markers.", count)
    return count


def extract_scene_mandates(raw_script: str) -> list[str]:
    """
    Extract the scene direction text from each `[X–Y sec: Label – ...]` bracket
    in the raw script. These become locked scene constraints injected into the
    Gemini beat prompt so it cannot substitute its own locations.

    Example input bracket: "[3–10 sec: Immersive scenes – Coffee shop flirt, beach walk, party]"
    Example output entry : "Coffee shop flirt, beach walk, party"
    """
    mandates: list[str] = []
    for raw_bracket in _BEAT_BRACKET_FULL.findall(raw_script):
        # Strip "[X–Y sec: Label –" prefix; keep everything after the first em/en-dash
        # following the label. If no inner dash, fall back to stripping the time prefix.
        cleaned = re.sub(
            r"^\[\s*\d+\s*[\-\u2013\u2014]\s*\d+\s*sec\s*:\s*[^\u2013\u2014\-]*[\u2013\u2014\-]\s*",
            "",
            raw_bracket,
            count=1,
            flags=re.IGNORECASE,
        )
        if cleaned == raw_bracket:
            cleaned = re.sub(
                r"^\[\s*\d+\s*[\-\u2013\u2014]\s*\d+\s*sec\s*:\s*",
                "",
                raw_bracket,
                count=1,
                flags=re.IGNORECASE,
            )
        cleaned = cleaned.rstrip("]").strip()
        if cleaned:
            mandates.append(cleaned)
    return mandates


from config import OUTPUT_DIR
from utils import llm_client

# ---------------------------------------------------------------------------
# Beat enhancer
# ---------------------------------------------------------------------------

_BEAT_SYSTEM_PROMPT = """
You are a video production prompt engineer specialising in short-form mobile marketing
videos for Indian apps targeting the Veo3 AI video generator.

OUTPUT FORMAT — return ONLY valid JSON, no markdown fences, no explanation:
{
  "opening_image_prompt": "<detailed Gemini image generation prompt for the very first frame>",
  "character_anchor": "<Single exhaustive paragraph (100-120 words) locking the male character's exact appearance: age, precise skin tone (e.g. warm medium-wheatish), hair (length, style, colour), facial hair (stubble length, shape), clothing (garment name, exact colour like 'slate-blue linen shirt', fit), footwear if visible, build, height estimate, and one distinctive feature like a small mole or watch. This block is copy-pasted verbatim into every segment prompt and must never change.>",
  "beats": [
    {
      "beat_label": "<short label e.g. Hook>",
      "scene_description": "<physical environment: exact location, time of day, lighting quality, background elements — 2-3 sentences>",
      "character_action": "<what the character physically does second-by-second during this 8-second clip — be frame-level specific>",
      "shot_description": "<exact camera movement, lens feel, framing changes across the 8 seconds>",
      "exact_voiceover": "<Copy the female voiceover text VERBATIM word-for-word from the raw script for this beat's time window. Do NOT paraphrase, summarise, rewrite, translate, or shorten. If the script uses '…' keep it. If it uses Hinglish words keep them. The only allowed change is removing stage-direction prefixes like 'Female Voiceover (warm, smiling energy):' if present.>",
      "audio_prompt": "<Veo3 music + ambience + delivery instruction (the words themselves come from exact_voiceover — do NOT repeat them here). Describe music tempo / instruments / mood, ambient sounds for the location, voice quality and pacing, and final mix levels — see Audio Prompt Rules below.>",
      "continuation_note": "<one sentence for scene/narrative continuity from previous beat; empty string for beat 1>"
    }
  ]
}

GLOBAL RULES:
- Always produce exactly {num_segments} beats.
- Do NOT include celebrity names, brand names (except "Wingman" — the app name is fine), or explicit content.
- Preserve the conversion intent: Hook → Immersive → Transformation → CTA.

CHARACTER ANCHOR RULES:
- `character_anchor` must be written first, before beats, because every beat references it.
- It must describe clothing so specifically that if you showed it to a costume designer,
  they could reproduce the exact look: fabric, colour code, fit, condition (clean/neat),
  visible accessories.
- It must not change across beats — if beat 1 has slate-blue shirt, beat 4 has slate-blue shirt.

OPENING IMAGE PROMPT RULES:
- Must depict a REAL-LOOKING aspirational Indian man, 26-28 years old.
- He is NOT a street vendor, NOT wet, NOT in a market. He looks like a confident urban
  professional — the target user of a dating confidence app.
- Exact clothing must match `character_anchor` exactly.
- Location: bright modern Mumbai café interior, warm golden-hour window light.
- Shot: waist-up, Sony A7R IV feel, 85mm f/1.8, shallow bokeh, 9:16 vertical.
- Expression: relaxed half-smile, slightly hopeful. Looking toward camera.
- Style: hyperrealistic photograph, candid portrait, natural skin texture with pores,
  cinematic warm colour grade, no digital artefacts.
- Banned words in this prompt: wet, market, street, night, cartoon, digital, render, CGI.

SCENE DESCRIPTION RULES (every beat must obey):
- Each beat MUST be set in a physically distinct location from every other beat.
  Suggested sequence for 3 beats: [café interior] → [outdoor Mumbai promenade or rooftop]
  → [his bedroom desk / warmly lit home setup].
  For 4 beats add: [house party / social gathering].
- Describe the environment so concretely that Veo3 can build it without guessing:
  include floor material, background furniture, light source direction, time of day.
- Always include at least one secondary element in the background that makes the scene
  feel alive (barista moving, pigeons on a parapet, city lights blinking).

CHARACTER ACTION RULES (eliminates the "static face zoom" problem):
- Describe second-by-second physical movement across the 8-second clip.
- The character must perform at least two distinct physical actions per beat.
- Example good action: "0-2s: sits down at café table, places phone face-up; 2-5s: picks
  up phone, eyes widen reading a message, slow grin spreads; 5-8s: leans back, exhales
  with a confident laugh, looks up toward camera."
- Example bad action (forbidden): "man stands smiling" / "camera zooms in on face".
- Forbidden: any static shot lasting more than 2 seconds, slow zoom with no action,
  character just standing or sitting without moving.

SHOT DESCRIPTION RULES:
- Must specify the exact camera motion at each moment of the 8 seconds.
- Always start from a wider framing and move to a closer one, OR use a tracking shot.
- Explicitly include one moment of rack focus, lens flare, or motivated camera bump
  to give the clip a natural handheld feel.

AUDIO PROMPT RULES (Veo3 native audio — no external TTS):
- Veo3 generates audio natively when given a detailed audio prompt. Use this field to
  drive voice, music, and ambience entirely within Veo3.
- Structure every audio_prompt as three layers:

  LAYER 1 — VOICEOVER DELIVERY (do NOT repeat the words here — they live in exact_voiceover):
  "A warm, confident Indian female voice (28-32 years old, mid-range tone, slight smile
  in her delivery, Hinglish accent) speaks the exact_voiceover text directly and clearly.
  Her pace is relaxed but purposeful — not rushed, not monotone. She sounds like a
  trusted friend giving good advice. Brand name 'Wingman' spoken with light emphasis."

  LAYER 2 — MUSIC:
  Describe tempo, instruments, and emotional feel appropriate to this beat's narrative
  role (Hook = curious/intriguing; Immersive = playful/upbeat; Transformation =
  swelling/empowering; CTA = warm urgent resolve). Include specific instrument cues
  e.g. "acoustic guitar pluck at 120 BPM, light hi-hat, no heavy bass".

  LAYER 3 — AMBIENCE:
  Describe the environmental sound of the scene location (café murmur, distant city
  traffic, gentle breeze). This grounds the viewer in the real world.

  Final instruction to always append: "The voiceover must sit clearly above the music
  mix. Music is -12dB relative to voice. Ambience is -18dB. No voice distortion.
  No echo. Clean studio-quality vocal presence."

TEXT ON SCREEN RULES (eliminates garbled text):
- Do NOT ask Veo3 to render any text overlays, chat bubble text, on-screen labels,
  or UI copy. Veo3 cannot produce legible text and will hallucinate gibberish.
- If the beat needs a "Tap below" or "₹2" message, describe it as a VISUAL METAPHOR
  only: e.g. "a glowing button-shaped light pulses at the bottom of frame" — do not
  specify any actual text content.
- For the CTA beat, instead of text, use bright visual signals: pulsing amber glow,
  character pointing toward camera, energetic eye contact, confident nod.

APP UI RULES (eliminates cartoon avatar problem):
- Do NOT ask Veo3 to show app screens, chat interfaces, or in-app UI.
- Instead, show the character's EMOTIONAL REACTION to what's on his phone:
  his face, his body language, his energy — not the phone screen itself.
- If the product demo feel is needed, describe: "he holds his phone up toward camera
  with a proud smile, screen facing him" — camera sees his face, not the screen.

EXACT VOICEOVER RULE (non-negotiable):
- The `exact_voiceover` field must be a verbatim copy of the voiceover lines from the
  raw script for this beat's time window.
- You are a transcriber for this field, NOT a writer. Do not improve, condense, or adapt.
- If a line feels too long for 8 seconds — include it anyway. The audio_prompt delivery
  instruction will handle pacing.
- Specifically preserve: Hinglish phrasing, "…" ellipses, rhetorical questions, the
  word "Wingman", and price mentions like "₹2".

CTA BEAT RULES (applies to the LAST beat only):
- The final beat is the Call-To-Action. It must feel urgent, warm, and direct.
- character_action for the CTA beat: the character looks directly into camera,
  leans slightly forward, and makes a direct "you" gesture (finger point or open
  palm toward lens) at least once during the 8 seconds.
- audio_prompt for the CTA beat must include:
    * Voice energy: friendly urgency, "smiling while talking" delivery
    * Music that swells or builds to a positive resolve — not flat or ambient
- scene_description for the CTA beat: bright, warm, well-lit interior — NOT dark,
  NOT moody. Energy should feel like a friend sealing a decision.
- Do NOT end the CTA beat on a static smiling portrait. The character must be MID-ACTION
  (mid-gesture, mid-laugh, mid-lean) so the video ends on energy, not a posed still.

APP NAME GROUNDING RULE:
- The app name "Wingman" MUST appear spoken aloud in at least one beat's exact_voiceover.
  Since you are copying verbatim from the script this happens automatically — verify it.
- In the beat where "Wingman" is spoken, the character_action must include the character
  glancing at or holding his phone with a pleased/proud expression — even if the screen
  is not visible. This grounds the app reference visually without requiring UI rendering.
- Do NOT replace "Wingman" with a generic description like "the app" or "this platform".

SECONDARY CHARACTER RULE (Beat 2 / Immersive beat only):
- Beat 2 MUST include a female character in the scene — shown from behind, at a distance,
  or in soft bokeh. She does NOT need to be the focus of the frame.
- She is present as environmental storytelling — sitting across from him at a café table,
  walking beside him on a promenade, laughing in a group at a party.
- Do NOT describe her face or name her. Keep her as a soft background presence.
- The male character's body language should be oriented toward her — leaning in,
  gesturing while talking, smiling in her direction.
- This is what makes the scene feel like a date/social scenario rather than
  a man sitting alone with his phone.

NEGATIVE CONSTRAINTS (append to every beat's shot_description):
"Do not render: cartoon characters, illustrated avatars, text overlays, speech bubbles,
app UI screens, static face zoom, wet or dishevelled appearance on the character,
any scene that contradicts the character_anchor clothing description."
""".strip()


def enhance_script_to_beats(raw_script: str, num_segments: int = 4) -> dict:
    """
    Use Gemini to break a raw marketing script into Veo3-ready beats.

    Returns a dict with keys:
      - opening_image_prompt: str
      - character_anchor: str
      - beats: list of dicts (beat_label, scene_description, character_action,
               shot_description, exact_voiceover, audio_prompt, continuation_note)
    """
    log.info("Enhancing script into %d beats via Gemini…", num_segments)

    system = _BEAT_SYSTEM_PROMPT.replace("{num_segments}", str(num_segments))

    # Extract locked scene mandates from the raw script and append as hard constraints
    scene_mandates = extract_scene_mandates(raw_script)
    if scene_mandates:
        mandate_block = "\n\nSCENE MANDATES FROM RAW SCRIPT (these are locked — do not substitute):\n"
        for idx, mandate in enumerate(scene_mandates[:num_segments], 1):
            mandate_block += f"Beat {idx} MUST show: {mandate}\n"
        mandate_block += (
            "These scene locations are non-negotiable. The script author chose them "
            "deliberately. Your job is to make them cinematic, not to replace them.\n"
        )
        system = system + mandate_block
        log.info("Injected %d scene mandates into system prompt.", len(scene_mandates[:num_segments]))

    plan = None
    last_exc = None

    for attempt in range(1, 4):
        raw = llm_client.complete(
            system=system,
            user=raw_script,
            model="gemini",
            max_tokens=8192,
        )

        # Strip markdown code fences if Gemini wraps output anyway
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        # If the response was truncated mid-JSON, try to close it gracefully
        # by locating the last complete beat object and truncating there.
        if not raw.endswith("}"):
            log.warning("Attempt %d — response appears truncated, attempting repair…", attempt)
            # Find the last fully closed beat object
            last_close = raw.rfind("}}")
            if last_close != -1:
                raw = raw[: last_close + 2] + "\n  ]\n}"
                log.info("Repaired JSON by closing at position %d.", last_close)

        try:
            plan = json.loads(raw)
            log.info("Attempt %d — JSON parsed successfully.", attempt)
            break
        except json.JSONDecodeError as exc:
            log.warning("Attempt %d — JSON parse failed: %s", attempt, exc)
            last_exc = exc
            if attempt < 3:
                log.info("Retrying with a tighter prompt…")
                # On subsequent attempts, tell Gemini to be more concise
                system = system + (
                    "\n\nCRITICAL: Keep each field under 200 characters. "
                    "Dialogue may be shortened to fit. Brevity is essential."
                )

    if plan is None:
        log.error("All attempts failed. Last raw output:\n%s", raw)
        raise RuntimeError(f"Script enhancer returned invalid JSON after 3 attempts: {last_exc}") from last_exc

    if "opening_image_prompt" not in plan or "beats" not in plan:
        raise RuntimeError(f"Script enhancer JSON missing required keys. Got: {list(plan.keys())}")

    beats = plan["beats"]
    if len(beats) != num_segments:
        log.warning(
            "Expected %d beats but got %d — truncating/padding to %d",
            num_segments, len(beats), num_segments,
        )
        # Truncate or duplicate last beat to match requested segment count
        while len(beats) < num_segments:
            beats.append(beats[-1])
        beats = beats[:num_segments]
        plan["beats"] = beats

    # ── Full prompt logging so you can see exactly what will be generated ──
    log.info("")
    log.info("┌─ OPENING IMAGE PROMPT " + "─" * 38)
    for line in plan["opening_image_prompt"].splitlines():
        log.info("│  %s", line)
    log.info("└" + "─" * 60)
    log.info("")

    anchor = plan.get("character_anchor", "")
    if anchor:
        log.info("┌─ CHARACTER ANCHOR " + "─" * 41)
        for line in anchor.splitlines():
            log.info("│  %s", line)
        log.info("└" + "─" * 60)
        log.info("")
    else:
        log.warning("No character_anchor in plan — character consistency will be unreliable!")

    for i, beat in enumerate(beats, 1):
        label = beat.get("beat_label", f"Beat {i}")
        log.info("┌─ BEAT %d / %d: %s " + "─" * max(0, 38 - len(label)), i, len(beats), label)
        log.info("│  [scene]    %s", beat.get("scene_description", ""))
        log.info("│  [action]   %s", beat.get("character_action", ""))
        log.info("│  [shot]     %s", beat.get("shot_description", ""))
        log.info("│  [vo]       %s", beat.get("exact_voiceover", beat.get("dialogue", "")))
        log.info("│  [audio]    %s", beat.get("audio_prompt", ""))
        cont = beat.get("continuation_note", "")
        if cont:
            log.info("│  [cont]     %s", cont)
        log.info("└" + "─" * 60)
        log.info("")

    return plan


# ---------------------------------------------------------------------------
# Opening image
# ---------------------------------------------------------------------------

_OPENING_IMAGE_PROMPT_SUFFIX = """
MANDATORY OVERRIDES — these supersede all other instructions:
- The man must look like an aspirational young urban Indian professional, NOT a street
  vendor, NOT wet, NOT in a market or nighttime outdoor scene.
- He must be indoors in a clean, modern, warmly-lit Mumbai café.
- His clothing must match the character_anchor exactly as described above.
- Expression: relaxed, half-smiling, slightly hopeful — not nervous, not sad.
- Background must show café furniture (wooden tables, plants, large windows) in soft bokeh.
- Lighting: warm golden-hour window light falling on his left cheek.
- Shot: waist-up portrait, 9:16 vertical, Sony A7R IV feel, 85mm f/1.8 bokeh.
- Photorealistic skin — visible pores, natural imperfections, not AI-smooth.
- Absolutely no: wet hair, dishevelled clothes, street market, night scene, cartoon style.
""".strip()


def generate_opening_image(prompt: str, character_anchor: str = "") -> str:
    """
    Generate the opening reference frame via Gemini image generation.

    The character_anchor (if provided) is prepended to the prompt so the image's
    appearance matches every subsequent Veo3 segment exactly.
    """
    anchor_block = (
        f"CHARACTER (must match exactly):\n{character_anchor}\n\n"
        if character_anchor else ""
    )
    final_prompt = f"{anchor_block}{prompt}\n\n{_OPENING_IMAGE_PROMPT_SUFFIX}"

    log.info("Generating opening reference image…")
    log.info("┌─ IMAGE GEN PROMPT " + "─" * 41)
    for line in final_prompt.splitlines():
        log.info("│  %s", line)
    log.info("└" + "─" * 60)
    from steps.image_generator import generate_image
    path = generate_image(final_prompt)
    log.info("Opening image saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Chained segment generation
# ---------------------------------------------------------------------------

def generate_chained_segments(plan: dict, opening_image_path: str, video_backend: str = "veo3") -> list:
    """
    Generate N chained video segments using the selected backend (Veo3 or Kling).

    Injects character_anchor into every segment prompt for identity consistency.
    Beat 1 uses opening_image_path as the starting frame; each subsequent beat
    uses the last frame of the previous segment for scene/lighting continuity.

    Returns a list of local .mp4 file paths, one per beat.
    """
    from steps.video_dispatcher import generate_video_segment, extract_last_frame

    beats = plan["beats"]
    character_anchor = plan.get("character_anchor", "")
    segment_paths = []
    current_image = opening_image_path

    for i, beat in enumerate(beats, 1):
        log.info("━━━ Generating segment %d / %d [%s] ━━━", i, len(beats), beat.get("beat_label", ""))
        log.info("  Reference image : %s", current_image)

        anchor_block = (
            f"CHARACTER (must be identical in every frame — never deviate):\n{character_anchor}\n\n"
            if character_anchor else ""
        )

        full_scene = (
            f"{anchor_block}"
            f"SCENE: {beat.get('scene_description', '')}\n\n"
            f"CHARACTER ACTION (second-by-second): {beat.get('character_action', '')}\n\n"
            f"CAMERA: {beat.get('shot_description', '')}\n\n"
            f"CONTINUITY: {beat.get('continuation_note', '')}"
        )

        exact_vo = beat.get("exact_voiceover", beat.get("dialogue", ""))
        audio_template = beat.get("audio_prompt", "")

        full_audio = (
            f"VOICEOVER (speak these exact words, verbatim, no deviation):\n"
            f'"{exact_vo}"\n\n'
            f"Voice: warm confident Indian female, 28-32 years old, slight smile in delivery, "
            f"Hinglish accent. Pace: natural conversational speed, not rushed. "
            f"Brand name 'Wingman' spoken with light emphasis.\n\n"
            f"{audio_template}"
        )

        log.info("  Scene desc      : %s", beat.get("scene_description", ""))
        log.info("  Action          : %s", beat.get("character_action", ""))
        log.info("  Shot desc       : %s", beat.get("shot_description", ""))
        log.info("  Voiceover       : %s", exact_vo)
        log.info("  Audio template  : %s", audio_template)
        cont = beat.get("continuation_note", "")
        if cont:
            log.info("  Continuation    : %s", cont)

        def _progress(elapsed: int, status: str, seg=i) -> None:
            log.info("  Segment %d — [%3ds] %s", seg, elapsed, status)

        mp4_path = generate_video_segment(
            image_path=current_image,
            scene_description=full_scene,
            shot_description="",
            segment_number=i,
            dialogue=full_audio,
            continuation_note="",
            duration=8,
            timeout=600,
            on_progress=_progress,
            backend=video_backend,
        )
        segment_paths.append(mp4_path)
        log.info("  Segment %d saved: %s", i, mp4_path)

        # Extract last frame for next segment's starting image (skip for final beat)
        if i < len(beats):
            log.info("  Extracting last frame for segment %d handoff…", i)
            current_image = extract_last_frame(mp4_path)
            log.info("  Last frame: %s", current_image)

    return segment_paths


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------

def stitch_segments(segment_paths: list, output_filename: str | None = None) -> str:
    """
    Concatenate segment .mp4 files with 0.4s crossfade transitions using ffmpeg xfade.

    Falls back to a hard-cut concat if the xfade filter graph fails for any reason.
    Returns the path to the stitched file.
    """
    import imageio_ffmpeg

    if not output_filename:
        output_filename = f"paywall_final_{uuid.uuid4().hex[:8]}.mp4"

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    if len(segment_paths) == 1:
        # Single segment — just copy directly
        import shutil
        shutil.copy2(segment_paths[0], output_path)
        size_mb = os.path.getsize(output_path) / 1_048_576
        log.info("Final video saved: %s (%.1f MB)", output_path, size_mb)
        return output_path

    FADE = 0.4        # seconds of overlap between segments
    SEG_DUR = 8.0     # duration of each segment

    # Build ffmpeg input args
    input_args: list[str] = []
    for p in segment_paths:
        input_args += ["-i", p]

    # Build xfade + acrossfade filter chain
    n = len(segment_paths)
    vf_parts: list[str] = []
    af_parts: list[str] = []

    prev_v = "[0:v]"
    prev_a = "[0:a]"

    for i in range(1, n):
        offset = round((i * SEG_DUR) - (i * FADE), 3)
        out_v = f"[xv{i}]" if i < n - 1 else ""
        out_a = f"[xa{i}]" if i < n - 1 else ""

        vf_parts.append(
            f"{prev_v}[{i}:v]xfade=transition=fade:duration={FADE}:offset={offset}{out_v}"
        )
        af_parts.append(
            f"{prev_a}[{i}:a]acrossfade=d={FADE}{out_a}"
        )
        prev_v = f"[xv{i}]"
        prev_a = f"[xa{i}]"

    filter_complex = ";".join(vf_parts + af_parts)

    cmd = [
        ffmpeg_exe, "-y",
        *input_args,
        "-filter_complex", filter_complex,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ]

    log.info("Stitching %d segments with xfade transitions → %s", n, output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.warning("xfade stitch failed, falling back to hard-cut concat:\n%s", result.stderr)
        # Graceful fallback to original concat method
        concat_list_path = os.path.join(OUTPUT_DIR, f"concat_{uuid.uuid4().hex[:6]}.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for path in segment_paths:
                safe_path = path.replace("\\", "/")
                f.write(f"file '{safe_path}'\n")
        fallback_cmd = [
            ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list_path, "-c", "copy", output_path,
        ]
        subprocess.run(fallback_cmd, check=True)
        try:
            os.remove(concat_list_path)
        except OSError:
            pass

    size_mb = os.path.getsize(output_path) / 1_048_576
    log.info("Final video saved: %s (%.1f MB)", output_path, size_mb)
    return output_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    raw_script: str,
    num_segments: int = 3,
    opening_image_path: str | None = None,
    keep_segments: bool = False,
    video_backend: str = "veo3",
) -> str:
    """
    Full paywall video generation pipeline.

    video_backend: "veo3" (default) or "kling"
    Returns the path to the final stitched MP4.
    """
    log.info("=" * 60)
    log.info("Paywall Video Generator  [backend=%s]", video_backend)
    log.info("Segments: %d × 8s = ~%ds final video", num_segments, num_segments * 8)
    log.info("=" * 60)

    # Step 1: Enhance script → beat plan
    plan = enhance_script_to_beats(raw_script, num_segments=num_segments)

    # Step 2: Opening reference image
    if opening_image_path:
        if not os.path.isfile(opening_image_path):
            raise FileNotFoundError(f"Provided opening image not found: {opening_image_path}")
        log.info("Using provided opening image: %s", opening_image_path)
    else:
        opening_image_path = generate_opening_image(
            plan["opening_image_prompt"],
            character_anchor=plan.get("character_anchor", ""),
        )

    # Step 3: Generate chained segments via selected backend
    segment_paths = generate_chained_segments(plan, opening_image_path, video_backend=video_backend)

    # Step 4: Stitch into one video
    final_path = stitch_segments(segment_paths)

    # Step 5: Optionally remove intermediate segments
    if not keep_segments:
        for p in segment_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        log.info("Intermediate segments removed (use --keep-segments to retain them).")

    log.info("=" * 60)
    log.info("DONE — Final video: %s", final_path)
    log.info("=" * 60)
    return final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a paywall promo video using Veo3 from a marketing script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_paywall_video.py
  python generate_paywall_video.py --keep-segments
  python generate_paywall_video.py --script-file my_script.txt --segments 4
  python generate_paywall_video.py --image hero.jpg --keep-segments
        """,
    )
    parser.add_argument(
        "--script-file", "-s",
        default="",
        metavar="PATH",
        help="Path to a .txt file containing the marketing script. Defaults to the built-in paywall script.",
    )
    parser.add_argument(
        "--segments", "-n",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Number of 8-second Veo3 segments to generate. "
            "If omitted, auto-detected from the script's time markers "
            "(e.g. 4 markers → 4 segments = ~32s video)."
        ),
    )
    parser.add_argument(
        "--image", "-i",
        default="",
        metavar="PATH",
        help="Path to an existing image to use as the opening frame. If omitted, one is auto-generated.",
    )
    parser.add_argument(
        "--keep-segments", "-k",
        action="store_true",
        help="Keep intermediate per-beat .mp4 files after stitching (useful for debugging).",
    )

    args = parser.parse_args()

    # Load script
    if args.script_file:
        if not os.path.isfile(args.script_file):
            log.error("Script file not found: %s", args.script_file)
            sys.exit(1)
        with open(args.script_file, encoding="utf-8") as fh:
            raw_script = fh.read().strip()
        log.info("Loaded script from: %s (%d chars)", args.script_file, len(raw_script))
    else:
        raw_script = PAYWALL_SCRIPT
        log.info("Using built-in paywall script (%d chars).", len(raw_script))

    # Auto-detect segment count from script time markers if not explicitly passed
    if args.segments == 0:
        args.segments = count_beats_in_script(raw_script)
        log.info("Auto-detected segment count: %d", args.segments)

    if args.segments < 1 or args.segments > 10:
        log.error("--segments must be between 1 and 10.")
        sys.exit(1)

    try:
        final_path = run(
            raw_script=raw_script,
            num_segments=args.segments,
            opening_image_path=args.image or None,
            keep_segments=args.keep_segments,
        )
        print(f"\nFinal video: {final_path}")
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        log.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
