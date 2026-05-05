# Veo3 Script Compliance Fixes

> Based on video `paywall_final_6444de1f.mp4` — character consistency is now ✅ working.
> These fixes target the remaining failures: wrong scenes, missing CTA, missing app
> references, and Gemini paraphrasing the dialogue instead of using exact words.

---

## What the Video Actually Shows vs. What the Script Demands

| Script Beat             | Script Says                           | What Video Shows                           | Verdict                      |
| ----------------------- | ------------------------------------- | ------------------------------------------ | ---------------------------- |
| Hook (0–3s)             | Nervous guy → smiles after chat       | Café, man on phone, nervous → smiles       | ✅ Correct scene             |
| Immersive (3–10s)       | Coffee flirt + beach walk + party     | **Only café, no beach, no girl, no party** | ❌ Wrong scene               |
| Transformation (10–16s) | Before/after confidence shift         | Rooftop bar, man socialising               | ⚠️ Close but no before/after |
| CTA (16–21s)            | ₹2 unlock, Wingman named, tap button  | Man smiling in bedroom — **zero CTA**      | ❌ Completely missing        |
| Dialogue                | Exact Hindi/English lines from script | Gemini rewrote them — **none are exact**   | ❌ All paraphrased           |
| App reference           | "Wingman" spoken + implied visually   | **Never mentioned or shown**               | ❌ Missing                   |

---

## Root Cause 1 — Wrong Segment Count (Biggest Issue)

**The script has 4 beats. The default is `--segments 3`. Gemini silently merges and drops beats.**

When you run with 3 segments, Gemini is instructed to produce exactly 3 beats from a
4-beat script. It compresses beats 2+3 together and drops the CTA entirely — which is
precisely the most commercially critical beat.

### Fix A — Auto-derive segment count from the raw script

Add a parser that counts `[X sec:` markers in the raw script and automatically sets
`num_segments`. Add this function to `generate_paywall_video.py`:

```python
import re

def count_beats_in_script(raw_script: str) -> int:
    """
    Count the number of explicit time-bracketed beats in the raw script.
    Looks for patterns like [0–3 sec:, [3–10 sec:, [16–21 sec: etc.
    Falls back to 4 if none found (safe default for this paywall script).
    """
    # Match patterns like [0–3 sec: or [16-21 sec: (en-dash or hyphen)
    matches = re.findall(r'\[\d+[\–\-]\d+\s*sec\s*:', raw_script)
    count = len(matches)
    if count < 2:
        log.warning(
            "Could not detect beat count from script markers — defaulting to 4 segments."
        )
        return 4
    log.info("Detected %d beats from script time markers → setting segments=%d", count, count)
    return count
```

Then in `main()`, replace the hardcoded default:

```python
# BEFORE — always defaults to 3 regardless of script
parser.add_argument("--segments", "-n", type=int, default=3, ...)

# AFTER — auto-detect from script if --segments not explicitly passed
args = parser.parse_args()

if args.segments == 3:  # i.e. user didn't override
    raw_script_for_count = open(args.script_file).read() if args.script_file else PAYWALL_SCRIPT
    args.segments = count_beats_in_script(raw_script_for_count)
    log.info("Auto-detected segment count: %d", args.segments)
```

---

## Root Cause 2 — Gemini Paraphrases Dialogue Instead of Copying It

The current rule says:

> "extract the female voiceover lines that naturally fit within ~8 seconds of speech"

**"Extract" and "naturally fit" gives Gemini full licence to rewrite, shorten, and paraphrase.**
The result is generic motivational lines instead of the actual script copy.

### Fix B — Replace `dialogue` with `exact_voiceover` with a strict copy rule

**In `_BEAT_SYSTEM_PROMPT`, change the JSON schema field and its rule:**

```python
# In the OUTPUT FORMAT JSON schema, rename the field:
# BEFORE:
"dialogue": "<the exact female voiceover line(s) spoken during this 8-second window — keep natural pacing>",

# AFTER:
"exact_voiceover": "<Copy the female voiceover text VERBATIM word-for-word from the raw script for this beat's time window. Do NOT paraphrase, summarise, rewrite, translate, or shorten. If the script uses '…' keep it. If it uses 'Hinglish' words keep them. The only allowed change is removing stage direction brackets like [voiceover] if present.>",
```

**Add this to the RULES section:**

```
EXACT VOICEOVER RULE (non-negotiable):
- The `exact_voiceover` field must be a verbatim copy of the voiceover lines from the
  raw script for this beat's time window.
- You are a transcriber for this field, NOT a writer. Do not improve, condense, or adapt.
- If a line feels too long for 8 seconds — include it anyway. The pacing instruction in
  the audio_prompt will handle delivery speed.
- Specifically preserve: Hinglish phrasing, "…" ellipses, rhetorical questions, the
  word "Wingman", and price mentions like "₹2".
```

**In `generate_chained_segments()`, update the audio_prompt builder to use the new field:**

```python
# BEFORE
full_audio = beat.get("audio_prompt", "")

# AFTER
exact_vo = beat.get("exact_voiceover", beat.get("dialogue", ""))
audio_template = beat.get("audio_prompt", "")

full_audio = f"""
VOICEOVER (speak these exact words, verbatim, no deviation):
"{exact_vo}"

Voice: warm confident Indian female, 28-32 years old, slight smile in delivery,
Hinglish accent. Pace: natural conversational speed, not rushed.
Brand name "Wingman" spoken with light emphasis.

{audio_template}
"""
```

---

## Root Cause 3 — Gemini Ignores the Raw Script's Scene Instructions

The `scene_description` rule tells Gemini to pick scenes freely. It doesn't tell Gemini
to read and honour the scene directions already written in the raw script like:
`[Coffee shop flirt, beach walk, fun party vibe]`.

### Fix C — Add a scene extraction pass before beat generation

Add a two-pass approach: first extract the scene mandates from the raw script, then
pass them as locked constraints into the beat generation.

**Add this function:**

```python
def extract_scene_mandates(raw_script: str) -> list[str]:
    """
    Extract the bracketed scene direction lines from the raw script.
    These become locked scene constraints for each beat.

    Example input:  "[3–10 sec: Immersive scenes – Coffee shop flirt, beach walk, party]"
    Example output: ["Coffee shop flirt, beach walk, fun party vibe"]
    """
    # Match the content inside brackets that contain sec: markers
    pattern = r'\[[\d\–\-]+\s*sec\s*:[^\]]+\]'
    matches = re.findall(pattern, raw_script)
    # Strip the time marker prefix, keep scene description
    mandates = []
    for m in matches:
        # Remove "[X–Y sec: Label –" prefix, keep the rest
        clean = re.sub(r'\[[\d\–\-]+\s*sec\s*:\s*[^–\-]*[\–\-]\s*', '', m)
        clean = clean.rstrip(']').strip()
        mandates.append(clean)
    return mandates
```

**Then pass the mandates into the beat system prompt dynamically:**

```python
def enhance_script_to_beats(raw_script: str, num_segments: int = 4) -> dict:
    scene_mandates = extract_scene_mandates(raw_script)

    # Build a mandate block to inject into the system prompt
    if scene_mandates:
        mandate_block = "\n\nSCENE MANDATES FROM RAW SCRIPT (these are locked — do not substitute):\n"
        for i, mandate in enumerate(scene_mandates[:num_segments], 1):
            mandate_block += f"Beat {i} MUST show: {mandate}\n"
        mandate_block += (
            "These scene locations are non-negotiable. The script author chose them "
            "deliberately. Your job is to make them cinematic, not to replace them.\n"
        )
    else:
        mandate_block = ""

    system = _BEAT_SYSTEM_PROMPT.replace("{num_segments}", str(num_segments)) + mandate_block
    # ... rest of function unchanged
```

---

## Root Cause 4 — The CTA Beat Has No CTA Behaviour

Beat 4 in the script is explicit: pulsing button, ₹2 price, "tap below", urgency.
But nothing in the current prompt system tells Gemini that beat 4 is a CTA beat or
that it must have specific commercial energy.

### Fix D — Add a CTA beat enforcer

Add this as a new mandatory rule section in `_BEAT_SYSTEM_PROMPT`:

```
CTA BEAT RULES (applies to the LAST beat only):
- The final beat is the Call-To-Action. It must feel urgent, warm, and direct.
- character_action for the CTA beat: the character looks directly into camera,
  leans slightly forward, and makes a direct "you" gesture (finger point or open
  palm toward lens) at least once during the 8 seconds.
- audio_prompt for the CTA beat must include:
    * The exact ₹2 price spoken aloud from the script
    * The phrase "cancel anytime" or its Hinglish equivalent from the script
    * Music that swells or builds to a positive resolve — not flat/ambient
    * Voice energy: friendly urgency, "smiling while talking" delivery
- scene_description for the CTA beat: bright, warm, well-lit interior — NOT dark,
  NOT moody. The energy should feel like a friend sealing a decision, not a sad goodbye.
- Do NOT end the CTA beat on a static smiling portrait. The character must be MID-ACTION
  (mid-gesture, mid-laugh, mid-lean) so the video ends on energy, not a posed still.
```

---

## Root Cause 5 — "Wingman" Is Never Visually or Verbally Referenced

The app name appears in the script multiple times but never surfaces in the video because:

1. The exact_voiceover paraphrase dropped it
2. No scene element grounds it visually

### Fix E — Add an app-name grounding rule

Add to `_BEAT_SYSTEM_PROMPT` RULES:

```
APP NAME GROUNDING RULE:
- The app name "Wingman" MUST appear spoken aloud in at least one beat's exact_voiceover.
  Since you are copying verbatim from the script, this happens automatically — verify it.
- In the beat where "Wingman" is spoken, the character_action must include the character
  glancing at or holding his phone with a pleased/proud expression — even if the screen
  is not visible. This grounds the app reference visually without requiring UI rendering.
- Do NOT replace "Wingman" with a generic description like "the app" or "this platform".
```

---

## Root Cause 6 — Immersive Beat Needs Multiple People, Not Just the Male Lead

The script beat 2 says "coffee shop flirt, beach walk, fun party vibe" — all of these
imply a female character in the scene. The current scene prompts only describe the male
character and the environment. Veo3 defaults to showing him alone.

### Fix F — Add secondary character rules for the immersive beat

Add to `_BEAT_SYSTEM_PROMPT` RULES:

```
SECONDARY CHARACTER RULE (Beat 2 / Immersive beat only):
- Beat 2 MUST include a female character in the scene — shown from behind, at a distance,
  or in soft bokeh. She does NOT need to be the focus of the frame.
- She is present as environmental storytelling — sitting across from him at a café table,
  walking beside him on a promenade, laughing in a group at a party.
- Do NOT describe her face or name her. Keep her as a soft background presence.
- The male character's body language should be oriented toward her — leaning in,
  gesturing while talking, smiling in her direction.
- This is what makes the scene feel like a "date/social scenario" rather than
  a man sitting alone with his phone.
```

---

## Summary of All Changes — Where Each Goes

| Fix                               | Location in code                                             | What it changes                                    |
| --------------------------------- | ------------------------------------------------------------ | -------------------------------------------------- |
| **A** — Auto-detect segment count | `main()` + new `count_beats_in_script()`                     | Stops beats being silently dropped                 |
| **B** — Exact voiceover copy      | `_BEAT_SYSTEM_PROMPT` schema + `generate_chained_segments()` | Forces verbatim dialogue from script               |
| **C** — Scene mandate extraction  | New `extract_scene_mandates()` + `enhance_script_to_beats()` | Locks beach/café/party from script                 |
| **D** — CTA beat enforcer         | `_BEAT_SYSTEM_PROMPT` new rule section                       | Gives beat 4 its commercial energy                 |
| **E** — Wingman grounding         | `_BEAT_SYSTEM_PROMPT` new rule section                       | Ensures app name appears spoken + visually implied |
| **F** — Secondary character       | `_BEAT_SYSTEM_PROMPT` new rule section                       | Puts a female presence in the immersive scene      |

Apply in order A → B → C → D → E → F. A and B are the highest leverage — they fix the
structural mismatch and dialogue problem that affect every beat simultaneously.

---

## The One Command Change to Make Immediately

Even before applying code changes, run with `--segments 4` explicitly:

```bash
python generate_paywall_video.py --segments 4 --keep-segments
```

This alone will stop the CTA beat from being dropped. Fixes B–F then ensure
each of those 4 segments has the right content.

---

## What Good Beat JSON Looks Like After These Fixes

```json
{
  "beat_label": "Immersive",
  "scene_description": "Outdoor Marine Drive promenade, Mumbai, late afternoon golden hour.
    Warm orange light on sea-facing path. Background: gentle waves, distant skyline,
    pedestrians walking in soft bokeh. A young woman in a light kurta walks beside
    the male lead, visible from behind in soft focus.",
  "character_action": "0-2s: walks along promenade, hands relaxed, gesturing lightly
    while talking toward the woman beside him. 2-5s: she laughs at something he said —
    he grins wide, looks briefly at camera then back at her. 5-8s: he pulls out his
    phone, glances at it with a knowing smile, pockets it confidently.",
  "shot_description": "Tracking shot alongside them at waist height. At 3s, gentle push-in
    to his face as she laughs. At 6s, wide shot showing both figures against the sunset
    sea. Warm lens flare at 7s. Do not render text, chat bubbles, app UI, or cartoon.",
  "exact_voiceover": "With Wingman, you dive straight into real, flirty role-play stories. You become the guy in the scene. One beautiful girl, one exciting situation — coffee dates, late-night talks, beach vibes… You chat with her, tease her, build the spark… and she responds to your words. All in complete privacy.",
  "audio_prompt": "MUSIC: Upbeat acoustic guitar, 125 BPM, light percussion, playful energy.
    Builds slightly from start to end. AMBIENCE: Gentle sea breeze, distant traffic murmur.
    LEVELS: Voice 0dB, music -12dB, ambience -18dB. Voice warm and excited.",
  "continuation_note": "Continuing from the café hook scene, now outdoors on a Mumbai
    promenade at golden hour — same character, same blue linen shirt."
}
```
