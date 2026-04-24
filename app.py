import time
import traceback

import streamlit as st

from config import validate_keys, CITY_ACCENT_MAP
from steps.audio_generator import generate_audio, list_voices
from steps.image_generator import generate_image, generate_images
from steps.media_merger import concatenate_segments
from steps.prompt_enhancer import (
    enhance_image_prompt,
    enhance_audio_prompt,
    generate_character_fields,
    generate_scenario_fields,
    generate_beat_fields,
    generate_storyline,
    critique_storyline,
    generate_video_scene_script,
    generate_poster_fields,
    assemble_poster_prompt,
)
from steps.veo3_generator import extract_last_frame, generate_video_segment

st.set_page_config(page_title="Wingman Creator", page_icon="🎬", layout="wide")
st.title("🎬 Wingman — Character & Scenario Creator")
st.caption("14 inputs → 40+ AI-generated fields → image · audio · video · DB")

# ── Session state init ───────────────────────────────────────────────────
_DEFAULTS: dict = {
    # form inputs
    "char_inputs": {},
    "scenario_inputs": {},
    "beat_inputs": {},
    # pipeline outputs
    "s1_char_fields": None,       # dict — LLM-generated character fields
    "s2_image_path": "",          # character portrait path (selected)
    "s2_image_paths": [],         # list of 2 portrait candidate paths
    "s2_image_prompt": "",        # editable portrait prompt
    "s3_audio_path": "",          # character voice path
    "s4_scenario_fields": None,   # dict — LLM-generated scenario fields
    "s5_scene_image_path": "",    # scenario poster path (selected)
    "s5_scene_image_paths": [],   # list of 2 poster candidate paths
    "s5_image_prompt": "",        # editable poster prompt (assembled from fields)
    "s5_poster_fields": {},       # structured poster input fields
    "s6_beat_fields": None,       # list — LLM-generated beat dicts
    "s7_storyline": "",           # str — final approved storyline narrative
    "s7_editor_log": [],          # list of {loop, score, improvements} dicts
    "s8_script": None,            # dict — video scene script
    "s8_script_prompt": "",       # editable screenplay prompt context
    "s9_segment_paths": [],       # list of .mp4 paths
    "s9_segment_prompts": {},     # dict {segment_number: editable_prompt}
    "s10_combined_path": "",      # concatenated video
    "s11_final_path": "",         # final video
    # step statuses
    **{f"s{n}_status": "pending" for n in range(1, 12)},
    # ui helpers
    "pipeline_started": False,
    "segment_duration": 5,
    "selected_voice_id": None,
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

BEAT_TYPES = ["HOOK", "BUILD", "TWIST", "CONSEQUENCE", "CLIFFHANGER"]
CITIES = list(CITY_ACCENT_MAP.keys())


# ── Helpers ──────────────────────────────────────────────────────────────

def _status_badge(n: int) -> str:
    s = st.session_state[f"s{n}_status"]
    return {
        "done":    ":green[✓ Done]",
        "error":   ":red[✗ Error]",
        "pending": ":gray[○ Pending]",
    }.get(s, ":gray[○ Pending]")


def step_header(n: int, title: str):
    st.markdown(f"### Step {n} — {title} &nbsp; {_status_badge(n)}")


def _clear_from(n: int):
    """Reset outputs and statuses for steps n–11."""
    data_keys = [
        "s1_char_fields", "s2_image_path", "s3_audio_path",
        "s4_scenario_fields", "s5_scene_image_path", "s6_beat_fields",
        "s7_storyline", "s8_script", "s9_segment_paths",
        "s10_combined_path", "s11_final_path",
    ]
    prompt_keys = {
        2: ["s2_image_prompt", "s2_image_paths"],
        5: ["s5_image_prompt", "s5_scene_image_paths", "s5_poster_fields"],
        7: ["s7_editor_log"],
        8: ["s8_script_prompt"],
        9: ["s9_segment_prompts"],
    }
    for i in range(n, 12):
        st.session_state[f"s{i}_status"] = "pending"
        key = data_keys[i - 1] if i <= len(data_keys) else None
        if key:
            st.session_state[key] = _DEFAULTS[key]
        for pk in prompt_keys.get(i, []):
            st.session_state[pk] = _DEFAULTS[pk]


def prev_done(n: int) -> bool:
    if n == 1:
        return st.session_state["pipeline_started"]
    if n == 4:
        # Step 3 is optional — Step 4 only needs Step 2.
        return st.session_state["s2_status"] == "done"
    return st.session_state[f"s{n-1}_status"] == "done"


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    missing = validate_keys()
    if missing:
        st.error(f"Missing keys in .env: {', '.join(missing)}")
        st.info("Copy `.env.example` → `.env` and fill in your keys.")
    else:
        st.success("All API keys configured ✓")

    st.divider()
    seg_dur = st.selectbox(
        "Veo3 segment duration",
        [5, 8],
        index=0,
        help="5 s → ~7 segments (~35 s). 8 s → ~5 segments (~40 s).",
    )
    st.session_state["segment_duration"] = seg_dur

    # try:
    #     available_voices = list_voices()
    # except Exception:
    #     available_voices = []

    # if available_voices:
    #     voice_labels = [v["name"] for v in available_voices]
    #     voice_ids    = [v["voice_id"] for v in available_voices]
    #     sel_label = st.selectbox("Voiceover voice", voice_labels)
    #     st.session_state["selected_voice_id"] = voice_ids[voice_labels.index(sel_label)]
    # else:
    #     st.warning("Could not load ElevenLabs voices — using account default.")
    #     st.session_state["selected_voice_id"] = None


# ══════════════════════════════════════════════════════════════════════════
# INPUT FORM — 3 tabs
# ══════════════════════════════════════════════════════════════════════════
st.subheader("📝 Create Character & Scenario")

tab_char, tab_scen, tab_beats = st.tabs(["🧍 Character", "🎬 Scenario", "🥊 Beats"])

with tab_char:
    st.caption("6 inputs → 13 DB fields generated by AI (name, backstory, voice, image prompt…)")
    ci = st.session_state.get("char_inputs", {})

    col1, col2 = st.columns(2)
    with col1:
        c1 = st.text_input(
            "C1 — Who is she in one punchy line? *",
            placeholder="e.g. The Girl Next Door Who Got Hot",
            value=ci.get("archetype_phrase", ""),
            help="Max 8 words. The creative seed — every other field traces back to this.",
        )
        c2 = st.text_area(
            "C2 — What is the one unresolved thing in her life right now? *",
            placeholder="e.g. 30+ DMs daily, tired of creeps, but still secretly hoping someone real shows up",
            height=90,
            value=ci.get("core_life_tension", ""),
        )
        c3 = st.selectbox(
            "C3 — Which city is she from? *",
            CITIES,
            index=CITIES.index(ci.get("city", CITIES[0])),
            help="City determines dialect, references, class texture — changes everything.",
        )
    with col2:
        c4 = st.text_area(
            "C4 — What is ONE specific thing about how she texts/talks that makes her feel real?",
            placeholder="e.g. Deliberately waits 5–10 minutes before replying even though she typed it instantly",
            height=90,
            value=ci.get("signature_comm_behavior", ""),
        )
        c5 = st.text_area(
            "C5 — What is one thing she NEVER does, no matter what?",
            placeholder="e.g. Never says she likes someone first — shows it through behavior instead",
            height=90,
            value=ci.get("what_she_never_does", ""),
        )
        c6 = st.text_input(
            "C6 — Describe how she looks and feels in 5 words",
            placeholder="e.g. Fit, casual, golden hour, homely",
            value=ci.get("physical_vibe", ""),
        )

with tab_scen:
    st.caption("5 inputs → 17 DB fields (title, setup, atmosphere, image prompt, initial messages…)")
    si = st.session_state.get("scenario_inputs", {})

    s1 = st.text_area(
        "S1 — What exactly is happening right now? Be hyper-specific. *",
        placeholder="e.g. She posted an Indori poha story. Tu 48th DM hai uske inbox mein.",
        height=100,
        value=si.get("trigger_detail", ""),
        help="The specificity of the trigger is what makes the scenario feel immersive.",
    )
    s2 = st.text_area(
        "S2 — What fear or insecurity does this situation poke at in the person playing it? *",
        placeholder="e.g. You're one of 47. You're invisible unless you do something actually different.",
        height=90,
        value=si.get("primal_fear", ""),
    )
    col3, col4 = st.columns(2)
    with col3:
        s3 = st.text_area(
            "S3 — How is she feeling at the exact moment this scenario starts?",
            placeholder="e.g. Mildly curious but fundamentally skeptical. She has been disappointed too many times.",
            height=90,
            value=si.get("emotional_state_now", ""),
        )
    with col4:
        s4 = st.text_input(
            "S4 — Where is she physically, and what time is it?",
            placeholder="e.g. Late night, her bedroom, post-gym, phone in hand",
            value=si.get("time_and_place", ""),
        )
    s5 = st.text_input(
        "S5 — Where does this scenario end if the user does everything right?",
        placeholder="e.g. From one of 47 strangers to the one she actually wants to talk to",
        value=si.get("arc_destination", ""),
    )

with tab_beats:
    st.caption("3 inputs → N beats generated (narrative context, flow + hook directives, advance logic…)")
    bi = st.session_state.get("beat_inputs", {})

    num_beats = st.slider(
        "B1 — How many beats does this scenario have?",
        min_value=1, max_value=5, value=bi.get("num_beats", 5),
        help="1=HOOK only, 5=full arc (HOOK→BUILD→TWIST→CONSEQUENCE→CLIFFHANGER)",
    )

    st.markdown("**B2 — Select beat type for each slot:**")
    beat_seq = bi.get("beat_sequence", BEAT_TYPES[:num_beats])
    # Adjust to num_beats
    while len(beat_seq) < num_beats:
        beat_seq.append(BEAT_TYPES[min(len(beat_seq), len(BEAT_TYPES)-1)])
    beat_seq = beat_seq[:num_beats]

    beat_seq_out = []
    bcols = st.columns(num_beats)
    for idx, col in enumerate(bcols):
        with col:
            default_bt = beat_seq[idx] if idx < len(beat_seq) else BEAT_TYPES[min(idx, len(BEAT_TYPES)-1)]
            chosen = col.selectbox(
                f"Beat {idx+1}",
                BEAT_TYPES,
                index=BEAT_TYPES.index(default_bt) if default_bt in BEAT_TYPES else 0,
                key=f"beat_sel_{idx}",
            )
            beat_seq_out.append(chosen)

    has_twist = "TWIST" in beat_seq_out
    b3 = ""
    if has_twist:
        b3 = st.text_area(
            "B3 — What does she do to test him at the TWIST beat? What is she actually checking for?",
            placeholder="e.g. She brings up Rohit casually. She's checking if he gets uncomfortable or honest.",
            height=80,
            value=bi.get("test_moment_desc", ""),
        )

# ── Start / Reset ────────────────────────────────────────────────────────
st.divider()
col_start, col_reset = st.columns([3, 1])

with col_start:
    start_label = "▶ Start Pipeline" if not st.session_state["pipeline_started"] else "🔄 Update & Restart Pipeline"
    start_clicked = st.button(start_label, type="primary", disabled=bool(missing))

with col_reset:
    if st.session_state["pipeline_started"]:
        if st.button("Reset All", type="secondary"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

required_char  = {"C1 — Archetype phrase": c1, "C3 — City": c3}
required_scen  = {"S1 — Trigger detail": s1, "S2 — Primal fear": s2}
missing_fields = [label for label, val in {**required_char, **required_scen}.items() if not str(val).strip()]

if start_clicked and missing_fields:
    st.warning(f"Please fill in required fields: {', '.join(missing_fields)}")

if start_clicked and not missing_fields:
    new_char = {
        "archetype_phrase": c1.strip(), "core_life_tension": c2.strip(),
        "city": c3, "signature_comm_behavior": c4.strip(),
        "what_she_never_does": c5.strip(), "physical_vibe": c6.strip(),
    }
    new_scen = {
        "trigger_detail": s1.strip(), "primal_fear": s2.strip(),
        "emotional_state_now": s3.strip(), "time_and_place": s4.strip(),
        "arc_destination": s5.strip(),
    }
    new_beat = {
        "num_beats": num_beats, "beat_sequence": beat_seq_out,
        "test_moment_desc": b3.strip() if has_twist else "",
    }
    changed = (
        new_char != st.session_state["char_inputs"]
        or new_scen != st.session_state["scenario_inputs"]
        or new_beat != st.session_state["beat_inputs"]
    )
    if changed:
        _clear_from(1)
    st.session_state["char_inputs"]     = new_char
    st.session_state["scenario_inputs"] = new_scen
    st.session_state["beat_inputs"]     = new_beat
    st.session_state["pipeline_started"] = True
    st.rerun()

if not st.session_state["pipeline_started"]:
    st.info("Fill in the tabs above and click **▶ Start Pipeline** to begin.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("⚙️ Pipeline")

char_inputs     = st.session_state["char_inputs"]
scenario_inputs = st.session_state["scenario_inputs"]
beat_inputs     = st.session_state["beat_inputs"]
seg_dur         = st.session_state["segment_duration"]
voice_id        = st.session_state["selected_voice_id"]


# ─── STEP 1 — Generate Character Fields ──────────────────────────────────
step_header(1, "Generate Character Fields")

if st.session_state["s1_status"] == "done":
    fields = st.session_state["s1_char_fields"]
    with st.expander("✅ Character fields preview (editable below)", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Name:** {fields.get('name','')}")
            st.markdown(f"**Age:** {fields.get('age','')}")
            st.markdown(f"**City:** {fields.get('city','')}")
            st.markdown(f"**Archetype:** {fields.get('archetype','')}")
            st.markdown(f"**Accent HSL:** {fields.get('accentHsl','')}")
            st.markdown(f"**Emoji usage:** {fields.get('emojiUsage','')}")
            st.markdown(f"**Texting speed:** {fields.get('textingSpeed','')}")
        with col_b:
            st.markdown("**Vibe Summary:**")
            st.caption(fields.get('vibeSummary',''))
            st.markdown("**Hard Limits:**")
            for hl in fields.get('hardLimits', []):
                st.caption(f"• {hl}")
    if st.button("Regenerate Character Fields", key="regen_s1"):
        _clear_from(1)
        st.rerun()

elif st.session_state["s1_status"] == "error":
    st.error(st.session_state.get("s1_error", "Unknown error"))
    if st.button("Retry Step 1"):
        _clear_from(1); st.rerun()

else:
    if st.button("✨ Generate Character Fields", key="run_s1", type="primary"):
        with st.spinner("Calling Gemini to generate character fields…"):
            try:
                fields = generate_character_fields(char_inputs)
                st.session_state["s1_char_fields"] = fields
                st.session_state["s1_status"] = "done"
            except Exception as e:
                st.session_state["s1_status"] = "error"
                st.session_state["s1_error"] = str(e)
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
        st.rerun()


# ─── STEP 2 — Generate Character Portrait ────────────────────────────────
st.divider()
step_header(2, "Generate Character Portrait")

if st.session_state["s2_status"] == "done":
    st.image(st.session_state["s2_image_path"], width=380)
    with st.expander("Portrait prompt used", expanded=False):
        st.caption(st.session_state.get("s2_image_prompt", ""))
    if st.button("Regenerate Portrait", key="regen_s2"):
        _clear_from(2); st.rerun()

elif st.session_state["s2_status"] == "error":
    st.error(st.session_state.get("s2_error", "Unknown error"))
    if st.button("Retry Step 2"):
        _clear_from(2); st.rerun()

else:
    if prev_done(2):
        # Phase 1: prepare prompt (LLM-enhanced) on first visit
        if not st.session_state["s2_image_prompt"]:
            if st.button("✨ Prepare Portrait Prompt", key="prep_s2"):
                avatar_prompt = st.session_state["s1_char_fields"].get("avatarPrompt", "")
                with st.spinner("Enhancing image prompt…"):
                    try:
                        st.session_state["s2_image_prompt"] = enhance_image_prompt(avatar_prompt)
                    except Exception as e:
                        st.error(f"Prompt preparation failed: {e}")
                st.rerun()
        elif not st.session_state["s2_image_paths"]:
            # Phase 2: show editable prompt + generate button
            st.session_state["s2_image_prompt"] = st.text_area(
                "Portrait generation prompt (edit to refine before generating)",
                value=st.session_state["s2_image_prompt"],
                height=220,
                key="s2_prompt_editor",
            )
            col_s2a, col_s2b = st.columns([3, 1])
            with col_s2a:
                if st.button("🖼️ Generate 2 Portraits", key="run_s2", type="primary"):
                    with st.spinner("Generating 2 portrait options…"):
                        try:
                            paths = generate_images(st.session_state["s2_image_prompt"], count=2)
                            st.session_state["s2_image_paths"] = paths
                        except Exception as e:
                            st.session_state["s2_status"] = "error"
                            st.session_state["s2_error"] = str(e)
                            with st.expander("Error details"):
                                st.code(traceback.format_exc())
                    st.rerun()
            with col_s2b:
                if st.button("↺ Re-prepare Prompt", key="reprep_s2"):
                    st.session_state["s2_image_prompt"] = ""
                    st.rerun()
        else:
            # Phase 3: show both portraits side-by-side, user picks one
            st.caption("Pick the portrait you want to use:")
            col_a, col_b = st.columns(2)
            paths = st.session_state["s2_image_paths"]
            with col_a:
                st.image(paths[0], use_container_width=True)
                if st.button("Select Portrait A", key="pick_s2_a", type="primary"):
                    st.session_state["s2_image_path"] = paths[0]
                    st.session_state["s2_status"] = "done"
                    st.rerun()
            with col_b:
                st.image(paths[1], use_container_width=True)
                if st.button("Select Portrait B", key="pick_s2_b", type="primary"):
                    st.session_state["s2_image_path"] = paths[1]
                    st.session_state["s2_status"] = "done"
                    st.rerun()
            if st.button("↺ Regenerate Both", key="regen_s2_both"):
                st.session_state["s2_image_paths"] = []
                st.rerun()
    else:
        st.caption("Complete Step 1 first.")


# ─── STEP 3 — Generate Character Voice ───────────────────────────────────
# st.divider()
# step_header(3, "Generate Character Voice (Optional Preview)")

# if st.session_state["s3_status"] == "done":
#     st.audio(st.session_state["s3_audio_path"])
#     if st.button("Regenerate Voice", key="regen_s3"):
#         _clear_from(3); st.rerun()

# elif st.session_state["s3_status"] == "error":
#     st.error(st.session_state.get("s3_error", "Unknown error"))
#     if st.button("Retry Step 3"):
#         _clear_from(3); st.rerun()

# else:
#     if st.button("🎙️ Generate Voice", key="run_s3", type="primary", disabled=not prev_done(3)):
#         fields     = st.session_state["s1_char_fields"]
#         city       = fields.get("city", char_inputs.get("city", ""))
#         accent_ctx = CITY_ACCENT_MAP.get(city, "")
#         voice_text = fields.get("voicePrompt", "")
#         # Build a short character intro passage for TTS
#         tts_passage = (
#             f"{fields.get('vibeSummary', '')} "
#             f"{fields.get('backstory', '')[:200]}"
#         ).strip()
#         paced = enhance_audio_prompt(tts_passage) if tts_passage else voice_text[:300]
#         with st.spinner(f"Generating voice ({city} accent) with ElevenLabs…"):
#             try:
#                 path = generate_audio(paced, voice_id)
#                 st.session_state["s3_audio_path"] = path
#                 st.session_state["s3_status"] = "done"
#             except Exception as e:
#                 st.session_state["s3_status"] = "error"
#                 st.session_state["s3_error"] = str(e)
#                 with st.expander("Error details"):
#                     st.code(traceback.format_exc())
#         st.rerun()
#     if not prev_done(3):
#         st.caption("Complete Step 2 first.")


# ─── STEP 4 — Generate Scenario Fields ───────────────────────────────────
st.divider()
step_header(4, "Generate Scenario Fields")

if st.session_state["s4_status"] == "done":
    sf = st.session_state["s4_scenario_fields"]
    with st.expander("✅ Scenario fields preview", expanded=False):
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown(f"**Title:** {sf.get('scenarioTitle','')}")
            st.markdown(f"**Difficulty:** {sf.get('difficulty','')}")
            st.markdown(f"**Tone:** {sf.get('tone','')}")
            st.markdown(f"**Time of day:** {sf.get('timeOfDay','')}")
            st.markdown(f"**Tagline:** _{sf.get('tagline','')}_")
            st.markdown(f"**Overall arc:** {sf.get('overallArc','')}")
        with col_d:
            st.markdown("**Situation setup:**")
            st.caption(sf.get('situationSetupForUser',''))
            st.markdown("**Initial messages:**")
            for msg in sf.get('initialMessages', []):
                st.caption(f"→ {msg}")
    if st.button("Regenerate Scenario Fields", key="regen_s4"):
        _clear_from(4); st.rerun()

elif st.session_state["s4_status"] == "error":
    st.error(st.session_state.get("s4_error", "Unknown error"))
    if st.button("Retry Step 4"):
        _clear_from(4); st.rerun()

else:
    if st.button("✨ Generate Scenario Fields", key="run_s4", type="primary", disabled=not prev_done(4)):
        char_fields = st.session_state["s1_char_fields"]
        with st.spinner("Calling Gemini to generate scenario fields…"):
            try:
                sf = generate_scenario_fields(scenario_inputs, char_fields)
                st.session_state["s4_scenario_fields"] = sf
                st.session_state["s4_status"] = "done"
            except Exception as e:
                st.session_state["s4_status"] = "error"
                st.session_state["s4_error"] = str(e)
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
        st.rerun()
    if not prev_done(4):
        st.caption("Complete Step 2 first.")


# ─── STEP 5 — Generate Scenario Poster Image ─────────────────────────────
st.divider()
step_header(5, "Generate Scenario Poster Image")

if st.session_state["s5_status"] == "done":
    st.image(st.session_state["s5_scene_image_path"], width=600)
    with st.expander("Poster prompt used", expanded=False):
        st.caption(st.session_state.get("s5_image_prompt", ""))
    if st.button("Regenerate Poster", key="regen_s5"):
        _clear_from(5); st.rerun()

elif st.session_state["s5_status"] == "error":
    st.error(st.session_state.get("s5_error", "Unknown error"))
    if st.button("Retry Step 5"):
        _clear_from(5); st.rerun()

else:
    if prev_done(5):
        # ── Phase 1: generate structured poster fields ────────────────────
        if not st.session_state["s5_poster_fields"]:
            if st.button("✨ Prepare Poster Fields", key="prep_s5"):
                char_fields = st.session_state["s1_char_fields"] or {}
                scenario_fields = st.session_state["s4_scenario_fields"] or {}
                with st.spinner("Generating structured poster fields…"):
                    try:
                        st.session_state["s5_poster_fields"] = generate_poster_fields(
                            char_fields, scenario_fields
                        )
                    except Exception as e:
                        st.error(f"Field generation failed: {e}")
                st.rerun()

        # ── Phase 2: edit fields + assemble prompt ────────────────────────
        elif not st.session_state["s5_image_prompt"]:
            pf = st.session_state["s5_poster_fields"]
            st.markdown("**Edit poster fields before building the prompt:**")

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                pf["character"] = st.text_area(
                    "Character", value=pf.get("character", ""), height=80, key="pf_character"
                )
                pf["scenario"] = st.text_area(
                    "Scenario / Tension", value=pf.get("scenario", ""), height=80, key="pf_scenario"
                )
                pf["emotion_vibe"] = st.text_area(
                    "Emotion / Vibe & Pose", value=pf.get("emotion_vibe", ""), height=80, key="pf_emotion"
                )
                pf["setting"] = st.text_area(
                    "Setting", value=pf.get("setting", ""), height=80, key="pf_setting"
                )
            with col_p2:
                pf["visual_style"] = st.text_area(
                    "Visual Style", value=pf.get("visual_style", ""), height=80, key="pf_vstyle"
                )
                pf["camera_lighting"] = st.text_area(
                    "Camera & Lighting", value=pf.get("camera_lighting", ""), height=80, key="pf_camera"
                )
                pf["wardrobe"] = st.text_area(
                    "Wardrobe & Details", value=pf.get("wardrobe", ""), height=80, key="pf_wardrobe"
                )
                # text_overlay: checkbox controls inclusion; AI may have returned null
                ai_overlay = pf.get("text_overlay") or ""
                use_overlay = st.checkbox(
                    "Include text overlay",
                    value=bool(ai_overlay.strip()),
                    key="pf_overlay_toggle",
                )
                if use_overlay:
                    pf["text_overlay"] = st.text_input(
                        "Text Overlay (title, font, placement)",
                        value=ai_overlay,
                        key="pf_overlay_text",
                    )
                else:
                    pf["text_overlay"] = ""

            st.session_state["s5_poster_fields"] = pf

            col_s5build, col_s5reset = st.columns([3, 1])
            with col_s5build:
                if st.button("🔨 Build Prompt from Fields", key="build_s5", type="primary"):
                    st.session_state["s5_image_prompt"] = assemble_poster_prompt(pf)
                    st.rerun()
            with col_s5reset:
                if st.button("↺ Re-generate Fields", key="reprep_s5"):
                    st.session_state["s5_poster_fields"] = {}
                    st.rerun()

        # ── Phase 3: edit assembled prompt + generate images ──────────────
        elif not st.session_state["s5_scene_image_paths"]:
            st.markdown("**Final poster prompt** (edit freely before generating):")
            st.session_state["s5_image_prompt"] = st.text_area(
                "Poster generation prompt",
                value=st.session_state["s5_image_prompt"],
                height=220,
                key="s5_prompt_editor",
                label_visibility="collapsed",
            )
            col_s5a, col_s5b, col_s5c = st.columns([3, 1, 1])
            with col_s5a:
                if st.button("🎬 Generate 2 Posters", key="run_s5", type="primary"):
                    with st.spinner("Generating 2 poster options…"):
                        try:
                            paths = generate_images(st.session_state["s5_image_prompt"], count=2)
                            st.session_state["s5_scene_image_paths"] = paths
                        except Exception as e:
                            st.session_state["s5_status"] = "error"
                            st.session_state["s5_error"] = str(e)
                            with st.expander("Error details"):
                                st.code(traceback.format_exc())
                    st.rerun()
            with col_s5b:
                if st.button("↺ Edit Fields", key="back_to_fields_s5"):
                    st.session_state["s5_image_prompt"] = ""
                    st.rerun()
            with col_s5c:
                if st.button("↺ Start Over", key="reprep_s5_full"):
                    st.session_state["s5_poster_fields"] = {}
                    st.session_state["s5_image_prompt"] = ""
                    st.rerun()

        # ── Phase 4: pick from generated posters ─────────────────────────
        else:
            st.caption("Pick the poster you want to use:")
            col_a, col_b = st.columns(2)
            paths = st.session_state["s5_scene_image_paths"]
            with col_a:
                st.image(paths[0], use_container_width=True)
                if st.button("Select Poster A", key="pick_s5_a", type="primary"):
                    st.session_state["s5_scene_image_path"] = paths[0]
                    st.session_state["s5_status"] = "done"
                    st.rerun()
            with col_b:
                st.image(paths[1], use_container_width=True)
                if st.button("Select Poster B", key="pick_s5_b", type="primary"):
                    st.session_state["s5_scene_image_path"] = paths[1]
                    st.session_state["s5_status"] = "done"
                    st.rerun()
            if st.button("↺ Regenerate Both", key="regen_s5_both"):
                st.session_state["s5_scene_image_paths"] = []
                st.rerun()
    else:
        st.caption("Complete Step 4 first.")


# ─── STEP 6 — Generate Beat Fields ───────────────────────────────────────
st.divider()
step_header(6, "Generate Beat Fields")

if st.session_state["s6_status"] == "done":
    beats = st.session_state["s6_beat_fields"]
    st.caption(f"{len(beats)} beats generated")
    for b in beats:
        with st.expander(f"Beat {b.get('beatNumber','?')} — {b.get('beatType','')}", expanded=False):
            st.markdown(f"**Narrative context:** {b.get('narrativeContext','')}")
            st.markdown(f"**Emotional state:** {b.get('characterEmotionalState','')}")
            col_fl, col_hk = st.columns(2)
            with col_fl:
                st.markdown("**Flow directive** _(user engaged)_")
                st.caption(b.get('flowDirective',''))
            with col_hk:
                st.markdown("**Hook directive** _(user disengaged)_")
                st.caption(b.get('hookDirective',''))
            st.caption(
                f"Min turns: {b.get('minTurnsInBeat','?')} · "
                f"Advance score: {b.get('engagedAdvanceScore','?')}"
            )
    if st.button("Regenerate Beats", key="regen_s6"):
        _clear_from(6); st.rerun()

elif st.session_state["s6_status"] == "error":
    st.error(st.session_state.get("s6_error", "Unknown error"))
    if st.button("Retry Step 6"):
        _clear_from(6); st.rerun()

else:
    if st.button("✨ Generate Beat Fields", key="run_s6", type="primary", disabled=not prev_done(6)):
        scenario_fields = st.session_state["s4_scenario_fields"]
        char_fields     = st.session_state["s1_char_fields"]
        with st.spinner("Generating beat fields…"):
            try:
                beats = generate_beat_fields(beat_inputs, scenario_fields, char_fields)
                st.session_state["s6_beat_fields"] = beats
                st.session_state["s6_status"] = "done"
            except Exception as e:
                st.session_state["s6_status"] = "error"
                st.session_state["s6_error"] = str(e)
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
        st.rerun()
    if not prev_done(6):
        st.caption("Complete Step 5 first.")


# ─── STEP 7 — Generate Storyline (Writer + Editor loop) ──────────────────
st.divider()
step_header(7, "Generate Storyline")

_MAX_EDITOR_LOOPS = 3
_SCORE_THRESHOLD  = 6.5

if st.session_state["s7_status"] == "done":
    storyline = st.session_state["s7_storyline"]
    editor_log = st.session_state.get("s7_editor_log", [])

    # Show editor loop history
    if editor_log:
        with st.expander(f"Editor loop history ({len(editor_log)} iteration(s))", expanded=False):
            for entry in editor_log:
                st.markdown(f"**Loop {entry['loop']}** — Score: `{entry['score']}/10`")
                if entry.get("improvements"):
                    for pt in entry["improvements"]:
                        st.caption(f"• {pt}")

    # Editable storyline — user can tweak before passing to Step 8
    st.session_state["s7_storyline"] = st.text_area(
        "Storyline narrative (edit if desired before generating screenplay in Step 8)",
        value=storyline,
        height=350,
        key="s7_storyline_editor",
    )
    if st.button("Regenerate Storyline", key="regen_s7"):
        _clear_from(7); st.rerun()

elif st.session_state["s7_status"] == "error":
    st.error(st.session_state.get("s7_error", "Unknown error"))
    if st.button("Retry Step 7"):
        _clear_from(7); st.rerun()

else:
    if prev_done(7):
        if st.button("📖 Generate Storyline", key="run_s7", type="primary"):
            ch = st.session_state["s1_char_fields"]
            sc = st.session_state["s4_scenario_fields"]
            bf = st.session_state["s6_beat_fields"]
            editor_log = []
            try:
                progress_ph = st.empty()
                progress_ph.info("Writer agent drafting storyline…")
                storyline = generate_storyline(ch, sc, bf)

                for loop in range(_MAX_EDITOR_LOOPS):
                    progress_ph.info(f"Editor agent reviewing storyline (loop {loop + 1}/{_MAX_EDITOR_LOOPS})…")
                    critique = critique_storyline(storyline, ch, sc, bf)
                    editor_log.append({
                        "loop": loop + 1,
                        "score": critique["score"],
                        "improvements": critique.get("improvements", []),
                    })
                    if critique["score"] >= _SCORE_THRESHOLD:
                        progress_ph.success(f"Storyline approved by editor (score {critique['score']}/10)")
                        break
                    progress_ph.info(
                        f"Score {critique['score']}/10 — rewriting with editor notes (loop {loop + 1})…"
                    )
                    storyline = generate_storyline(ch, sc, bf, editor_notes=critique["improvements"])
                else:
                    progress_ph.warning(
                        f"Max editor loops reached — using best version (score {editor_log[-1]['score']}/10)"
                    )

                st.session_state["s7_storyline"]   = storyline
                st.session_state["s7_editor_log"]  = editor_log
                st.session_state["s7_status"]      = "done"
            except Exception as e:
                st.session_state["s7_status"] = "error"
                st.session_state["s7_error"]  = str(e)
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
            st.rerun()
    else:
        st.caption("Complete Step 6 first.")


# ─── STEP 8 — Generate Video Screenplay ──────────────────────────────────
st.divider()
step_header(8, "Generate Video Screenplay")

if st.session_state["s8_status"] == "done":
    script = st.session_state["s8_script"]
    segs   = script.get("segments", [])
    st.caption(f"{len(segs)} segments · {len(segs) * seg_dur}s total")
    for seg in segs:
        sn = seg["segment_number"]
        with st.expander(f"Segment {sn}", expanded=(sn == 1)):
            c1, c2, c3 = st.columns(3)
            c1.markdown("**Scene**");    c1.caption(seg.get("scene_description", ""))
            c2.markdown("**Shot**");     c2.caption(seg.get("shot_description", ""))
            c3.markdown("**Dialogue**"); c3.caption(seg.get("dialogue", ""))
    if st.button("Regenerate Script", key="regen_s8"):
        _clear_from(8); st.rerun()

elif st.session_state["s8_status"] == "error":
    st.error(st.session_state.get("s8_error", "Unknown error"))
    if st.button("Retry Step 8"):
        _clear_from(8); st.rerun()

else:
    if prev_done(8):
        if st.button("📜 Generate Video Screenplay", key="run_s8", type="primary"):
            sc = st.session_state["s4_scenario_fields"]
            ch = st.session_state["s1_char_fields"]
            target = 35
            num_segs = max(4, round(target / seg_dur))
            with st.spinner("Writing roleplay teaser screenplay…"):
                try:
                    script = generate_video_scene_script(sc, ch, num_segs, seg_dur)
                    st.session_state["s8_script"] = script
                    st.session_state["s8_status"] = "done"
                except Exception as e:
                    st.session_state["s8_status"] = "error"
                    st.session_state["s8_error"] = str(e)
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
            st.rerun()
    else:
        st.caption("Complete Step 7 first.")


# ─── STEP 9 — Generate Video Segments (Veo3, chained) ────────────────────
st.divider()
step_header(9, "Generate Video Segments (Veo3)")
_s8_done = st.session_state["s8_status"] == "done"


def _build_segment_prompt(seg: dict) -> str:
    """Build the default Veo3 prompt for a single segment."""
    parts = [seg.get("scene_description", "").strip(), seg.get("shot_description", "").strip()]
    dialogue = seg.get("dialogue", "").strip()
    if dialogue:
        parts.append(f'The character speaks: "{dialogue}"')
    cont = seg.get("continuation_note", "").strip()
    if cont:
        parts.append(cont)
    return " ".join(p for p in parts if p)


def _run_step9():
    script      = st.session_state["s8_script"]
    segments    = script["segments"]
    total       = script["total_segments"]
    done_so_far = st.session_state["s9_segment_paths"]
    resume_from = len(done_so_far)
    seg_prompts = st.session_state.get("s9_segment_prompts", {})

    current_ref   = st.session_state["s2_image_path"] if resume_from == 0 else extract_last_frame(done_so_far[-1])
    segment_paths = list(done_so_far)

    try:
        for i, seg in enumerate(segments[resume_from:], start=resume_from):
            n    = seg["segment_number"]
            cont = seg.get("continuation_note", "") if n > 1 else ""

            # Use user-edited prompt if available, else build default
            custom_prompt = seg_prompts.get(str(n), "").strip()

            status_text = st.empty()
            seg_start   = time.time()
            status_text.info(f"**Segment {n}/{total}** — submitting to Veo3…")

            def _make_cb(ph, sn, st_total, t0):
                def _cb(elapsed, status):
                    mins, secs = divmod(elapsed, 60)
                    tstr = f"{mins}m {secs}s" if mins else f"{secs}s"
                    ph.info(f"**Segment {sn}/{st_total}** — Veo3 generating… Status: `{status}` · {tstr}")
                return _cb

            if custom_prompt:
                # User provided a full custom prompt — pass scene/shot as empty to avoid duplication
                from steps.veo3_generator import _critique_veo3_prompt, _rewrite_veo3_prompt
                _MAX_TRIES = 3
                prompt = custom_prompt
                for _att in range(_MAX_TRIES):
                    _cr = _critique_veo3_prompt(prompt)
                    if _cr["passes"]:
                        break
                    prompt = _rewrite_veo3_prompt(prompt, _cr["issues"])
                video_path = generate_video_segment(
                    image_path=current_ref,
                    scene_description=prompt,
                    shot_description="",
                    segment_number=n,
                    dialogue="",
                    continuation_note="",
                    duration=seg_dur,
                    on_progress=_make_cb(status_text, n, total, seg_start),
                )
            else:
                video_path = generate_video_segment(
                    image_path=current_ref,
                    scene_description=seg.get("scene_description", ""),
                    shot_description=seg.get("shot_description", ""),
                    segment_number=n,
                    dialogue=seg.get("dialogue", ""),
                    continuation_note=cont,
                    duration=seg_dur,
                    on_progress=_make_cb(status_text, n, total, seg_start),
                )

            seg_elapsed = int(time.time() - seg_start)
            segment_paths.append(video_path)
            st.session_state["s9_segment_paths"] = segment_paths
            status_text.success(f"Segment {n}/{total} done in {seg_elapsed}s")
            st.video(video_path)

            if n < total:
                current_ref = extract_last_frame(video_path)
                cooldown = 10
                status_text.info(f"Cooling down {cooldown}s before segment {n+1}…")
                time.sleep(cooldown)

        st.session_state["s9_status"] = "done"

    except Exception as e:
        st.session_state["s9_status"] = "error"
        st.session_state["s9_error"] = str(e)
        st.error(f"Failed on segment {len(segment_paths)+1}: {e}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        st.rerun()


if st.session_state["s9_status"] == "done":
    for idx, vp in enumerate(st.session_state["s9_segment_paths"], 1):
        st.caption(f"Segment {idx}")
        st.video(vp)
    if st.button("Regenerate All Segments", key="regen_s9"):
        _clear_from(9); st.rerun()

elif st.session_state["s9_status"] == "error":
    done = st.session_state["s9_segment_paths"]
    if done:
        st.warning(f"{len(done)} segment(s) completed. Resume will continue from segment {len(done)+1}.")
        for idx, vp in enumerate(done, 1):
            st.caption(f"Segment {idx} (done)"); st.video(vp)
    st.error(st.session_state.get("s9_error", "Unknown error"))
    col_ra, col_rb = st.columns(2)
    with col_ra:
        if st.button("Resume from last completed", key="resume_s9", type="primary"):
            st.session_state["s9_status"] = "pending"; st.rerun()
    with col_rb:
        if st.button("Restart All", key="restart_s9"):
            _clear_from(9); st.rerun()

else:
    if _s8_done:
        # Show editable per-segment prompts before generation
        script    = st.session_state["s8_script"]
        segments  = script.get("segments", [])
        seg_prompts = st.session_state.get("s9_segment_prompts", {})
        if segments:
            st.caption("Review and edit per-segment Veo3 prompts before generating. Leave blank to use auto-built prompts.")
            for seg in segments:
                sn = seg["segment_number"]
                default_p = _build_segment_prompt(seg)
                edited = st.text_area(
                    f"Segment {sn} prompt",
                    value=seg_prompts.get(str(sn), default_p),
                    height=100,
                    key=f"s9_prompt_seg{sn}",
                )
                seg_prompts[str(sn)] = edited
            st.session_state["s9_segment_prompts"] = seg_prompts

        if st.button("🎬 Generate Video Segments", key="run_s9", type="primary"):
            _run_step9(); st.rerun()
    else:
        st.button("🎬 Generate Video Segments", key="run_s9_dis", type="primary", disabled=True)
        st.caption("Complete Step 8 first.")

if st.session_state["s9_status"] == "pending" and _s8_done and st.session_state["s9_segment_paths"]:
    _run_step9(); st.rerun()


# ─── STEP 10 — Concatenate Segments ──────────────────────────────────────
st.divider()
step_header(10, "Concatenate Video Segments")

if st.session_state["s10_status"] == "done":
    st.video(st.session_state["s10_combined_path"])
    if st.button("Regenerate Concat", key="regen_s10"):
        _clear_from(10); st.rerun()

elif st.session_state["s10_status"] == "error":
    st.error(st.session_state.get("s10_error", "Unknown error"))
    if st.button("Retry Step 10"):
        _clear_from(10); st.rerun()

else:
    if st.button("🔗 Concatenate Segments", key="run_s10", type="primary", disabled=not prev_done(10)):
        with st.spinner("Concatenating…"):
            try:
                combined = concatenate_segments(st.session_state["s9_segment_paths"])
                st.session_state["s10_combined_path"] = combined
                st.session_state["s10_status"] = "done"
            except Exception as e:
                st.session_state["s10_status"] = "error"
                st.session_state["s10_error"] = str(e)
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
        st.rerun()
    if not prev_done(10):
        st.caption("Complete Step 9 first.")


# ─── STEP 11 — Final Video ────────────────────────────────────────────────
st.divider()
step_header(11, "Final Video")

if st.session_state["s11_status"] == "done":
    st.video(st.session_state["s11_final_path"])
    if st.button("Reset Final", key="regen_s11"):
        _clear_from(11); st.rerun()

elif st.session_state["s11_status"] == "error":
    st.error(st.session_state.get("s11_error", "Unknown error"))
    if st.button("Retry Step 11"):
        _clear_from(11); st.rerun()

else:
    if st.button("✅ Set as Final Video", key="run_s11", type="primary", disabled=not prev_done(11)):
        try:
            st.session_state["s11_final_path"] = st.session_state["s10_combined_path"]
            st.session_state["s11_status"] = "done"
        except Exception as e:
            st.session_state["s11_status"] = "error"
            st.session_state["s11_error"] = str(e)
        st.rerun()
    if not prev_done(11):
        st.caption("Complete Step 10 first.")
    else:
        st.caption("Veo3 generates audio natively — the concatenated video is your final output.")


# ══════════════════════════════════════════════════════════════════════════
# 💾 SAVE ALL — Upload media to Supabase + write to DB
# ══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("💾 Save to Database")

all_media_done = (
    st.session_state["s2_image_path"]
    and st.session_state["s4_scenario_fields"]
    and st.session_state["s5_scene_image_path"]
    and st.session_state["s6_beat_fields"]
)

if not all_media_done:
    st.info("Complete at least Steps 1–6 to enable saving.")
else:
    if st.button("💾 Save Character, Scenario & Beats to DB", type="primary"):
        from utils.supabase_client import upload_image, upload_audio, upload_video
        from utils.db_saver import save_character, save_scenario, save_beats

        with st.spinner("Uploading media to Supabase & saving to DB…"):
            try:
                progress = st.empty()

                # 1. Upload character portrait
                progress.info("Uploading character portrait…")
                char_image_url = upload_image(st.session_state["s2_image_path"], prefix="characters/images")

                # 2. Upload character voice
                char_audio_url = None
                if st.session_state.get("s3_audio_path"):
                    progress.info("Uploading character voice…")
                    char_audio_url = upload_audio(st.session_state["s3_audio_path"], prefix="characters/audio")

                # 3. Save character to DB
                progress.info("Saving character to DB…")
                char_data = {
                    **st.session_state["s1_char_fields"],
                    "imageUrl": char_image_url,
                    "voiceAudioUrl": char_audio_url,
                }
                char_id = save_character(char_data)

                # 4. Upload scenario scene image
                progress.info("Uploading scenario image…")
                scene_image_url = upload_image(st.session_state["s5_scene_image_path"], prefix="scenarios/images")

                # 5. Upload final video (if done) or skip
                video_url = None
                if st.session_state["s11_final_path"]:
                    progress.info("Uploading final video…")
                    video_url = upload_video(st.session_state["s11_final_path"], prefix="scenarios/videos")

                # 6. Save scenario to DB
                progress.info("Saving scenario to DB…")
                scenario_data = {
                    **st.session_state["s4_scenario_fields"],
                    "imageUrl": scene_image_url,
                    "videoUrl": video_url,
                }
                scenario_id = save_scenario(scenario_data, char_id)

                # 7. Save beats to DB
                progress.info("Saving beats to DB…")
                save_beats(st.session_state["s6_beat_fields"], scenario_id, char_id)

                progress.empty()
                st.success(
                    f"✅ Saved! Character ID: `{char_id}` · Scenario ID: `{scenario_id}` · "
                    f"{len(st.session_state['s6_beat_fields'])} beats"
                )
                st.balloons()

            except Exception as e:
                st.error(f"Save failed: {e}")
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
