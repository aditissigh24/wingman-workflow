import time
import traceback

import streamlit as st

from config import validate_keys
from steps.audio_generator import generate_audio, list_voices
from steps.image_generator import generate_image
from steps.media_merger import concatenate_segments, merge_audio_video
from steps.prompt_enhancer import enhance_audio_prompt, enhance_image_prompt, generate_multi_scene_script
from steps.video_generator import extract_last_frame, generate_video_segment

st.set_page_config(page_title="AI Media Pipeline", page_icon="🎬", layout="wide")
st.title("AI Media Pipeline")
st.caption("Input → Portrait → Script → Audio → Multi-segment Kling Video → Final")

# ── Session state initialisation ───────────────────────────────────────────
_STEP_OUTPUTS = {
    "s1_image_prompt": "",
    "s2_image_path": "",
    "s3_script": None,
    "s4_enhanced_dialogue": "",
    "s5_audio_path": "",
    "s6_segment_paths": [],
    "s7_combined_path": "",
    "s8_final_path": "",
}
_STEP_STATUSES = {f"s{n}_status": "pending" for n in range(1, 9)}
_EDIT_KEYS = {
    "edit_image_prompt": "",
    "edit_enhanced_dialogue": "",
}

for key, default in {**_STEP_OUTPUTS, **_STEP_STATUSES, **_EDIT_KEYS}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if "pipeline_started" not in st.session_state:
    st.session_state["pipeline_started"] = False
if "structured_input" not in st.session_state:
    st.session_state["structured_input"] = {}
if "segment_duration" not in st.session_state:
    st.session_state["segment_duration"] = 5
if "selected_voice_id" not in st.session_state:
    st.session_state["selected_voice_id"] = None


# ── Helpers ─────────────────────────────────────────────────────────────────

def reset_from_step(n: int):
    """Clear all outputs and statuses for steps n through 8."""
    step_output_map = {
        1: ["s1_image_prompt", "edit_image_prompt"],
        2: ["s2_image_path"],
        3: ["s3_script"],
        4: ["s4_enhanced_dialogue", "edit_enhanced_dialogue"],
        5: ["s5_audio_path"],
        6: ["s6_segment_paths"],
        7: ["s7_combined_path"],
        8: ["s8_final_path"],
    }
    # Keys that are bound to st widgets must be *popped* (not assigned) so that
    # Streamlit resets them to their default on the next render.  Assigning to
    # them after the widget has been instantiated raises StreamlitAPIException.
    widget_keys = {"edit_image_prompt", "edit_enhanced_dialogue"}

    non_widget_defaults = {
        "s1_image_prompt": "",
        "s2_image_path": "",
        "s3_script": None,
        "s4_enhanced_dialogue": "",
        "s5_audio_path": "",
        "s6_segment_paths": [],
        "s7_combined_path": "",
        "s8_final_path": "",
    }

    # Also clear per-segment edit keys when resetting step 3+
    if n <= 3 and st.session_state.get("s3_script"):
        script = st.session_state["s3_script"]
        if script:
            for seg in script.get("segments", []):
                sn = seg["segment_number"]
                for field in ("scene", "shot", "dialogue"):
                    st.session_state.pop(f"edit_seg_{sn}_{field}", None)

    for step in range(n, 9):
        st.session_state[f"s{step}_status"] = "pending"
        for key in step_output_map.get(step, []):
            if key in widget_keys:
                st.session_state.pop(key, None)
            elif key in non_widget_defaults:
                st.session_state[key] = non_widget_defaults[key]


def step_header(n: int, title: str):
    """Render a step title with a status badge."""
    status = st.session_state[f"s{n}_status"]
    if status == "done":
        badge = ":green[✓ Done]"
    elif status == "error":
        badge = ":red[✗ Error]"
    else:
        badge = ":gray[○ Pending]"
    st.markdown(f"### Step {n} — {title} &nbsp; {badge}")


def prev_step_done(n: int) -> bool:
    """Return True if the step before n is complete (or n == 1 and pipeline started)."""
    if n == 1:
        return st.session_state["pipeline_started"]
    return st.session_state[f"s{n-1}_status"] == "done"


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    missing = validate_keys()
    if missing:
        st.error(f"Missing API keys in .env: {', '.join(missing)}")
        st.info("Copy `.env.example` to `.env` and fill in your keys.")
    else:
        st.success("All API keys configured")

    st.divider()
    st.subheader("Settings")

    segment_duration = st.selectbox(
        "Segment duration (seconds)",
        [5, 10],
        index=0,
        help=(
            "Each Kling clip is this long. "
            "5 s → ~7 segments (~35 s total). "
            "10 s → ~4 segments (~40 s total)."
        ),
    )
    st.session_state["segment_duration"] = segment_duration

    try:
        available_voices = list_voices()
    except Exception:
        available_voices = []

    if available_voices:
        voice_labels = [v["name"] for v in available_voices]
        voice_ids = [v["voice_id"] for v in available_voices]
        selected_voice_label = st.selectbox(
            "Voiceover voice",
            voice_labels,
            index=0,
            help="Voices available on your ElevenLabs account.",
        )
        st.session_state["selected_voice_id"] = voice_ids[voice_labels.index(selected_voice_label)]
    else:
        st.warning("Could not load ElevenLabs voices. Using account default.")
        st.session_state["selected_voice_id"] = None

# ── Input Form ──────────────────────────────────────────────────────────────
st.subheader("Describe Your Video")

with st.expander("Character", expanded=not st.session_state["pipeline_started"]):
    st.caption("Who is in the video? This drives the portrait image generation.")
    col_char1, col_char2 = st.columns(2)
    with col_char1:
        character_description = st.text_area(
            "Character description *",
            placeholder="e.g., 30-year-old South Asian woman, sharp jawline, dark wavy hair, warm brown eyes",
            height=100,
            value=st.session_state["structured_input"].get("character_description", ""),
            help="Physical appearance — age, gender, ethnicity, facial features, hair.",
        )
        personality = st.text_input(
            "Personality / vibe",
            placeholder="e.g., confident, warm, intellectually curious",
            value=st.session_state["structured_input"].get("personality", ""),
            help="The character's inner quality — reflected in expression and posture.",
        )
    with col_char2:
        outfit = st.text_area(
            "Outfit / clothing",
            placeholder="e.g., navy blazer over a white shirt, minimalist silver jewelry",
            height=100,
            value=st.session_state["structured_input"].get("outfit", ""),
            help="What the character is wearing.",
        )
        character_action = st.text_input(
            "Character action / energy",
            placeholder="e.g., gesturing naturally, calm and composed, leaning forward",
            value=st.session_state["structured_input"].get("character_action", ""),
            help="How does the character move and carry themselves?",
        )

with st.expander("Scene & Visual Style", expanded=not st.session_state["pipeline_started"]):
    st.caption("Where does the scene take place? What does it look like?")
    col_scene1, col_scene2, col_scene3 = st.columns(3)
    with col_scene1:
        location = st.text_input(
            "Location / environment",
            placeholder="e.g., modern minimalist studio, outdoor cliffside at golden hour",
            value=st.session_state["structured_input"].get("location", ""),
            help="The physical setting of the scene.",
        )
    with col_scene2:
        mood = st.text_input(
            "Mood / atmosphere",
            placeholder="e.g., calm and authoritative, energetic and motivational",
            value=st.session_state["structured_input"].get("mood", ""),
            help="The emotional tone of the visual environment.",
        )
    with col_scene3:
        _style_options = ["Cinematic", "Documentary", "Commercial / Ad", "Vlog / Authentic", "Artistic / Editorial"]
        _style_saved = st.session_state["structured_input"].get("visual_style", "Cinematic")
        _style_idx = _style_options.index(_style_saved) if _style_saved in _style_options else 0
        visual_style = st.selectbox(
            "Visual style",
            _style_options,
            index=_style_idx,
            help="Overall visual aesthetic — affects lighting, framing, and color grade.",
        )

with st.expander("Content & Audience", expanded=not st.session_state["pipeline_started"]):
    st.caption("What is the video about and who is it for?")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        intent = st.text_area(
            "Video intent / purpose *",
            placeholder="e.g., Explain the top 3 benefits of meditation to beginners",
            height=100,
            value=st.session_state["structured_input"].get("intent", ""),
            help="What should the viewer walk away knowing or feeling?",
        )
        target_audience = st.text_input(
            "Target audience",
            placeholder="e.g., first-time founders aged 25–40, busy parents, fitness beginners",
            value=st.session_state["structured_input"].get("target_audience", ""),
            help="Who is watching? Age, context, level of knowledge.",
        )
    with col_c2:
        _vtype_options = [
            "Motivational", "Tutorial / How-to", "Product Promo",
            "Testimonial / Story", "Educational", "News / Announcement", "Storytelling",
        ]
        _vtype_saved = st.session_state["structured_input"].get("video_type", "Motivational")
        _vtype_idx = _vtype_options.index(_vtype_saved) if _vtype_saved in _vtype_options else 0
        video_type = st.selectbox(
            "Video type / format",
            _vtype_options,
            index=_vtype_idx,
            help="The genre of the video — shapes structure, pacing, and tone.",
        )
        additional_context = st.text_area(
            "Additional context",
            placeholder="e.g., Brand name: MindFlow. Key stats: 80% of users see results in 7 days.",
            height=100,
            value=st.session_state["structured_input"].get("additional_context", ""),
            help="Brand names, product details, key facts, statistics, or background information.",
        )

with st.expander("Script & Dialogue", expanded=not st.session_state["pipeline_started"]):
    st.caption("What does the character say? This drives the full voiceover script.")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        key_message = st.text_area(
            "Key message / topic",
            placeholder="e.g., Meditation reduces stress, improves focus, and takes only 5 minutes a day",
            height=100,
            value=st.session_state["structured_input"].get("key_message", ""),
            help="The core content of the dialogue — main points to communicate.",
        )
        emotional_arc = st.text_area(
            "Emotional arc / narrative journey",
            placeholder="e.g., Open with the viewer's pain point → build empathy → introduce the solution → inspire action",
            height=100,
            value=st.session_state["structured_input"].get("emotional_arc", ""),
            help="How should the viewer feel at each stage? This shapes the script structure.",
        )
    with col_d2:
        call_to_action = st.text_input(
            "Call to action",
            placeholder="e.g., Download the app today, Visit our website, Subscribe now",
            value=st.session_state["structured_input"].get("call_to_action", ""),
            help="What should the viewer do after watching?",
        )
        _tone_options = ["Conversational", "Professional", "Inspiring", "Storytelling", "Authoritative", "Energetic", "Empathetic"]
        _tone_saved = st.session_state["structured_input"].get("speaking_tone", "Conversational")
        _tone_idx = _tone_options.index(_tone_saved) if _tone_saved in _tone_options else 0
        speaking_tone = st.selectbox(
            "Speaking tone",
            _tone_options,
            index=_tone_idx,
            help="The delivery style for the dialogue.",
        )

# ── Start / Reset buttons ───────────────────────────────────────────────────
col_start, col_reset = st.columns([2, 1])

with col_start:
    start_clicked = st.button(
        "Start Pipeline" if not st.session_state["pipeline_started"] else "Update & Restart Pipeline",
        type="primary",
        disabled=bool(missing),
    )

with col_reset:
    if st.session_state["pipeline_started"]:
        if st.button("Reset All", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

missing_fields = [
    label
    for label, val in {
        "Character description": character_description,
        "Video intent / purpose": intent,
    }.items()
    if not val.strip()
]

if start_clicked and missing_fields:
    st.warning(f"Please fill in the required fields: {', '.join(missing_fields)}")

if start_clicked and not missing_fields:
    new_input = {
        "character_description": character_description.strip(),
        "personality": personality.strip(),
        "outfit": outfit.strip(),
        "location": location.strip(),
        "mood": mood.strip(),
        "visual_style": visual_style,
        "intent": intent.strip(),
        "character_action": character_action.strip(),
        "target_audience": target_audience.strip(),
        "video_type": video_type,
        "additional_context": additional_context.strip(),
        "key_message": key_message.strip(),
        "emotional_arc": emotional_arc.strip(),
        "call_to_action": call_to_action.strip(),
        "speaking_tone": speaking_tone,
    }
    if new_input != st.session_state["structured_input"]:
        # Input changed — full reset
        reset_from_step(1)
    st.session_state["structured_input"] = new_input
    st.session_state["pipeline_started"] = True
    st.rerun()

# ── Pipeline steps (only shown after pipeline is started) ───────────────────
if not st.session_state["pipeline_started"]:
    st.info("Fill in the form above and click **Start Pipeline** to begin.")
    st.stop()

st.divider()
st.subheader("Pipeline")

structured_input = st.session_state["structured_input"]
seg_dur = st.session_state["segment_duration"]
voice_id = st.session_state["selected_voice_id"]

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Enhance image prompt
# ═══════════════════════════════════════════════════════════════════════════
step_header(1, "Enhance Portrait Prompt")

s1_status = st.session_state["s1_status"]

if s1_status == "done":
    st.text_area(
        "Portrait prompt (editable — changes feed into Step 2)",
        key="edit_image_prompt",
        height=160,
    )
    col_r1a, col_r1b = st.columns([1, 4])
    with col_r1a:
        if st.button("Regenerate", key="regen_s1"):
            reset_from_step(1)
            st.rerun()

elif s1_status == "error":
    st.error(st.session_state.get("s1_error", "Unknown error"))
    if st.button("Retry Step 1", key="retry_s1"):
        reset_from_step(1)
        st.rerun()

else:
    if st.button("Generate Portrait Prompt", key="run_s1", type="primary"):
        with st.spinner("Calling Gemini to enhance portrait prompt..."):
            try:
                result = enhance_image_prompt(structured_input)
                st.session_state["s1_image_prompt"] = result
                st.session_state["edit_image_prompt"] = result
                st.session_state["s1_status"] = "done"
            except Exception as e:
                st.session_state["s1_status"] = "error"
                st.session_state["s1_error"] = str(e)
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Generate character portrait
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
step_header(2, "Generate Character Portrait")

s2_status = st.session_state["s2_status"]
_s1_done = st.session_state["s1_status"] == "done"

if s2_status == "done":
    st.image(st.session_state["s2_image_path"], width=400)
    if st.button("Regenerate Portrait", key="regen_s2"):
        reset_from_step(2)
        st.rerun()

elif s2_status == "error":
    st.error(st.session_state.get("s2_error", "Unknown error"))
    if st.button("Retry Step 2", key="retry_s2"):
        reset_from_step(2)
        st.rerun()

else:
    if st.button("Generate Portrait", key="run_s2", type="primary", disabled=not _s1_done):
        prompt_to_use = st.session_state.get("edit_image_prompt") or st.session_state["s1_image_prompt"]
        with st.spinner("Generating portrait with Gemini..."):
            try:
                path = generate_image(prompt_to_use)
                st.session_state["s2_image_path"] = path
                st.session_state["s2_status"] = "done"
            except Exception as e:
                st.session_state["s2_status"] = "error"
                st.session_state["s2_error"] = str(e)
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
        st.rerun()
    if not _s1_done:
        st.caption("Complete Step 1 first.")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Generate multi-scene screenplay
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
step_header(3, "Generate Multi-Scene Screenplay")

s3_status = st.session_state["s3_status"]
_s2_done = st.session_state["s2_status"] == "done"

if s3_status == "done":
    script = st.session_state["s3_script"]
    segments = script["segments"]
    st.caption(f"{len(segments)} segments · {len(segments) * seg_dur}s total — all fields are editable")
    if st.session_state.get("s3_truncation_warning"):
        st.warning(st.session_state["s3_truncation_warning"])

    for seg in segments:
        n = seg["segment_number"]
        # Initialise edit keys on first render after generation
        for field, val in [("scene", seg["scene_description"]), ("shot", seg["shot_description"]), ("dialogue", seg["dialogue"])]:
            k = f"edit_seg_{n}_{field}"
            if k not in st.session_state:
                st.session_state[k] = val

        with st.expander(f"Segment {n} of {script['total_segments']}", expanded=(n == 1)):
            col_s, col_sh, col_d = st.columns(3)
            with col_s:
                st.markdown("**Scene (environment)**")
                st.text_area(
                    label=f"scene_{n}",
                    key=f"edit_seg_{n}_scene",
                    height=120,
                    label_visibility="collapsed",
                )
            with col_sh:
                st.markdown("**Shot (character motion)**")
                st.text_area(
                    label=f"shot_{n}",
                    key=f"edit_seg_{n}_shot",
                    height=120,
                    label_visibility="collapsed",
                )
            with col_d:
                st.markdown("**Dialogue**")
                st.text_area(
                    label=f"dialogue_{n}",
                    key=f"edit_seg_{n}_dialogue",
                    height=120,
                    label_visibility="collapsed",
                )
            if seg.get("continuation_note"):
                st.caption(f"Continuation note: {seg['continuation_note']}")

    if st.button("Regenerate Script", key="regen_s3"):
        reset_from_step(3)
        st.rerun()

elif s3_status == "error":
    st.error(st.session_state.get("s3_error", "Unknown error"))
    if st.button("Retry Step 3", key="retry_s3"):
        reset_from_step(3)
        st.rerun()

else:
    if st.button("Generate Script", key="run_s3", type="primary", disabled=not _s2_done):
        prompt_used = st.session_state.get("edit_image_prompt") or st.session_state["s1_image_prompt"]
        with st.spinner("Writing multi-scene screenplay with Gemini..."):
            try:
                script = generate_multi_scene_script(
                    structured_input, prompt_used, segment_duration=seg_dur
                )
                # Validate the model returned all segments
                expected = script.get("total_segments", 0)
                actual = len(script.get("segments", []))
                if actual < expected:
                    # Patch total_segments to match what was actually returned so
                    # downstream steps don't silently miss segments
                    script["total_segments"] = actual
                    st.session_state["s3_truncation_warning"] = (
                        f"Gemini only returned {actual} of {expected} requested segments. "
                        "You can regenerate to try again, or proceed with fewer segments."
                    )
                else:
                    st.session_state.pop("s3_truncation_warning", None)
                st.session_state["s3_script"] = script
                st.session_state["s3_status"] = "done"
            except Exception as e:
                st.session_state["s3_status"] = "error"
                st.session_state["s3_error"] = str(e)
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
        st.rerun()
    if not _s2_done:
        st.caption("Complete Step 2 first.")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Enhance dialogue for TTS
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
step_header(4, "Enhance Dialogue for TTS Pacing")

s4_status = st.session_state["s4_status"]
_s3_done = st.session_state["s3_status"] == "done"

if s4_status == "done":
    st.text_area(
        "Pacing-enhanced dialogue (editable — changes feed into Step 5 audio generation)",
        key="edit_enhanced_dialogue",
        height=160,
    )
    if st.button("Regenerate", key="regen_s4"):
        reset_from_step(4)
        st.rerun()

elif s4_status == "error":
    st.error(st.session_state.get("s4_error", "Unknown error"))
    if st.button("Retry Step 4", key="retry_s4"):
        reset_from_step(4)
        st.rerun()

else:
    if st.button("Enhance Dialogue", key="run_s4", type="primary", disabled=not _s3_done):
        script = st.session_state["s3_script"]
        segments = script["segments"]
        # Use edited dialogue if the user changed it, otherwise fall back to
        # the raw value from the script. Never silently drop a segment.
        def _seg_dialogue(seg):
            key = f"edit_seg_{seg['segment_number']}_dialogue"
            edited = st.session_state.get(key, "").strip()
            return edited if edited else seg["dialogue"]

        full_dialogue = " ".join(_seg_dialogue(seg) for seg in segments)
        st.caption(f"Building audio from {len(segments)} segment(s).")
        with st.spinner("Adding TTS pacing cues with Gemini..."):
            try:
                enhanced = enhance_audio_prompt(full_dialogue)
                st.session_state["s4_enhanced_dialogue"] = enhanced
                st.session_state["edit_enhanced_dialogue"] = enhanced
                st.session_state["s4_status"] = "done"
            except Exception as e:
                st.session_state["s4_status"] = "error"
                st.session_state["s4_error"] = str(e)
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
        st.rerun()
    if not _s3_done:
        st.caption("Complete Step 3 first.")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Generate voiceover audio
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
step_header(5, "Generate Voiceover")

s5_status = st.session_state["s5_status"]
_s4_done = st.session_state["s4_status"] == "done"

if s5_status == "done":
    st.audio(st.session_state["s5_audio_path"])
    if st.button("Regenerate Audio", key="regen_s5"):
        reset_from_step(5)
        st.rerun()

elif s5_status == "error":
    st.error(st.session_state.get("s5_error", "Unknown error"))
    if st.button("Retry Step 5", key="retry_s5"):
        reset_from_step(5)
        st.rerun()

else:
    if st.button("Generate Audio", key="run_s5", type="primary", disabled=not _s4_done):
        dialogue_to_use = st.session_state.get("edit_enhanced_dialogue") or st.session_state["s4_enhanced_dialogue"]
        with st.spinner("Generating voiceover with ElevenLabs..."):
            try:
                path = generate_audio(dialogue_to_use, voice_id)
                st.session_state["s5_audio_path"] = path
                st.session_state["s5_status"] = "done"
            except Exception as e:
                st.session_state["s5_status"] = "error"
                st.session_state["s5_error"] = str(e)
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
        st.rerun()
    if not _s4_done:
        st.caption("Complete Step 4 first.")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — Generate video segments (Kling, chained)
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
step_header(6, "Generate Video Segments (Kling)")

s6_status = st.session_state["s6_status"]
_s5_done = st.session_state["s5_status"] == "done"

def _run_step6():
    """Generate Kling video segments, resuming from the last completed one."""
    script = st.session_state["s3_script"]
    segments = script["segments"]
    total = script["total_segments"]
    already_done = st.session_state["s6_segment_paths"]  # clips already generated
    resume_from = len(already_done)  # 0-based index into segments list

    # Determine the reference image for the segment we're resuming from
    if resume_from == 0:
        current_ref = st.session_state["s2_image_path"]
    else:
        # Extract last frame of the most recently completed clip
        current_ref = extract_last_frame(already_done[-1])

    segment_paths = list(already_done)
    pipeline_start = time.time()

    try:
        for i, seg in enumerate(segments[resume_from:], start=resume_from):
            n = seg["segment_number"]
            scene = st.session_state.get(f"edit_seg_{n}_scene", seg["scene_description"])
            shot = st.session_state.get(f"edit_seg_{n}_shot", seg["shot_description"])
            cont_note = seg.get("continuation_note", "") if n > 1 else ""

            status_text = st.empty()
            seg_start = time.time()

            status_text.info(
                f"**Segment {n}/{total}** — submitting to Kling…  "
                f"(segments remaining: {total - n + 1})"
            )

            def _make_progress_cb(placeholder, seg_n, seg_total, t0):
                def _on_progress(elapsed: int, status: str):
                    mins, secs = divmod(elapsed, 60)
                    time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
                    placeholder.info(
                        f"**Segment {seg_n}/{seg_total}** — Kling is generating…  \n"
                        f"Status: `{status}` · Elapsed: **{time_str}** "
                        f"(segments remaining: {seg_total - seg_n})"
                    )
                return _on_progress

            video_path = generate_video_segment(
                image_path=current_ref,
                scene_description=scene,
                shot_description=shot,
                segment_number=n,
                continuation_note=cont_note,
                duration=seg_dur,
                on_progress=_make_progress_cb(status_text, n, total, seg_start),
            )

            seg_elapsed = int(time.time() - seg_start)
            pipeline_elapsed = int(time.time() - pipeline_start)
            mins_s, secs_s = divmod(seg_elapsed, 60)
            mins_p, secs_p = divmod(pipeline_elapsed, 60)
            seg_time_str = f"{mins_s}m {secs_s}s" if mins_s else f"{secs_s}s"
            pipe_time_str = f"{mins_p}m {secs_p}s" if mins_p else f"{secs_p}s"

            segment_paths.append(video_path)
            st.session_state["s6_segment_paths"] = segment_paths
            status_text.success(
                f"Segment {n}/{total} done in **{seg_time_str}** "
                f"(total pipeline time so far: {pipe_time_str})"
            )
            st.video(video_path)

            if n < total:
                current_ref = extract_last_frame(video_path)
                # Brief cooldown so Kling releases the previous task slot
                # before the next submission (avoids immediate 429 rate limit)
                cooldown = 15
                status_text.info(
                    f"Segment {n} done. Waiting {cooldown}s before submitting segment {n + 1}…"
                )
                time.sleep(cooldown)

        st.session_state["s6_status"] = "done"

    except Exception as e:
        st.session_state["s6_status"] = "error"
        st.session_state["s6_error"] = str(e)
        st.error(f"Failed on segment {len(segment_paths) + 1}: {e}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        st.rerun()


if s6_status == "done":
    seg_paths = st.session_state["s6_segment_paths"]
    for idx, vp in enumerate(seg_paths, start=1):
        st.caption(f"Segment {idx}")
        st.video(vp)
    if st.button("Regenerate All Segments", key="regen_s6_all"):
        reset_from_step(6)
        st.rerun()

elif s6_status == "error":
    completed = st.session_state["s6_segment_paths"]
    if completed:
        st.warning(f"{len(completed)} segment(s) already generated. Resume will continue from segment {len(completed) + 1}.")
        for idx, vp in enumerate(completed, start=1):
            st.caption(f"Segment {idx} (completed)")
            st.video(vp)
    st.error(st.session_state.get("s6_error", "Unknown error"))
    col_r6a, col_r6b = st.columns(2)
    with col_r6a:
        if st.button("Resume from last completed segment", key="resume_s6", type="primary"):
            st.session_state["s6_status"] = "pending"
            st.rerun()
    with col_r6b:
        if st.button("Restart All Segments", key="restart_s6"):
            reset_from_step(6)
            st.rerun()

else:
    if _s5_done:
        col_6a, col_6b = st.columns(2)
        with col_6a:
            if st.button("Generate Video Segments", key="run_s6", type="primary"):
                _run_step6()
                st.rerun()
    else:
        st.button("Generate Video Segments", key="run_s6_dis", type="primary", disabled=True)
        st.caption("Complete Step 5 first.")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — Concatenate segments
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
step_header(7, "Concatenate Video Segments")

s7_status = st.session_state["s7_status"]
_s6_done = st.session_state["s6_status"] == "done"

if s7_status == "done":
    st.video(st.session_state["s7_combined_path"])
    if st.button("Regenerate Concatenation", key="regen_s7"):
        reset_from_step(7)
        st.rerun()

elif s7_status == "error":
    st.error(st.session_state.get("s7_error", "Unknown error"))
    if st.button("Retry Step 7", key="retry_s7"):
        reset_from_step(7)
        st.rerun()

else:
    if st.button("Concatenate Segments", key="run_s7", type="primary", disabled=not _s6_done):
        with st.spinner("Concatenating all segments..."):
            try:
                combined = concatenate_segments(st.session_state["s6_segment_paths"])
                st.session_state["s7_combined_path"] = combined
                st.session_state["s7_status"] = "done"
            except Exception as e:
                st.session_state["s7_status"] = "error"
                st.session_state["s7_error"] = str(e)
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
        st.rerun()
    if not _s6_done:
        st.caption("Complete Step 6 first.")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8 — Merge audio + video
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
step_header(8, "Merge Voiceover into Video")

s8_status = st.session_state["s8_status"]
_s7_done = st.session_state["s7_status"] == "done"
_s5_done_for_merge = st.session_state["s5_status"] == "done"

if s8_status == "done":
    st.subheader("Final Video")
    st.video(st.session_state["s8_final_path"])
    with open(st.session_state["s8_final_path"], "rb") as f:
        st.download_button(
            "Download Final Video",
            data=f,
            file_name="ai_generated_video.mp4",
            mime="video/mp4",
        )
    if st.button("Regenerate Final Merge", key="regen_s8"):
        reset_from_step(8)
        st.rerun()

elif s8_status == "error":
    st.error(st.session_state.get("s8_error", "Unknown error"))
    if st.button("Retry Step 8", key="retry_s8"):
        reset_from_step(8)
        st.rerun()

else:
    _merge_ready = _s7_done and _s5_done_for_merge
    if st.button("Merge Audio + Video", key="run_s8", type="primary", disabled=not _merge_ready):
        with st.spinner("Merging voiceover into video..."):
            try:
                final = merge_audio_video(
                    st.session_state["s7_combined_path"],
                    st.session_state["s5_audio_path"],
                )
                st.session_state["s8_final_path"] = final
                st.session_state["s8_status"] = "done"
            except Exception as e:
                st.session_state["s8_status"] = "error"
                st.session_state["s8_error"] = str(e)
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
        st.rerun()
    if not _merge_ready:
        st.caption("Complete Steps 6 and 7 first.")
