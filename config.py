import os
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
FAL_KEY = os.environ.get("FAL_KEY", "")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "wingman-media")

ELEVENLABS_MODEL = "eleven_multilingual_v2"
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George — default premade voice

SEGMENT_DURATION = 5  # seconds per video clip

# fal.ai model IDs
FAL_IMAGE_MODEL      = os.environ.get("FAL_IMAGE_MODEL",      "fal-ai/flux-pro/v1.1")
FAL_VIDEO_MODEL      = os.environ.get("FAL_VIDEO_MODEL",      "bytedance/seedance-2.0/fast/image-to-video")
FAL_ELEVENLABS_MODEL = os.environ.get("FAL_ELEVENLABS_MODEL", "fal-ai/elevenlabs/tts/eleven-v3")
FAL_LIPSYNC_MODEL    = os.environ.get("FAL_LIPSYNC_MODEL",    "fal-ai/heygen/v3/lipsync/precision")

# OpenRouter model IDs for LLM text calls
OR_CLAUDE_MODEL = os.environ.get("OR_CLAUDE_MODEL", "anthropic/claude-sonnet-4-6")
OR_GEMINI_MODEL = os.environ.get("OR_GEMINI_MODEL", "google/gemini-2.5-flash")

# City → ElevenLabs accent hint mapping (used in TTS voice direction)
CITY_ACCENT_MAP = {
    "Indore":     "Madhya Pradesh Hinglish, warm bhai energy, middle-class cadence",
    "Lucknow":    "Lucknow Hindi, tehzeeb, formal warmth, slightly literary",
    "Mumbai":     "Mumbai code-switching, fast, aspirational, independent",
    "Delhi":      "Delhi confident, slightly aggressive, brand-aware, direct",
    "Pune":       "Pune Marathi-inflected Hindi, chill, progressive",
    "Jaipur":     "Rajasthani-inflected Hindi, colourful, slightly formal",
    "Chandigarh": "Punjabi-inflected Hindi, very direct, expressive",
    "Hyderabad":  "Hyderabadi Urdu-Hindi mix, warm, laid-back",
    "Bengaluru":  "South Indian English-Hindi mix, calm, techie",
    "Kolkata":    "Bengali-inflected Hindi, intellectual, nostalgic",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def validate_keys() -> list[str]:
    """Return list of missing API key names."""
    missing = []
    if not FAL_KEY:
        missing.append("FAL_KEY")
    if not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    if not ELEVENLABS_API_KEY:
        missing.append("ELEVENLABS_API_KEY")
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_SERVICE_ROLE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    return missing
