import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "wingman-media")

GEMINI_TEXT_MODEL = "gemini-2.5-flash"
GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"
CLAUDE_MODEL = "claude-sonnet-4-5"
ELEVENLABS_MODEL = "eleven_multilingual_v2"
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George — default premade voice

VEO3_MODEL = "veo-3.1-generate-preview"

SEGMENT_DURATION = 5   # seconds per Veo3 clip (5 or 10)

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
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    if not ELEVENLABS_API_KEY:
        missing.append("ELEVENLABS_API_KEY")
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_SERVICE_ROLE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    return missing
