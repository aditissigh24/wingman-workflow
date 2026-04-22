import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
SYNCLABS_API_KEY = os.environ.get("SYNCLABS_API_KEY", "")
KLING_ACCESS_KEY = os.environ.get("KLING_ACCESS_KEY", "")
KLING_SECRET_KEY = os.environ.get("KLING_SECRET_KEY", "")

GEMINI_TEXT_MODEL = "gemini-2.5-flash"
GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
ELEVENLABS_MODEL = "eleven_multilingual_v2"
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George — default premade voice (free-plan compatible)
SYNCLABS_MODEL = "lipsync-2"

KLING_MODEL = "kling-v2-master"
KLING_BASE_URL = "https://api.klingai.com"

SEGMENT_DURATION = 5   # seconds per Kling clip (5 or 10)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def validate_keys():
    """Return list of missing API key names."""
    missing = []
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    if not ELEVENLABS_API_KEY:
        missing.append("ELEVENLABS_API_KEY")
    if not KLING_ACCESS_KEY:
        missing.append("KLING_ACCESS_KEY")
    if not KLING_SECRET_KEY:
        missing.append("KLING_SECRET_KEY")
    return missing
