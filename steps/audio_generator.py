import os
import uuid
from typing import Optional
from elevenlabs import ElevenLabs
from config import ELEVENLABS_API_KEY, ELEVENLABS_MODEL, ELEVENLABS_VOICE_ID, OUTPUT_DIR


client = ElevenLabs(api_key=ELEVENLABS_API_KEY)


def generate_audio(voiceover_script: str, voice_id: Optional[str] = None) -> str:
    """
    Generate voiceover audio from the script using ElevenLabs TTS.
    Returns the file path to the saved audio file.
    """
    audio_iterator = client.text_to_speech.convert(
        text=voiceover_script,
        voice_id=voice_id or ELEVENLABS_VOICE_ID,
        model_id=ELEVENLABS_MODEL,
        output_format="mp3_44100_128",
    )

    filename = f"audio_{uuid.uuid4().hex[:8]}.mp3"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "wb") as f:
        for chunk in audio_iterator:
            f.write(chunk)

    return filepath


def list_voices() -> list[dict]:
    """Return available voices as a list of {voice_id, name} dicts."""
    response = client.voices.get_all()
    return [{"voice_id": v.voice_id, "name": v.name} for v in response.voices]
