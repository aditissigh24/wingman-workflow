"""
Database saver — writes Character, Scenario, and ScenarioBeat records to Postgres via Prisma.
Uses the existing schema.prisma models (no ORM reimplementation needed).
"""

import asyncio
import logging

logger = logging.getLogger("db_saver")


def _run_async(coro):
    """Run an async coroutine synchronously (for use in Streamlit)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _trunc(value, max_len: int) -> str:
    """Truncate a string to max_len characters."""
    s = str(value) if value is not None else ""
    return s[:max_len]


async def _save_character_async(data: dict) -> str:
    from prisma import Prisma
    db = Prisma()
    await db.connect()
    try:
        # Character.id is @id @db.VarChar(10) — must be provided by caller or generated
        char_id = _trunc(data.get("id") or _short_id(), 10)
        record = await db.character.upsert(
            where={"id": char_id},
            data={
                "create": {
                    "id": char_id,
                    "name": _trunc(data.get("name", ""), 100),
                    "age": int(data.get("age", 22)),
                    "gender": data.get("gender", "FEMALE"),
                    "city": _trunc(data.get("city", ""), 100),
                    "archetype": _trunc(data.get("archetype", ""), 255),
                    "vibeSummary": data.get("vibeSummary", ""),
                    "backstory": data.get("backstory", ""),
                    "speakingStyle": data.get("speakingStyle", ""),
                    "emojiUsage": _trunc(data.get("emojiUsage", ""), 150),
                    "textingSpeed": _trunc(data.get("textingSpeed", ""), 150),
                    "voicePrompt": data.get("voicePrompt", ""),
                    "hardLimits": data.get("hardLimits", []),
                    "avatarPrompt": data.get("avatarPrompt", ""),
                    "accentHsl": _trunc(data.get("accentHsl", ""), 50),
                    "imageUrl": data.get("imageUrl") or [],
                    "voiceAudioUrl": data.get("voiceAudioUrl"),
                },
                "update": {
                    "name": _trunc(data.get("name", ""), 100),
                    "age": int(data.get("age", 22)),
                    "gender": data.get("gender", "FEMALE"),
                    "city": _trunc(data.get("city", ""), 100),
                    "archetype": _trunc(data.get("archetype", ""), 255),
                    "vibeSummary": data.get("vibeSummary", ""),
                    "backstory": data.get("backstory", ""),
                    "speakingStyle": data.get("speakingStyle", ""),
                    "emojiUsage": _trunc(data.get("emojiUsage", ""), 150),
                    "textingSpeed": _trunc(data.get("textingSpeed", ""), 150),
                    "voicePrompt": data.get("voicePrompt", ""),
                    "hardLimits": data.get("hardLimits", []),
                    "avatarPrompt": data.get("avatarPrompt", ""),
                    "accentHsl": _trunc(data.get("accentHsl", ""), 50),
                    "imageUrl": data.get("imageUrl") or [],
                    "voiceAudioUrl": data.get("voiceAudioUrl"),
                },
            },
        )
        logger.info("Character saved: id=%s name=%s", record.id, record.name)
        return record.id
    finally:
        await db.disconnect()


async def _save_scenario_async(data: dict, character_id: str) -> str:
    from prisma import Prisma
    db = Prisma()
    await db.connect()
    try:
        scenario_id = data.get("id") or _short_id()
        record = await db.scenario.upsert(
            where={"id": scenario_id},
            data={
                "create": {
                    "id": scenario_id,
                    "characterId": character_id,
                    "scenarioTitle": data.get("scenarioTitle", ""),
                    "difficulty": data.get("difficulty", "Medium"),
                    "situationSetupForUser": data.get("situationSetupForUser", ""),
                    "imagePrompt": data.get("imagePrompt", ""),
                    "imageUrl": data.get("imageUrl"),
                    "audioUrl": data.get("audioUrl"),
                    "videoUrl": data.get("videoUrl"),
                    "tagline": data.get("tagline", ""),
                    "learningObjective": data.get("learningObjective", ""),
                    "goodOutcome": data.get("goodOutcome", ""),
                    "badOutcome": data.get("badOutcome", ""),
                    "primalHook": data.get("primalHook", ""),
                    "initialMessages": data.get("initialMessages", []),
                    "initialChips": data.get("initialChips", []),
                    "settingDescription": data.get("settingDescription", ""),
                    "atmosphere": data.get("atmosphere", ""),
                    "tone": data.get("tone", ""),
                    "timeOfDay": data.get("timeOfDay", ""),
                    "overallArc": data.get("overallArc", ""),
                },
                "update": {
                    "imageUrl": data.get("imageUrl"),
                    "audioUrl": data.get("audioUrl"),
                    "videoUrl": data.get("videoUrl"),
                },
            },
        )
        logger.info("Scenario saved: id=%s title=%s", record.id, record.scenarioTitle)
        return record.id
    finally:
        await db.disconnect()


async def _save_beats_async(beats: list, scenario_id: str, character_id: str) -> None:
    from prisma import Prisma
    db = Prisma()
    await db.connect()
    try:
        for beat in beats:
            beat_type_str = beat.get("beatType", "HOOK")
            # Map any new-style beat types to existing enum values as fallback
            _beat_type_map = {
                "DEEPEN": "BUILD", "TEST": "TWIST", "PEAK": "CONSEQUENCE",
                "RESIST": "CONSEQUENCE", "BREAK": "CLIFFHANGER",
            }
            beat_type = _beat_type_map.get(beat_type_str, beat_type_str)

            await db.scenariobeat.create(
                data={
                    "scenarioId": scenario_id,
                    "characterId": character_id,
                    "beatNumber": int(beat.get("beatNumber", 1)),
                    "beatType": beat_type,
                    "narrativeContext": beat.get("narrativeContext", ""),
                    "characterEmotionalState": beat.get("characterEmotionalState", ""),
                    "flowDirective": beat.get("flowDirective", ""),
                    "hookDirective": beat.get("hookDirective", ""),
                    "minTurnsInBeat": int(beat.get("minTurnsInBeat", 2)),
                    "engagedAdvanceScore": float(beat.get("engagedAdvanceScore", 3.0)),
                    "maxTurnsInBeat": int(beat.get("maxTurnsInBeat", len(beats))),
                }
            )
        logger.info("Saved %d beats for scenario %s", len(beats), scenario_id)
    finally:
        await db.disconnect()


def _short_id(length: int = 8) -> str:
    """Generate a short random alphanumeric ID (VarChar(10) compatible)."""
    import random, string
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


# ── Public sync wrappers ─────────────────────────────────────────────────

def save_character(data: dict) -> str:
    """Upsert a Character record. Returns the character id."""
    return _run_async(_save_character_async(data))


def save_scenario(data: dict, character_id: str) -> str:
    """Upsert a Scenario record. Returns the scenario id."""
    return _run_async(_save_scenario_async(data, character_id))


def save_beats(beats: list, scenario_id: str, character_id: str) -> None:
    """Create ScenarioBeat records for a scenario."""
    _run_async(_save_beats_async(beats, scenario_id, character_id))
