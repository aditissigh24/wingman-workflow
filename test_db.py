"""
Quick DB connectivity test — tries to upsert a dummy Character record.
Run with:  python test_db.py
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

TEST_CHAR_ID = "test_dbchk"  # <= 10 chars, VarChar(10)
TEST_SCEN_ID = "test_scen1"


async def main():
    try:
        from prisma import Prisma
    except ImportError:
        print("[FAIL] prisma package not installed. Run:  pip install prisma")
        sys.exit(1)

    db = Prisma()
    print(f"Connecting to DB...")
    print(f"  DATABASE_URL = {os.getenv('DATABASE_URL', '(not set)')[:60]}...")

    try:
        await db.connect()
        print("[OK] Connected to database.")
    except Exception as e:
        print(f"[FAIL] Could not connect: {e}")
        sys.exit(1)

    try:
        # --- Character test ---
        print(f"\nUpserting dummy Character (id={TEST_CHAR_ID})...")
        record = await db.character.upsert(
            where={"id": TEST_CHAR_ID},
            data={
                "create": {
                    "id": TEST_CHAR_ID,
                    "name": "Test Character",
                    "age": 25,
                    "gender": "FEMALE",
                    "city": "Mumbai",
                    "archetype": "Test Dummy",
                    "vibeSummary": "A dummy record used for DB connectivity testing.",
                    "backstory": "Created by test_db.py to verify Prisma can write to the DB.",
                    "speakingStyle": "Casual",
                    "emojiUsage": "Minimal",
                    "textingSpeed": "Fast",
                    "voicePrompt": "Friendly and warm.",
                    "hardLimits": [],
                    "avatarPrompt": "Simple avatar",
                    "accentHsl": "210 50% 60%",
                },
                "update": {
                    "vibeSummary": "Updated by test_db.py",
                },
            },
        )
        print(f"[OK] Character saved successfully!")
        print(f"     id       = {record.id}")
        print(f"     name     = {record.name}")
        print(f"     gender   = {record.gender}")
        print(f"     city     = {record.city}")
        print(f"     createdAt= {record.createdAt}")

        # --- ScenarioBeat test (exercises maxTurnsInBeat) ---
        print(f"\nUpserting dummy Scenario (id={TEST_SCEN_ID})...")
        await db.scenario.upsert(
            where={"id": TEST_SCEN_ID},
            data={
                "create": {
                    "id": TEST_SCEN_ID,
                    "characterId": TEST_CHAR_ID,
                    "scenarioTitle": "Test Scenario",
                },
                "update": {"scenarioTitle": "Test Scenario"},
            },
        )
        print("[OK] Scenario saved.")

        print("Creating dummy ScenarioBeat with maxTurnsInBeat...")
        await db.scenariobeat.delete_many(where={"scenarioId": TEST_SCEN_ID})
        beat = await db.scenariobeat.create(data={
            "scenarioId": TEST_SCEN_ID,
            "characterId": TEST_CHAR_ID,
            "beatNumber": 1,
            "beatType": "HOOK",
            "narrativeContext": "Test narrative context.",
            "characterEmotionalState": "Curious and warm.",
            "flowDirective": "Let the conversation breathe.",
            "hookDirective": "Say something unexpectedly observant.",
            "minTurnsInBeat": 2,
            "engagedAdvanceScore": 3.5,
            "maxTurnsInBeat": 5,
        })
        print(f"[OK] ScenarioBeat saved!")
        print(f"     id              = {beat.id}")
        print(f"     beatType        = {beat.beatType}")
        print(f"     minTurnsInBeat  = {beat.minTurnsInBeat}")
        print(f"     maxTurnsInBeat  = {beat.maxTurnsInBeat}")
        print(f"     engagedAdvScore = {beat.engagedAdvanceScore}")

        print(f"\n[INFO] Records left in DB for manual verification:")
        print(f"       Character  id = {TEST_CHAR_ID}")
        print(f"       Scenario   id = {TEST_SCEN_ID}")
        print(f"       ScenarioBeat id = {beat.id}")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await db.disconnect()
        print("\nDisconnected.")

    print("\n[PASS] DB test passed - Prisma can read/write to the database.")


if __name__ == "__main__":
    asyncio.run(main())
