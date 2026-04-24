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
        print(f"\nUpserting dummy Character (id={TEST_CHAR_ID})...")
        # NOTE: voiceAudioUrl and imageUrl are excluded because those columns
        # may not exist yet in the DB if migrations haven't been fully applied.
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
    except Exception as e:
        print(f"[FAIL] Could not save Character: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await db.disconnect()
        print("\nDisconnected.")

    print("\n✓ DB test passed — Prisma can read/write to the database.")


if __name__ == "__main__":
    asyncio.run(main())
