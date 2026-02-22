"""Upload voice samples to ElevenLabs and create a cloned voice.

Usage:
    python scraper/upload_voice.py

Reads all .wav files from audio_samples/ and uploads them to ElevenLabs
to create a professional voice clone. Prints the Voice ID to use in .env.

Requirements:
    - ELEVENLABS_API_KEY set in .env
    - Clean audio samples in audio_samples/ (no music, single speaker only)
    - ElevenLabs account with voice cloning enabled (paid plan)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs import ElevenLabs

load_dotenv()

AUDIO_DIR = Path(__file__).parent.parent / "audio_samples"
API_KEY = os.getenv("ELEVENLABS_API_KEY")


def main():
    if not API_KEY:
        print("ERROR: ELEVENLABS_API_KEY not set in .env")
        sys.exit(1)

    # Find all audio files
    audio_files = []
    for ext in ("*.wav", "*.mp3", "*.m4a"):
        audio_files.extend(AUDIO_DIR.glob(ext))

    if not audio_files:
        print(f"No audio files found in {AUDIO_DIR}")
        print("Run the scraper first: python scraper/youtube_scraper.py")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files:")
    for f in audio_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")

    # Limit to 25 files (ElevenLabs limit)
    if len(audio_files) > 25:
        print(f"\nWARNING: ElevenLabs allows max 25 files. Using first 25.")
        audio_files = audio_files[:25]

    print(f"\nUploading {len(audio_files)} files to ElevenLabs...")

    client = ElevenLabs(api_key=API_KEY)

    # Open all files for upload
    file_handles = [open(f, "rb") for f in audio_files]

    try:
        voice = client.clone(
            name="Andrew Tate Clone",
            description="AI voice clone of Andrew Tate for HoloBot project. "
                        "Confident, bold, energetic speaking style.",
            files=file_handles,
        )

        print(f"\n{'='*50}")
        print(f"Voice cloned successfully!")
        print(f"Voice ID: {voice.voice_id}")
        print(f"Voice Name: {voice.name}")
        print(f"\nAdd this to your .env file:")
        print(f"  ELEVENLABS_VOICE_ID={voice.voice_id}")
        print(f"{'='*50}")

    finally:
        for fh in file_handles:
            fh.close()


if __name__ == "__main__":
    main()
