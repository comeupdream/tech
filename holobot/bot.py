"""HoloBot — Talk to an AI voice clone through your mic and speakers.

No accounts. No APIs. No extra installs. Just pip install and run.

Setup:
    1. pip install -r requirements.txt
    2. Drop a ~10-30s .wav of your target voice in audio_samples/reference.wav
    3. python bot.py

That's it. Models auto-download on first run.

Controls:
    Ctrl+C  — Quit
"""

import asyncio
import logging
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

import config
from pipeline import (
    VoiceActivityDetector,
    ConversationMemory,
    process_utterance,
    synthesize_speech,
    get_tts_sample_rate,
    _get_whisper_model,
    _get_tts_model,
    _get_llm,
)
from personality import CHARACTER_NAME, GREETING

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("holobot")


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def preflight_checks():
    """Make sure the basics are in place before loading models."""
    errors = []

    # Reference audio
    ref = Path(config.REFERENCE_AUDIO)
    if not ref.exists():
        errors.append(
            f"Reference audio not found: {ref}\n"
            f"  Place a ~10-30s .wav clip of the target voice there.\n"
            f"  Or run: python scraper/youtube_scraper.py"
        )

    # Microphone available
    try:
        input_device = sd.query_devices(kind="input")
        logger.info(f"Mic: {input_device['name']}")
    except Exception:
        errors.append(
            "No microphone detected.\n"
            "  Make sure a mic is connected and not muted."
        )

    if errors:
        print("\n" + "=" * 60)
        print("SETUP ISSUES — fix these before running:\n")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}\n")
        print("=" * 60)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Audio playback
# ---------------------------------------------------------------------------

def play_audio(audio_np: np.ndarray, sample_rate: int):
    """Play a numpy float32 audio array through the speakers. Blocks until done."""
    audio_np = np.clip(audio_np, -1.0, 1.0)
    sd.play(audio_np, samplerate=sample_rate)
    sd.wait()


# ---------------------------------------------------------------------------
# Main conversation loop
# ---------------------------------------------------------------------------

async def conversation_loop():
    """Core loop: listen → transcribe → think → speak → repeat."""
    vad = VoiceActivityDetector()
    memory = ConversationMemory()

    print(f"\n{'='*60}")
    print(f"  {CHARACTER_NAME} is ready. Start talking.")
    print(f"  (Ctrl+C to quit)")
    print(f"{'='*60}\n")

    # Play greeting
    logger.info("Generating greeting...")
    greeting_audio = await synthesize_speech(GREETING)
    tts_sr = get_tts_sample_rate()
    play_audio(greeting_audio, tts_sr)

    # Mic capture settings
    chunk_duration_ms = 30  # read 30ms chunks from the mic
    chunk_samples = int(config.MIC_SAMPLE_RATE * chunk_duration_ms / 1000)

    print("\nListening... (speak now)\n")

    while True:
        try:
            # Record one chunk from the mic (blocking, but only 30ms)
            audio_chunk = sd.rec(
                chunk_samples,
                samplerate=config.MIC_SAMPLE_RATE,
                channels=config.MIC_CHANNELS,
                dtype="int16",
            )
            sd.wait()

            # Feed into VAD
            pcm_bytes = audio_chunk.tobytes()
            utterance = vad.process_chunk(pcm_bytes)

            if utterance is not None:
                print("  [heard you, thinking...]")
                response_audio = await process_utterance(utterance, memory)

                print("  [speaking...]")
                play_audio(response_audio, tts_sr)
                print("  [listening...]\n")

        except KeyboardInterrupt:
            print(f"\n\n{CHARACTER_NAME} has left. Stay dangerous.")
            break
        except Exception:
            logger.exception("Error in conversation loop")


# ---------------------------------------------------------------------------
# Model preloader
# ---------------------------------------------------------------------------

def preload_models():
    """Load all AI models upfront so the first response is fast.
    First run auto-downloads everything (~5-6GB total, one time only)."""
    print("Loading AI models...")
    print("(First run downloads them automatically — one time only)\n")

    print("  [1/4] Whisper (speech recognition)...")
    _get_whisper_model()

    print("  [2/4] Llama 3.1 (AI brain)...")
    _get_llm()

    print("  [3/4] Chatterbox (voice cloning)...")
    _get_tts_model()

    print("  [4/4] Silero VAD (voice detection)...")
    VoiceActivityDetector()

    print("\nAll models loaded. Ready to talk.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"\n  HoloBot — Talk to {CHARACTER_NAME}")
    print(f"  100% local. No accounts. No APIs. No extra installs.\n")

    preflight_checks()
    preload_models()
    asyncio.run(conversation_loop())


if __name__ == "__main__":
    main()
