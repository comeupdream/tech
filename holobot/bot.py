"""HoloBot — Talk to an AI voice clone through your mic and speakers.

No accounts. No APIs. No Discord. Just run it and talk.

Prerequisites:
    1. Install Ollama: https://ollama.com
    2. Pull a model:   ollama pull llama3.1
    3. Place a ~10-30s reference .wav in audio_samples/reference.wav
    4. pip install -r requirements.txt
    5. python bot.py

Controls:
    Ctrl+C  — Quit
    r       — Reset conversation memory (type in terminal while running)
"""

import asyncio
import logging
import shutil
import sys
import threading
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
    """Make sure everything is set up before starting."""
    errors = []

    # Reference audio
    ref = Path(config.REFERENCE_AUDIO)
    if not ref.exists():
        errors.append(
            f"Reference audio not found: {ref}\n"
            f"  Place a ~10-30s .wav clip of the target voice there.\n"
            f"  Or run: python scraper/youtube_scraper.py"
        )

    # Ollama running
    try:
        import urllib.request
        urllib.request.urlopen(
            config.OLLAMA_BASE_URL.replace("/v1", ""), timeout=3
        )
    except Exception:
        errors.append(
            f"Ollama not reachable at {config.OLLAMA_BASE_URL}\n"
            f"  Install: https://ollama.com\n"
            f"  Then run: ollama pull {config.OLLAMA_MODEL}\n"
            f"  Start it: ollama serve"
        )

    # Microphone available
    try:
        devices = sd.query_devices()
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
    # Clamp to [-1, 1] to avoid clipping distortion
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
            # Record one chunk from the mic (blocking, but it's only 30ms)
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
                # Got a complete utterance — process it
                print("  [heard you, thinking...]")
                response_audio = await process_utterance(utterance, memory)

                print(f"  [speaking...]")
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
    """Load all AI models upfront so the first response is fast."""
    print("Loading AI models (first run downloads them, may take a few minutes)...")
    print("  - Whisper (speech recognition)...")
    _get_whisper_model()
    print("  - Chatterbox (voice cloning)...")
    _get_tts_model()
    print("  - Silero VAD (voice detection)... ", end="")
    VoiceActivityDetector()
    print("done")
    print("All models loaded.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"\n  HoloBot — Talk to {CHARACTER_NAME}")
    print(f"  100% local. No accounts. No APIs.\n")

    preflight_checks()
    preload_models()
    asyncio.run(conversation_loop())


if __name__ == "__main__":
    main()
