import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Discord (only "signup" needed — free at discord.com/developers)
# ---------------------------------------------------------------------------
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
AUDIO_SAMPLES_DIR = PROJECT_ROOT / "audio_samples"

# Reference audio clip for voice cloning (just one clean .wav file of Tate talking)
# Chatterbox can clone from a single ~10-30 second clip
REFERENCE_AUDIO = os.getenv(
    "REFERENCE_AUDIO",
    str(AUDIO_SAMPLES_DIR / "reference.wav"),
)

# ---------------------------------------------------------------------------
# Audio settings — Discord sends 48kHz stereo PCM16
# ---------------------------------------------------------------------------
DISCORD_SAMPLE_RATE = 48000
DISCORD_CHANNELS = 2
PIPELINE_SAMPLE_RATE = 16000  # Whisper expects 16kHz mono
PIPELINE_CHANNELS = 1

# ---------------------------------------------------------------------------
# VAD — Silero (local, no signup)
# ---------------------------------------------------------------------------
VAD_THRESHOLD = 0.5           # speech probability threshold
SILENCE_DURATION_MS = 1200    # ms of silence before we consider speech "done"
MIN_SPEECH_DURATION_MS = 500  # ignore utterances shorter than this

# ---------------------------------------------------------------------------
# STT — faster-whisper (local, no signup)
# ---------------------------------------------------------------------------
# Model sizes: "tiny", "base", "small", "medium", "large-v3"
# Bigger = more accurate but slower. "base" is a good starting point.
# "large-v3" is best quality but needs ~4GB VRAM.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
# "cuda" for GPU, "cpu" for CPU-only (slower but works everywhere)
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
# "float16" for GPU, "int8" for CPU, "float32" for fallback
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")

# ---------------------------------------------------------------------------
# LLM — Ollama (local, no signup)
# ---------------------------------------------------------------------------
# Install: https://ollama.com then run: ollama pull llama3.1
# Ollama serves an OpenAI-compatible API on localhost
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
LLM_MAX_TOKENS = 300          # keep responses punchy, not essays
LLM_TEMPERATURE = 0.9         # some creative flair

# ---------------------------------------------------------------------------
# TTS — Chatterbox by Resemble AI (local, no signup, MIT licensed)
# ---------------------------------------------------------------------------
# Zero-shot voice cloning from a single reference audio clip.
# Needs ~4GB VRAM on GPU, or runs on CPU (slower).
CHATTERBOX_DEVICE = os.getenv("CHATTERBOX_DEVICE", "auto")
# Exaggeration: 0.0 = monotone, 0.5 = normal, 1.0 = very expressive
CHATTERBOX_EXAGGERATION = float(os.getenv("CHATTERBOX_EXAGGERATION", "0.6"))
# CFG/pace weight — higher = closer to reference voice style
CHATTERBOX_CFG_WEIGHT = float(os.getenv("CHATTERBOX_CFG_WEIGHT", "0.5"))
