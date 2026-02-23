import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
AUDIO_SAMPLES_DIR = PROJECT_ROOT / "audio_samples"

# Reference audio clip for voice cloning (one clean .wav of the target speaking)
# Chatterbox clones from a single ~10-30 second clip
REFERENCE_AUDIO = os.getenv(
    "REFERENCE_AUDIO",
    str(AUDIO_SAMPLES_DIR / "reference.wav"),
)

# ---------------------------------------------------------------------------
# Microphone / speaker settings
# ---------------------------------------------------------------------------
MIC_SAMPLE_RATE = 16000       # capture at 16kHz mono (what Whisper wants)
MIC_CHANNELS = 1
PLAYBACK_SAMPLE_RATE = 24000  # Chatterbox outputs 24kHz — play directly at that rate
PLAYBACK_CHANNELS = 1

# ---------------------------------------------------------------------------
# VAD — Silero (local)
# ---------------------------------------------------------------------------
VAD_THRESHOLD = 0.5           # speech probability threshold
SILENCE_DURATION_MS = 1200    # ms of silence before we consider speech "done"
MIN_SPEECH_DURATION_MS = 500  # ignore utterances shorter than this

# ---------------------------------------------------------------------------
# STT — faster-whisper (local)
# ---------------------------------------------------------------------------
# Model sizes: "tiny", "base", "small", "medium", "large-v3"
# Bigger = more accurate but slower. "base" is a good starting point.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
# "cuda" for GPU, "cpu" for CPU-only (slower but works)
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
# "float16" for GPU, "int8" for CPU, "float32" for fallback
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")

# ---------------------------------------------------------------------------
# LLM — llama-cpp-python (local, auto-downloads model on first run)
# ---------------------------------------------------------------------------
# HuggingFace repo and filename for the GGUF model
# Default: Llama 3.1 8B Instruct (Q4_K_M quantization, ~4.7GB download once)
LLM_REPO_ID = os.getenv(
    "LLM_REPO_ID", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
)
LLM_MODEL_FILE = os.getenv(
    "LLM_MODEL_FILE", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)
# Context window (how much conversation history to keep)
LLM_CONTEXT_SIZE = int(os.getenv("LLM_CONTEXT_SIZE", "4096"))
# GPU layers: -1 = offload everything to GPU, 0 = CPU only
# Auto-detected: uses GPU if available, otherwise CPU
LLM_GPU_LAYERS = int(os.getenv("LLM_GPU_LAYERS", "-1"))
LLM_MAX_TOKENS = 300          # keep responses punchy
LLM_TEMPERATURE = 0.9         # creative flair

# ---------------------------------------------------------------------------
# TTS — Chatterbox (local, MIT licensed)
# ---------------------------------------------------------------------------
CHATTERBOX_DEVICE = os.getenv("CHATTERBOX_DEVICE", "auto")
# Exaggeration: 0.0 = monotone, 0.5 = normal, 1.0 = very expressive
CHATTERBOX_EXAGGERATION = float(os.getenv("CHATTERBOX_EXAGGERATION", "0.6"))
# CFG/pace weight — higher = closer to reference voice style
CHATTERBOX_CFG_WEIGHT = float(os.getenv("CHATTERBOX_CFG_WEIGHT", "0.5"))
