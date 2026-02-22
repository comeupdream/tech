import os
from dotenv import load_dotenv

load_dotenv()

# Discord
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Deepgram (STT)
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# OpenAI (LLM brain)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ElevenLabs (TTS + voice clone)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# Audio settings — Discord sends 48kHz stereo PCM16
DISCORD_SAMPLE_RATE = 48000
DISCORD_CHANNELS = 2
PIPELINE_SAMPLE_RATE = 16000  # most STT models expect 16kHz mono
PIPELINE_CHANNELS = 1

# VAD settings
VAD_THRESHOLD = 0.5           # speech probability threshold
SILENCE_DURATION_MS = 1200    # ms of silence before we consider speech "done"
MIN_SPEECH_DURATION_MS = 500  # ignore utterances shorter than this

# LLM settings
LLM_MODEL = "gpt-4o"
LLM_MAX_TOKENS = 300          # keep responses punchy, not essays
LLM_TEMPERATURE = 0.9         # some creative flair

# TTS settings
TTS_MODEL = "eleven_turbo_v2_5"
