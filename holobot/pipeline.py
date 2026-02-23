"""Voice processing pipeline: VAD → STT → LLM → TTS.

Fully local — no APIs, no signups, no cloud calls.
Talks to your mic and speakers directly.

  STT:  faster-whisper  (runs Whisper locally)
  LLM:  Ollama          (runs Llama/Mistral locally)
  TTS:  Chatterbox      (zero-shot voice clone, MIT licensed)
  VAD:  Silero          (speech boundary detection)
"""

import asyncio
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from openai import AsyncOpenAI

import config
from personality import SYSTEM_PROMPT, FALLBACK_RESPONSE

logger = logging.getLogger("holobot.pipeline")


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _resolve_device(setting: str) -> str:
    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return setting


def _resolve_compute_type(setting: str) -> str:
    if setting == "auto":
        return "float16" if torch.cuda.is_available() else "int8"
    return setting


# ---------------------------------------------------------------------------
# VAD — Silero Voice Activity Detection
# ---------------------------------------------------------------------------

class VoiceActivityDetector:
    """Detects speech boundaries in streaming 16kHz mono PCM audio."""

    def __init__(self):
        self.model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self.threshold = config.VAD_THRESHOLD
        self.silence_samples = int(
            config.MIC_SAMPLE_RATE * config.SILENCE_DURATION_MS / 1000
        )
        self.min_speech_samples = int(
            config.MIC_SAMPLE_RATE * config.MIN_SPEECH_DURATION_MS / 1000
        )
        self.reset()

    def reset(self):
        self.model.reset_states()
        self._speech_buffer = bytearray()
        self._silence_counter = 0
        self._is_speaking = False

    def process_chunk(self, pcm_16k_mono: bytes) -> bytes | None:
        """Feed a chunk of 16kHz mono PCM16. Returns complete utterance bytes
        when speech ends, or None if still accumulating."""
        audio = np.frombuffer(pcm_16k_mono, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio)

        chunk_size = 512  # Silero expects 512 samples at 16kHz
        for i in range(0, len(tensor), chunk_size):
            chunk = tensor[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))

            prob = self.model(chunk, config.MIC_SAMPLE_RATE).item()

            if prob >= self.threshold:
                self._is_speaking = True
                self._silence_counter = 0
                self._speech_buffer.extend(pcm_16k_mono)
                return None
            elif self._is_speaking:
                self._silence_counter += len(chunk)
                self._speech_buffer.extend(pcm_16k_mono)

                if self._silence_counter >= self.silence_samples:
                    utterance = bytes(self._speech_buffer)
                    total_samples = len(utterance) // 2
                    self.reset()
                    if total_samples >= self.min_speech_samples:
                        return utterance
                    return None

        return None


# ---------------------------------------------------------------------------
# STT — faster-whisper (local)
# ---------------------------------------------------------------------------

_whisper_model = None


def _get_whisper_model():
    """Lazy-load the Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        device = _resolve_device(config.WHISPER_DEVICE)
        compute = _resolve_compute_type(config.WHISPER_COMPUTE_TYPE)
        logger.info(f"Loading Whisper '{config.WHISPER_MODEL}' on {device} ({compute})...")
        _whisper_model = WhisperModel(
            config.WHISPER_MODEL, device=device, compute_type=compute
        )
        logger.info("Whisper loaded.")
    return _whisper_model


async def transcribe(pcm_16k_mono: bytes) -> str:
    """Transcribe 16kHz mono PCM16 audio to text."""
    audio = np.frombuffer(pcm_16k_mono, dtype=np.int16).astype(np.float32) / 32768.0
    model = _get_whisper_model()

    segments, _ = await asyncio.to_thread(
        model.transcribe, audio, language="en", beam_size=3
    )
    transcript = " ".join(seg.text.strip() for seg in segments)
    logger.info(f"STT: {transcript}")
    return transcript.strip()


# ---------------------------------------------------------------------------
# LLM — Ollama (local)
# ---------------------------------------------------------------------------

class ConversationMemory:
    """Rolling conversation history."""

    def __init__(self, max_turns: int = 20):
        self.messages: deque[dict] = deque(maxlen=max_turns * 2)

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})

    def get_messages(self) -> list[dict]:
        return [{"role": "system", "content": SYSTEM_PROMPT}] + list(self.messages)


async def generate_response(memory: ConversationMemory, user_text: str) -> str:
    """Generate an in-character response from the local LLM."""
    if not user_text or len(user_text) < 3:
        return FALLBACK_RESPONSE

    memory.add_user(user_text)

    client = AsyncOpenAI(
        base_url=config.OLLAMA_BASE_URL,
        api_key="ollama",
    )

    response = await client.chat.completions.create(
        model=config.OLLAMA_MODEL,
        messages=memory.get_messages(),
        max_tokens=config.LLM_MAX_TOKENS,
        temperature=config.LLM_TEMPERATURE,
    )

    reply = response.choices[0].message.content.strip()
    memory.add_assistant(reply)
    logger.info(f"LLM: {reply}")
    return reply


# ---------------------------------------------------------------------------
# TTS — Chatterbox (local, zero-shot voice clone)
# ---------------------------------------------------------------------------

_tts_model = None


def _get_tts_model():
    """Lazy-load the Chatterbox TTS model."""
    global _tts_model
    if _tts_model is None:
        from chatterbox.tts import ChatterboxTTS
        device = _resolve_device(config.CHATTERBOX_DEVICE)
        logger.info(f"Loading Chatterbox TTS on {device}...")
        _tts_model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("Chatterbox loaded.")
    return _tts_model


def _get_reference_audio() -> str:
    """Validate and return path to the voice cloning reference audio."""
    ref_path = Path(config.REFERENCE_AUDIO)
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference audio not found at {ref_path}\n"
            f"Place a ~10-30s .wav clip of your target voice there.\n"
            f"Or run: python scraper/youtube_scraper.py"
        )
    return str(ref_path)


async def synthesize_speech(text: str) -> np.ndarray:
    """Convert text to speech with the cloned voice.
    Returns a float32 numpy array at Chatterbox's native sample rate."""
    model = _get_tts_model()
    ref_audio = _get_reference_audio()

    wav_tensor = await asyncio.to_thread(
        model.generate,
        text,
        audio_prompt_path=ref_audio,
        exaggeration=config.CHATTERBOX_EXAGGERATION,
        cfg_weight=config.CHATTERBOX_CFG_WEIGHT,
    )

    # Convert torch tensor → numpy float32
    if wav_tensor.dim() > 1:
        wav_tensor = wav_tensor.squeeze()
    audio_np = wav_tensor.cpu().numpy().astype(np.float32)

    logger.info(f"TTS: {len(audio_np)} samples at {model.sr}Hz")
    return audio_np


def get_tts_sample_rate() -> int:
    """Return the sample rate of the TTS model output."""
    return _get_tts_model().sr


# ---------------------------------------------------------------------------
# Full pipeline: speech in → speech out
# ---------------------------------------------------------------------------

async def process_utterance(pcm_16k_mono: bytes, memory: ConversationMemory) -> np.ndarray:
    """Full pipeline: transcribe → think → speak.
    Returns float32 numpy audio at Chatterbox's native sample rate."""
    start = time.monotonic()

    transcript = await transcribe(pcm_16k_mono)
    if not transcript:
        return await synthesize_speech(FALLBACK_RESPONSE)

    reply_text = await generate_response(memory, transcript)
    audio_out = await synthesize_speech(reply_text)

    elapsed = time.monotonic() - start
    logger.info(
        f"Pipeline: {elapsed:.1f}s | '{transcript}' → '{reply_text[:60]}...'"
    )
    return audio_out
