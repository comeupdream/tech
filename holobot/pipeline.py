"""Voice processing pipeline: VAD → STT → LLM → TTS.

Fully local — no paid APIs, no signups, no cloud calls.

  STT:  faster-whisper  (runs Whisper locally)
  LLM:  Ollama          (runs Llama/Mistral locally)
  TTS:  Chatterbox      (zero-shot voice clone, MIT licensed)
  VAD:  Silero          (speech boundary detection)
"""

import asyncio
import io
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.signal import resample_poly
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
# VAD — Silero Voice Activity Detection (local)
# ---------------------------------------------------------------------------

class VoiceActivityDetector:
    """Wraps Silero VAD to detect speech boundaries in streaming audio."""

    def __init__(self):
        self.model, self.utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self.threshold = config.VAD_THRESHOLD
        self.silence_samples = int(
            config.PIPELINE_SAMPLE_RATE * config.SILENCE_DURATION_MS / 1000
        )
        self.min_speech_samples = int(
            config.PIPELINE_SAMPLE_RATE * config.MIN_SPEECH_DURATION_MS / 1000
        )
        self.reset()

    def reset(self):
        self.model.reset_states()
        self._speech_buffer = bytearray()
        self._silence_counter = 0
        self._is_speaking = False

    def process_chunk(self, pcm_16k_mono: bytes) -> bytes | None:
        """Feed a chunk of 16kHz mono PCM16. Returns complete utterance bytes
        when speech ends, or None if still listening."""
        audio = np.frombuffer(pcm_16k_mono, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio)

        # Silero expects chunks of 512 samples at 16kHz
        chunk_size = 512
        for i in range(0, len(tensor), chunk_size):
            chunk = tensor[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))

            prob = self.model(chunk, config.PIPELINE_SAMPLE_RATE).item()

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
                    total_samples = len(utterance) // 2  # 16-bit = 2 bytes/sample
                    self.reset()
                    if total_samples >= self.min_speech_samples:
                        return utterance
                    return None

        if not self._is_speaking:
            return None
        return None


# ---------------------------------------------------------------------------
# Audio conversion helpers
# ---------------------------------------------------------------------------

def discord_pcm_to_16k_mono(pcm_48k_stereo: bytes) -> bytes:
    """Convert Discord's 48kHz stereo PCM16 to 16kHz mono PCM16."""
    samples = np.frombuffer(pcm_48k_stereo, dtype=np.int16)
    # Stereo to mono — average left and right channels
    if len(samples) % 2 == 0:
        stereo = samples.reshape(-1, 2)
        mono = stereo.mean(axis=1).astype(np.int16)
    else:
        mono = samples
    # Resample 48kHz → 16kHz (ratio 1:3)
    resampled = resample_poly(mono.astype(np.float32), up=1, down=3).astype(np.int16)
    return resampled.tobytes()


def pcm16k_to_float32(pcm_data: bytes) -> np.ndarray:
    """Convert 16kHz mono PCM16 bytes to float32 numpy array [-1.0, 1.0]."""
    return np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0


def audio_to_discord_pcm(audio_tensor: torch.Tensor, source_sr: int) -> bytes:
    """Convert a torch audio tensor to Discord-ready 48kHz stereo PCM16 bytes."""
    # Ensure mono
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    # Resample to 48kHz
    if source_sr != config.DISCORD_SAMPLE_RATE:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, source_sr, config.DISCORD_SAMPLE_RATE
        )

    # Float → int16
    samples = (audio_tensor.squeeze().clamp(-1.0, 1.0) * 32767).to(torch.int16).numpy()

    # Mono → stereo (duplicate channel)
    stereo = np.column_stack([samples, samples]).flatten().astype(np.int16)
    return stereo.tobytes()


# ---------------------------------------------------------------------------
# STT — faster-whisper (local, no API key)
# ---------------------------------------------------------------------------

_whisper_model = None


def _get_whisper_model():
    """Lazy-load the Whisper model (takes a few seconds on first call)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        device = _resolve_device(config.WHISPER_DEVICE)
        compute = _resolve_compute_type(config.WHISPER_COMPUTE_TYPE)
        logger.info(f"Loading Whisper model '{config.WHISPER_MODEL}' on {device} ({compute})")
        _whisper_model = WhisperModel(
            config.WHISPER_MODEL, device=device, compute_type=compute
        )
        logger.info("Whisper model loaded.")
    return _whisper_model


async def transcribe(pcm_16k_mono: bytes) -> str:
    """Transcribe a PCM16 audio utterance to text using local Whisper."""
    audio_float = pcm16k_to_float32(pcm_16k_mono)
    model = _get_whisper_model()

    # faster-whisper is synchronous, so run in a thread
    segments, info = await asyncio.to_thread(
        model.transcribe, audio_float, language="en", beam_size=3
    )
    # Collect all segment texts
    transcript = " ".join(seg.text.strip() for seg in segments)
    logger.info(f"STT: {transcript}")
    return transcript.strip()


# ---------------------------------------------------------------------------
# LLM — Ollama via OpenAI-compatible API (local, no API key)
# ---------------------------------------------------------------------------

class ConversationMemory:
    """Rolling conversation history with a fixed window."""

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.messages: deque[dict] = deque(maxlen=max_turns * 2)

    def add_user(self, text: str, speaker: str = "Someone"):
        self.messages.append({"role": "user", "content": f"[{speaker}]: {text}"})

    def add_assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})

    def get_messages(self) -> list[dict]:
        return [{"role": "system", "content": SYSTEM_PROMPT}] + list(self.messages)


async def generate_response(
    memory: ConversationMemory, user_text: str, speaker: str = "Someone"
) -> str:
    """Generate an in-character response from the local LLM via Ollama."""
    if not user_text or len(user_text) < 3:
        return FALLBACK_RESPONSE

    memory.add_user(user_text, speaker)

    # Ollama serves an OpenAI-compatible API — we just point the client at localhost
    client = AsyncOpenAI(
        base_url=config.OLLAMA_BASE_URL,
        api_key="ollama",  # Ollama doesn't need a real key but the client requires one
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
# TTS — Chatterbox by Resemble AI (local, MIT licensed, zero-shot voice clone)
# ---------------------------------------------------------------------------

_tts_model = None
_reference_audio = None


def _get_tts_model():
    """Lazy-load the Chatterbox TTS model."""
    global _tts_model
    if _tts_model is None:
        from chatterbox.tts import ChatterboxTTS
        device = _resolve_device(config.CHATTERBOX_DEVICE)
        logger.info(f"Loading Chatterbox TTS on {device}...")
        _tts_model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("Chatterbox TTS loaded.")
    return _tts_model


def _get_reference_audio() -> str:
    """Get the path to the reference audio file for voice cloning."""
    global _reference_audio
    if _reference_audio is None:
        ref_path = Path(config.REFERENCE_AUDIO)
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Reference audio not found at {ref_path}\n"
                f"Place a ~10-30 second .wav clip of Andrew Tate speaking at:\n"
                f"  {ref_path}\n"
                f"Or set REFERENCE_AUDIO in .env to point to your clip."
            )
        _reference_audio = str(ref_path)
        logger.info(f"Using reference audio: {_reference_audio}")
    return _reference_audio


async def synthesize_speech(text: str) -> bytes:
    """Convert text to speech using Chatterbox with zero-shot voice cloning.
    Returns raw PCM16 48kHz stereo bytes ready for Discord playback."""
    model = _get_tts_model()
    ref_audio = _get_reference_audio()

    # Chatterbox generate is synchronous + GPU-bound, run in thread
    wav_tensor = await asyncio.to_thread(
        model.generate,
        text,
        audio_prompt_path=ref_audio,
        exaggeration=config.CHATTERBOX_EXAGGERATION,
        cfg_weight=config.CHATTERBOX_CFG_WEIGHT,
    )

    # Chatterbox outputs at its own sample rate (usually 24kHz)
    source_sr = model.sr
    pcm_48k_stereo = audio_to_discord_pcm(wav_tensor, source_sr)

    logger.info(f"TTS: synthesized {len(pcm_48k_stereo)} bytes of audio")
    return pcm_48k_stereo


# ---------------------------------------------------------------------------
# Full pipeline: audio in → audio out
# ---------------------------------------------------------------------------

async def process_utterance(
    pcm_16k_mono: bytes,
    memory: ConversationMemory,
    speaker: str = "Someone",
) -> bytes:
    """Full pipeline: transcribe → think → speak. Returns Discord-ready PCM audio."""
    start = time.monotonic()

    # 1. STT — local Whisper
    transcript = await transcribe(pcm_16k_mono)
    if not transcript:
        return await synthesize_speech(FALLBACK_RESPONSE)

    # 2. LLM — local Ollama
    reply_text = await generate_response(memory, transcript, speaker)

    # 3. TTS — local Chatterbox
    audio_out = await synthesize_speech(reply_text)

    elapsed = time.monotonic() - start
    logger.info(f"Pipeline took {elapsed:.2f}s for: '{transcript}' → '{reply_text[:80]}...'")
    return audio_out
