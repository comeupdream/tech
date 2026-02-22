"""Voice processing pipeline: VAD → STT → LLM → TTS.

Takes raw PCM audio from Discord, detects speech, transcribes it,
generates an in-character response, and synthesizes speech audio.
"""

import asyncio
import io
import struct
import logging
import time
from collections import deque

import numpy as np
import torch
from scipy.signal import resample_poly
from openai import AsyncOpenAI
from deepgram import DeepgramClient, PrerecordedOptions
from elevenlabs import AsyncElevenLabs

import config
from personality import SYSTEM_PROMPT, FALLBACK_RESPONSE

logger = logging.getLogger("holobot.pipeline")

# ---------------------------------------------------------------------------
# VAD — Silero Voice Activity Detection
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
                    total_samples = len(utterance) // 2  # 16-bit = 2 bytes per sample
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


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw PCM16 in a WAV header for APIs that expect WAV input."""
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# STT — Deepgram transcription
# ---------------------------------------------------------------------------

async def transcribe(pcm_16k_mono: bytes) -> str:
    """Transcribe a PCM16 audio utterance to text using Deepgram."""
    client = DeepgramClient(config.DEEPGRAM_API_KEY)
    wav_data = pcm_to_wav_bytes(pcm_16k_mono)

    options = PrerecordedOptions(
        model="nova-2",
        language="en",
        smart_format=True,
    )

    source = {"buffer": wav_data, "mimetype": "audio/wav"}
    response = await asyncio.to_thread(
        client.listen.rest.v("1").transcribe_file, source, options
    )

    transcript = (
        response.results.channels[0].alternatives[0].transcript
    )
    logger.info(f"STT: {transcript}")
    return transcript.strip()


# ---------------------------------------------------------------------------
# LLM — Character response generation
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


async def generate_response(memory: ConversationMemory, user_text: str, speaker: str = "Someone") -> str:
    """Generate an in-character response from the LLM."""
    if not user_text or len(user_text) < 3:
        return FALLBACK_RESPONSE

    memory.add_user(user_text, speaker)

    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=memory.get_messages(),
        max_tokens=config.LLM_MAX_TOKENS,
        temperature=config.LLM_TEMPERATURE,
    )

    reply = response.choices[0].message.content.strip()
    memory.add_assistant(reply)
    logger.info(f"LLM: {reply}")
    return reply


# ---------------------------------------------------------------------------
# TTS — ElevenLabs text-to-speech with cloned voice
# ---------------------------------------------------------------------------

async def synthesize_speech(text: str) -> bytes:
    """Convert text to speech using ElevenLabs with the cloned voice.
    Returns raw PCM16 48kHz stereo bytes ready for Discord playback."""
    client = AsyncElevenLabs(api_key=config.ELEVENLABS_API_KEY)

    # Generate audio — ElevenLabs returns mp3 by default
    audio_iterator = await client.text_to_speech.convert(
        voice_id=config.ELEVENLABS_VOICE_ID,
        text=text,
        model_id=config.TTS_MODEL,
        output_format="pcm_24000",  # raw PCM 24kHz mono
    )

    # Collect all chunks
    pcm_24k = bytearray()
    async for chunk in audio_iterator:
        pcm_24k.extend(chunk)

    pcm_24k = bytes(pcm_24k)

    # Convert 24kHz mono → 48kHz stereo for Discord
    samples = np.frombuffer(pcm_24k, dtype=np.int16).astype(np.float32)
    upsampled = resample_poly(samples, up=2, down=1).astype(np.int16)
    # Mono to stereo — duplicate the channel
    stereo = np.column_stack([upsampled, upsampled]).flatten()
    pcm_48k_stereo = stereo.astype(np.int16).tobytes()

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

    # 1. STT
    transcript = await transcribe(pcm_16k_mono)
    if not transcript:
        return await synthesize_speech(FALLBACK_RESPONSE)

    # 2. LLM
    reply_text = await generate_response(memory, transcript, speaker)

    # 3. TTS
    audio_out = await synthesize_speech(reply_text)

    elapsed = time.monotonic() - start
    logger.info(f"Pipeline took {elapsed:.2f}s for: '{transcript}' → '{reply_text[:80]}...'")
    return audio_out
