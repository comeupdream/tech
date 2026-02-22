"""HoloBot — AI voice character bot for Discord.

Usage:
    python bot.py

The bot responds to these slash commands:
    /join   — Bot joins your voice channel
    /leave  — Bot leaves the voice channel
    /reset  — Clear conversation memory
"""

import asyncio
import io
import logging
import discord
from discord import FFmpegPCMAudio
from discord.ext import commands

import config
from pipeline import (
    VoiceActivityDetector,
    ConversationMemory,
    discord_pcm_to_16k_mono,
    process_utterance,
    synthesize_speech,
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
# Bot setup
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Per-guild state
voice_sessions: dict[int, "VoiceSession"] = {}


class VoiceSession:
    """Tracks state for one active voice channel session."""

    def __init__(self, voice_client: discord.VoiceClient, text_channel):
        self.vc = voice_client
        self.text_channel = text_channel
        self.vad = VoiceActivityDetector()
        self.memory = ConversationMemory()
        self.processing = False
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def start(self):
        """Start the audio processing loop and play the greeting."""
        self._task = asyncio.create_task(self._process_loop())
        greeting_audio = await synthesize_speech(GREETING)
        await self._play_audio(greeting_audio)

    async def stop(self):
        if self._task:
            self._task.cancel()
            self._task = None

    def feed_audio(self, pcm_48k_stereo: bytes, user: discord.Member):
        """Called by the sink for each chunk of user audio."""
        if self.processing:
            return  # don't buffer while we're responding

        pcm_16k = discord_pcm_to_16k_mono(pcm_48k_stereo)
        utterance = self.vad.process_chunk(pcm_16k)
        if utterance:
            self._audio_queue.put_nowait((utterance, user.display_name))

    async def _process_loop(self):
        """Main loop: wait for complete utterances, run through the pipeline."""
        while True:
            try:
                utterance_pcm, speaker = await self._audio_queue.get()
                self.processing = True
                logger.info(f"Processing utterance from {speaker}")

                response_audio = await process_utterance(
                    utterance_pcm, self.memory, speaker
                )
                await self._play_audio(response_audio)
                self.processing = False

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in processing loop")
                self.processing = False

    async def _play_audio(self, pcm_48k_stereo: bytes):
        """Play raw PCM audio through the Discord voice connection."""
        if not self.vc or not self.vc.is_connected():
            return

        # Write PCM to a temporary buffer and play via FFmpeg
        # Discord.py/py-cord expects an AudioSource
        source = RawPCMAudio(pcm_48k_stereo)

        # Wait for any current playback to finish
        while self.vc.is_playing():
            await asyncio.sleep(0.1)

        event = asyncio.Event()
        self.vc.play(source, after=lambda e: event.set())
        await event.wait()


class RawPCMAudio(discord.AudioSource):
    """AudioSource that reads from a raw PCM16 48kHz stereo byte buffer."""

    def __init__(self, pcm_data: bytes):
        self._buffer = io.BytesIO(pcm_data)
        # Discord expects 20ms frames of 48kHz stereo PCM16
        # 48000 * 2 channels * 2 bytes * 0.020s = 3840 bytes per frame
        self.frame_size = 3840

    def read(self) -> bytes:
        data = self._buffer.read(self.frame_size)
        if len(data) < self.frame_size:
            return b""  # signal end of audio
        return data

    def is_opus(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Custom audio sink — captures per-user audio from the voice channel
# ---------------------------------------------------------------------------

class HoloBotSink(discord.sinks.Sink):
    """Custom sink that feeds audio into our VoiceSession's VAD in real-time."""

    def __init__(self, session: VoiceSession):
        super().__init__()
        self.session = session

    def write(self, data: bytes, user: int):
        """Called by py-cord with PCM audio data for each speaking user."""
        member = self.session.vc.guild.get_member(user)
        if member and not member.bot:
            self.session.feed_audio(data, member)

    def cleanup(self):
        pass


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

@bot.slash_command(name="join", description=f"Summon {CHARACTER_NAME} to your voice channel")
async def join(ctx: discord.ApplicationContext):
    if not ctx.author.voice:
        await ctx.respond("You need to be in a voice channel first, brother.", ephemeral=True)
        return

    channel = ctx.author.voice.channel
    guild_id = ctx.guild.id

    # Leave existing session if any
    if guild_id in voice_sessions:
        old = voice_sessions[guild_id]
        await old.stop()
        if old.vc.is_connected():
            await old.vc.disconnect()

    vc = await channel.connect()
    session = VoiceSession(vc, ctx.channel)
    voice_sessions[guild_id] = session

    await ctx.respond(f"**{CHARACTER_NAME}** just joined **{channel.name}**. 🎙️")

    # Start recording (listening) and processing
    await session.start()
    vc.start_recording(HoloBotSink(session), lambda sink, channel: None, ctx.channel)


@bot.slash_command(name="leave", description=f"{CHARACTER_NAME} leaves the voice channel")
async def leave(ctx: discord.ApplicationContext):
    guild_id = ctx.guild.id
    if guild_id not in voice_sessions:
        await ctx.respond("I'm not in a voice channel.", ephemeral=True)
        return

    session = voice_sessions.pop(guild_id)
    await session.stop()
    if session.vc.is_recording():
        session.vc.stop_recording()
    await session.vc.disconnect()
    await ctx.respond(f"**{CHARACTER_NAME}** has left. Stay dangerous. 💨")


@bot.slash_command(name="reset", description="Clear conversation memory")
async def reset(ctx: discord.ApplicationContext):
    guild_id = ctx.guild.id
    if guild_id in voice_sessions:
        voice_sessions[guild_id].memory = ConversationMemory()
        await ctx.respond("Memory wiped. Fresh start, brother.")
    else:
        await ctx.respond("No active session.", ephemeral=True)


@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    logger.info(f"Character: {CHARACTER_NAME}")
    logger.info("Ready to enter voice channels.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not config.DISCORD_BOT_TOKEN:
        raise RuntimeError("DISCORD_BOT_TOKEN not set. Copy .env.example to .env and fill it in.")
    bot.run(config.DISCORD_BOT_TOKEN)
