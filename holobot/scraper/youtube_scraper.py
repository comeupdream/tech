"""Scrape YouTube videos for voice reference samples.

Downloads audio from YouTube videos, extracts clean vocal segments,
and prepares a reference.wav for Chatterbox voice cloning.

Usage:
    # Download specific videos
    python scraper/youtube_scraper.py --urls URL1 URL2 ...

    # Search and download
    python scraper/youtube_scraper.py --search "Andrew Tate interview" --max-results 5

    # Download, split into 30s segments, and auto-pick the best one
    python scraper/youtube_scraper.py --urls URL1 --split-segments --segment-length 30

Chatterbox voice cloning tips:
    - Only needs ONE clean clip of ~10-30 seconds
    - Single speaker only (no music, no other people talking)
    - Clear audio quality, minimal background noise
    - The scraper creates a reference.wav automatically from the best segment
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "audio_samples"
REFERENCE_FILE = OUTPUT_DIR / "reference.wav"


def download_audio(url: str, output_dir: Path, prefix: str = "sample") -> Path | None:
    """Download audio from a YouTube URL using yt-dlp."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f"{prefix}_%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        # 24kHz mono — Chatterbox's native sample rate
        "--postprocessor-args", "-ar 24000 -ac 1",
        "--output", output_template,
        "--no-playlist",
        url,
    ]

    print(f"Downloading: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:200]}")
        return None

    # Find the downloaded file
    for line in result.stdout.splitlines():
        if "[ExtractAudio] Destination:" in line:
            path = line.split("Destination:", 1)[1].strip()
            print(f"  Saved: {path}")
            return Path(path)

    # Fallback — look for newest .wav in the output dir
    wavs = sorted(output_dir.glob("*.wav"), key=os.path.getmtime, reverse=True)
    if wavs:
        print(f"  Saved: {wavs[0]}")
        return wavs[0]

    print("  WARNING: Could not find downloaded file")
    return None


def search_and_download(query: str, max_results: int, output_dir: Path) -> list[Path]:
    """Search YouTube and download top results."""
    cmd = [
        "yt-dlp",
        f"ytsearch{max_results}:{query}",
        "--get-url", "--get-title", "--get-id",
        "--no-playlist",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Search failed: {result.stderr[:200]}")
        return []

    # Parse video IDs from output
    lines = result.stdout.strip().splitlines()
    video_ids = [line for line in lines if len(line) == 11 and " " not in line]

    downloaded = []
    for i, vid_id in enumerate(video_ids[:max_results]):
        url = f"https://www.youtube.com/watch?v={vid_id}"
        path = download_audio(url, output_dir, prefix=f"sample_{i:03d}")
        if path:
            downloaded.append(path)

    return downloaded


def split_audio(input_path: Path, segment_length: int, output_dir: Path) -> list[Path]:
    """Split a long audio file into segments using ffmpeg."""
    stem = input_path.stem
    output_pattern = str(output_dir / f"{stem}_seg_%03d.wav")

    cmd = [
        "ffmpeg", "-i", str(input_path),
        "-f", "segment",
        "-segment_time", str(segment_length),
        "-ar", "24000", "-ac", "1",
        "-c:a", "pcm_s16le",
        output_pattern,
        "-y",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Split failed: {result.stderr[:200]}")
        return []

    segments = sorted(output_dir.glob(f"{stem}_seg_*.wav"))
    print(f"  Split into {len(segments)} segments of ~{segment_length}s each")
    return segments


def extract_reference(input_path: Path, output_path: Path, start: float = 0, duration: float = 25):
    """Extract a clean reference clip from a longer audio file.
    Chatterbox works best with 10-30 seconds of clean speech."""
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-ss", str(start),
        "-t", str(duration),
        "-ar", "24000", "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path),
        "-y",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Extract failed: {result.stderr[:200]}")
        return False

    size_kb = output_path.stat().st_size / 1024
    print(f"  Reference clip: {output_path} ({size_kb:.0f} KB, {duration}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download YouTube audio for voice cloning")
    parser.add_argument("--urls", nargs="+", help="YouTube URLs to download")
    parser.add_argument("--search", type=str, help="Search query to find videos")
    parser.add_argument("--max-results", type=int, default=3, help="Max search results")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--split-segments", action="store_true", help="Split into segments")
    parser.add_argument("--segment-length", type=int, default=30, help="Segment length in seconds")
    parser.add_argument(
        "--ref-start", type=float, default=10,
        help="Start time (seconds) for reference clip extraction (skip intros)"
    )
    parser.add_argument(
        "--ref-duration", type=float, default=25,
        help="Duration (seconds) of the reference clip"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check dependencies
    if not shutil.which("yt-dlp"):
        print("ERROR: yt-dlp not found. Install it: pip install yt-dlp")
        sys.exit(1)
    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found. Install it: sudo apt install ffmpeg")
        sys.exit(1)

    downloaded = []

    if args.urls:
        for url in args.urls:
            path = download_audio(url, output_dir)
            if path:
                downloaded.append(path)

    if args.search:
        downloaded.extend(search_and_download(args.search, args.max_results, output_dir))

    if not args.urls and not args.search:
        # Default: search for Andrew Tate clips
        print("No URLs or search query provided. Searching for Andrew Tate interview clips...")
        queries = [
            "Andrew Tate interview podcast solo",
            "Andrew Tate motivational speech",
            "Andrew Tate funny moments talking",
        ]
        for q in queries:
            downloaded.extend(search_and_download(q, 2, output_dir))

    if args.split_segments and downloaded:
        all_segments = []
        for path in downloaded:
            segments = split_audio(path, args.segment_length, output_dir)
            all_segments.extend(segments)
        print(f"\nTotal segments: {len(all_segments)}")

    # Auto-create reference.wav from the first download
    if downloaded and not REFERENCE_FILE.exists():
        print(f"\nCreating reference clip for Chatterbox voice cloning...")
        extract_reference(
            downloaded[0], REFERENCE_FILE,
            start=args.ref_start, duration=args.ref_duration
        )

    print(f"\n{'='*60}")
    print(f"Downloaded {len(downloaded)} audio files to: {output_dir}")
    if REFERENCE_FILE.exists():
        print(f"\nReference audio ready: {REFERENCE_FILE}")
        print(f"  Chatterbox will use this to clone the voice.")
        print(f"  Listen to it — if the audio quality is bad or has")
        print(f"  other speakers, replace it with a cleaner clip.")
    print(f"\nNext steps:")
    print(f"  1. Listen to audio_samples/reference.wav")
    print(f"     - Should be ~10-30s of ONLY the target speaking")
    print(f"     - No music, no other voices, clear audio")
    print(f"  2. If it's not clean, manually pick a better segment:")
    print(f"     python scraper/youtube_scraper.py --urls YOUR_URL --ref-start 45 --ref-duration 25")
    print(f"  3. Run the bot: python bot.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
