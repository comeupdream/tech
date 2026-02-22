"""Scrape YouTube videos for voice reference samples.

Downloads audio from YouTube videos/playlists, extracts clean vocal segments,
and prepares them for ElevenLabs voice cloning.

Usage:
    python scraper/youtube_scraper.py --urls URL1 URL2 ...
    python scraper/youtube_scraper.py --search "Andrew Tate interview" --max-results 5
    python scraper/youtube_scraper.py --urls URL1 --split-segments --segment-length 30

ElevenLabs voice cloning tips:
    - Upload 1-30 minutes of CLEAN speech (no music, no other speakers)
    - Consistent audio quality across samples
    - Variety of tones/emotions helps (calm, energetic, laughing, serious)
    - Remove background noise and music before uploading
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "audio_samples"


def download_audio(url: str, output_dir: Path, prefix: str = "sample") -> Path | None:
    """Download audio from a YouTube URL using yt-dlp."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f"{prefix}_%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "-ar 24000 -ac 1",  # 24kHz mono for ElevenLabs
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


def main():
    parser = argparse.ArgumentParser(description="Download YouTube audio for voice cloning")
    parser.add_argument("--urls", nargs="+", help="YouTube URLs to download")
    parser.add_argument("--search", type=str, help="Search query to find videos")
    parser.add_argument("--max-results", type=int, default=3, help="Max search results")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--split-segments", action="store_true", help="Split into segments")
    parser.add_argument("--segment-length", type=int, default=30, help="Segment length in seconds")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"\n{'='*50}")
    print(f"Downloaded {len(downloaded)} audio files to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Listen to the files and remove any with music/other speakers")
    print(f"  2. Go to https://elevenlabs.io/voice-cloning")
    print(f"  3. Upload the clean audio files to create a cloned voice")
    print(f"  4. Copy the Voice ID and set ELEVENLABS_VOICE_ID in your .env file")


if __name__ == "__main__":
    main()
