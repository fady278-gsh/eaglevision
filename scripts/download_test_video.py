#!/usr/bin/env python3
"""
scripts/download_test_video.py
==============================
Downloads a short excavator clip from YouTube for testing.
Uses yt-dlp for reliable downloads.

Usage:
    pip install yt-dlp
    python scripts/download_test_video.py
    # or specify custom URL:
    python scripts/download_test_video.py --url "https://www.youtube.com/watch?v=..."
"""

import argparse
import subprocess
import sys
import os

# Short, fixed-camera excavator clips (30-90 seconds)
DEFAULT_URLS = [
    "https://www.youtube.com/watch?v=JyxxkKFc6fU",   # the one you shared
]

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "input.mp4")


def download(url: str, output: str, max_seconds: int = 60):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    print(f"⬇  Downloading (first {max_seconds}s): {url}")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "--download-sections", f"*0-{max_seconds}",
        "--force-keyframes-at-cuts",
        "-o", output,
        url,
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("❌ yt-dlp failed. Try: pip install -U yt-dlp")
        sys.exit(1)
    print(f"✅ Saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",     default=DEFAULT_URLS[0])
    parser.add_argument("--output",  default=OUTPUT_FILE)
    parser.add_argument("--seconds", type=int, default=60)
    args = parser.parse_args()
    download(args.url, args.output, args.seconds)
