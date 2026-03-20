"""
Werewolf Among Us — Utterance Clip Extractor

For each utterance in the YouTube and Ego4D splits, extracts a short MP4 clip
from the corresponding game video. Clips are saved to:

    dataset/WerewolfAmongUs/clips/youtube/{YT_ID}_{Game_ID}/utt_{Rec_Id:04d}.mp4
    dataset/WerewolfAmongUs/clips/ego4d/{EG_ID}_{Game_ID}/utt_{Rec_Id:04d}.mp4

Clip duration: from utterance timestamp to the next utterance's timestamp,
capped at MAX_CLIP_DURATION seconds. This ensures each clip is tight around
the spoken utterance, which is important for downstream model inference
(AffWild2, GazeFollow, VideoCoAttention, etc.).

Ego4D games are split into multiple parts (e.g. _1.mp4, _2.mp4).
Part boundaries are detected by timestamp resets (timestamp decreases vs previous).
"""

import json
import os
import subprocess

from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEREWOLF_DIR = os.path.join(ROOT_DIR, "data", "WerewolfAmongUs")

YOUTUBE_DIR = os.path.join(WEREWOLF_DIR, "Youtube")
EGO4D_DIR = os.path.join(WEREWOLF_DIR, "Ego4D")

CLIPS_DIR = os.path.join(WEREWOLF_DIR, "clips")
YOUTUBE_CLIPS_DIR = os.path.join(CLIPS_DIR, "youtube")
EGO4D_CLIPS_DIR = os.path.join(CLIPS_DIR, "ego4d")

SPLITS = ["train", "val", "test"]

# Max clip duration in seconds (caps clips between two far-apart utterances)
MAX_CLIP_DURATION = 15.0
# Padding added after the last utterance in a part
LAST_UTT_PADDING = 5.0


# ==========================================
# 2. HELPERS
# ==========================================


def ts_to_seconds(ts: str) -> float:
    """Convert 'MM:SS' timestamp string to float seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    raise ValueError(f"Unrecognised timestamp format: {ts!r}")


def get_video_duration(path: str) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def extract_clip(input_path: str, start: float, duration: float, output_path: str):
    """
    Extract a clip from input_path starting at `start` seconds for `duration` seconds.
    Re-encodes to H.264 at 16fps to match the project's video standard.
    Skips if output already exists.
    """
    if os.path.exists(output_path):
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-ss",
            str(start),
            "-i",
            input_path,
            "-t",
            str(duration),
            "-r",
            "16",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-preset",
            "fast",
            "-c:a",
            "aac",
            "-y",
            output_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def split_utterances_by_part(dialogue: list) -> list[list[dict]]:
    """
    Split a flat dialogue list into parts based on timestamp resets.
    A reset is detected when the current timestamp is significantly earlier
    than the previous one (threshold: 30 seconds).

    Returns a list of parts, each part being a list of utterance dicts.
    """
    parts = []
    current_part = []
    prev_seconds = -1

    for utt in dialogue:
        try:
            t = ts_to_seconds(utt["timestamp"])
        except ValueError:
            continue

        if prev_seconds >= 0 and t < prev_seconds - 30:
            # Timestamp reset → new part
            parts.append(current_part)
            current_part = []

        current_part.append(utt)
        prev_seconds = t

    if current_part:
        parts.append(current_part)

    return parts


def compute_clip_windows(part_utts: list, part_duration: float) -> list[tuple]:
    """
    For each utterance in a part, compute (start_sec, clip_duration_sec).
    Clip ends at the start of the next utterance, capped at MAX_CLIP_DURATION.
    The last utterance in a part gets LAST_UTT_PADDING seconds of tail.
    """
    windows = []
    for i, utt in enumerate(part_utts):
        start = ts_to_seconds(utt["timestamp"])
        if i + 1 < len(part_utts):
            next_start = ts_to_seconds(part_utts[i + 1]["timestamp"])
            duration = min(next_start - start, MAX_CLIP_DURATION)
        else:
            duration = min(LAST_UTT_PADDING, part_duration - start, MAX_CLIP_DURATION)
        duration = max(duration, 0.5)  # at least 0.5s
        windows.append((start, duration))
    return windows


# ==========================================
# 3. YOUTUBE EXTRACTION
# ==========================================


def collect_youtube_jobs():
    """Pre-collect all (video_file, start, duration, out_path) jobs for YouTube."""
    videos_dir = os.path.join(YOUTUBE_DIR, "videos")
    jobs = []
    missing = 0

    for split in SPLITS:
        split_path = os.path.join(YOUTUBE_DIR, "split", f"{split}.json")
        if not os.path.exists(split_path):
            continue

        with open(split_path) as f:
            games = json.load(f)

        for game in games:
            video_name = game["video_name"]
            game_id = game["Game_ID"]
            yt_id = game["YT_ID"]

            video_file = os.path.join(videos_dir, f"{video_name}_{game_id}.mp4")
            if not os.path.exists(video_file):
                missing += len(game["Dialogue"])
                continue

            video_duration = get_video_duration(video_file)
            dialogue = game["Dialogue"]
            windows = compute_clip_windows(dialogue, video_duration)
            game_key = f"{yt_id}_{game_id}"
            out_dir = os.path.join(YOUTUBE_CLIPS_DIR, game_key)

            for utt, (start, dur) in zip(dialogue, windows):
                rec_id = utt["Rec_Id"]
                out_path = os.path.join(out_dir, f"utt_{rec_id:04d}.mp4")
                jobs.append((video_file, start, dur, out_path))

    return jobs, missing


def process_youtube():
    print("  Scanning YouTube games...")
    jobs, missing = collect_youtube_jobs()

    if missing:
        print(f"  ⚠️  {missing} utterances skipped (video not found)")

    errors = 0
    with tqdm(total=len(jobs), unit="clip", desc="  YouTube") as pbar:
        for video_file, start, duration, out_path in jobs:
            try:
                extract_clip(video_file, start, duration, out_path)
            except subprocess.CalledProcessError:
                errors += 1
            pbar.update(1)

    done = len(jobs) - errors
    print(f"  YouTube: {done} clips extracted, {missing} skipped, {errors} errors")


# ==========================================
# 4. EGO4D EXTRACTION
# ==========================================


def collect_ego4d_jobs():
    """Pre-collect all (video_file, start, duration, out_path) jobs for Ego4D."""
    videos_dir = os.path.join(EGO4D_DIR, "videos")
    jobs = []
    missing = 0

    for split in SPLITS:
        split_path = os.path.join(EGO4D_DIR, "split", f"{split}.json")
        if not os.path.exists(split_path):
            continue

        with open(split_path) as f:
            games = json.load(f)

        for game in games:
            eg_id = game["EG_ID"]
            game_id = game["Game_ID"]
            game_num = game_id.replace("Game", "")

            dialogue = game["Dialogue"]
            parts = split_utterances_by_part(dialogue)
            game_key = f"{eg_id}_{game_id}"
            out_dir = os.path.join(EGO4D_CLIPS_DIR, game_key)

            for part_idx, part_utts in enumerate(parts, start=1):
                video_file = os.path.join(
                    videos_dir, f"{eg_id}_Game{game_num}_{part_idx}.mp4"
                )
                if not os.path.exists(video_file):
                    missing += len(part_utts)
                    continue

                video_duration = get_video_duration(video_file)
                windows = compute_clip_windows(part_utts, video_duration)

                for utt, (start, dur) in zip(part_utts, windows):
                    rec_id = utt["Rec_Id"]
                    out_path = os.path.join(out_dir, f"utt_{rec_id:04d}.mp4")
                    jobs.append((video_file, start, dur, out_path))

    return jobs, missing


def process_ego4d():
    print("  Scanning Ego4D games...")
    jobs, missing = collect_ego4d_jobs()

    if missing:
        print(f"  ⚠️  {missing} utterances skipped (part video not found)")

    errors = 0
    with tqdm(total=len(jobs), unit="clip", desc="  Ego4D ") as pbar:
        for video_file, start, duration, out_path in jobs:
            try:
                extract_clip(video_file, start, duration, out_path)
            except subprocess.CalledProcessError:
                errors += 1
            pbar.update(1)

    done = len(jobs) - errors
    print(f"  Ego4D: {done} clips extracted, {missing} skipped, {errors} errors")


# ==========================================
# 5. MAIN
# ==========================================

if __name__ == "__main__":
    os.makedirs(YOUTUBE_CLIPS_DIR, exist_ok=True)
    os.makedirs(EGO4D_CLIPS_DIR, exist_ok=True)

    print("\n🎬 Extracting YouTube utterance clips...")
    process_youtube()

    print("\n🎬 Extracting Ego4D utterance clips...")
    process_ego4d()

    print("\n✨ Done! Clips saved to:", CLIPS_DIR)
