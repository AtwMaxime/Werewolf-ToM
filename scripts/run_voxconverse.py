"""
Step 2h — Speaker Diarization  (pyannote.audio)

Diarization is run on the **full game audio** (not per-clip) to get
consistent speaker labels across an entire game.  The resulting speaker
timeline is then intersected with each utterance's time window to assign
a dominant speaker label.

pip install pyannote.audio
Requires a HuggingFace token with access to pyannote/speaker-diarization-3.1.
Run: huggingface-cli login

This is a **per-utterance** annotation.

Output — annotations/diarization_{subset}.json:
{
  "youtube/GameKey/utt_0002": {
    "diarization_speaker": "SPEAKER_01",   // pyannote speaker ID
    "diarization_overlap": false           // true if >1 speaker in window
  }
}
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import torch

# PyTorch ≥2.6 defaults weights_only=True which breaks pyannote/lightning ckpts
_orig_torch_load = torch.load
def _torch_load_unsafe(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_unsafe

from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    DATASET_DIR,
    SPLITS,
    SUBSETS,
    iter_utterances,
    save_annotation,
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

MODEL_ID = "pyannote/speaker-diarization-3.1"

# ──────────────────────────────────────────────
# Game-level audio extraction
# ──────────────────────────────────────────────

def game_video_path(subset: str, game_key: str) -> str | None:
    """Return the path to the full game video (first part for Ego4D)."""
    if subset == "youtube":
        videos_dir = os.path.join(DATASET_DIR, "Youtube", "videos")
        # game_key = "{YT_ID}_{Game_ID}" → video = "{YT_ID}_{Game_ID}.mp4"
        path = os.path.join(videos_dir, f"{game_key}.mp4")
        return path if os.path.exists(path) else None
    else:
        videos_dir = os.path.join(DATASET_DIR, "Ego4D", "videos")
        # Ego4D games have parts; use part 1 for diarization of full timeline
        # game_key = "{EG_ID}_{Game_ID}" → video = "{EG_ID}_GameN_1.mp4"
        eg_id, game_id = game_key.rsplit("_", 1)
        game_num = game_id.replace("Game", "")
        path = os.path.join(videos_dir, f"{eg_id}_Game{game_num}_1.mp4")
        return path if os.path.exists(path) else None


def extract_game_audio(video_path: str, out_wav: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", "-vn", out_wav],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )


# ──────────────────────────────────────────────
# Timestamp helpers
# ──────────────────────────────────────────────

def ts_to_seconds(ts: str) -> float:
    parts = ts.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


# ──────────────────────────────────────────────
# Diarization → utterance mapping
# ──────────────────────────────────────────────

def diarize_game_audio(pipeline, wav_path: str) -> list[tuple]:
    """
    Return list of (start_sec, end_sec, speaker_label).
    """
    diarization = pipeline(wav_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))
    return segments


def assign_to_utterance(
    segments: list[tuple],
    utt_start: float,
    utt_end: float,
) -> tuple[str | None, bool]:
    """
    Find the dominant speaker within [utt_start, utt_end].
    Returns (speaker_label, has_overlap).
    """
    coverage = defaultdict(float)
    for seg_start, seg_end, speaker in segments:
        overlap = max(0.0, min(seg_end, utt_end) - max(seg_start, utt_start))
        if overlap > 0:
            coverage[speaker] += overlap

    if not coverage:
        return None, False

    dominant = max(coverage, key=coverage.get)
    has_overlap = len(coverage) > 1
    return dominant, has_overlap


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def build_utt_time_index(subset: str) -> dict:
    """
    Returns {utt_key: (start_sec, end_sec)} from split JSONs.
    end_sec is approximated as the next utterance's start (or start+15).
    """
    if subset == "youtube":
        split_root = os.path.join(DATASET_DIR, "Youtube", "split")
        key_fields = ("YT_ID", "Game_ID")
    else:
        split_root = os.path.join(DATASET_DIR, "Ego4D", "split")
        key_fields = ("EG_ID", "Game_ID")

    index = {}
    for split in SPLITS:
        split_file = os.path.join(split_root, f"{split}.json")
        if not os.path.exists(split_file):
            continue
        with open(split_file) as f:
            games = json.load(f)

        for game in games:
            id_a = game[key_fields[0]]
            id_b = game[key_fields[1]]
            game_key = f"{id_a}_{id_b}"
            dialogue = game["Dialogue"]
            for i, utt in enumerate(dialogue):
                rec_id = utt["Rec_Id"]
                utt_key = f"{subset}/{game_key}/utt_{rec_id:04d}"
                try:
                    start = ts_to_seconds(utt["timestamp"])
                except (KeyError, ValueError):
                    continue
                if i + 1 < len(dialogue):
                    try:
                        end = min(ts_to_seconds(dialogue[i + 1]["timestamp"]), start + 15.0)
                    except (KeyError, ValueError):
                        end = start + 5.0
                else:
                    end = start + 5.0
                index[utt_key] = (start, end, game_key)

    return index


def run(subset: str):
    print(f"\n[diarization] {subset}")

    from pyannote.audio import Pipeline   # noqa: late import (optional dep)
    pipeline = Pipeline.from_pretrained(MODEL_ID, use_auth_token=True)

    utt_time_index = build_utt_time_index(subset)

    # Group utterances by game so we diarize each game once
    by_game: dict[str, list[str]] = defaultdict(list)
    for utt_key, (start, end, game_key) in utt_time_index.items():
        by_game[game_key].append(utt_key)

    output = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for game_key, utt_keys in tqdm(by_game.items(), desc=f"  {subset}", unit="game"):
            video_path = game_video_path(subset, game_key)
            if video_path is None:
                continue

            wav_path = os.path.join(tmpdir, f"{game_key}.wav")
            try:
                extract_game_audio(video_path, wav_path)
                segments = diarize_game_audio(pipeline, wav_path)
            except Exception as e:
                print(f"  WARN {game_key}: {e}")
                continue

            for utt_key in utt_keys:
                start, end, _ = utt_time_index[utt_key]
                speaker, overlap = assign_to_utterance(segments, start, end)
                output[utt_key] = {
                    "diarization_speaker": speaker,
                    "diarization_overlap": overlap,
                }

            # Clean up per-game WAV to save disk space
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    save_annotation(output, "diarization", subset)


if __name__ == "__main__":
    for subset in SUBSETS:
        run(subset)
