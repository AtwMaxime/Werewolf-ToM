"""
Step 2e — Context Emotion  (CocoER)

CocoER requires its own conda environment (Python 3.7, torch 1.11).
This script drives it via subprocess: it writes a temporary batch input
file, calls CocoER's inference entrypoint, and reads back the results.

Sampled at COCOER_FPS; held for in-between frames.

Repo:  https://github.com/bisno/CocoER
Clone: models/emotion/CocoER/

Output — annotations/context_emotion_{subset}.json:
{
  "youtube/GameKey/utt_0002": {
    "n_frames":  48,
    "video_fps": 16.0,
    "sample_fps": 2,
    "frames": [
      {
        "idx": 0,
        "sampled": true,
        "persons": [
          {
            "track_id": 0,
            "context_emotions": ["Disapproval", "Doubt"],
            "context_emotion_scores": {"Disapproval": 0.61, …}
          }
        ]
      },
      …
    ]
  }
}
"""

import json
import os
import subprocess
import sys
import tempfile

import cv2
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    SUBSETS,
    expand_sampled,
    iter_utterances,
    iter_sampled_frames,
    load_annotation,
    open_video,
    save_annotation,
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

COCOER_FPS   = 2
COCOER_DIR   = os.path.join(ROOT_DIR, "models", "emotion", "CocoER")
COCOER_CONDA = "cocoer"    # conda env name
TOP_K        = 3           # number of top emotions to report as active

# CocoER's 26 emotion categories (EMOTIC)
COCOER_CLASSES = [
    "Affection", "Anger", "Annoyance", "Anticipation", "Aversion",
    "Confidence", "Disapproval", "Disconnection", "Disquietment", "Doubt",
    "Embarrassment", "Engagement", "Esteem", "Excitement", "Fatigue",
    "Fear", "Happiness", "Pain", "Peace", "Pleasure", "Sadness",
    "Sensitivity", "Suffering", "Surprise", "Sympathy", "Yearning",
]

# ──────────────────────────────────────────────
# Subprocess wrapper
# ──────────────────────────────────────────────

COCOER_RUNNER = os.path.join(os.path.dirname(__file__), "_cocoer_worker.py")


def _write_worker_script():
    """Write a minimal CocoER inference worker that reads/writes JSON."""
    script = '''"""CocoER worker — invoked inside the cocoer conda env."""
import json, sys, os
import numpy as np
import torch
from PIL import Image

COCOER_DIR = sys.argv[1]
INPUT_JSON  = sys.argv[2]
OUTPUT_JSON = sys.argv[3]

sys.path.insert(0, COCOER_DIR)
from models.cocoer import CocoER   # noqa

CKPT = os.path.join(COCOER_DIR, "checkpoints", "cocoer_best.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CocoER()
state = torch.load(CKPT, map_location=device)
model.load_state_dict(state["model"] if "model" in state else state)
model = model.to(device).eval()

with open(INPUT_JSON) as f:
    batch = json.load(f)

results = {}
for key, entry in batch.items():
    img = np.array(Image.open(entry["image_path"]).convert("RGB"))
    h, w = img.shape[:2]
    x1, y1, x2, y2 = entry["face_bbox"]
    # CocoER expects full image + normalised person bbox
    bbox_norm = [x1/w, y1/h, x2/w, y2/h]
    with torch.no_grad():
        scores = model.infer(img, bbox_norm, device=device)  # (26,) numpy
    results[key] = scores.tolist()

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f)
'''
    with open(COCOER_RUNNER, "w") as f:
        f.write(script)


# ──────────────────────────────────────────────
# Per-utterance processing
# ──────────────────────────────────────────────

def run_cocoer_batch(batch: dict) -> dict:
    """
    batch: {key: {"image_path": str, "face_bbox": [x1,y1,x2,y2]}}
    Returns: {key: [score×26]}
    """
    if not batch:
        return {}

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as fin:
        json.dump(batch, fin)
        input_path = fin.name

    output_path = input_path.replace(".json", "_out.json")

    cmd = [
        "conda", "run", "-n", COCOER_CONDA, "--no-capture-output",
        "python", COCOER_RUNNER,
        COCOER_DIR, input_path, output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    with open(output_path) as f:
        results = json.load(f)

    os.unlink(input_path)
    os.unlink(output_path)
    return results


def scores_to_entry(track_id: int, raw_scores: list) -> dict:
    score_dict = {cls: float(raw_scores[i]) for i, cls in enumerate(COCOER_CLASSES)}
    top = sorted(score_dict, key=score_dict.get, reverse=True)[:TOP_K]
    return {
        "track_id":               track_id,
        "context_emotions":       top,
        "context_emotion_scores": score_dict,
    }


def process_clip(utt: dict, detections: dict) -> dict:
    det_entry = detections.get(utt["utt_key"])
    if det_entry is None:
        return {}

    n_frames   = det_entry["n_frames"]
    video_fps  = det_entry["video_fps"]
    det_frames = det_entry["frames"]

    cap, _, _ = open_video(utt["clip_path"])
    sampled_frames = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for frame_idx, frame in iter_sampled_frames(cap, n_frames, video_fps, COCOER_FPS):
            det_frame = det_frames[frame_idx] if frame_idx < len(det_frames) else {}
            persons   = det_frame.get("persons", [])

            # Save frame to disk for subprocess
            frame_path = os.path.join(tmpdir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)

            batch = {}
            for p in persons:
                if p.get("face_bbox") is None:
                    continue
                key = f"{frame_idx}_{p['track_id']}"
                batch[key] = {"image_path": frame_path, "face_bbox": p["face_bbox"]}

            if batch:
                raw = run_cocoer_batch(batch)
            else:
                raw = {}

            persons_out = []
            for p in persons:
                key = f"{frame_idx}_{p['track_id']}"
                if key in raw:
                    persons_out.append(scores_to_entry(p["track_id"], raw[key]))
                else:
                    persons_out.append({
                        "track_id":               p["track_id"],
                        "context_emotions":       None,
                        "context_emotion_scores": None,
                    })
            sampled_frames.append({"persons": persons_out})

    cap.release()

    all_frames = expand_sampled(sampled_frames, n_frames, video_fps, COCOER_FPS)
    return {
        "n_frames":   n_frames,
        "video_fps":  video_fps,
        "sample_fps": COCOER_FPS,
        "frames":     all_frames,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run(subset: str):
    print(f"\n[cocoer] {subset}")
    _write_worker_script()
    detections = load_annotation("detections", subset)

    output = {}
    utts = list(iter_utterances(subset))
    for utt in tqdm(utts, desc=f"  {subset}", unit="clip"):
        try:
            result = process_clip(utt, detections)
            if result:
                output[utt["utt_key"]] = result
        except Exception as e:
            print(f"  WARN {utt['utt_key']}: {e}")

    save_annotation(output, "context_emotion", subset)


if __name__ == "__main__":
    for subset in SUBSETS:
        run(subset)
