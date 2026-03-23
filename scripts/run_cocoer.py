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
import json, sys, os, types
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

COCOER_DIR = sys.argv[1]
INPUT_JSON  = sys.argv[2]
OUTPUT_JSON = sys.argv[3]

os.chdir(COCOER_DIR)  # model_GWT loads ./checkpoints/VI_weights/w.pth via relative path
sys.path.insert(0, COCOER_DIR)
from models_sw import model_GWT  # noqa

# ── build minimal args namespace ──
args = types.SimpleNamespace(
    num_class=26, num_block1=1, num_block2=1,
    gpu="0",       # model_GWT reads args.gpu[0] -> "0"
    loss_type="bce",
    inside_lr=1.0,
)

CKPT   = os.path.join(COCOER_DIR, "checkpoints", "GWT.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = model_GWT(args)
state = torch.load(CKPT, map_location=device)
model.load_state_dict(state["model"] if "model" in state else state)
model = model.to(device).eval()

trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])

def scale_coord(coord, src, tgt):
    x1, y1, x2, y2 = coord
    h0, w0 = src
    h1, w1 = tgt
    return (int(x1*w1/w0), int(y1*h1/h0), int(x2*w1/w0), int(y2*h1/h0))

with open(INPUT_JSON) as f:
    batch = json.load(f)

results = {}
for key, entry in batch.items():
    try:
        pil_img = Image.open(entry["image_path"]).convert("RGB")
        W, H = pil_img.size

        face_bbox = entry["face_bbox"]   # [x1,y1,x2,y2]
        body_bbox = entry.get("body_bbox") or face_bbox  # fallback to face if no body

        ctx  = trans(pil_img).unsqueeze(0).to(device)
        body = trans(pil_img.crop(body_bbox)).unsqueeze(0).to(device)
        head = trans(pil_img.crop(face_bbox)).unsqueeze(0).to(device)

        cb = torch.tensor([scale_coord(body_bbox, (H,W), (224,224))]).to(device)
        ch = torch.tensor([scale_coord(face_bbox, (H,W), (224,224))]).to(device)

        with torch.no_grad():
            pred, _, _ = model(ctx, body, head, cb, ch, None)
        scores = torch.sigmoid(pred[0]).cpu().numpy()
        results[key] = scores.tolist()
    except Exception as e:
        results[key] = None

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
        "/local_scratch/mattwood/miniconda3/bin/conda",
        "run", "-n", COCOER_CONDA, "--no-capture-output",
        "python", COCOER_RUNNER,
        COCOER_DIR, input_path, output_path,
    ]
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)

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


CLIP_BATCH_SIZE = 30  # clips per subprocess call to amortise model-load cost


def collect_clip_frames(utt: dict, detections: dict, tmpdir: str) -> dict:
    """
    Extract sampled frames for one clip, save to tmpdir, return metadata.
    Returns {utt_key: {"n_frames", "video_fps", "frame_data": [(idx, persons)],
                        "batch": {key: {image_path, face_bbox, body_bbox}}}}
    or None if clip has no detection entry.
    """
    det_entry = detections.get(utt["utt_key"])
    if det_entry is None:
        return None

    n_frames   = det_entry["n_frames"]
    video_fps  = det_entry["video_fps"]
    det_frames = det_entry["frames"]

    cap, _, _ = open_video(utt["clip_path"])
    frame_data = []
    clip_batch = {}
    utt_prefix = utt["utt_key"].replace("/", "__")

    for frame_idx, frame in iter_sampled_frames(cap, n_frames, video_fps, COCOER_FPS):
        det_frame = det_frames[frame_idx] if frame_idx < len(det_frames) else {}
        persons   = det_frame.get("persons", [])
        frame_path = os.path.join(tmpdir, f"{utt_prefix}_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        for p in persons:
            if p.get("face_bbox") is None:
                continue
            key = f"{utt['utt_key']}|{frame_idx}|{p['track_id']}"
            clip_batch[key] = {
                "image_path": frame_path,
                "face_bbox":  p["face_bbox"],
                "body_bbox":  p.get("pose_bbox"),
            }
        frame_data.append((frame_idx, persons))
    cap.release()

    return {
        "n_frames":   n_frames,
        "video_fps":  video_fps,
        "frame_data": frame_data,
        "batch":      clip_batch,
    }


def assemble_clip_result(clip_info: dict, raw: dict) -> dict:
    """Build the per-clip output dict from subprocess scores."""
    utt_key   = clip_info["utt_key"]
    n_frames  = clip_info["n_frames"]
    video_fps = clip_info["video_fps"]

    sampled_frames = []
    for frame_idx, persons in clip_info["frame_data"]:
        persons_out = []
        for p in persons:
            key = f"{utt_key}|{frame_idx}|{p['track_id']}"
            scores = raw.get(key)
            if scores is not None:
                persons_out.append(scores_to_entry(p["track_id"], scores))
            else:
                persons_out.append({
                    "track_id":               p["track_id"],
                    "context_emotions":       None,
                    "context_emotion_scores": None,
                })
        sampled_frames.append({"persons": persons_out})

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

    with tempfile.TemporaryDirectory() as tmpdir:
        for batch_start in tqdm(range(0, len(utts), CLIP_BATCH_SIZE),
                                desc=f"  {subset}", unit="batch"):
            batch_utts = utts[batch_start: batch_start + CLIP_BATCH_SIZE]
            clip_infos = []
            combined_batch = {}

            for utt in batch_utts:
                try:
                    info = collect_clip_frames(utt, detections, tmpdir)
                    if info is None:
                        continue
                    info["utt_key"] = utt["utt_key"]
                    clip_infos.append(info)
                    combined_batch.update(info["batch"])
                except Exception as e:
                    print(f"  WARN {utt['utt_key']} (collect): {e}")

            try:
                raw = run_cocoer_batch(combined_batch) if combined_batch else {}
                for ci in clip_infos:
                    result = assemble_clip_result(ci, raw)
                    if result:
                        output[ci["utt_key"]] = result
            except Exception as e:
                print(f"  WARN batch {batch_start}: {e}")

    save_annotation(output, "context_emotion", subset)


if __name__ == "__main__":
    for subset in SUBSETS:
        run(subset)
