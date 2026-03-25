"""
Step 2e — Context Emotion  (CocoER)

Run directly with the cocoer conda env (Python 3.7):
  CUDA_VISIBLE_DEVICES=1 \\
  /local_scratch/mattwood/miniconda3/envs/cocoer/bin/python scripts/run_cocoer.py

Model (model_GWT) is loaded ONCE per subset and called in-process — no subprocess
overhead. Sampled at COCOER_FPS; held for in-between frames.

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
            "context_emotion_scores": {"Disapproval": 0.61, ...}
          }
        ]
      },
      ...
    ]
  }
}
"""

from __future__ import annotations

import json
import os
import sys
import types

import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))   # absolute — safe after chdir
COCOER_DIR  = os.path.join(ROOT_DIR, "models", "emotion", "CocoER")

# Import OUR pipeline utils under a private name so CocoER's utils.py
# stays importable as 'utils' for models_sw.py's `from utils import *`.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_pipeline_utils",
                                     os.path.join(SCRIPTS_DIR, "utils.py"))
_pu = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pu)
SUBSETS           = _pu.SUBSETS
expand_sampled    = _pu.expand_sampled
iter_utterances   = _pu.iter_utterances
iter_sampled_frames = _pu.iter_sampled_frames
load_annotation   = _pu.load_annotation
open_video        = _pu.open_video
save_annotation   = _pu.save_annotation

# Now set up CocoER imports (utils.py here refers to CocoER's, not ours)
os.chdir(COCOER_DIR)          # model_GWT loads ./checkpoints/VI_weights/w.pth
sys.path.insert(0, COCOER_DIR)

# Explicitly load CocoER's utils into sys.modules['utils'] so that
# models_sw.py's `from utils import *` picks up DiscreteLoss etc.
import importlib as _il
sys.modules.pop("utils", None)
_il.import_module("utils")

# Patch torch.load so VI weights (saved on cuda:1) deserialise correctly
# when CUDA_VISIBLE_DEVICES remaps device indices.
_orig_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs.setdefault("map_location", "cpu")
    return _orig_load(*args, **kwargs)
torch.load = _safe_load

from models_sw import model_GWT  # noqa: E402

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

COCOER_FPS = 2
TOP_K      = 3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

COCOER_CLASSES = [
    "Affection", "Anger", "Annoyance", "Anticipation", "Aversion",
    "Confidence", "Disapproval", "Disconnection", "Disquietment", "Doubt",
    "Embarrassment", "Engagement", "Esteem", "Excitement", "Fatigue",
    "Fear", "Happiness", "Pain", "Peace", "Pleasure", "Sadness",
    "Sensitivity", "Suffering", "Surprise", "Sympathy", "Yearning",
]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])

# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_model():
    args = types.SimpleNamespace(
        num_class=26, num_block1=1, num_block2=1,
        gpu="0", loss_type="bce", inside_lr=1.0,
    )
    model = model_GWT(args)
    ckpt  = os.path.join(COCOER_DIR, "checkpoints", "GWT.pth")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(DEVICE).eval()


# ──────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────

INFER_BATCH = 64   # max persons per GPU forward pass

def _scale_coord(coord, src_hw, tgt_hw=(224, 224)):
    x1, y1, x2, y2 = coord
    h0, w0 = src_hw
    h1, w1 = tgt_hw
    return (int(x1*w1/w0), int(y1*h1/h0), int(x2*w1/w0), int(y2*h1/h0))


# ──────────────────────────────────────────────
# Per-utterance processing  (fully batched)
# ──────────────────────────────────────────────

def infer_person(model, ctx_t, pil_img, face_bbox, body_bbox, H, W):
    """Single-person forward pass — matches the original working API exactly."""
    ctx  = ctx_t.unsqueeze(0).to(DEVICE)
    body = _transform(pil_img.crop(body_bbox)).unsqueeze(0).to(DEVICE)
    head = _transform(pil_img.crop(face_bbox)).unsqueeze(0).to(DEVICE)
    cb   = torch.tensor([_scale_coord(body_bbox, (H, W))]).to(DEVICE)
    ch   = torch.tensor([_scale_coord(face_bbox, (H, W))]).to(DEVICE)
    with torch.no_grad():
        pred, _, _ = model(ctx, body, head, cb, ch, None)
    if pred is None:
        return None
    scores_np = torch.sigmoid(pred[0]).cpu().numpy()
    return {cls: float(scores_np[i]) for i, cls in enumerate(COCOER_CLASSES)}


def process_clip(model, utt, detections):
    """
    Pre-load all sampled frames in one sequential video pass,
    then run one forward pass per person (batch=1, matching the
    original working API).
    """
    det_entry = detections.get(utt["utt_key"])
    if det_entry is None:
        return {}

    n_frames   = det_entry["n_frames"]
    video_fps  = det_entry["video_fps"]
    det_frames = det_entry["frames"]

    cap, _, _ = open_video(utt["clip_path"])
    sampled_frames_out = []

    for frame_idx, frame_bgr in iter_sampled_frames(cap, n_frames, video_fps, COCOER_FPS):
        det_frame = det_frames[frame_idx] if frame_idx < len(det_frames) else {}
        persons   = det_frame.get("persons", [])

        H, W = frame_bgr.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        ctx_t   = _transform(pil_img)

        frame_persons = []
        for p in persons:
            face_bbox = p.get("face_bbox")
            if face_bbox is None:
                frame_persons.append({
                    "track_id": p["track_id"],
                    "context_emotions": None,
                    "context_emotion_scores": None,
                })
                continue
            body_bbox = p.get("pose_bbox") or face_bbox
            try:
                score_dict = infer_person(model, ctx_t, pil_img,
                                          face_bbox, body_bbox, H, W)
                if score_dict is None:
                    raise ValueError("model returned None")
                top = sorted(score_dict, key=score_dict.get, reverse=True)[:TOP_K]
                frame_persons.append({
                    "track_id": p["track_id"],
                    "context_emotions": top,
                    "context_emotion_scores": score_dict,
                })
            except Exception:
                frame_persons.append({
                    "track_id": p["track_id"],
                    "context_emotions": None,
                    "context_emotion_scores": None,
                })

        sampled_frames_out.append({"persons": frame_persons})

    cap.release()

    all_frames = expand_sampled(sampled_frames_out, n_frames, video_fps, COCOER_FPS)
    return {
        "n_frames":   n_frames,
        "video_fps":  video_fps,
        "sample_fps": COCOER_FPS,
        "frames":     all_frames,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

CHECKPOINT_EVERY = 200   # save partial results every N clips


def run(subset):
    print("\n[cocoer] {}".format(subset))
    model      = load_model()
    detections = load_annotation("detections", subset)

    # Load any existing partial results so we can resume
    out_path = os.path.join(ROOT_DIR, "annotations",
                            "context_emotion_{}.json".format(subset))
    if os.path.exists(out_path):
        with open(out_path) as f:
            output = json.load(f)
        print("  Resuming from {} existing entries".format(len(output)))
    else:
        output = {}

    utts = [u for u in iter_utterances(subset)
            if u["utt_key"] not in output]
    print("  {} clips remaining".format(len(utts)))

    for i, utt in enumerate(tqdm(utts, desc="  {}".format(subset), unit="clip")):
        try:
            result = process_clip(model, utt, detections)
            if result:
                output[utt["utt_key"]] = result
        except Exception as e:
            print("  WARN {}: {}".format(utt["utt_key"], e))

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_annotation(output, "context_emotion", subset)

    save_annotation(output, "context_emotion", subset)


if __name__ == "__main__":
    for subset in SUBSETS:
        run(subset)
