"""
Step 2c — Gaze Target Estimation  (Gazelle)

Runs Gazelle (GazeFollow + VideoAttentionTarget variants) on each tracked
face to estimate where that person is looking within the frame.

Sampled at GAZE_FPS; held for in-between frames.

Repo: https://github.com/fkryan/gazelle
Clone to: models/gaze/gazelle/

Output — annotations/gaze_{subset}.json:
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
            "track_id":        0,
            "gaze_point":      [0.52, 0.48],   // normalised (x,y) ∈ [0,1]
            "gaze_heatmap":    null,            // omitted by default (large)
            "gaze_conf":       0.83,
            "gaze_inout":      "in",            // "in" | "out" | null
            "gaze_inout_score": 0.91
          }
        ]
      },
      …
    ]
  }
}
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "models", "gaze", "gazelle"))

from utils import (
    SUBSETS,
    crop_bbox,
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

GAZE_FPS    = 2
STORE_HEATMAP = False   # heatmaps are large; set True to keep them
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_gazelle():
    """Load Gazelle vitl14+inout via hubconf (auto-downloads weights)."""
    model, transform = torch.hub.load(
        os.path.join(ROOT_DIR, "models", "gaze", "gazelle"),
        "gazelle_dinov2_vitl14_inout",
        pretrained=True,
        source="local",
        trust_repo=True,
    )
    model = model.to(DEVICE).eval()
    return model, transform


# ──────────────────────────────────────────────
# Per-frame inference
# ──────────────────────────────────────────────

def run_gazelle_frame(model, transform, frame_bgr, persons: list) -> list[dict]:
    """
    Run Gazelle on one frame for all tracked persons (batched per frame).
    Returns a list of per-person gaze dicts.

    Correct API (gazelle_dinov2_vitl14):
      input  = {"images": (1,3,H,W), "bboxes": [[(x1,y1,x2,y2), ...]]}
      output = {"heatmap": [tensor(1,H,W) per person], "inout": tensor or None}
    """
    from PIL import Image as _PILImage
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    pil_img = _PILImage.fromarray(frame_rgb)

    # Split into persons with and without face bbox
    valid_idx   = []   # indices into persons list
    valid_bboxes = []  # normalised (x1,y1,x2,y2) tuples

    for i, person in enumerate(persons):
        fb = person.get("face_bbox")
        if fb is not None:
            x1, y1, x2, y2 = fb
            valid_idx.append(i)
            valid_bboxes.append((x1 / w, y1 / h, x2 / w, y2 / h))

    # Default null result for every person
    results = [{
        "track_id":         p["track_id"],
        "gaze_point":       None,
        "gaze_conf":        None,
        "gaze_inout":       None,
        "gaze_inout_score": None,
    } for p in persons]

    if not valid_bboxes:
        return results

    try:
        img_t = transform(pil_img).unsqueeze(0).to(DEVICE)
        inp   = {"images": img_t, "bboxes": [valid_bboxes]}
        with torch.no_grad():
            out = model(inp)

        # out["heatmap"]: list of length 1 (per image); element is (N_persons, H, W)
        heatmaps = out["heatmap"][0].cpu().numpy()  # (N_persons, H, W)
        inouts   = out.get("inout") # tensor or None

        for j, i in enumerate(valid_idx):
            heatmap  = heatmaps[j]  # (H, W)
            flat_idx = int(np.argmax(heatmap))
            gy = flat_idx // heatmap.shape[1]
            gx = flat_idx  % heatmap.shape[1]
            gaze_point = [gx / heatmap.shape[1], gy / heatmap.shape[0]]
            gaze_conf  = float(heatmap.max())

            if inouts is not None:
                inout_score = float(torch.sigmoid(inouts[0][j]).cpu())
                inout_label = "in" if inout_score >= 0.5 else "out"
            else:
                inout_score = None
                inout_label = None

            entry = {
                "track_id":         persons[i]["track_id"],
                "gaze_point":       gaze_point,
                "gaze_conf":        gaze_conf,
                "gaze_inout":       inout_label,
                "gaze_inout_score": inout_score,
            }
            if STORE_HEATMAP:
                entry["gaze_heatmap"] = heatmap.tolist()
            results[i] = entry

    except Exception as e:
        print(f"    gaze frame error: {e}", flush=True)

    return results


# ──────────────────────────────────────────────
# Per-utterance processing
# ──────────────────────────────────────────────

def process_clip(model, transform, utt: dict, detections: dict) -> dict:
    det_entry = detections.get(utt["utt_key"])
    if det_entry is None:
        return {}

    n_frames   = det_entry["n_frames"]
    video_fps  = det_entry["video_fps"]
    det_frames = det_entry["frames"]

    cap, _, _ = open_video(utt["clip_path"])
    sampled_frames = []

    for frame_idx, frame in iter_sampled_frames(cap, n_frames, video_fps, GAZE_FPS):
        det_frame = det_frames[frame_idx] if frame_idx < len(det_frames) else {}
        persons   = det_frame.get("persons", [])
        gaze_results = run_gazelle_frame(model, transform, frame, persons)
        sampled_frames.append({"persons": gaze_results})

    cap.release()

    all_frames = expand_sampled(sampled_frames, n_frames, video_fps, GAZE_FPS)
    return {
        "n_frames":   n_frames,
        "video_fps":  video_fps,
        "sample_fps": GAZE_FPS,
        "frames":     all_frames,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run(subset: str):
    print(f"\n[gazelle] {subset}")
    model, transform = load_gazelle()
    detections = load_annotation("detections", subset)

    output = {}
    utts = list(iter_utterances(subset))
    for utt in tqdm(utts, desc=f"  {subset}", unit="clip"):
        try:
            result = process_clip(model, transform, utt, detections)
            if result:
                output[utt["utt_key"]] = result
        except Exception as e:
            print(f"  WARN {utt['utt_key']}: {e}")

    save_annotation(output, "gaze", subset)


if __name__ == "__main__":
    for subset in SUBSETS:
        run(subset)
