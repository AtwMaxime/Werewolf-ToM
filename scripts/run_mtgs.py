"""
Step 2d — Social Gaze  (MTGS)

MTGS is a temporal model that estimates mutual gaze and shared attention
between all persons visible in a video clip.  It is run on a sliding window
of frames (not frame-by-frame) to exploit temporal context.

We bypass MTGS's built-in YOLOv5 head detector by injecting our YOLOv8-Face
bboxes directly.

Repo:  https://github.com/idiap/MTGS
Clone: models/gaze/MTGS/   (pip install -e .)

Output — annotations/social_gaze_{subset}.json:
{
  "youtube/GameKey/utt_0002": {
    "n_frames":  48,
    "video_fps": 16.0,
    "sample_fps": 4,
    "frames": [
      {
        "idx": 0,
        "sampled": true,
        "mutual_gaze_pairs": [
          {"track_id_a": 0, "track_id_b": 1, "score": 0.87}
        ],
        "shared_attention": false,
        "shared_attention_score": 0.22
      },
      …
    ]
  }
}
"""

import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "models", "gaze", "MTGS"))

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

MTGS_FPS        = 4
MTGS_MODEL_ID   = "mtgs-vsgaze"   # HuggingFace model ID
MUTUAL_THRESH   = 0.50
SHARED_THRESH   = 0.50
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_mtgs():
    from mtgs import MTGSPipeline   # noqa: installed from repo
    pipeline = MTGSPipeline.from_pretrained(MTGS_MODEL_ID)
    pipeline.model = pipeline.model.to(DEVICE).eval()
    return pipeline


# ──────────────────────────────────────────────
# Per-utterance processing
# ──────────────────────────────────────────────

def process_clip(pipeline, utt: dict, detections: dict) -> dict:
    det_entry = detections.get(utt["utt_key"])
    if det_entry is None:
        return {}

    n_frames   = det_entry["n_frames"]
    video_fps  = det_entry["video_fps"]
    det_frames = det_entry["frames"]

    # Collect sampled frames + corresponding head bboxes
    cap, _, _ = open_video(utt["clip_path"])
    frames_bgr  = []
    frame_infos = []   # (frame_idx, persons_list)

    for frame_idx, frame in iter_sampled_frames(cap, n_frames, video_fps, MTGS_FPS):
        frames_bgr.append(frame)
        det_frame = det_frames[frame_idx] if frame_idx < len(det_frames) else {}
        frame_infos.append((frame_idx, det_frame.get("persons", [])))

    cap.release()

    if not frames_bgr:
        return {}

    h, w = frames_bgr[0].shape[:2]

    # Build per-frame head bbox lists in MTGS format (normalised [x1,y1,x2,y2])
    all_head_bboxes = []
    for _, persons in frame_infos:
        heads = []
        for p in persons:
            fb = p.get("face_bbox")
            if fb:
                heads.append([fb[0]/w, fb[1]/h, fb[2]/w, fb[3]/h])
        all_head_bboxes.append(heads)

    sampled_frames_out = []

    try:
        with torch.no_grad():
            # MTGS processes the full temporal clip at once
            frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
            result = pipeline(frames_rgb, all_head_bboxes)
            # result is a list of per-frame dicts with keys:
            #   "mutual_gaze": NxN matrix, "shared_attention": float

        for fi, (frame_idx, persons) in enumerate(frame_infos):
            frame_res = result[fi] if fi < len(result) else {}

            mut_matrix = frame_res.get("mutual_gaze", [])
            shared_score = float(frame_res.get("shared_attention", 0.0))

            pairs = []
            for i, pi in enumerate(persons):
                for j, pj in enumerate(persons):
                    if j <= i:
                        continue
                    score = float(mut_matrix[i][j]) if i < len(mut_matrix) and j < len(mut_matrix[i]) else 0.0
                    pairs.append({
                        "track_id_a": pi["track_id"],
                        "track_id_b": pj["track_id"],
                        "score":      score,
                    })

            sampled_frames_out.append({
                "mutual_gaze_pairs":      pairs,
                "shared_attention":       shared_score >= SHARED_THRESH,
                "shared_attention_score": shared_score,
            })

    except Exception as e:
        # Fallback: fill with empty results
        for frame_idx, persons in frame_infos:
            sampled_frames_out.append({
                "mutual_gaze_pairs":      [],
                "shared_attention":       False,
                "shared_attention_score": None,
            })

    all_frames = expand_sampled(sampled_frames_out, n_frames, video_fps, MTGS_FPS)
    return {
        "n_frames":   n_frames,
        "video_fps":  video_fps,
        "sample_fps": MTGS_FPS,
        "frames":     all_frames,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run(subset: str):
    print(f"\n[mtgs] {subset}")
    pipeline   = load_mtgs()
    detections = load_annotation("detections", subset)

    output = {}
    utts = list(iter_utterances(subset))
    for utt in tqdm(utts, desc=f"  {subset}", unit="clip"):
        try:
            result = process_clip(pipeline, utt, detections)
            if result:
                output[utt["utt_key"]] = result
        except Exception as e:
            print(f"  WARN {utt['utt_key']}: {e}")

    save_annotation(output, "social_gaze", subset)


if __name__ == "__main__":
    for subset in SUBSETS:
        run(subset)
