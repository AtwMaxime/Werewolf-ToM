"""
Step 2b — Proxemics  (YOLOv8-pose keypoints, no extra model)

Uses the body keypoints already stored in detections_*.json to compute
pairwise physical-proximity signals between every pair of tracked persons
on each frame.

Distances are computed in pixel space between wrist/hand, shoulder, and
torso keypoints.  The frame resolution is stored so downstream code can
normalise if needed.

COCO keypoint indices used:
  5  left_shoulder    6  right_shoulder
  7  left_elbow       8  right_elbow
  9  left_wrist      10  right_wrist
  11 left_hip        12 right_hip

Output — annotations/proxemics_{subset}.json:
{
  "youtube/GameKey/utt_0002": {
    "n_frames":  48,
    "video_fps": 16.0,
    "frame_w": 640,
    "frame_h": 360,
    "frames": [
      {
        "idx": 0,
        "sampled": true,         // inherited from detection
        "pairs": [
          {
            "track_id_a": 0,
            "track_id_b": 1,
            "wrist_wrist_min_px":  82.3,   // min across all wrist combinations
            "shoulder_dist_px":   145.7,   // centroid shoulder distance
            "torso_dist_px":      163.2    // centroid hip distance
          }
        ]
      },
      …
    ]
  }
}
"""

import math
import os
import sys

import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    SUBSETS,
    iter_utterances,
    load_annotation,
    open_video,
    save_annotation,
)

# ──────────────────────────────────────────────
# COCO keypoint indices
# ──────────────────────────────────────────────

L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST,    R_WRIST    = 9, 10
L_HIP,      R_HIP      = 11, 12

KPT_CONF_THRESH = 0.30   # ignore low-confidence keypoints


def kpt(keypoints: list, idx: int):
    """Return (x, y) or None if the keypoint is absent/low-confidence."""
    if idx >= len(keypoints):
        return None
    k = keypoints[idx]
    if len(k) >= 3 and k[2] < KPT_CONF_THRESH:
        return None
    return (k[0], k[1])


def dist(a, b) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def centroid(p1, p2):
    if p1 is None and p2 is None:
        return None
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


# ──────────────────────────────────────────────
# Pair computation
# ──────────────────────────────────────────────

def compute_pair(pa: dict, pb: dict) -> dict:
    kp_a = pa.get("keypoints", [])
    kp_b = pb.get("keypoints", [])

    # Wrist-to-wrist: min of all 4 cross-combinations
    wrist_dists = []
    for ai in (L_WRIST, R_WRIST):
        for bi in (L_WRIST, R_WRIST):
            d = dist(kpt(kp_a, ai), kpt(kp_b, bi))
            if d is not None:
                wrist_dists.append(d)
    wrist_min = min(wrist_dists) if wrist_dists else None

    shoulder_dist = dist(
        centroid(kpt(kp_a, L_SHOULDER), kpt(kp_a, R_SHOULDER)),
        centroid(kpt(kp_b, L_SHOULDER), kpt(kp_b, R_SHOULDER)),
    )

    torso_dist = dist(
        centroid(kpt(kp_a, L_HIP), kpt(kp_a, R_HIP)),
        centroid(kpt(kp_b, L_HIP), kpt(kp_b, R_HIP)),
    )

    return {
        "track_id_a":         pa["track_id"],
        "track_id_b":         pb["track_id"],
        "wrist_wrist_min_px": wrist_min,
        "shoulder_dist_px":   shoulder_dist,
        "torso_dist_px":      torso_dist,
    }


# ──────────────────────────────────────────────
# Per-utterance processing
# ──────────────────────────────────────────────

def process_clip(utt: dict, detections: dict) -> dict:
    det_entry = detections.get(utt["utt_key"])
    if det_entry is None:
        return {}

    n_frames   = det_entry["n_frames"]
    video_fps  = det_entry["video_fps"]
    det_frames = det_entry["frames"]

    # Get frame dimensions from the clip
    cap = cv2.VideoCapture(utt["clip_path"])
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    all_frames = []
    for det_frame in det_frames:
        persons = det_frame.get("persons", [])
        pairs = []
        for i, pa in enumerate(persons):
            for pb in persons[i + 1:]:
                pairs.append(compute_pair(pa, pb))

        all_frames.append({
            "idx":     det_frame["idx"],
            "sampled": det_frame["sampled"],
            "pairs":   pairs,
        })

    return {
        "n_frames":  n_frames,
        "video_fps": video_fps,
        "frame_w":   frame_w,
        "frame_h":   frame_h,
        "frames":    all_frames,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run(subset: str):
    print(f"\n[proxemics] {subset}")
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

    save_annotation(output, "proxemics", subset)


if __name__ == "__main__":
    for subset in SUBSETS:
        run(subset)
