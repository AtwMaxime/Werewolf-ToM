"""
Step 2a — Facial Expression + Valence/Arousal  (HSEmotion)

Runs HSEmotion on each tracked face.  Sampled at EXPR_FPS (same cadence as
detection so no extra video decode is needed) and held for in-between frames.

Output — annotations/expression_{subset}.json:
{
  "youtube/GameKey/utt_0002": {
    "n_frames":  48,
    "video_fps": 16.0,
    "sample_fps": 4,
    "frames": [
      {
        "idx": 0,
        "sampled": true,
        "persons": [
          {
            "track_id": 0,
            "expression": "Anger",          // one of 8 AffectNet classes
            "expression_scores": {…},       // class → probability
            "valence": -0.42,
            "arousal":  0.61
          }
        ]
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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
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

# pip install hsemotion
from hsemotion.facial_emotions import HSEmotionRecognizer

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

EXPR_FPS    = 4    # must be ≤ DETECT_FPS so every sampled frame has detections
MODEL_NAME  = "enet_b0_8_best_afew"   # lightest accurate variant

EXPRESSION_CLASSES = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happiness", "Neutral", "Sadness", "Surprise",
]

# ──────────────────────────────────────────────
# Per-utterance processing
# ──────────────────────────────────────────────

def process_clip(fer, utt: dict, detections: dict) -> dict:
    det_entry = detections.get(utt["utt_key"])
    if det_entry is None:
        return {}

    n_frames  = det_entry["n_frames"]
    video_fps = det_entry["video_fps"]
    det_frames = det_entry["frames"]   # list indexed by frame idx

    cap, _, _ = open_video(utt["clip_path"])
    sampled_frames = []

    for frame_idx, frame in iter_sampled_frames(cap, n_frames, video_fps, EXPR_FPS):
        det_frame = det_frames[frame_idx] if frame_idx < len(det_frames) else {}
        persons_out = []

        for person in det_frame.get("persons", []):
            face_bbox = person.get("face_bbox")
            if face_bbox is None:
                persons_out.append({
                    "track_id": person["track_id"],
                    "expression": None,
                    "expression_scores": None,
                    "valence": None,
                    "arousal": None,
                })
                continue

            face_img = crop_bbox(frame, face_bbox, pad=0.1)
            if face_img is None or face_img.size == 0:
                persons_out.append({
                    "track_id": person["track_id"],
                    "expression": None,
                    "expression_scores": None,
                    "valence": None,
                    "arousal": None,
                })
                continue

            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            emotion_label, scores = fer.predict_emotions(face_rgb, logits=False)
            valence, arousal = fer.predict_valence_arousal(face_rgb)

            # scores is a 1-D numpy array ordered by EXPRESSION_CLASSES
            score_dict = {
                cls: float(scores[i])
                for i, cls in enumerate(EXPRESSION_CLASSES)
                if i < len(scores)
            }

            persons_out.append({
                "track_id":          person["track_id"],
                "expression":        emotion_label,
                "expression_scores": score_dict,
                "valence":           float(valence),
                "arousal":           float(arousal),
            })

        sampled_frames.append({"persons": persons_out})

    cap.release()

    all_frames = expand_sampled(sampled_frames, n_frames, video_fps, EXPR_FPS)
    return {
        "n_frames":   n_frames,
        "video_fps":  video_fps,
        "sample_fps": EXPR_FPS,
        "frames":     all_frames,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run(subset: str):
    print(f"\n[hsemotion] {subset}")
    fer        = HSEmotionRecognizer(model_name=MODEL_NAME)
    detections = load_annotation("detections", subset)

    output = {}
    utts = list(iter_utterances(subset))
    for utt in tqdm(utts, desc=f"  {subset}", unit="clip"):
        try:
            result = process_clip(fer, utt, detections)
            if result:
                output[utt["utt_key"]] = result
        except Exception as e:
            print(f"  WARN {utt['utt_key']}: {e}")

    save_annotation(output, "expression", subset)


if __name__ == "__main__":
    for subset in SUBSETS:
        run(subset)
