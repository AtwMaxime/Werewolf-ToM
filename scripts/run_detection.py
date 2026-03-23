"""
Step 1 — Face + Person Detection

For every utterance clip, runs:
  • YOLOv8-Face  → face bboxes + 5-point landmarks
  • YOLOv8-pose  → person bboxes + 17-point body keypoints

Sampled at DETECT_FPS; all other frames carry the last sampled result
(faces move slowly within a 15-second clip).

Within each clip a simple IoU-based tracker assigns persistent track_ids.
For YouTube clips the annotated speaker ("Identity") is mapped to whichever
tracked person has the largest face in the first sampled frame (heuristic
for talking-head game recordings; can be refined with ReID later).

Output — annotations/detections_{subset}.json:
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
            "track_id":       0,
            "player_name":    "Mitchell",   // null for Ego4D
            "is_speaker":     true,
            "face_bbox":      [x1,y1,x2,y2],  // null if no face matched
            "face_conf":      0.95,
            "face_landmarks": [[x,y],…],       // 5 pts, null if absent
            "pose_bbox":      [x1,y1,x2,y2],
            "pose_conf":      0.88,
            "keypoints":      [[x,y,conf],…]   // 17 pts
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
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    ROOT_DIR,
    SUBSETS,
    bbox_iou,
    bbox_area,
    expand_sampled,
    iter_utterances,
    load_annotation,
    open_video,
    iter_sampled_frames,
    save_annotation,
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

DETECT_FPS = 4          # inference rate; in-between frames are held
FACE_CONF  = 0.40       # minimum face detection confidence
POSE_CONF  = 0.40       # minimum person detection confidence
IOU_TRACK  = 0.30       # min IoU to link a detection to an existing track
LOST_TTL   = 2          # sampled frames a track survives without a match

FACE_WEIGHTS = os.path.join(
    ROOT_DIR, "models", "detection", "YOLOv8-Face", "weights", "yolov8m-face.pt"
)

# ──────────────────────────────────────────────
# IoU tracker
# ──────────────────────────────────────────────

class IoUTracker:
    """Greedy IoU-based tracker for bboxes across sampled frames."""

    def __init__(self, iou_thresh: float = IOU_TRACK, lost_ttl: int = LOST_TTL):
        self.iou_thresh = iou_thresh
        self.lost_ttl = lost_ttl
        self.tracks: dict[int, dict] = {}   # track_id → {bbox, ttl, meta}
        self._next_id = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """
        detections: list of dicts with 'bbox' key ([x1,y1,x2,y2]).
        Returns detections enriched with 'track_id'.
        """
        # Decrement TTL on existing tracks
        for t in self.tracks.values():
            t["ttl"] -= 1

        matched_tracks = set()
        results = []

        for det in detections:
            best_id, best_iou = None, 0.0
            for tid, track in self.tracks.items():
                if tid in matched_tracks:
                    continue
                if track["ttl"] < 0:
                    continue
                iou = bbox_iou(det["bbox"], track["bbox"])
                if iou > best_iou:
                    best_iou, best_id = iou, tid

            if best_id is not None and best_iou >= self.iou_thresh:
                track = self.tracks[best_id]
                track["bbox"] = det["bbox"]
                track["ttl"]  = self.lost_ttl
                matched_tracks.add(best_id)
                det = {**det, "track_id": best_id}
            else:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = {"bbox": det["bbox"], "ttl": self.lost_ttl}
                matched_tracks.add(tid)
                det = {**det, "track_id": tid}

            results.append(det)

        # Prune dead tracks
        self.tracks = {k: v for k, v in self.tracks.items() if v["ttl"] >= 0}
        return results


# ──────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────

def detect_faces(face_model, frame) -> list[dict]:
    results = face_model(frame, verbose=False)[0]
    detections = []
    for i, box in enumerate(results.boxes):
        conf = float(box.conf[0])
        if conf < FACE_CONF:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        landmarks = None
        if results.keypoints is not None and i < len(results.keypoints.xy):
            landmarks = results.keypoints.xy[i].tolist()  # [[x,y]×5]
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "conf": conf,
            "landmarks": landmarks,
        })
    return detections


def detect_poses(pose_model, frame) -> list[dict]:
    results = pose_model(frame, verbose=False)[0]
    detections = []
    for i, box in enumerate(results.boxes):
        conf = float(box.conf[0])
        if conf < POSE_CONF:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        kpts = []
        if results.keypoints is not None and i < len(results.keypoints.data):
            kpts = results.keypoints.data[i].tolist()   # [[x,y,conf]×17]
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "conf": conf,
            "keypoints": kpts,
        })
    return detections


def associate_faces_to_poses(faces: list, poses: list) -> list[dict]:
    """
    For each pose, find the face with highest IoU. Merge into a unified person dict.
    Faces with no matching pose are attached as standalone persons.
    """
    used_faces = set()
    persons = []

    for pose in poses:
        best_fi, best_iou = None, 0.0
        for fi, face in enumerate(faces):
            if fi in used_faces:
                continue
            iou = bbox_iou(face["bbox"], pose["bbox"])
            if iou > best_iou:
                best_iou, best_fi = iou, fi

        face = faces[best_fi] if best_fi is not None and best_iou > 0.10 else None
        if best_fi is not None and best_iou > 0.10:
            used_faces.add(best_fi)

        persons.append({
            "bbox": pose["bbox"],           # use pose bbox as canonical
            "face_bbox": face["bbox"] if face else None,
            "face_conf": face["conf"] if face else None,
            "face_landmarks": face["landmarks"] if face else None,
            "pose_bbox": pose["bbox"],
            "pose_conf": pose["conf"],
            "keypoints": pose["keypoints"],
        })

    # Leftover faces (no matching pose body)
    for fi, face in enumerate(faces):
        if fi in used_faces:
            continue
        persons.append({
            "bbox": face["bbox"],
            "face_bbox": face["bbox"],
            "face_conf": face["conf"],
            "face_landmarks": face["landmarks"],
            "pose_bbox": None,
            "pose_conf": None,
            "keypoints": [],
        })

    return persons


def format_person(raw: dict, track_id: int, speaker_track: int | None, speaker: str | None) -> dict:
    return {
        "track_id":       track_id,
        "player_name":    speaker if track_id == speaker_track else None,
        "is_speaker":     track_id == speaker_track,
        "face_bbox":      raw["face_bbox"],
        "face_conf":      raw["face_conf"],
        "face_landmarks": raw["face_landmarks"],
        "pose_bbox":      raw["pose_bbox"],
        "pose_conf":      raw["pose_conf"],
        "keypoints":      raw["keypoints"],
    }


# ──────────────────────────────────────────────
# Per-utterance processing
# ──────────────────────────────────────────────

def process_clip(face_model, pose_model, utt: dict) -> dict:
    cap, n_frames, video_fps = open_video(utt["clip_path"])
    tracker = IoUTracker()
    sampled_frames = []
    speaker_track = None   # track_id of the known speaker (YouTube only)

    for frame_idx, frame in iter_sampled_frames(cap, n_frames, video_fps, DETECT_FPS):
        faces  = detect_faces(face_model, frame)
        poses  = detect_poses(pose_model, frame)
        merged = associate_faces_to_poses(faces, poses)

        # Track using bbox field
        tracked = tracker.update([{"bbox": p["bbox"], **p} for p in merged])

        # First sampled frame: pick speaker as the largest-face track
        if frame_idx == 0 and utt.get("speaker") and tracked:
            best = max(
                [p for p in tracked if p["face_bbox"] is not None],
                key=lambda p: bbox_area(p["face_bbox"]),
                default=tracked[0],
            )
            speaker_track = best["track_id"]

        persons = [
            format_person(p, p["track_id"], speaker_track, utt.get("speaker"))
            for p in tracked
        ]
        sampled_frames.append({"persons": persons})

    cap.release()

    all_frames = expand_sampled(sampled_frames, n_frames, video_fps, DETECT_FPS)
    return {
        "n_frames":  n_frames,
        "video_fps": video_fps,
        "sample_fps": DETECT_FPS,
        "frames": all_frames,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run(subset: str):
    print(f"\n[detection] {subset}")
    face_model = YOLO(FACE_WEIGHTS)
    pose_model = YOLO("yolov8m-pose.pt")    # auto-downloads on first use

    output = {}
    utts = list(iter_utterances(subset))
    for utt in tqdm(utts, desc=f"  {subset}", unit="clip"):
        try:
            output[utt["utt_key"]] = process_clip(face_model, pose_model, utt)
        except Exception as e:
            print(f"  WARN {utt['utt_key']}: {e}")

    save_annotation(output, "detections", subset)


if __name__ == "__main__":
    for subset in SUBSETS:
        run(subset)
