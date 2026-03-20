"""
Shared utilities for WOLF-ToM annotation scripts.
"""

import json
import os
from typing import Generator, Iterator

import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset", "WerewolfAmongUs")
ANNOTATIONS_DIR = os.path.join(ROOT_DIR, "annotations")

SUBSETS = ["youtube", "ego4d"]
SPLITS = ["train", "val", "test"]

# ──────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────

def clip_path(subset: str, game_key: str, rec_id: int) -> str:
    return os.path.join(
        DATASET_DIR, "clips", subset, game_key, f"utt_{rec_id:04d}.mp4"
    )


def annotation_path(name: str, subset: str) -> str:
    return os.path.join(ANNOTATIONS_DIR, f"{name}_{subset}.json")


# ──────────────────────────────────────────────
# JSON I/O
# ──────────────────────────────────────────────

def load_annotation(name: str, subset: str) -> dict:
    path = annotation_path(name, subset)
    with open(path) as f:
        return json.load(f)


def save_annotation(data: dict, name: str, subset: str) -> None:
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    path = annotation_path(name, subset)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  Saved → {path}  ({len(data)} utterances)")


# ──────────────────────────────────────────────
# Dataset iteration
# ──────────────────────────────────────────────

def iter_utterances(subset: str) -> Iterator[dict]:
    """
    Yield one dict per utterance across all splits:
      {
        "utt_key":   "youtube/GameKey/utt_0002",
        "game_key":  "GameKey",
        "rec_id":    2,
        "speaker":   "Mitchell",          # None for Ego4D
        "clip_path": "/path/to/utt.mp4",
        "text":      "I think it's you",  # from annotation
      }
    Skips utterances whose clip file doesn't exist.
    """
    if subset == "youtube":
        split_root = os.path.join(DATASET_DIR, "Youtube", "split")
        key_fields = ("YT_ID", "Game_ID")
    else:
        split_root = os.path.join(DATASET_DIR, "Ego4D", "split")
        key_fields = ("EG_ID", "Game_ID")

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

            for utt in game["Dialogue"]:
                rec_id = utt["Rec_Id"]
                path = clip_path(subset, game_key, rec_id)
                if not os.path.exists(path):
                    continue
                yield {
                    "utt_key": f"{subset}/{game_key}/utt_{rec_id:04d}",
                    "game_key": game_key,
                    "rec_id": rec_id,
                    "speaker": utt.get("Identity"),
                    "clip_path": path,
                    "text": utt.get("Utterance", ""),
                }


# ──────────────────────────────────────────────
# Frame sampling
# ──────────────────────────────────────────────

def open_video(path: str):
    """Return (cap, n_frames, video_fps). Caller must cap.release()."""
    cap = cv2.VideoCapture(path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    return cap, n_frames, fps


def iter_sampled_frames(
    cap: cv2.VideoCapture,
    n_frames: int,
    video_fps: float,
    sample_fps: float,
) -> Generator[tuple[int, object], None, None]:
    """
    Yield (frame_idx, bgr_frame) for every sampled frame.
    sample_fps ≤ video_fps.  Frames are read in order (no random seek).
    """
    step = max(1, round(video_fps / sample_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    while frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            yield frame_idx, frame
        frame_idx += 1


def expand_sampled(
    sampled: list,       # list of per-sampled-frame dicts (no idx key needed)
    n_frames: int,
    video_fps: float,
    sample_fps: float,
) -> list[dict]:
    """
    Given results only for sampled frames, produce a list of length n_frames
    where non-sampled frames carry the data from the nearest preceding sample.
    Each output dict gains 'idx' and 'sampled' fields.
    """
    step = max(1, round(video_fps / sample_fps))
    out = []
    sample_cursor = 0
    last_data: dict = {}

    for i in range(n_frames):
        if i % step == 0 and sample_cursor < len(sampled):
            last_data = sampled[sample_cursor]
            sample_cursor += 1
            is_sampled = True
        else:
            is_sampled = False

        row = {"idx": i, "sampled": is_sampled}
        row.update(last_data)
        out.append(row)

    return out


# ──────────────────────────────────────────────
# Bounding-box helpers
# ──────────────────────────────────────────────

def bbox_iou(a: list, b: list) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def bbox_area(b: list) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def crop_bbox(img, bbox: list, pad: float = 0.0):
    """Crop and return the region defined by [x1,y1,x2,y2] with optional padding."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if pad > 0:
        pw = int((x2 - x1) * pad)
        ph = int((y2 - y1) * pad)
        x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
        x2, y2 = min(w, x2 + pw), min(h, y2 + ph)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]
