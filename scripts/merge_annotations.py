"""
Step 3 — Merge all annotation layers into master.json

Combines every per-subset JSON into a single master file keyed by
utterance key.  The merged schema has:

  • Frame-level layers (list of per-frame dicts):
      detections, expression, gaze, social_gaze, context_emotion, proxemics

  • Utterance-level layers (flat dict):
      speech_emotion, vocalsound, diarization

The frame-level layers may have different sample_fps values.  merge_frames()
aligns them by frame index using the highest-resolution layer (detections)
as the index spine.

Output — annotations/master.json:
{
  "youtube/GameKey/utt_0002": {
    "speaker":    "Mitchell",
    "text":       "I think it's you",
    "n_frames":   48,
    "video_fps":  16.0,

    // per-utterance
    "speech_emotion":        "anger",
    "speech_emotion_scores": {…},
    "vocal_sound":           null,
    "vocal_sound_scores":    {…},
    "diarization_speaker":   "SPEAKER_01",
    "diarization_overlap":   false,

    // per-frame (list of n_frames dicts)
    "frames": [
      {
        "idx":     0,
        "sampled_detection":      true,
        "sampled_expression":     true,
        "sampled_gaze":           false,
        "sampled_social_gaze":    true,
        "sampled_context_emotion":false,
        "sampled_proxemics":      true,   // proxemics re-uses detection cadence

        "persons": [
          {
            "track_id":               0,
            "player_name":            "Mitchell",
            "is_speaker":             true,

            // detection
            "face_bbox":              [x1,y1,x2,y2],
            "face_conf":              0.95,
            "face_landmarks":         [[x,y],…],
            "pose_bbox":              [x1,y1,x2,y2],
            "keypoints":              [[x,y,conf],…],

            // expression
            "expression":             "Anger",
            "expression_scores":      {…},
            "valence":                -0.42,
            "arousal":                0.61,

            // gaze
            "gaze_point":             [0.52, 0.48],
            "gaze_conf":              0.83,
            "gaze_inout":             "in",
            "gaze_inout_score":       0.91,

            // context emotion
            "context_emotions":       ["Disapproval", "Doubt"],
            "context_emotion_scores": {…}
          }
        ],

        // social gaze (scene-level, not per-person)
        "mutual_gaze_pairs":       [{track_id_a, track_id_b, score}],
        "shared_attention":        false,
        "shared_attention_score":  0.22,

        // proxemics (pair-level)
        "proxemics_pairs": [
          {
            "track_id_a":        0,
            "track_id_b":        1,
            "wrist_wrist_min_px": 82.3,
            "shoulder_dist_px":  145.7,
            "torso_dist_px":     163.2
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
import sys

from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    ANNOTATIONS_DIR,
    SUBSETS,
    iter_utterances,
    load_annotation,
)

# ──────────────────────────────────────────────
# Frame-level merge helpers
# ──────────────────────────────────────────────

def index_by_frame(frames: list) -> dict[int, dict]:
    return {f["idx"]: f for f in frames}


def merge_person_layers(
    det_persons: list,
    expr_persons: list | None,
    gaze_persons: list | None,
    ctx_persons: list | None,
) -> list[dict]:
    """Merge per-person dicts from different layers by track_id."""
    # Index each layer by track_id
    def by_tid(persons):
        return {p["track_id"]: p for p in (persons or [])}

    expr_idx = by_tid(expr_persons)
    gaze_idx = by_tid(gaze_persons)
    ctx_idx  = by_tid(ctx_persons)

    merged = []
    for p in det_persons:
        tid = p["track_id"]
        ep  = expr_idx.get(tid, {})
        gp  = gaze_idx.get(tid, {})
        cp  = ctx_idx.get(tid, {})

        merged.append({
            # detection
            "track_id":               tid,
            "player_name":            p.get("player_name"),
            "is_speaker":             p.get("is_speaker", False),
            "face_bbox":              p.get("face_bbox"),
            "face_conf":              p.get("face_conf"),
            "face_landmarks":         p.get("face_landmarks"),
            "pose_bbox":              p.get("pose_bbox"),
            "keypoints":              p.get("keypoints"),
            # expression
            "expression":             ep.get("expression"),
            "expression_scores":      ep.get("expression_scores"),
            "valence":                ep.get("valence"),
            "arousal":                ep.get("arousal"),
            # gaze
            "gaze_point":             gp.get("gaze_point"),
            "gaze_conf":              gp.get("gaze_conf"),
            "gaze_inout":             gp.get("gaze_inout"),
            "gaze_inout_score":       gp.get("gaze_inout_score"),
            # context emotion
            "context_emotions":       cp.get("context_emotions"),
            "context_emotion_scores": cp.get("context_emotion_scores"),
        })

    return merged


def merge_frames(
    det_entry: dict,
    expr_entry: dict | None,
    gaze_entry: dict | None,
    sg_entry: dict | None,
    ctx_entry: dict | None,
    prox_entry: dict | None,
) -> list[dict]:
    n_frames = det_entry["n_frames"]
    det_idx  = index_by_frame(det_entry.get("frames", []))
    expr_idx = index_by_frame(expr_entry.get("frames", [])) if expr_entry else {}
    gaze_idx = index_by_frame(gaze_entry.get("frames", [])) if gaze_entry else {}
    sg_idx   = index_by_frame(sg_entry.get("frames", []))   if sg_entry   else {}
    ctx_idx  = index_by_frame(ctx_entry.get("frames", []))  if ctx_entry  else {}
    prox_idx = index_by_frame(prox_entry.get("frames", [])) if prox_entry else {}

    frames = []
    for i in range(n_frames):
        df   = det_idx.get(i, {})
        ef   = expr_idx.get(i, {})
        gf   = gaze_idx.get(i, {})
        sgf  = sg_idx.get(i, {})
        cf   = ctx_idx.get(i, {})
        pf   = prox_idx.get(i, {})

        persons = merge_person_layers(
            df.get("persons", []),
            ef.get("persons"),
            gf.get("persons"),
            cf.get("persons"),
        )

        frame = {
            "idx":                       i,
            "sampled_detection":         df.get("sampled", False),
            "sampled_expression":        ef.get("sampled", False),
            "sampled_gaze":              gf.get("sampled", False),
            "sampled_social_gaze":       sgf.get("sampled", False),
            "sampled_context_emotion":   cf.get("sampled", False),
            "persons":                   persons,
            "mutual_gaze_pairs":         sgf.get("mutual_gaze_pairs", []),
            "shared_attention":          sgf.get("shared_attention", False),
            "shared_attention_score":    sgf.get("shared_attention_score"),
            "proxemics_pairs":           pf.get("pairs", []),
        }
        frames.append(frame)

    return frames


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run():
    master = {}

    for subset in SUBSETS:
        print(f"\n[merge] loading {subset} annotations…")

        def _load(name):
            try:
                return load_annotation(name, subset)
            except FileNotFoundError:
                print(f"  MISSING: {name}_{subset}.json — skipping layer")
                return {}

        detections     = _load("detections")
        expressions    = _load("expression")
        gazes          = _load("gaze")
        social_gazes   = _load("social_gaze")
        ctx_emotions   = _load("context_emotion")
        proxemics      = _load("proxemics")
        speech_emotion = _load("speech_emotion")
        vocalsound     = _load("vocalsound")
        diarization    = _load("diarization")

        utts = list(iter_utterances(subset))
        for utt in tqdm(utts, desc=f"  merging {subset}", unit="utt"):
            key = utt["utt_key"]

            det_entry = detections.get(key)
            if det_entry is None:
                continue

            frames = merge_frames(
                det_entry,
                expressions.get(key),
                gazes.get(key),
                social_gazes.get(key),
                ctx_emotions.get(key),
                proxemics.get(key),
            )

            se  = speech_emotion.get(key, {})
            vs  = vocalsound.get(key, {})
            dia = diarization.get(key, {})

            master[key] = {
                "speaker":   utt.get("speaker"),
                "text":      utt.get("text", ""),
                "n_frames":  det_entry["n_frames"],
                "video_fps": det_entry["video_fps"],
                # utterance-level
                "speech_emotion":        se.get("speech_emotion"),
                "speech_emotion_scores": se.get("speech_emotion_scores"),
                "vocal_sound":           vs.get("vocal_sound"),
                "vocal_sound_scores":    vs.get("vocal_sound_scores"),
                "diarization_speaker":   dia.get("diarization_speaker"),
                "diarization_overlap":   dia.get("diarization_overlap"),
                # frame-level
                "frames": frames,
            }

    out_path = os.path.join(ANNOTATIONS_DIR, "master.json")
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(master, f)

    print(f"\n  Saved → {out_path}  ({len(master)} utterances)")


if __name__ == "__main__":
    run()
