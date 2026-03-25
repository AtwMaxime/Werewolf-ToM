"""
WOLF-ToM Annotation Visualizer
================================
Run with:
  streamlit run scripts/visualize.py

Access remotely via SSH tunnel:
  ssh -L 8501:localhost:8501 auriga
  Then open http://localhost:8501 in your browser.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure conda base site-packages are on the path (needed when streamlit
# forks a subprocess that may not inherit the full conda environment).
_site = "/local_scratch/mattwood/miniconda3/lib/python3.13/site-packages"
if _site not in sys.path:
    sys.path.insert(0, _site)

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = ROOT_DIR / "data"
ANNOTATIONS_DIR = ROOT_DIR / "annotations"
CLIPS_DIR      = DATA_DIR / "clips"

ANNOTATION_NAMES = [
    "detections", "expression", "gaze", "proxemics",
    "social_gaze", "speech_emotion", "vocalsound",
    "diarization", "context_emotion",
]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading split data…")
def load_splits(subset: str) -> dict:
    """Returns {game_key: game_dict} for all splits."""
    if subset == "youtube":
        split_root = DATA_DIR / "Youtube" / "split"
        id_fields  = ("YT_ID", "Game_ID")
    else:
        split_root = DATA_DIR / "Ego4D" / "split"
        id_fields  = ("EG_ID", "Game_ID")

    games = {}
    for split in ("train", "val", "test"):
        path = split_root / f"{split}.json"
        if not path.exists():
            continue
        for game in json.loads(path.read_text()):
            key = f"{game[id_fields[0]]}_{game[id_fields[1]]}"
            games[key] = {**game, "_split": split}
    return games


INDEX_DIR = ANNOTATIONS_DIR / "indexed"


@st.cache_data(show_spinner=False)
def load_utt_annotation(name: str, subset: str, utt_key: str) -> dict:
    """Load a single utterance's annotation from the indexed per-utterance files."""
    fname = utt_key.replace("/", "__") + ".json"
    path  = INDEX_DIR / f"{name}_{subset}" / fname
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_annotation(name: str, subset: str) -> dict:
    """Thin shim so the rest of the code stays the same."""
    return _LazyAnnotation(name, subset)


class _LazyAnnotation(dict):
    """Dict-like that loads individual utterances on demand from indexed files."""
    def __init__(self, name: str, subset: str):
        self._name   = name
        self._subset = subset

    def get(self, utt_key, default=None):
        result = load_utt_annotation(self._name, self._subset, utt_key)
        return result if result else default


# ─────────────────────────────────────────────────────────────────────────────
# Utterance helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_utterances(subset: str, game_key: str, games: dict) -> list[dict]:
    """Return all utterances for a game that have a clip file."""
    game = games.get(game_key, {})
    dialogue = game.get("Dialogue", [])
    utts = []
    for utt in dialogue:
        rec_id = utt.get("Rec_Id") or utt.get("rec_id")
        if rec_id is None:
            continue
        clip = CLIPS_DIR / subset / game_key / f"utt_{rec_id:04d}.mp4"
        if clip.exists():
            utts.append({**utt, "_rec_id": rec_id, "_clip": str(clip),
                         "_utt_key": f"{subset}/{game_key}/utt_{rec_id:04d}"})
    return utts


# ─────────────────────────────────────────────────────────────────────────────
# Frame rendering
# ─────────────────────────────────────────────────────────────────────────────

ROLE_COLORS = {
    "Werewolf":      (220,  50,  50),
    "Minion":        (200,  80,  80),
    "Seer":          ( 50, 150, 220),
    "Robber":        ( 50, 200, 150),
    "Troublemaker":  (200, 150,  50),
    "Tanner":        (180,  80, 200),
    "Drunk":         (200, 200,  50),
    "Hunter":        ( 80, 200, 100),
    "Mason":         ( 80, 160, 200),
    "Insomniac":     (160, 200,  80),
    "Villager":      (180, 180, 180),
}

TRACK_PALETTE = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (255, 180,  60), (180,  60, 255), ( 60, 255, 180),
]

POSE_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
]

def track_color(track_id: int):
    return TRACK_PALETTE[track_id % len(TRACK_PALETTE)]


def draw_annotations(frame_bgr, frame_data: dict, annotations: dict,
                     utt_key: str, show_flags: dict) -> np.ndarray:
    img = frame_bgr.copy()
    H, W = img.shape[:2]

    # Collect per-person data from every annotation type
    person_data: dict[int, dict] = {}   # track_id -> merged info

    if show_flags.get("detections"):
        ann = annotations.get("detections", {}).get(utt_key, {})
        frames = ann.get("frames", [])
        fi = frame_data.get("_idx", 0)
        if fi < len(frames):
            for p in frames[fi].get("persons", []):
                tid = p["track_id"]
                person_data.setdefault(tid, {}).update({
                    "body_bbox": p.get("pose_bbox"),   # field is pose_bbox
                    "face_bbox": p.get("face_bbox"),
                    "keypoints": p.get("keypoints"),
                    "player_name": p.get("player_name"),
                    "is_speaker": p.get("is_speaker"),
                })

    for ann_name in ("expression", "proxemics", "social_gaze", "context_emotion"):
        if not show_flags.get(ann_name):
            continue
        ann = annotations.get(ann_name, {}).get(utt_key, {})
        frames = ann.get("frames", [])
        fi = frame_data.get("_idx", 0)
        if fi < len(frames):
            for p in frames[fi].get("persons", []):
                tid = p["track_id"]
                person_data.setdefault(tid, {}).update(p)

    # Draw each person
    for tid, pdata in person_data.items():
        color = track_color(tid)

        # Pose skeleton
        if show_flags.get("detections") and pdata.get("keypoints"):
            kps = pdata["keypoints"]  # list of [x, y, conf]
            for i, j in POSE_PAIRS:
                if i < len(kps) and j < len(kps):
                    xi, yi, ci = kps[i]
                    xj, yj, cj = kps[j]
                    if ci > 0.3 and cj > 0.3:
                        cv2.line(img, (int(xi), int(yi)), (int(xj), int(yj)),
                                 color, 2)
            for kp in kps:
                x, y, c = kp
                if c > 0.3:
                    cv2.circle(img, (int(x), int(y)), 3, color, -1)

        # Body bbox
        if show_flags.get("detections") and pdata.get("body_bbox"):
            x1, y1, x2, y2 = [int(v) for v in pdata["body_bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Face bbox
        if show_flags.get("detections") and pdata.get("face_bbox"):
            x1, y1, x2, y2 = [int(v) for v in pdata["face_bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Gaze arrow: from face center to gaze_point (normalized coords)
        if show_flags.get("gaze"):
            gaze_ann = annotations.get("gaze", {}).get(utt_key, {})
            gaze_frames = gaze_ann.get("frames", []) if isinstance(gaze_ann, dict) else []
            fi = frame_data.get("_idx", 0)
            if fi < len(gaze_frames):
                for gp in gaze_frames[fi].get("persons", []):
                    if gp["track_id"] != tid:
                        continue
                    gaze_pt = gp.get("gaze_point")
                    if gaze_pt and pdata.get("face_bbox"):
                        fx1, fy1, fx2, fy2 = pdata["face_bbox"]
                        face_cx = int((fx1 + fx2) / 2)
                        face_cy = int((fy1 + fy2) / 2)
                        gx = int(gaze_pt[0] * W)
                        gy = int(gaze_pt[1] * H)
                        cv2.arrowedLine(img, (face_cx, face_cy), (gx, gy),
                                        (0, 255, 255), 2, tipLength=0.15)
                        cv2.circle(img, (gx, gy), 5, (0, 255, 255), -1)

        # Labels above body bbox (or face)
        anchor_bbox = pdata.get("body_bbox") or pdata.get("face_bbox")
        if anchor_bbox:
            bx1, by1 = int(anchor_bbox[0]), int(anchor_bbox[1])
            name_str = pdata.get("player_name") or f"#{tid}"
            spk_mark = " 🎤" if pdata.get("is_speaker") else ""
            lines = [f"{name_str}{spk_mark}"]
            if show_flags.get("expression") and pdata.get("expression"):
                lines.append(f"exp: {pdata['expression']}")
            if show_flags.get("gaze") and pdata.get("gaze_inout"):
                lines.append(f"gaze: {pdata['gaze_inout']}")
            if show_flags.get("proxemics") and pdata.get("proxemics"):
                lines.append(f"prox: {pdata['proxemics']}")
            if show_flags.get("context_emotion") and pdata.get("context_emotions"):
                emo = ", ".join(pdata["context_emotions"][:2])
                lines.append(f"ctx: {emo}")

            font_scale, thickness = 0.45, 1
            line_h = 16
            total_h = len(lines) * line_h
            txt_y = max(total_h, by1 - 4)
            for li, line in enumerate(lines):
                y = txt_y - (len(lines) - 1 - li) * line_h
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                              font_scale, thickness)
                cv2.rectangle(img, (bx1, y - th - 2), (bx1 + tw + 2, y + 2),
                              (0, 0, 0), -1)
                cv2.putText(img, line, (bx1, y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, color, thickness)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def role_badge(role: str) -> str:
    color_map = {
        "Werewolf": "#dc3232", "Minion": "#c85050",
        "Seer": "#3296dc",     "Villager": "#888",
        "Troublemaker": "#c89632", "Tanner": "#b450c8",
        "Robber": "#32c896",   "Drunk": "#c8c832",
        "Hunter": "#50c864",   "Mason": "#50a0c8",
        "Insomniac": "#a0c850",
    }
    bg = color_map.get(role, "#555")
    return f'<span style="background:{bg};color:#fff;padding:2px 7px;border-radius:4px;font-size:0.85em">{role}</span>'


def annotation_chip(label: str, value: str) -> str:
    return (f'<span style="background:#1e3a5f;color:#7eb8f7;padding:2px 8px;'
            f'border-radius:4px;font-size:0.82em;margin:2px">'
            f'<b>{label}</b> {value}</span>')


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="WOLF-ToM Visualizer", layout="wide",
                       page_icon="🐺")
    st.title("🐺 WOLF-ToM Annotation Visualizer")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Navigation")
        subset = st.selectbox("Subset", ["youtube", "ego4d"])
        games = load_splits(subset)

        game_keys = sorted(games.keys())
        game_key  = st.selectbox("Game", game_keys,
                                 format_func=lambda k: f"{k} ({games[k]['_split']})")

        utts = get_utterances(subset, game_key, games)
        if not utts:
            st.warning("No clips found for this game.")
            return

        def utt_label(u):
            spk = u.get("speaker") or u.get("Identity") or "?"
            text = (u.get("utterance") or u.get("Utterance") or "")[:40]
            return f"#{u['_rec_id']:04d} {spk}: {text}"

        utt_idx = st.selectbox("Utterance", range(len(utts)),
                               format_func=lambda i: utt_label(utts[i]))
        utt = utts[utt_idx]

        st.divider()
        st.subheader("Overlays")
        show = {
            "detections":     st.checkbox("Detections (bbox + pose)", value=True),
            "expression":     st.checkbox("Expression", value=True),
            "gaze":           st.checkbox("Gaze", value=True),
            "proxemics":      st.checkbox("Proxemics", value=False),
            "social_gaze":    st.checkbox("Social gaze", value=False),
            "context_emotion": st.checkbox("Context emotion", value=True),
        }

    # ── Load annotations (only what's needed for this subset) ─────────────────
    needed = ["speech_emotion", "vocalsound", "diarization"]
    if show["detections"]:
        needed.append("detections")
    if show["expression"]:
        needed.append("expression")
    if show["gaze"]:
        needed.append("gaze")
    if show["proxemics"]:
        needed.append("proxemics")
    if show["social_gaze"]:
        needed.append("social_gaze")
    if show["context_emotion"]:
        needed.append("context_emotion")
    annotations = {name: load_annotation(name, subset) for name in needed}

    game  = games[game_key]
    utt_key = utt["_utt_key"]
    players = game.get("playerNames", [])
    start_roles = game.get("startRoles", [])
    end_roles   = game.get("endRoles",   [])
    vote_outcome = game.get("votingOutcome", [])

    # ── Game info ──────────────────────────────────────────────────────────────
    with st.expander("Game info", expanded=True):
        cols = st.columns(len(players)) if players else [st]
        for i, (col, name) in enumerate(zip(cols, players)):
            with col:
                start = start_roles[i] if i < len(start_roles) else "?"
                end   = end_roles[i]   if i < len(end_roles)   else "?"
                voted = bool(vote_outcome[i]) if i < len(vote_outcome) else False
                st.markdown(f"**{name}**", unsafe_allow_html=True)
                st.markdown(f"Start: {role_badge(start)}", unsafe_allow_html=True)
                st.markdown(f"End: {role_badge(end)}", unsafe_allow_html=True)
                if voted:
                    st.markdown("🗳️ voted out")

    # ── Utterance info ────────────────────────────────────────────────────────
    speaker   = utt.get("speaker") or utt.get("Identity") or "Unknown"
    text      = utt.get("utterance") or utt.get("Utterance") or ""
    strategy  = utt.get("annotation") or []
    if isinstance(strategy, list):
        strategy = ", ".join(strategy)

    sp_emo   = annotations["speech_emotion"].get(utt_key, {})
    voc      = annotations["vocalsound"].get(utt_key, {})
    diar     = annotations["diarization"].get(utt_key, {})
    # field name aliases
    voc_label  = voc.get("vocal_sound") or voc.get("vocalsound")
    voc_scores = voc.get("vocal_sound_scores") or voc.get("vocalsound_scores", {})

    chips = []
    if strategy:
        chips.append(annotation_chip("strategy:", strategy))
    if sp_emo.get("speech_emotion"):
        chips.append(annotation_chip("speech emotion:", sp_emo["speech_emotion"]))
    if voc_label:
        chips.append(annotation_chip("vocalsound:", voc_label))
    diar_spk = diar.get("diarization_speaker") or diar.get("speaker")
    if diar_spk:
        chips.append(annotation_chip("diarization:", diar_spk))

    st.markdown(f"### 💬 {speaker}: _{text}_")
    if chips:
        st.markdown(" ".join(chips), unsafe_allow_html=True)
    st.caption(f"key: `{utt_key}` | split: {game['_split']}")

    # ── Frame viewer ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(utt["_clip"])
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    cap.release()

    frame_idx = st.slider("Frame", 0, max(0, n_frames - 1), 0)

    # Read chosen frame
    cap = cv2.VideoCapture(utt["_clip"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret:
        st.error("Could not read frame.")
        return

    # Build frame_data dict for draw_annotations
    frame_data_ctx = {"_idx": frame_idx}

    annotated = draw_annotations(frame_bgr, frame_data_ctx, annotations,
                                 utt_key, show)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    col_img, col_scores = st.columns([3, 1])

    with col_img:
        st.image(annotated_rgb, caption=f"Frame {frame_idx}/{n_frames}  ({fps:.1f} fps)",
                 use_container_width=True)

    with col_scores:
        st.subheader("Scores")

        # Expression scores
        if show["expression"]:
            expr_ann = annotations["expression"].get(utt_key, {})
            expr_frames = expr_ann.get("frames", [])
            if frame_idx < len(expr_frames):
                for p in expr_frames[frame_idx].get("persons", []):
                    scores = p.get("expression_scores")
                    if scores:
                        st.caption(f"Expression — track #{p['track_id']}")
                        top = sorted(scores, key=scores.get, reverse=True)[:5]
                        for label in top:
                            st.progress(float(scores[label]),
                                        text=f"{label}: {scores[label]:.2f}")

        # Context emotion scores
        if show["context_emotion"]:
            ctx_ann = annotations["context_emotion"].get(utt_key, {})
            ctx_frames = ctx_ann.get("frames", [])
            if frame_idx < len(ctx_frames):
                for p in ctx_frames[frame_idx].get("persons", []):
                    scores = p.get("context_emotion_scores")
                    if scores:
                        st.caption(f"Context emotion — track #{p['track_id']}")
                        top = sorted(scores, key=scores.get, reverse=True)[:5]
                        for label in top:
                            st.progress(float(scores[label]),
                                        text=f"{label}: {scores[label]:.2f}")

        # Gaze scores
        if show["gaze"]:
            gaze_ann = annotations["gaze"].get(utt_key, {})
            gaze_frames = gaze_ann.get("frames", []) if isinstance(gaze_ann, dict) else []
            if frame_idx < len(gaze_frames):
                for p in gaze_frames[frame_idx].get("persons", []):
                    inout = p.get("gaze_inout")
                    score = p.get("gaze_inout_score")
                    conf  = p.get("gaze_conf")
                    if inout is not None:
                        st.caption(f"Gaze — track #{p['track_id']}")
                        if score is not None:
                            st.progress(float(score),
                                        text=f"in-frame: {score:.2f}")
                        if conf is not None:
                            st.progress(float(conf),
                                        text=f"conf: {conf:.2f}")

        # Speech emotion scores (utterance-level)
        if sp_emo.get("speech_emotion_scores"):
            st.caption("Speech emotion")
            scores = sp_emo["speech_emotion_scores"]
            top = sorted(scores, key=scores.get, reverse=True)[:4]
            for label in top:
                st.progress(float(scores[label]),
                            text=f"{label}: {scores[label]:.2f}")

        # Vocalsound scores
        if voc_scores:
            st.caption("Vocalsound")
            top = sorted(voc_scores, key=voc_scores.get, reverse=True)[:4]
            for label in top:
                st.progress(float(voc_scores[label]),
                            text=f"{label}: {voc_scores[label]:.2f}")


if __name__ == "__main__":
    main()
