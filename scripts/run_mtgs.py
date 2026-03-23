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

MTGS_FPS       = 4
MTGS_HF_REPO   = "Idiap/mtgs-vsgaze"
MUTUAL_THRESH  = 0.50
SHARED_THRESH  = 0.50
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# Model config (matches mtgs/config/config.yaml defaults for mtgs-vsgaze)
MTGS_CFG = dict(
    temporal_context   = 2,
    image_size         = 448,
    patch_size         = 14,
    head_size          = 224,
    decoder_feature_dim= 128,
    decoder_use_bn     = True,
    img_mean           = [0.44232, 0.40506, 0.36457],
    img_std            = [0.28674, 0.27776, 0.27995],
)

# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_mtgs():
    import importlib.util
    from huggingface_hub import hf_hub_download

    # Import GazePredictor directly from file to bypass demo/__init__.py
    # which checks for a yolov5crowdhuman folder we don't need.
    _spec = importlib.util.spec_from_file_location(
        "gaze_prediction",
        os.path.join(ROOT_DIR, "models", "gaze", "MTGS",
                     "mtgs", "demo", "gaze_prediction.py"),
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    GazePredictor = _mod.GazePredictor

    ckpt = hf_hub_download(repo_id=MTGS_HF_REPO, filename="mtgs-vsgaze.ckpt")
    predictor = GazePredictor(
        checkpoint_file    = ckpt,
        temporal_context   = MTGS_CFG["temporal_context"],
        image_size         = MTGS_CFG["image_size"],
        patch_size         = MTGS_CFG["patch_size"],
        decoder_feature_dim= MTGS_CFG["decoder_feature_dim"],
        decoder_use_bn     = MTGS_CFG["decoder_use_bn"],
        device             = DEVICE,
    )
    return predictor


def prepare_sample(frame_bgr, head_bboxes_px, cfg, device):
    """
    Replicate DemoProcessor.prepare_input using our pre-computed bboxes.
    head_bboxes_px: list of [x1,y1,x2,y2] in pixel coords.
    """
    import torchvision.transforms.functional as TF
    from PIL import Image as PILImage
    from mtgs.utils import square_bbox

    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(frame_rgb)

    head_bboxes_t = torch.tensor(head_bboxes_px, dtype=torch.float32)  # (N,4)
    sq_bboxes = square_bbox(head_bboxes_t, w, h)

    heads = []
    head_size = cfg["head_size"]
    for bbox in sq_bboxes:
        head = TF.resize(
            TF.to_tensor(pil_img.crop(bbox.numpy())),
            [head_size, head_size], antialias=True,
        )
        heads.append(head)
    heads = torch.stack(heads)
    heads = TF.normalize(heads, mean=cfg["img_mean"], std=cfg["img_std"])

    image_size = cfg["image_size"]
    image = TF.to_tensor(pil_img)
    image = TF.resize(image, [image_size, image_size], antialias=True)
    image = TF.normalize(image, mean=cfg["img_mean"], std=cfg["img_std"])
    image = image.unsqueeze(0)

    # Normalise bboxes
    nb = sq_bboxes.clone().float()
    nb[:, 0] /= w; nb[:, 2] /= w
    nb[:, 1] /= h; nb[:, 3] /= h

    thsize = torch.zeros((1, 3, head_size, head_size), dtype=torch.float32)
    heads  = torch.cat([thsize, heads])
    nb     = torch.cat([torch.zeros((1, 4), dtype=torch.float32), nb])

    sample = {
        "image":            image.unsqueeze(0).to(device),
        "num_valid_people": len(head_bboxes_px),
        "heads":            heads.unsqueeze(0).unsqueeze(0).to(device),
        "head_bboxes":      nb.unsqueeze(0).unsqueeze(0).to(device),
    }
    return sample


# ──────────────────────────────────────────────
# Per-utterance processing
# ──────────────────────────────────────────────

def process_clip(predictor, utt: dict, detections: dict) -> dict:
    from mtgs.utils.social_gaze import get_social_gaze_predictions
    from mtgs.utils.utils import spatial_argmax2d

    det_entry = detections.get(utt["utt_key"])
    if det_entry is None:
        return {}

    n_frames   = det_entry["n_frames"]
    video_fps  = det_entry["video_fps"]
    det_frames = det_entry["frames"]

    cap, _, _ = open_video(utt["clip_path"])
    sampled_frames_out = []

    for frame_idx, frame in iter_sampled_frames(cap, n_frames, video_fps, MTGS_FPS):
        det_frame = det_frames[frame_idx] if frame_idx < len(det_frames) else {}
        persons   = det_frame.get("persons", [])

        head_bboxes_px = [p["face_bbox"] for p in persons if p.get("face_bbox")]
        if not head_bboxes_px:
            sampled_frames_out.append({
                "mutual_gaze_pairs": [], "shared_attention": False,
                "shared_attention_score": None,
            })
            continue

        try:
            h, w = frame.shape[:2]
            sample = prepare_sample(frame, head_bboxes_px, MTGS_CFG, DEVICE)
            with torch.no_grad():
                _, _, gaze_heatmaps, inouts, lah, laeo, coatt = (
                    predictor.predictor(sample)
                )
                laeo  = laeo.squeeze(0).sigmoid().cpu()
                coatt = coatt.squeeze(0).sigmoid().cpu()

            n = len(head_bboxes_px)
            social = get_social_gaze_predictions(
                {"lah": lah, "laeo": laeo, "coatt": coatt,
                 "gaze_points": spatial_argmax2d(
                     gaze_heatmaps.squeeze(0).squeeze(0)[1:], normalize=True),
                 "pids": torch.arange(n),
                 "head_bboxes": torch.tensor(head_bboxes_px),
                 "inouts": inouts.squeeze(0).squeeze(0)[1:].sigmoid().cpu(),
                 "gaze_heatmaps": gaze_heatmaps.squeeze(0).squeeze(0)[1:].cpu(),
                 },
                w, h, n,
            )
            laeo_mat = social[1].cpu()  # (n, n)

            pairs = []
            face_persons = [p for p in persons if p.get("face_bbox")]
            for i, pi in enumerate(face_persons):
                for j, pj in enumerate(face_persons):
                    if j <= i:
                        continue
                    score = float(laeo_mat[i, j]) if i < laeo_mat.shape[0] and j < laeo_mat.shape[1] else 0.0
                    pairs.append({"track_id_a": pi["track_id"], "track_id_b": pj["track_id"], "score": score})

            coatt_score = float(social[2].mean().cpu()) if social[2].numel() > 0 else 0.0
            sampled_frames_out.append({
                "mutual_gaze_pairs":      pairs,
                "shared_attention":       coatt_score >= SHARED_THRESH,
                "shared_attention_score": coatt_score,
            })

        except Exception as e:
            sampled_frames_out.append({
                "mutual_gaze_pairs": [], "shared_attention": False,
                "shared_attention_score": None,
            })

    cap.release()

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
    predictor  = load_mtgs()
    detections = load_annotation("detections", subset)

    output = {}
    utts = list(iter_utterances(subset))
    for utt in tqdm(utts, desc=f"  {subset}", unit="clip"):
        try:
            result = process_clip(predictor, utt, detections)
            if result:
                output[utt["utt_key"]] = result
        except Exception as e:
            print(f"  WARN {utt['utt_key']}: {e}")

    save_annotation(output, "social_gaze", subset)


if __name__ == "__main__":
    for subset in SUBSETS:
        run(subset)
