# WOLF-ToM

Pseudo-annotation pipeline for the **Werewolf Among Us** dataset.

Runs SOTA specialized models on per-utterance video clips to generate rich multimodal annotation layers (gaze, emotion, face) that can then be consumed by the `data/` pipeline to build fine-tuning and evaluation datasets for Theory-of-Mind MLLMs.

---

## Project Structure

```
WOLF-ToM/
├── data/
│   └── WerewolfAmongUs/          # Raw dataset (not tracked by git)
│       ├── clips/
│       │   ├── youtube/          # Per-utterance clips: utt_XXXX.mp4
│       │   └── ego4d/
│       ├── Youtube/split/        # train/val/test annotation JSONs
│       └── Ego4D/split/
├── models/
│   ├── gaze/                     # Gaze detection model repo(s)
│   ├── emotion/                  # Emotion recognition model repo(s)
│   └── face/                     # Face detection / landmark model repo(s)
├── annotations/                  # Inference output caches (not tracked)
│   ├── gaze_youtube.json
│   ├── gaze_ego4d.json
│   ├── emotion_youtube.json
│   └── emotion_ego4d.json
├── scripts/
│   ├── run_gaze.py               # Inference: gaze model → annotations/gaze_*.json
│   ├── run_emotion.py            # Inference: emotion model → annotations/emotion_*.json
│   └── merge_annotations.py     # Merge all annotation JSONs into one master file
├── requirements.txt
└── README.md
```

---

## Annotation Schema

Each inference script produces a JSON file mapping `{subset}/{game_key}/utt_{rec_id:04d}` to a dict of model predictions:

```json
{
  "youtube/part10_Game1/utt_0002": {
    "gaze_x": 0.52,
    "gaze_y": 0.31,
    "gaze_confidence": 0.87
  },
  ...
}
```

The `merge_annotations.py` script combines all outputs into a single `annotations/master.json` keyed by the same path, ready to be consumed by the Werewolf builder in the `data/` project.

---

## Models

| # | Dataset / Task | Model | Repo | Weights |
|---|---|---|---|---|
| 1a | **Face detection + landmarks** (prerequisite) | YOLOv8-Face | [Yusepp/YOLOv8-Face](https://github.com/Yusepp/YOLOv8-Face) | see below |
| 1b | **Person detection + keypoints** (prerequisite) | YOLOv8-pose | [ultralytics](https://github.com/ultralytics/ultralytics) | auto-download |
| 1c | **Face re-ID** — Ego4D identity clustering | ArcFace via DeepFace | `pip install deepface` | auto-download |
| 2 | **GazeFollow** — image gaze target | Gazelle | [fkryan/gazelle](https://github.com/fkryan/gazelle) | TBD |
| 3 | **VideoAttentionTarget** — video gaze target | Gazelle | [fkryan/gazelle](https://github.com/fkryan/gazelle) | TBD |
| 4 | **VideoCoAttention / social gaze** — shared attention + mutual gaze | MTGS | [idiap/MTGS](https://github.com/idiap/MTGS) | HuggingFace (see below) |
| 5 | **AffWild2** — facial expression (8 classes) + valence/arousal | HSEmotion | `pip install hsemotion` | auto-download |
| 6 | **EMOTIC** — context emotion (26 discrete categories) | CocoER | [bisno/CocoER](https://github.com/bisno/CocoER) | see below |
| 8 | **MELD** — speech emotion (7 classes) | DRKF | [PANPANKK/DRKF](https://github.com/PANPANKK/DRKF_Decoupled_Representations_with_Knowledge_Fusion_for_Multimodal_Emotion_Recognition) | bundled locally |
| 10 | **Proxemics** — physical contact | YOLOv8-pose (keypoint distances) | already in 1b | auto-download |
| 11 | **MUStARD** — sarcasm detection | TBD | `models/` | TBD |
| 12 | **RLDD** — deception detection | TBD | `models/` | TBD |
| 13 | **UR-FUNNY** — humor detection | TBD | `models/` | TBD |
| 14 | **VocalSound** — vocal sound classification | AST | `pip install transformers` | auto-download (HuggingFace) |
| 15 | **VoxConverse** — speaker diarization | pyannote.audio | `pip install pyannote.audio` | HuggingFace (requires token) |

---

## Setup

### 1. Clone model repos

```bash
git clone https://github.com/Yusepp/YOLOv8-Face models/face/YOLOv8-Face
git clone https://github.com/fkryan/gazelle models/gaze/gazelle
git clone https://github.com/idiap/MTGS models/gaze/MTGS
git clone https://github.com/bisno/CocoER models/emotion/CocoER
git clone https://github.com/PANPANKK/DRKF_Decoupled_Representations_with_Knowledge_Fusion_for_Multimodal_Emotion_Recognition models/emotion/DRKF
# ... (fill in remaining repos as decided)
```

### 2. Download YOLOv8-Face weights

Place weights under `models/face/YOLOv8-Face/weights/`:

| Variant | Download |
|---------|----------|
| Large (v0.1) | [Google Drive](https://drive.google.com/file/d/1iHL-XjvzpbrE8ycVqEbGla4yc1dWlSWU/view?usp=sharing) |
| Medium (v0.2) | [Google Drive](https://drive.google.com/file/d/1IJZBcyMHGhzAi0G4aZLcqryqZSjPsps-/view?usp=sharing) |
| Nano (v0.1) | [Google Drive](https://drive.google.com/file/d/1ZD_CEsbo3p3_dd8eAtRfRxHDV44M0djK/view?usp=sharing) |

> We use **Medium** as the default balance between speed and accuracy.

### 3. Install MTGS

```bash
cd models/gaze/MTGS
pip install -r requirements.txt
pip install -e .
cd ../../..
```

MTGS pretrained weights are available on HuggingFace:

| Variant | Description |
|---------|-------------|
| `mtgs-vsgaze` | **Default** — temporal model trained on VSGaze (multi-person social gaze) |
| `mtgs-static-vsgaze` | Static (no temporal) — faster, slightly lower accuracy |
| `mtgs-static-gazefollow` | Static model trained on GazeFollow |

> Note: MTGS ships with YOLOv5 for head detection. We bypass this by feeding our YOLOv8-Face bboxes directly.

### CocoER (EMOTIC — 26 discrete categories)

```bash
cd models/emotion/CocoER
conda create -n cocoer python=3.7
conda activate cocoer
pip install torch==1.11.0+cu113 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
cd ../../..
```

Download pretrained weights and place them under `models/emotion/CocoER/checkpoints/` (links provided in the CocoER repo README).

### DRKF (MELD — speech emotion, 7 classes)

```bash
cd models/emotion/DRKF
conda env create -f environment.yaml -n drkf
conda activate drkf
cd ../../..
```

> Note: DRKF has no inference script — `scripts/run_meld.py` wraps the model's forward pass directly (wav2vec2 for audio + RoBERTa for text). Pretrained backbone weights are bundled locally in the repo.

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

> YOLOv8-pose weights (`yolov8m-pose.pt`) and DeepFace/ArcFace weights download automatically on first use.

### 5. pyannote.audio (speaker diarization)

pyannote.audio models are gated on HuggingFace. Accept the license and generate a token at https://huggingface.co/pyannote/speaker-diarization-3.1, then:

```bash
huggingface-cli login
```

---

## Workflow

```bash
# 1. Run face/person detection + build per-game player registry (prerequisite)
python scripts/run_detection.py

# 2. Run inference — one script per task (can be parallelised)
python scripts/run_gazelle.py          # GazeFollow + VideoAttentionTarget
python scripts/run_mtgs.py             # VideoCoAttention + social/mutual gaze (MTGS)
python scripts/run_hsemotion.py        # Facial expression (8 classes) + valence/arousal
python scripts/run_cocoer.py           # Context emotion (26 discrete categories, CocoER)
python scripts/run_meld.py             # Speech emotion (7 classes, DRKF — custom inference wrapper)
python scripts/run_proxemics.py        # Physical contact (pairwise keypoint distances from YOLOv8-pose)
python scripts/run_mustard.py          # Sarcasm
python scripts/run_rldd.py             # Deception
python scripts/run_urfunny.py          # Humor
python scripts/run_vocalsound.py       # Vocal sounds (AST)
python scripts/run_voxconverse.py      # Speaker diarization (pyannote.audio)

# 3. Merge all annotation JSONs
python scripts/merge_annotations.py
# → annotations/master.json

# 4. Use master.json in the data/ pipeline Werewolf builder
```

---

## Dataset

**Werewolf Among Us** (ACL Findings 2023) — 199 social deduction games (151 YouTube + 48 Ego4D), ~24,000 utterance clips with persuasion strategy and voting outcome annotations.

- [Paper](https://aclanthology.org/2023.findings-acl.411.pdf)
- [Project Page](https://bolinlai.github.io/projects/Werewolf-Among-Us/)
