# WOLF-ToM

Pseudo-annotation pipeline for the **Werewolf Among Us** dataset.

Runs SOTA specialized models on per-utterance video clips to generate rich multimodal annotation layers (gaze, emotion, audio, proxemics) that enrich the dataset with Theory-of-Mind relevant signals, feeding into the `data/` pipeline for MLLM fine-tuning and evaluation.

---

## Project Structure

```
WOLF-ToM/
├── data/
│   └── WerewolfAmongUs/              # Raw dataset (not tracked by git)
│       ├── clips/
│       │   ├── youtube/              # Per-utterance clips: utt_XXXX.mp4
│       │   └── ego4d/
│       ├── Youtube/split/            # train/val/test annotation JSONs
│       └── Ego4D/split/
├── models/
│   ├── detection/                    # YOLOv8-Face + YOLOv8-pose (prerequisites)
│   │   └── YOLOv8-Face/
│   ├── gaze/                         # Gazelle (GazeFollow + VAT), MTGS (social gaze)
│   │   ├── gazelle/
│   │   └── MTGS/
│   ├── emotion/                      # CocoER (context), DRKF (speech)
│   │   ├── CocoER/
│   │   └── DRKF/
│   └── audio/                        # AST (VocalSound), pyannote.audio (diarization)
├── annotations/                      # Inference output caches (not tracked by git)
│   ├── detections_youtube.json       # Face/person bboxes, keypoints, player registry
│   ├── detections_ego4d.json
│   ├── gaze_youtube.json             # Gazelle: per-utterance gaze points
│   ├── gaze_ego4d.json
│   ├── social_gaze_youtube.json      # MTGS: mutual gaze + shared attention
│   ├── social_gaze_ego4d.json
│   ├── expression_youtube.json       # HSEmotion: expression (8 classes) + valence/arousal
│   ├── expression_ego4d.json
│   ├── context_emotion_youtube.json  # CocoER: 26 discrete emotion categories per player
│   ├── context_emotion_ego4d.json
│   ├── speech_emotion_youtube.json   # DRKF: 7-class speech emotion per utterance
│   ├── speech_emotion_ego4d.json
│   ├── proxemics_youtube.json        # YOLOv8-pose: pairwise keypoint contact distances
│   ├── proxemics_ego4d.json
│   ├── vocalsound_youtube.json       # AST: vocal sound classification per utterance
│   ├── vocalsound_ego4d.json
│   ├── diarization_youtube.json      # pyannote.audio: speaker timeline per game
│   ├── diarization_ego4d.json
│   └── master.json                   # Merged output — all tasks, all utterances
├── scripts/
│   ├── run_detection.py              # YOLOv8-Face + YOLOv8-pose → detections_*.json
│   ├── run_gazelle.py                # Gazelle → gaze_*.json
│   ├── run_mtgs.py                   # MTGS → social_gaze_*.json
│   ├── run_hsemotion.py              # HSEmotion → expression_*.json
│   ├── run_cocoer.py                 # CocoER → context_emotion_*.json
│   ├── run_meld.py                   # DRKF → speech_emotion_*.json
│   ├── run_proxemics.py              # keypoint distances → proxemics_*.json
│   ├── run_vocalsound.py             # AST → vocalsound_*.json
│   ├── run_voxconverse.py            # pyannote.audio → diarization_*.json
│   └── merge_annotations.py         # All JSONs → master.json
├── requirements.txt
└── README.md
```

---

## Annotation Schema

Each inference script produces a JSON file keyed by `{subset}/{game_key}/utt_{rec_id:04d}`:

```json
{
  "youtube/part10_Game1/utt_0002": {
    "speaker": "Mitchell",
    "expression": "Anger",
    "valence": -0.42,
    "arousal": 0.61,
    "gaze_point": [524, 310],
    "mutual_gaze": true,
    "speech_emotion": "anger",
    "context_emotions": ["Disapproval", "Doubt"],
    "vocal_sound": null,
    "proxemics": []
  }
}
```

`merge_annotations.py` combines all per-task JSONs into `annotations/master.json`, ready for the Werewolf builder in the `data/` project.

---

## Models

| # | Task | Model | Source | Weights |
|---|------|-------|--------|---------|
| 1 | **Face detection + landmarks** *(prerequisite)* | YOLOv8-Face | [Yusepp/YOLOv8-Face](https://github.com/Yusepp/YOLOv8-Face) → `models/detection/` | see below |
| 2 | **Person detection + keypoints** *(prerequisite)* | YOLOv8-pose | `pip install ultralytics` | auto-download |
| 3 | **Face re-ID** — Ego4D player identity clustering | ArcFace via DeepFace | `pip install deepface` | auto-download |
| 4 | **GazeFollow** — image-level gaze target | Gazelle | [fkryan/gazelle](https://github.com/fkryan/gazelle) → `models/gaze/` | TBD |
| 5 | **VideoAttentionTarget** — video-level gaze target | Gazelle | [fkryan/gazelle](https://github.com/fkryan/gazelle) → `models/gaze/` | TBD |
| 6 | **Social gaze** — mutual gaze + shared attention | MTGS | [idiap/MTGS](https://github.com/idiap/MTGS) → `models/gaze/` | HuggingFace (see below) |
| 7 | **Facial expression** (8 classes) + valence/arousal | HSEmotion | `pip install hsemotion` | auto-download |
| 8 | **Context emotion** — 26 discrete categories | CocoER | [bisno/CocoER](https://github.com/bisno/CocoER) → `models/emotion/` | see below |
| 9 | **Speech emotion** — 7 classes | DRKF | [PANPANKK/DRKF](https://github.com/PANPANKK/DRKF_Decoupled_Representations_with_Knowledge_Fusion_for_Multimodal_Emotion_Recognition) → `models/emotion/` | bundled locally |
| 10 | **Proxemics** — pairwise physical contact | YOLOv8-pose keypoints | already in #2 | — |
| 11 | **Vocal sound** — laughter, sigh, cough, etc. | AST | `pip install transformers` | auto-download (HuggingFace) |
| 12 | **Speaker diarization** — who speaks when | pyannote.audio | `pip install pyannote.audio` | HuggingFace (requires token) |

---

## Setup

### 1. Clone model repos

```bash
# Detection
git clone https://github.com/Yusepp/YOLOv8-Face models/detection/YOLOv8-Face

# Gaze
git clone https://github.com/fkryan/gazelle models/gaze/gazelle
git clone https://github.com/idiap/MTGS models/gaze/MTGS

# Emotion
git clone https://github.com/bisno/CocoER models/emotion/CocoER
git clone https://github.com/PANPANKK/DRKF_Decoupled_Representations_with_Knowledge_Fusion_for_Multimodal_Emotion_Recognition models/emotion/DRKF
```

### 2. Download YOLOv8-Face weights

Place weights under `models/detection/YOLOv8-Face/weights/`:

| Variant | Download |
|---------|----------|
| Large (v0.1) | [Google Drive](https://drive.google.com/file/d/1iHL-XjvzpbrE8ycVqEbGla4yc1dWlSWU/view?usp=sharing) |
| Medium (v0.2) — **default** | [Google Drive](https://drive.google.com/file/d/1IJZBcyMHGhzAi0G4aZLcqryqZSjPsps-/view?usp=sharing) |
| Nano (v0.1) | [Google Drive](https://drive.google.com/file/d/1ZD_CEsbo3p3_dd8eAtRfRxHDV44M0djK/view?usp=sharing) |

### 3. Install MTGS

```bash
cd models/gaze/MTGS
pip install -r requirements.txt
pip install -e .
cd ../../..
```

MTGS pretrained weights (HuggingFace):

| Variant | Description |
|---------|-------------|
| `mtgs-vsgaze` | **Default** — temporal, trained on VSGaze (multi-person social gaze) |
| `mtgs-static-vsgaze` | Static — faster, slightly lower accuracy |
| `mtgs-static-gazefollow` | Static, trained on GazeFollow |

> MTGS ships with YOLOv5 for head detection — we bypass it by injecting our YOLOv8-Face bboxes directly.

### 4. Install CocoER

```bash
cd models/emotion/CocoER
conda create -n cocoer python=3.7
conda activate cocoer
pip install torch==1.11.0+cu113 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
cd ../../..
```

Download pretrained weights into `models/emotion/CocoER/checkpoints/` (links in the CocoER repo README).

### 5. Install DRKF

```bash
cd models/emotion/DRKF
conda env create -f environment.yaml -n drkf
conda activate drkf
cd ../../..
```

> DRKF has no inference script — `scripts/run_meld.py` wraps the model forward pass directly (wav2vec2 for audio + RoBERTa for text). Backbone weights are bundled in the repo.

### 6. Install Python dependencies (main env)

```bash
pip install -r requirements.txt
```

> YOLOv8-pose (`yolov8m-pose.pt`), HSEmotion, AST, and ArcFace weights all download automatically on first use.

### 7. Authenticate pyannote.audio

pyannote models are gated on HuggingFace. Accept the license at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1), then:

```bash
huggingface-cli login
```

---

## Workflow

```bash
# Step 1 — Detection (prerequisite for all vision tasks)
python scripts/run_detection.py

# Step 2 — Inference (independent, can be parallelised)
python scripts/run_gazelle.py       # GazeFollow + VideoAttentionTarget
python scripts/run_mtgs.py          # Social gaze + mutual gaze
python scripts/run_hsemotion.py     # Facial expression + valence/arousal
python scripts/run_cocoer.py        # Context emotion (26 categories)
python scripts/run_meld.py          # Speech emotion (7 classes)
python scripts/run_proxemics.py     # Physical contact (keypoint distances)
python scripts/run_vocalsound.py    # Vocal sound classification
python scripts/run_voxconverse.py   # Speaker diarization

# Step 3 — Merge
python scripts/merge_annotations.py
# → annotations/master.json

# Step 4 — Use master.json in the data/ Werewolf builder
```

> Note: CocoER and DRKF require their own conda environments. Their inference scripts invoke them as subprocesses.

---

## Dataset

**Werewolf Among Us** (ACL Findings 2023) — 199 social deduction games (151 YouTube + 48 Ego4D), ~24,000 per-utterance clips with persuasion strategy and voting outcome annotations.

- [Paper](https://aclanthology.org/2023.findings-acl.411.pdf)
- [Project Page](https://bolinlai.github.io/projects/Werewolf-Among-Us/)
