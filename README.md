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

| Task | Model | Repo |
|------|-------|------|
| Gaze target detection | VideoAttentionTarget | `models/gaze/` |
| Facial emotion recognition | AffWild2 / ABAW | `models/emotion/` |
| Face detection | (TBD) | `models/face/` |

---

## Workflow

```bash
# 1. Clone model repos into models/
git clone <gaze-repo> models/gaze/videoattentiontarget
git clone <emotion-repo> models/emotion/affwild2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run inference (one script per model)
python scripts/run_gaze.py
python scripts/run_emotion.py

# 4. Merge all annotations
python scripts/merge_annotations.py
# → annotations/master.json

# 5. Use master.json in the data/ pipeline Werewolf builder
```

---

## Dataset

**Werewolf Among Us** (ACL Findings 2023) — 199 social deduction games (151 YouTube + 48 Ego4D), ~24,000 utterance clips with persuasion strategy and voting outcome annotations.

- [Paper](https://aclanthology.org/2023.findings-acl.411.pdf)
- [Project Page](https://bolinlai.github.io/projects/Werewolf-Among-Us/)
