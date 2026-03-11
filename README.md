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
| 1 | **Face & person detection** (prerequisite) | TBD | `models/face/` | TBD |
| 2 | **GazeFollow** — image gaze target | Gazelle | [fkryan/gazelle](https://github.com/fkryan/gazelle) | TBD |
| 3 | **VideoAttentionTarget** — video gaze target | Gazelle | [fkryan/gazelle](https://github.com/fkryan/gazelle) | TBD |
| 4 | **VideoCoAttention** — shared attention | TBD | TBD | TBD |
| 5 | **AffWild2** — facial expression, valence/arousal, AUs | TBD | `models/emotion/` | TBD |
| 6 | **EMOTIC** — context emotion (26 cats + VAD) | TBD | `models/emotion/` | TBD |
| 7 | **MEVIEW / MMEW** — micro-expression + AUs | TBD | `models/emotion/` | TBD |
| 8 | **MELD** — speech emotion (7 classes) | TBD | `models/emotion/` | TBD |
| 9 | **PISC** — social relationship | TBD | `models/` | TBD |
| 10 | **Proxemics** — physical contact | TBD | `models/` | TBD |
| 11 | **MUStARD** — sarcasm detection | TBD | `models/` | TBD |
| 12 | **RLDD** — deception detection | TBD | `models/` | TBD |
| 13 | **UR-FUNNY** — humor detection | TBD | `models/` | TBD |
| 14 | **VocalSound** — vocal sound classification | TBD | `models/` | TBD |
| 15 | **VoxConverse** — speaker diarization | TBD | `models/` | TBD |

---

## Workflow

```bash
# 1. Clone model repos (see Models table above)
git clone https://github.com/fkryan/gazelle models/gaze/gazelle
# ... (fill in remaining repos as decided)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run face/person detection first (prerequisite)
python scripts/run_detection.py

# 4. Run inference — one script per task (can be parallelised)
python scripts/run_gazelle.py          # GazeFollow + VideoAttentionTarget
python scripts/run_coattention.py      # VideoCoAttention
python scripts/run_affwild2.py         # Facial expression / VA / AUs
python scripts/run_emotic.py           # Context emotion
python scripts/run_microexpr.py        # Micro-expression (MEVIEW/MMEW)
python scripts/run_meld.py             # Speech emotion
python scripts/run_pisc.py             # Social relationship
python scripts/run_proxemics.py        # Physical contact
python scripts/run_mustard.py          # Sarcasm
python scripts/run_rldd.py             # Deception
python scripts/run_urfunny.py          # Humor
python scripts/run_vocalsound.py       # Vocal sounds
python scripts/run_voxconverse.py      # Speaker diarization

# 5. Merge all annotation JSONs
python scripts/merge_annotations.py
# → annotations/master.json

# 6. Use master.json in the data/ pipeline Werewolf builder
```

---

## Dataset

**Werewolf Among Us** (ACL Findings 2023) — 199 social deduction games (151 YouTube + 48 Ego4D), ~24,000 utterance clips with persuasion strategy and voting outcome annotations.

- [Paper](https://aclanthology.org/2023.findings-acl.411.pdf)
- [Project Page](https://bolinlai.github.io/projects/Werewolf-Among-Us/)
