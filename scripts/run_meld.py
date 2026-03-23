"""
Step 2f — Speech Emotion  (EmoBERTa / tae898)

Uses tae898/emoberta-base — a RoBERTa model fine-tuned on MELD — to predict
one of 7 emotion classes per utterance from text.

This is a **per-utterance** annotation (not per-frame): the emotion
label applies uniformly to every frame in the clip.

HuggingFace: tae898/emoberta-base
Classes: neutral, surprise, fear, sadness, joy, disgust, anger

Output — annotations/speech_emotion_{subset}.json:
{
  "youtube/GameKey/utt_0002": {
    "speech_emotion":        "anger",
    "speech_emotion_scores": {"neutral": 0.05, "anger": 0.74, …}
  }
}
"""

from __future__ import annotations

import os
import sys

import torch
from tqdm import tqdm
from transformers import pipeline

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    SUBSETS,
    iter_utterances,
    save_annotation,
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

MODEL_NAME  = "tae898/emoberta-base"
BATCH_SIZE  = 64
DEVICE      = 0 if torch.cuda.is_available() else -1   # transformers device index

# Map emoberta label IDs to MELD class names
EMOBERTA_LABELS = [
    "neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"
]

# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def load_model():
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,
        device=DEVICE,
        truncation=True,
        max_length=128,
    )


def run_batch(pipe, texts: list[str]) -> list[dict]:
    """Return list of {label: score} dicts, one per text."""
    results = pipe(texts, batch_size=BATCH_SIZE)
    out = []
    for res in results:
        scores = {r["label"]: r["score"] for r in res}
        # Ensure all labels present (fill 0 if absent)
        full = {lbl: scores.get(lbl, 0.0) for lbl in EMOBERTA_LABELS}
        best = max(full, key=full.get)
        out.append({"label": best, "scores": full})
    return out


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run(subset: str):
    print(f"\n[emoberta] {subset}")
    pipe = load_model()

    utts  = list(iter_utterances(subset))
    output: dict = {}

    for batch_start in tqdm(range(0, len(utts), BATCH_SIZE),
                            desc=f"  {subset}", unit="batch"):
        batch_utts = utts[batch_start: batch_start + BATCH_SIZE]
        texts = [u.get("text", "") or "" for u in batch_utts]
        try:
            preds = run_batch(pipe, texts)
            for utt, pred in zip(batch_utts, preds):
                output[utt["utt_key"]] = {
                    "speech_emotion":        pred["label"],
                    "speech_emotion_scores": pred["scores"],
                }
        except Exception as e:
            print(f"  WARN batch {batch_start}: {e}")

    save_annotation(output, "speech_emotion", subset)


if __name__ == "__main__":
    for subset in SUBSETS:
        run(subset)
