"""
Step 2g — Vocal Sound Classification  (AST)

Classifies non-speech vocalisations (laughter, crying, coughing, sighing, etc.)
in each utterance clip using an Audio Spectrogram Transformer fine-tuned on
VocalSound.

This is a **per-utterance** annotation.

Model: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
       (fall back to the VocalSound-specific checkpoint when available)

pip install transformers librosa soundfile

Output — annotations/vocalsound_{subset}.json:
{
  "youtube/GameKey/utt_0002": {
    "vocal_sound":        "laughter",   // null if model confidence < threshold
    "vocal_sound_scores": {"laughter": 0.72, "silence": 0.14, …}
  }
}
"""

import os
import subprocess
import sys
import tempfile

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTForAudioClassification

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    SUBSETS,
    iter_utterances,
    save_annotation,
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

MODEL_ID   = "MIT/ast-finetuned-audioset-10-10-0.4593"
CONF_THRESH = 0.40    # min confidence to report a label (else null)
SAMPLE_RATE = 16000
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# VocalSound target labels (AudioSet subset)
VOCAL_LABELS = {
    "laughter", "crying", "cough", "sneeze", "sigh",
    "breathing", "throat_clearing", "humming",
}

# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_model():
    extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    model     = ASTForAudioClassification.from_pretrained(MODEL_ID)
    model     = model.to(DEVICE).eval()
    return extractor, model


# ──────────────────────────────────────────────
# Audio extraction
# ──────────────────────────────────────────────

def extract_audio_array(clip_path: str) -> np.ndarray:
    """Return a 16kHz mono float32 numpy array."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    subprocess.run(
        ["ffmpeg", "-y", "-i", clip_path, "-ac", "1", "-ar", str(SAMPLE_RATE), "-vn", wav_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    os.unlink(wav_path)
    return audio


# ──────────────────────────────────────────────
# Per-utterance inference
# ──────────────────────────────────────────────

def classify_vocalsound(extractor, model, audio: np.ndarray) -> dict:
    inputs  = extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inputs  = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs  = torch.softmax(logits, dim=-1).cpu().numpy()
    labels = model.config.id2label

    # Build score dict; keep only vocal-sound-relevant labels
    score_dict = {}
    for idx, prob in enumerate(probs):
        label = labels[idx].lower().replace(" ", "_")
        if any(vl in label for vl in VOCAL_LABELS):
            score_dict[label] = float(prob)

    if not score_dict:
        return {"vocal_sound": None, "vocal_sound_scores": {}}

    best_label = max(score_dict, key=score_dict.get)
    best_score = score_dict[best_label]
    return {
        "vocal_sound":        best_label if best_score >= CONF_THRESH else None,
        "vocal_sound_scores": score_dict,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run(subset: str):
    print(f"\n[vocalsound] {subset}")
    extractor, model = load_model()

    output = {}
    utts = list(iter_utterances(subset))
    for utt in tqdm(utts, desc=f"  {subset}", unit="clip"):
        try:
            audio  = extract_audio_array(utt["clip_path"])
            result = classify_vocalsound(extractor, model, audio)
            output[utt["utt_key"]] = result
        except Exception as e:
            print(f"  WARN {utt['utt_key']}: {e}")

    save_annotation(output, "vocalsound", subset)


if __name__ == "__main__":
    for subset in SUBSETS:
        run(subset)
