"""
Step 2f — Speech Emotion  (DRKF)

DRKF is a multimodal model (wav2vec2 audio + RoBERTa text) that predicts
one of 7 emotion classes per utterance.  It requires its own conda env.

This is a **per-utterance** annotation (not per-frame): the emotion
label applies uniformly to every frame in the clip.

Repo:  https://github.com/PANPANKK/DRKF_Decoupled_Representations_with_Knowledge_Fusion_for_Multimodal_Emotion_Recognition
Clone: models/emotion/DRKF/

Output — annotations/speech_emotion_{subset}.json:
{
  "youtube/GameKey/utt_0002": {
    "speech_emotion":        "anger",
    "speech_emotion_scores": {"neutral": 0.05, "anger": 0.74, …}
  }
}
"""

import json
import os
import subprocess
import sys
import tempfile

from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    SUBSETS,
    iter_utterances,
    load_annotation,
    save_annotation,
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

DRKF_DIR    = os.path.join(ROOT_DIR, "models", "emotion", "DRKF")
DRKF_CONDA  = "drkf"
DRKF_CLASSES = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
BATCH_SIZE  = 32    # utterances per subprocess call

# ──────────────────────────────────────────────
# Worker script
# ──────────────────────────────────────────────

DRKF_RUNNER = os.path.join(os.path.dirname(__file__), "_drkf_worker.py")


def _write_worker_script():
    script = '''"""DRKF worker — invoked inside the drkf conda env."""
import json, os, sys
import torch, torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, RobertaTokenizer

DRKF_DIR    = sys.argv[1]
INPUT_JSON  = sys.argv[2]
OUTPUT_JSON = sys.argv[3]

sys.path.insert(0, DRKF_DIR)
from model.DRKF import DRKF   # noqa

CLASSES = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

ckpt = os.path.join(DRKF_DIR, "checkpoints", "drkf_meld.pt")
model = DRKF(num_classes=len(CLASSES))
state = torch.load(ckpt, map_location=DEVICE)
model.load_state_dict(state["model"] if "model" in state else state)
model = model.to(DEVICE).eval()

with open(INPUT_JSON) as f:
    batch = json.load(f)

results = {}
for utt_key, entry in batch.items():
    audio_path = entry["audio_path"]
    text       = entry.get("text", "")

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    wav_np = waveform[0].numpy()

    audio_inputs = processor(wav_np, sampling_rate=16000, return_tensors="pt", padding=True)
    text_inputs  = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        logits = model(
            audio_inputs.input_values.to(DEVICE),
            text_inputs.input_ids.to(DEVICE),
            text_inputs.attention_mask.to(DEVICE),
        )
    probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
    results[utt_key] = {
        "label":  CLASSES[int(np.argmax(probs))],
        "scores": {c: float(probs[i]) for i, c in enumerate(CLASSES)},
    }

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f)
'''
    with open(DRKF_RUNNER, "w") as f:
        f.write(script)


# ──────────────────────────────────────────────
# Audio extraction helper
# ──────────────────────────────────────────────

def extract_audio(clip_path: str, out_wav: str):
    """Extract 16kHz mono WAV from the clip using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", clip_path,
            "-ac", "1", "-ar", "16000",
            "-vn", out_wav,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


# ──────────────────────────────────────────────
# Batch inference
# ──────────────────────────────────────────────

def run_drkf_batch(batch: dict) -> dict:
    """batch: {utt_key: {"audio_path": str, "text": str}}"""
    if not batch:
        return {}

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as fin:
        json.dump(batch, fin)
        input_path = fin.name

    output_path = input_path.replace(".json", "_out.json")

    cmd = [
        "conda", "run", "-n", DRKF_CONDA, "--no-capture-output",
        "python", DRKF_RUNNER,
        DRKF_DIR, input_path, output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    with open(output_path) as f:
        results = json.load(f)

    os.unlink(input_path)
    os.unlink(output_path)
    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run(subset: str):
    print(f"\n[drkf] {subset}")
    _write_worker_script()

    utts = list(iter_utterances(subset))
    output = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Process in batches to amortise subprocess startup cost
        for batch_start in tqdm(range(0, len(utts), BATCH_SIZE),
                                desc=f"  {subset}", unit="batch"):
            batch_utts = utts[batch_start: batch_start + BATCH_SIZE]
            batch = {}

            for utt in batch_utts:
                wav_path = os.path.join(tmpdir, f"{utt['rec_id']:06d}.wav")
                try:
                    extract_audio(utt["clip_path"], wav_path)
                    batch[utt["utt_key"]] = {
                        "audio_path": wav_path,
                        "text":       utt.get("text", ""),
                    }
                except subprocess.CalledProcessError:
                    pass

            try:
                raw = run_drkf_batch(batch)
                for utt_key, res in raw.items():
                    output[utt_key] = {
                        "speech_emotion":        res["label"],
                        "speech_emotion_scores": res["scores"],
                    }
            except Exception as e:
                print(f"  WARN batch {batch_start}: {e}")

    save_annotation(output, "speech_emotion", subset)


if __name__ == "__main__":
    for subset in SUBSETS:
        run(subset)
