"""
One-time preprocessing: split large annotation JSONs into per-utterance files.

  python3 scripts/index_annotations.py

Creates annotations/indexed/{name}_{subset}/{utt_key_escaped}.json
where utt_key_escaped replaces '/' with '__'.
The visualizer reads these tiny files instead of the full multi-GB JSONs.
"""

from __future__ import annotations
import json
import os
from pathlib import Path

ROOT_DIR        = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = ROOT_DIR / "annotations"
INDEX_DIR       = ANNOTATIONS_DIR / "indexed"

NAMES   = ["detections", "expression", "gaze", "proxemics",
           "social_gaze", "speech_emotion", "vocalsound",
           "diarization", "context_emotion"]
SUBSETS = ["youtube", "ego4d"]


def index_file(name: str, subset: str):
    src = ANNOTATIONS_DIR / f"{name}_{subset}.json"
    if not src.exists():
        print(f"  skip {src.name} (not found)")
        return

    out_dir = INDEX_DIR / f"{name}_{subset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check how many already done
    existing = set(p.stem for p in out_dir.glob("*.json"))

    print(f"  loading {src.name} ({src.stat().st_size // 1_000_000} MB)…", flush=True)
    data = json.loads(src.read_text())
    total = len(data)
    skipped = 0

    for i, (utt_key, val) in enumerate(data.items()):
        fname = utt_key.replace("/", "__")
        if fname in existing:
            skipped += 1
            continue
        (out_dir / f"{fname}.json").write_text(json.dumps(val))
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{total}", flush=True)

    print(f"  done — {total - skipped} written, {skipped} skipped")


if __name__ == "__main__":
    for subset in SUBSETS:
        for name in NAMES:
            print(f"[{name}] {subset}")
            index_file(name, subset)
    print("All done.")
