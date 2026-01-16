from pyannote.audio import Pipeline
import sys

import os
TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")

def try_load(model):
    print(f"Attempting to load {model}...")
    try:
        pipeline = Pipeline.from_pretrained(model, token=TOKEN)
        print(f"SUCCESS: Loaded {model}")
    except Exception as e:
        print(f"FAILED: {model}")
        print(f"Error: {e}")

if __name__ == "__main__":
    try_load("pyannote/speaker-diarization-3.1")
    try_load("pyannote/speaker-diarization-3.0")
