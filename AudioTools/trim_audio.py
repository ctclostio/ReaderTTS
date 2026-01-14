import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

INPUT_DIR = Path("SampleAudio/Refined_Kratos")
OUTPUT_DIR = Path("SampleAudio/Trimmed_Kratos")

def trim_and_split(file_path, output_base):
    print(f"Processing {file_path.name}...")
    y, sr = librosa.load(file_path, sr=None)
    
    # 1. Trim leading/trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    
    # 2. Split on silence to remove internal dead air
    # top_db=30 is a reasonable default for speech
    intervals = librosa.effects.split(y_trimmed, top_db=30, frame_length=2048, hop_length=512)
    
    file_idx = 0
    for start, end in intervals:
        segment = y_trimmed[start:end]
        
        # Filter out very short segments (e.g. < 0.5s) that might be just breath/noise
        if len(segment) / sr < 0.5:
            continue
            
        out_name = output_base / f"{file_path.stem}_seg{file_idx}.wav"
        
        # Verify segment isn't too quiet overall
        rms = np.sqrt(np.mean(segment**2))
        if rms < 0.01: # Skip near-silent chunks
             continue
             
        sf.write(str(out_name), segment, sr)
        file_idx += 1
    
    print(f" -> Created {file_idx} segments.")

def main():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
        
    files = list(INPUT_DIR.glob("*.wav"))
    if not files:
        print(f"No files found in {INPUT_DIR}")
        return

    print(f"Found {len(files)} files to process.")
    
    for f in files:
        trim_and_split(f, OUTPUT_DIR)

    print("\nDone! Trimmed clips are in SampleAudio/Trimmed_Kratos")

if __name__ == "__main__":
    main()
