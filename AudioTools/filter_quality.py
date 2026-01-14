import os
import shutil
import librosa
import numpy as np
from pathlib import Path

INPUT_DIR = Path("SampleAudio/Trimmed_Kratos")
REJECT_DIR = Path("SampleAudio/Rejected")

def main():
    if not REJECT_DIR.exists():
        REJECT_DIR.mkdir(parents=True)
        
    print("Starting Quality Filter (removing noise/short clips)...")
    
    files = list(INPUT_DIR.glob("*.wav"))
    if not files:
        print("No files found!")
        return

    print(f"Scanning {len(files)} files...")
    
    rejected_count = 0
    
    for f in files:
        try:
            # fast load duration
            duration = librosa.get_duration(path=f)
            
            # 1. Length Filter
            # Anything under 0.5s is rarely a useful phoneme for cloning and often just a click/breath
            if duration < 0.5:
                print(f"REJECT (Too Short {duration:.2f}s): {f.name}")
                shutil.move(str(f), str(REJECT_DIR / f.name))
                rejected_count += 1
                continue
                
            # 2. RMS (Volume) Filter is already done in trimming, but double check
            y, sr = librosa.load(f, sr=None)
            rms = np.sqrt(np.mean(y**2))
            
            if rms < 0.01:
                print(f"REJECT (Too Quiet {rms:.4f}): {f.name}")
                shutil.move(str(f), str(REJECT_DIR / f.name))
                rejected_count += 1
                continue
                
        except Exception as e:
            print(f"Error processing {f.name}: {e}")

    print(f"\nFilter Complete.")
    print(f"Rejected: {rejected_count}")
    print(f"Kept: {len(files) - rejected_count}")
    print(f"Rejected files moved to {REJECT_DIR}")

if __name__ == "__main__":
    main()
