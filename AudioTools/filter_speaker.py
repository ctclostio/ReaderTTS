import os
import shutil
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("SampleAudio/Trimmed_Kratos")
REJECT_DIR = Path("SampleAudio/Rejected")

# Threshold for cosine similarity (ECAPA-VoxCeleb typical threshold is ~0.25 for verification)
# We'll use a slightly looser one to avoid deleting "acting" voices (shouting/whispering) which shift the embedding.
THRESHOLD = 0.20 

def main():
    if not REJECT_DIR.exists():
        REJECT_DIR.mkdir(parents=True)
        
    print("Downloading/Loading Model...")
    # This will download the model to ~/.cache/speechbrain ideally
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="tmp_model"
    )
    print("Model loaded successfully.")
    
    files = list(INPUT_DIR.glob("*.wav"))
    if not files:
        print("No files found!")
        return

    print(f"Found {len(files)} files to process.")

    # 1. Establish Ground Truth
    # We use "Real_Life_Judge" clips as the anchor because they are clean studio recordings of the actor.
    anchors = [f for f in files if "Real_Life_Judge" in f.name]
    if not anchors:
        print("Error: No 'Real_Life_Judge' clips found to use as anchors. Using first 5 clips instead.")
        anchors = files[:5]
        
    print(f"Using {len(anchors)} anchor clips to build speaker profile.")
    
    anchor_embeddings = []
    for anchor in anchors:
        signal, fs = torchaudio.load(anchor)
        emb = verification.encode_batch(signal)
        anchor_embeddings.append(emb)
    
    # Average embedding not strictly supported by interface cleanly, 
    # so we will compare against ALL anchors and take the MAX score (best match to any anchor).
    # This handles variety (whisper vs shout) better than an average vector.
    
    print(f"Scanning {len(files)} files...")
    
    rejected_count = 0
    scores = []
    
    for f in tqdm(files):
        signal, fs = torchaudio.load(f)
        
        # Verify against all anchors and take the max similarity
        # (Is it similar to ANY of the known good clips?)
        emb = verification.encode_batch(signal)
        
        max_score = -1.0
        for anchor_emb in anchor_embeddings:
            # Cosine similarity
            score = torch.nn.functional.cosine_similarity(emb, anchor_emb).item()
            if score > max_score:
                max_score = score
        
        scores.append((f, max_score))
        
        if max_score < THRESHOLD:
            print(f"REJECT: {f.name} (Score: {max_score:.3f})")
            shutil.move(str(f), str(REJECT_DIR / f.name))
            rejected_count += 1
            
    print(f"\nScan Complete.")
    print(f"Rejected: {rejected_count}")
    print(f"Kept: {len(files) - rejected_count}")
    print(f"Rejected files moved to {REJECT_DIR}")

if __name__ == "__main__":
    main()
