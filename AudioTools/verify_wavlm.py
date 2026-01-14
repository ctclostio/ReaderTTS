import os
import shutil
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("SampleAudio/Trimmed_Kratos")
REJECT_DIR = Path("SampleAudio/Rejected")
MODEL_NAME = "microsoft/wavlm-base-plus-sv"

# Similarity Threshold
# WavLM SV outputs cosine similarity. 
# Increasing threshold to 0.75 for strict filtering.
THRESHOLD = 0.75

def main():
    if not REJECT_DIR.exists():
        REJECT_DIR.mkdir(parents=True)
        
    print(f"Loading WavLM Model: {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = WavLMForXVector.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"Model loaded on {device}.")
    
    files = list(INPUT_DIR.glob("*.wav"))
    if not files:
        print("No files found!")
        return

    # 1. Establish Ground Truth (Anchors)
    # Round 4: Using "Kratos_Anchor" (Read It Boy) as the sole ground truth.
    anchors = [f for f in files if "Kratos_Anchor" in f.name]
    if not anchors:
        print("Error: No 'Kratos_Anchor' clips found.")
        return

    print(f"Computing Anchor Embeddings ({len(anchors)} clips)...")
    anchor_embeddings = []
    
    with torch.no_grad():
        for anchor in anchors:
            wav, sr = torchaudio.load(anchor)
            # Resample if needed (WavLM expects 16k usually)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000).to(wav.device)
                wav = resampler(wav)
            
            inputs = feature_extractor(wav.squeeze().numpy(), return_tensors="pt", sampling_rate=16000)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            emb = model(**inputs).embeddings # [1, 512]
            anchor_embeddings.append(emb)

    # Compute Centroid (Average Embedding)
    if not anchor_embeddings:
         print("Failed to compute anchor embeddings.")
         return
         
    anchor_stack = torch.cat(anchor_embeddings, dim=0)
    centroid = torch.mean(anchor_stack, dim=0, keepdim=True)
    
    # 2. Scan All Files
    print(f"Scanning {len(files)} files against Ground Truth...")
    rejected = 0
    scores = []
    
    with torch.no_grad():
        for f in tqdm(files):
            # Skip checking anchors against themselves (they will pass)
            
            try:
                wav, sr = torchaudio.load(f)
                if wav.shape[1] < 1000: # Skip extremely short
                    continue
                    
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    wav = resampler(wav)
                
                inputs = feature_extractor(wav.squeeze().numpy(), return_tensors="pt", sampling_rate=16000)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                emb = model(**inputs).embeddings
                
                # Cosine Similarity
                score = torch.nn.functional.cosine_similarity(emb, centroid).item()
                scores.append((f.name, score))
                
                if score < THRESHOLD:
                    # Move to reject
                    # print(f"REJECT: {f.name} (Score: {score:.3f})")
                    shutil.move(str(f), str(REJECT_DIR / f.name))
                    rejected += 1
            except Exception as e:
                print(f"Error processing {f.name}: {e}")
                
    print(f"\nVerification Complete.")
    print(f"Rejected: {rejected}")
    print(f"Kept: {len(files) - rejected}")
    
    # Optional: Print lowest scores of KEPT files to see boundary
    scores.sort(key=lambda x: x[1])
    print("\nLowest scores (borderline kept):")
    for name, score in scores[:10]:
        if score >= THRESHOLD:
            print(f"{name}: {score:.3f}")

if __name__ == "__main__":
    main()
