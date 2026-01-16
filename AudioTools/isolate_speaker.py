import os
import shutil
import argparse
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm
import speechbrain.utils.fetching as sb_fetching

# --- Monkeypatch SpeechBrain to fix Windows Symlink Error ---
def patched_link_with_strategy(src, dst, local_strategy):
    """Force copy strategy to avoid WinError 1314"""
    if local_strategy == sb_fetching.LocalStrategy.NO_LINK:
        return src
        
    src = Path(src).absolute()
    dst = Path(dst).absolute()
    
    if src == dst:
        return dst
        
    # Always copy
    if dst.exists():
        dst.unlink()
    shutil.copy(str(src), str(dst))
    return dst

sb_fetching.link_with_strategy = patched_link_with_strategy

# --- Configuration ---
MODEL_DIR = Path("tmp_model")

def load_model():
    print("Loading SpeechBrain SpeakerRecognition model...")
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir=MODEL_DIR,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

def load_vad_model():
    print("Loading Silero VAD model...")
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=False)
    return model, utils

def get_embedding(model, wav_tensor, sample_rate=16000):
    # Wav tensor should be [1, T] or [T]
    # ECAPA-VoxCeleb expects 16k usually. SpeechBrain handles resampling if you use their `load_audio` but we are using torchaudio tensor.
    # We must ensure it's 16k for the model if passing tensor directly? 
    # Actually SpeechBrain's `encode_batch` handles it if we follow their pipeline, but safe to resample.
    
    # We will assume input is already resampled to 16k for consistency
    
    if wav_tensor.dim() == 2 and wav_tensor.shape[0] > 1:
        # Mix to mono
        wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
        
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0) # [1, T]

    emb = model.encode_batch(wav_tensor)
    if len(emb.shape) == 3:
        emb = emb.squeeze(1)
    return emb

def compute_average_anchor(model, anchor_path):
    print(f"Computing average anchor from: {anchor_path}")
    anchor_path = Path(anchor_path)
    embeddings = []
    
    files = []
    if anchor_path.is_dir():
        files = list(anchor_path.glob("*.wav")) + list(anchor_path.glob("*.mp3"))
    else:
        files = [anchor_path]
        
    if not files:
        print("No anchor files found!")
        return None

    for f in tqdm(files, desc="Processing Anchor Files"):
        try:
            # Load and resample to 16k for the model
            sig, fs = torchaudio.load(f)
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(fs, 16000)
                sig = resampler(sig)
            
            emb = get_embedding(model, sig)
            embeddings.append(emb)
        except Exception as e:
            print(f"Skipping anchor {f}: {e}")
            
    if not embeddings:
        return None
        
    # Stack and Mean
    # embeddings are [1, 192]
    emb_stack = torch.cat(embeddings, dim=0) # [N, 192]
    mean_emb = emb_stack.mean(dim=0, keepdim=True) # [1, 192]
    
    # Normalize? Cosine sim usually handles direction, but normalizing is good practice
    mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=1)
    
    print(f"Computed average anchor from {len(embeddings)} clips.")
    return mean_emb

def isolate_speaker(target_path, anchor_path, output_path, threshold=0.55):
    target_file = Path(target_path)
    output_file = Path(output_path)
    
    # 1. Load Models
    spk_model = load_model()
    vad_model, vad_utils = load_vad_model()
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils

    # 2. Compute Anchor
    anchor_emb = compute_average_anchor(spk_model, anchor_path)
    if anchor_emb is None:
        print("Failed to compute anchor.")
        return

    # 3. Load Target Audio
    print(f"Loading target: {target_file}")
    wav, fs = torchaudio.load(target_file)
    
    # Mix to mono for VAD
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    # Resample to 16k for VAD + SPK
    if fs != 16000:
        print("Resampling to 16kHz...")
        resampler = torchaudio.transforms.Resample(fs, 16000)
        wav = resampler(wav)
        fs = 16000

    wav = wav.squeeze() # VAD expects [T]
    
    print("Running VAD (Voice Activity Detection)...")
    # Stricter VAD settings
    speech_timestamps = get_speech_timestamps(
        wav, 
        vad_model, 
        sampling_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=400, # Increased from default 250
        min_silence_duration_ms=500
    )
    
    if not speech_timestamps:
        print("No speech detected by VAD.")
        return

    print(f"Detected {len(speech_timestamps)} speech segments.")
    
    # --- Adaptive Anchor Strategy ---
    pass1_kept = []
    pass1_embeddings = []
    
    print("Pass 1: Identifying high-confidence segments for Session Anchor...")
    high_conf_threshold = 0.75
    
    for ts in tqdm(speech_timestamps, desc="Pass 1 Analysis"):
        start = ts['start']
        end = ts['end']
        segment = wav[start:end]
        
        if len(segment) < 16000: continue # Skip < 1s
        
        seg_tensor = segment.unsqueeze(0)
        try:
            emb = get_embedding(spk_model, seg_tensor)
            score = torch.nn.functional.cosine_similarity(emb, anchor_emb).item()
            
            if score >= high_conf_threshold:
                pass1_kept.append(segment)
                pass1_embeddings.append(emb)
        except Exception as e:
            continue
            
    session_anchor = None
    if pass1_embeddings:
        print(f"Found {len(pass1_embeddings)} high-confidence segments. Computing Session Anchor...")
        emb_stack = torch.cat(pass1_embeddings, dim=0)
        session_anchor = emb_stack.mean(dim=0, keepdim=True)
        session_anchor = torch.nn.functional.normalize(session_anchor, p=2, dim=1)
    else:
        print("Warning: No high-confidence segments found. Modeling from global anchor only (less accurate for specific sessions).")
        session_anchor = anchor_emb
        
    # --- Pass 2: Filtering with Session Anchor ---
    kept_audio = []
    # Slightly lower threshold for session anchor match since it's exact-match
    final_threshold = threshold if session_anchor is anchor_emb else (threshold + 0.05) 
    
    print(f"Pass 2: Filtering using Session Anchor (Threshold: {final_threshold:.2f})...")
    
    for ts in tqdm(speech_timestamps, desc="Final Filtering"):
        start = ts['start']
        end = ts['end']
        segment = wav[start:end]
        
        if len(segment) < 12000: continue # Skip very short snippets in final pass too (<0.75s)
        
        seg_tensor = segment.unsqueeze(0)
        try:
            emb = get_embedding(spk_model, seg_tensor)
            score = torch.nn.functional.cosine_similarity(emb, session_anchor).item()
            
            if score >= final_threshold:
                kept_audio.append(segment)
        except Exception:
            continue

    if kept_audio:
        print(f"Kept {len(kept_audio)} / {len(speech_timestamps)} segments.")
        print("Concatenating and saving...")
        final_wav = torch.cat(kept_audio, dim=0).unsqueeze(0) # [1, T_total]
        
        torchaudio.save(output_file, final_wav, 16000)
        print(f"Saved to: {output_file}")
    else:
        print("No segments matched the speaker profile.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_file", help="Path to input audio")
    parser.add_argument("--anchor", required=True, help="Path to anchor file or directory")
    parser.add_argument("--output", required=True, help="Path to output wav")
    parser.add_argument("--threshold", type=float, default=0.60, help="Similarity threshold (Default 0.60)")
    
    args = parser.parse_args()
    
    isolate_speaker(args.target_file, args.anchor, args.output, args.threshold)
