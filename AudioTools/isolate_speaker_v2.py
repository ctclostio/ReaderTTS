import os
import argparse
import subprocess
import torch
import torchaudio

# Force soundfile backend and Monkeypatch AudioDecoder BEFORE other imports
try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass

import sys
from dataclasses import dataclass
import torchaudio

# --- Runtime Patch for Pyannote on Windows (Missing torchcodec) ---
# Pyannote 4.x relies on torchcodec, which fails on Windows. 
# We inject a mock AudioDecoder that uses torchaudio/soundfile instead.

@dataclass
class MockMetadata:
    sample_rate: int
    duration_seconds_from_header: float

@dataclass
class MockSamples:
    data: torch.Tensor
    sample_rate: int

class MockAudioDecoder:
    def __init__(self, path):
        self.path = str(path)
        self.info = torchaudio.info(self.path)
        
    @property
    def metadata(self):
        return MockMetadata(
            sample_rate=self.info.sample_rate,
            duration_seconds_from_header=self.info.num_frames / self.info.sample_rate
        )
        
    def get_all_samples(self):
        wav, sr = torchaudio.load(self.path)
        return MockSamples(data=wav, sample_rate=sr)
        
    def get_samples_played_in_range(self, start, end):
        # Calculate frames
        sr = self.info.sample_rate
        frame_start = int(start * sr)
        num_frames = int((end - start) * sr)
        
        wav, sr = torchaudio.load(self.path, frame_offset=frame_start, num_frames=num_frames)
        return MockSamples(data=wav, sample_rate=sr)

# Inject into sys.modules or patch pyannote.audio.core.io after import
import pyannote.audio.core.io
pyannote.audio.core.io.AudioDecoder = MockAudioDecoder
pyannote.audio.core.io.AudioStreamMetadata = MockMetadata
# Also inject torchcodec if it completely failed to import
if "torchcodec" not in sys.modules:
    import types
    torchcodec = types.ModuleType("torchcodec")
    sys.modules["torchcodec"] = torchcodec
    torchcodec.decoders = types.ModuleType("decoders")
    torchcodec.decoders.AudioDecoder = MockAudioDecoder
    
print("Applied Runtime Patch: AudioDecoder -> torchaudio fallback")
# ------------------------------------------------------------------

import shutil
from pathlib import Path
from pyannote.audio import Pipeline
from speechbrain.inference.speaker import SpeakerRecognition
from tqdm import tqdm

# --- Configuration ---
# Pyannote pipeline requires an auth token. 
# It usually looks for "HF_TOKEN" env var, or you can pass use_auth_token=True if logged in via `huggingface-cli login`.
PIPELINE_NAME = "pyannote/speaker-diarization-3.1"

def load_verification_model():
    print("Loading SpeechBrain SpeakerRecognition model (for identification)...")
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="tmp_model",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

def compute_average_anchor(model, anchor_path):
    print(f"Computing average anchor from: {anchor_path}")
    anchor_path = Path(anchor_path)
    embeddings = []
    
    files = []
    if anchor_path.is_dir():
        files = list(anchor_path.glob("*.wav")) + list(anchor_path.glob("*.mp3"))
    else:
        files = [anchor_path]
        
    for f in tqdm(files, desc="Processing Anchor Files"):
        try:
            sig, fs = torchaudio.load(f)
            # ECAPA-VoxCeleb expects 16k
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(fs, 16000)
                sig = resampler(sig)
            
            # Mix to mono
            if sig.shape[0] > 1:
                sig = sig.mean(dim=0, keepdim=True)
                
            emb = model.encode_batch(sig)
            embeddings.append(emb.squeeze(1))
        except Exception as e:
            print(f"Skipping anchor {f}: {e}")
            
    if not embeddings: return None
        
    emb_stack = torch.cat(embeddings, dim=0)
    mean_emb = emb_stack.mean(dim=0, keepdim=True)
    return torch.nn.functional.normalize(mean_emb, p=2, dim=1)

def run_demucs(input_path, output_dir):
    """Runs Demucs to separate vocals."""
    print(f"Running Demucs on {input_path}...")
    # demucs -n htdemucs --two-stems vocals -o output_dir input_path
    # Expected output: output_dir/htdemucs/{filename}/vocals.wav
    filename = Path(input_path).stem
    vocals_path = Path(output_dir) / "htdemucs" / filename / "vocals.wav"
    
    if vocals_path.exists():
        print(f"Demucs output found at {vocals_path}. Skipping separation.")
        return vocals_path

    cmd = [
        "demucs",
        "-n", "htdemucs",
        "--two-stems", "vocals",
        "-o", str(output_dir),
        str(input_path)
    ]
    subprocess.run(cmd, check=True)
    
    # Expected output: output_dir/htdemucs/{filename}/vocals.wav
    filename = Path(input_path).stem
    vocals_path = Path(output_dir) / "htdemucs" / filename / "vocals.wav"
    
    if not vocals_path.exists():
        raise FileNotFoundError(f"Demucs output not found at {vocals_path}")
        
    return vocals_path

def isolate_speaker_v2(target_path, anchor_path, output_path, hf_token=None, clean_audio=False):
    target_file = Path(target_path)
    output_file = Path(output_path)
    
    # 0. Optional: Clean Audio with Demucs
    if clean_audio:
        print("--- Pre-processing: Demucs Cleaning ---")
        temp_demucs_dir = Path("AudioTools/temp_demucs")
        temp_demucs_dir.mkdir(parents=True, exist_ok=True)
        try:
            cleaned_vocals = run_demucs(target_file, temp_demucs_dir)
            print(f"Using cleaned vocals: {cleaned_vocals}")
            # Switch target file to the cleaned vocals
            target_file = cleaned_vocals
        except Exception as e:
            print(f"Demucs cleaning failed: {e}")
            print("Falling back to original file...")
    
    # 1. Load Diarization Pipeline
    print(f"Loading Diarization Pipeline: {PIPELINE_NAME}...")
    try:
        pipeline = Pipeline.from_pretrained(
            PIPELINE_NAME, 
            token=hf_token if hf_token else True
        )
    except Exception as e:
        print("\nFATAL ERROR: Could not load Pyannote pipeline.")
        print(f"Error: {e}")
        print("Please ensure you have accepted the user agreement on HuggingFace for `pyannote/speaker-diarization-3.1`")
        print("And that you have a valid HF_TOKEN set or passed.")
        return

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pipeline.to(device)

    # 2. Run Diarization
    print(f"Diarizing {target_file}...")
    # REMOVED try/except to see full traceback
    diarization = pipeline(str(target_file))

    # 3. Load Verification Model (SpeechBrain)
    spk_model = load_verification_model()
    anchor_emb = compute_average_anchor(spk_model, anchor_path)
    
    # 4. Process Speakers
    # Handle Pyannote 4.0 DiarizeOutput wrapper
    annotation = diarization
    if hasattr(diarization, "speaker_diarization"):
        annotation = diarization.speaker_diarization
    
    # Group segments by speaker
    speaker_segments = {}
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append((turn.start, turn.end))
        
    print(f"Found {len(speaker_segments)} speakers: {list(speaker_segments.keys())}")
    
    # Load Main Audio
    wav, fs = torchaudio.load(target_file)
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(fs, 16000)
        wav = resampler(wav)
        fs = 16000
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze() # [T]

    best_score = -1.0
    best_speaker = None
    
    # Evaluate each speaker
    for speaker, segments in speaker_segments.items():
        # Collect audio for this speaker to test
        # We'll take the longest segments (up to 30s total) to verify identity
        # This avoids processing the whole track for verification if not needed
        
        # Calculate total duration
        total_dur = sum([end - start for start, end in segments])
        
        # Extract audio (concatenate all pieces for robustness?)
        # Let's simple concatenate all pieces
        speaker_audio_pieces = []
        for start, end in segments:
            s_frame = int(start * fs)
            e_frame = int(end * fs)
            speaker_audio_pieces.append(wav[s_frame:e_frame])
            
        full_speaker_wav = torch.cat(speaker_audio_pieces)
        
        # Get embedding
        # Chunk logic might be needed if too long, but ECAPA handles variable length well.
        # Let's limit to first 60 seconds for identity check to save memory/time if track is huge
        check_wav = full_speaker_wav[:16000*60] 
        
        emb = spk_model.encode_batch(check_wav.unsqueeze(0)).squeeze(1)
        score = torch.nn.functional.cosine_similarity(emb, anchor_emb).item()
        
        print(f"Speaker {speaker}: {total_dur:.1f}s | Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_speaker = speaker

    # 5. Export Best Speaker
    if best_speaker and best_score > 0.40: # 0.4 is a safe bet for ECAPA-VoxCeleb positive match
        print(f"Selected Target Speaker: {best_speaker} (Score: {best_score:.4f})")
        
        # Re-extract all audio for this speaker
        final_pieces = []
        # sort segments by time just in case
        segments = sorted(speaker_segments[best_speaker], key=lambda x: x[0])
        
        for start, end in segments:
            s_frame = int(start * fs)
            e_frame = int(end * fs)
            final_pieces.append(wav[s_frame:e_frame])
            
        final_wav = torch.cat(final_pieces).unsqueeze(0)
        torchaudio.save(output_file, final_wav, 16000)
        print(f"Saved to: {output_file}")
    else:
        print(f"No matching speaker found! Best score was {best_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_file", help="Path to input audio")
    parser.add_argument("--anchor", required=True, help="Path to anchor file or directory")
    parser.add_argument("--output", required=True, help="Path to output wav")
    parser.add_argument("--token", help="HuggingFace Auth Token", default=None)
    parser.add_argument("--clean", action="store_true", help="Run Demucs to isolate vocals before processing")
    
    args = parser.parse_args()
    
    isolate_speaker_v2(args.target_file, args.anchor, args.output, args.token, args.clean)
