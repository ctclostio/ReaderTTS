import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition
from pathlib import Path
from tqdm import tqdm

# Load Model
MODEL_DIR = Path("tmp_model")
spk_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir=MODEL_DIR,
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

def get_embedding(wav_tensor):
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
    return spk_model.encode_batch(wav_tensor).squeeze(1)

def compute_anchor(anchor_path):
    print(f"Loading anchor: {anchor_path}")
    files = list(Path(anchor_path).glob("*.wav"))
    embeddings = []
    for f in files:
        w, fs = torchaudio.load(f)
        if fs != 16000:
            w = torchaudio.transforms.Resample(fs, 16000)(w)
        embeddings.append(get_embedding(w))
    
    emb_stack = torch.cat(embeddings, dim=0)
    mean_emb = emb_stack.mean(dim=0, keepdim=True)
    return torch.nn.functional.normalize(mean_emb, p=2, dim=1)

def analyze(target_path, anchor_path):
    anchor = compute_anchor(anchor_path)
    
    # Load VAD
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    (get_speech_timestamps, _, _, _, _) = utils
    
    wav, fs = torchaudio.load(target_path)
    if fs != 16000:
        wav = torchaudio.transforms.Resample(fs, 16000)(wav)
    wav_mono = wav.mean(dim=0, keepdim=True).squeeze()
    
    timestamps = get_speech_timestamps(wav_mono, model, sampling_rate=16000, threshold=0.5, min_speech_duration_ms=400)
    
    print(f"\n--- Analyzing first 10 segments of {Path(target_path).name} ---")
    
    for i, ts in enumerate(timestamps[:10]):
        seg = wav_mono[ts['start']:ts['end']]
        emb = get_embedding(seg)
        score = torch.nn.functional.cosine_similarity(emb, anchor).item()
        
        duration = len(seg) / 16000
        print(f"Seg {i}: {duration:.2f}s | Score: {score:.4f} | {'[LIKELY MATCH]' if score > 0.5 else '[REJECT]'}")

def check_anchors(anchor_path):
    print(f"\n--- Checking Anchor Consistency in {anchor_path} ---")
    files = list(Path(anchor_path).glob("*.wav"))
    embeddings = []
    names = []
    
    for f in files:
        w, fs = torchaudio.load(f)
        if w.shape[0] > 1: w = w.mean(dim=0, keepdim=True)
        if fs != 16000: w = torchaudio.transforms.Resample(fs, 16000)(w)
        embeddings.append(get_embedding(w))
        names.append(f.name)
        
    # Check each against mean of others
    bad_files = []
    for i in range(len(embeddings)):
        others = torch.cat(embeddings[:i] + embeddings[i+1:], dim=0)
        mean_others = others.mean(dim=0, keepdim=True)
        mean_others = torch.nn.functional.normalize(mean_others, p=2, dim=1)
        
        score = torch.nn.functional.cosine_similarity(embeddings[i], mean_others).item()
        print(f"{names[i]}: Score vs Others = {score:.4f} {'[SUSPICIOUS]' if score < 0.6 else ''}")
        
        if score < 0.6:
            bad_files.append(names[i])
            
    return bad_files

if __name__ == "__main__":
    check_anchors("SampleAudio/Trump_KnownGood")
    # analyze("SampleAudio/Trump/In_conversation_with_President_Trump.wav", "SampleAudio/Trump_KnownGood")
