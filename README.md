# ReaderTTS: Kratos Edition (VibeVoice)

A highly-tuned TTS pipeline designed to clone the voice of **Kratos (Christopher Judge)** using the **VibeVoice** engine.
This implementation features an "Exact Slice" audio sourcing strategy for maximum character fidelity and reduced hallucinations.

## 🚀 Quick Start

### 1. Environment Setup
The project runs in a specialized virtual environment.
```powershell
& .\VibeVoice_new\venv_vibevoice_new\Scripts\Activate.ps1
```

### 2. Audio Sourcing ("Exact Slice" Strategy)
Instead of using hundreds of random clips, we use **surgical precision**. We download specific time segments from high-quality sources to ensure a "Pure" voice prompt.

Use the custom downloader tool:
```powershell
# Format: python AudioTools/download_exact.py <URL> <START_TIME> <END_TIME> <OUTPUT_NAME>
python AudioTools/download_exact.py https://www.youtube.com/watch?v=Whsq6npp8rM 00:00 00:21 segment_1
```
*Current optimized clips are stored in `SampleAudio/Exact_Reference`.*

### 3. Audio Processing Pipeline (Demucs + Pyannote)
To prepare high-quality datasets effectively, we use a two-stage pipeline:

**Step 1: Install Dependencies**
```powershell
pip install pyannote.audio demucs ffmpeg-python webrtcvad-wheels
```

**Step 2: Isolate & Clean**
Use the `isolate_speaker_v2.py` script. This automatically runs **Demucs** to remove background noise/music (extracting vocals), then uses **Pyannote 3.1** to identify and isolate the target speaker.

```powershell
# --clean: Enables Demucs pre-processing (Recommended for rallies/interviews)
# --anchor: Path to "Known Good" reference audio (for identification)
# --token: HuggingFace Token (required for Pyannote)
python AudioTools/isolate_speaker_v2.py "RawAudio/Interview.wav" --anchor "SampleAudio/Trump_KnownGood" --output "SampleAudio/Trump_Clean.wav" --clean --token "YOUR_HF_TOKEN"
```

## 🎙️ Generation Commands

### Option A: VibeVoice 1.5B (Standard)
**Best for**: Speed, Lower VRAM (requires ~6GB), Rapid Prototyping.
**Default Model**: `microsoft/VibeVoice-1.5b`

```powershell
python main.py input.txt --ref_audio SampleAudio/Exact_Reference --engine vibevoice --model_name "microsoft/VibeVoice-1.5b" --nfe_step 30 --cfg_strength 3.0 --output output_1.5b.wav
```

### Option B: VibeVoice 7B (High Capability)
**Best for**: Maximum nuance, Complex prosody.
**Requirements**: ~14GB VRAM (Weights) + ~4GB (Context) = ~18GB Total on 24GB card.
**Context Limit**: ~100s-200s max on 24GB.
**Recommended Model**: `aoi-ot/VibeVoice-7B` (Community mirror).

```powershell
python main.py input.txt --ref_audio SampleAudio/Exact_Reference --engine vibevoice --model_name "aoi-ot/VibeVoice-7B" --nfe_step 30 --cfg_strength 3.0 --output output_7b.wav
```

## ⚙️ Key Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cfg_strength` | `1.5` | Guidance strength. increased to **3.0** for Kratos to force strict tone adherence. |
| `--nfe_step` | `30` | Inference steps. Higher quality, slower generation. |
| `--model_name` | `1.5b` | Switch between `microsoft/VibeVoice-1.5b` and `aoi-ot/VibeVoice-7B`. |
| `--ref_audio` | Required | Path to your clean reference clips folder. |

## 🧠 Technical Notes
- **Context Window**: The engine is tuned to use up to **100 seconds** of reference audio. If you provide more, it uses a "Dynamic Shuffle" to pick a random 100s subset, preventing hallucinations.
- **7B Optimization**: The 7B model automatically loads in Half Precision (`float16`) to prevent OOM errors on consumer GPUs (3090/4090).
