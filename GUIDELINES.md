# ReaderTTS: Kratos VibeVoice Edition

This project is tuned to generate high-quality Kratos voice lines using the **VibeVoice** engine and an **"Exact Slice"** audio sourcing strategy.

## 1. Environment Setup
Always activate the specialized VibeVoice environment before running commands:
```powershell
& .\VibeVoice_new\venv_vibevoice_new\Scripts\Activate.ps1
```

## 2. Managing Reference Audio (The "Exact Slice" Method)
We abandoned bulk downloading in favor of surgical precision. We download exact time segments from high-quality sources to ensure "Pure Kratos" without background music or secondary characters.

### Tool: `download_exact.py`
Located in `AudioTools/`, this script downloads specific timestamps from YouTube without re-encoding.

**Usage:**
```powershell
python AudioTools/download_exact.py <YouTube_URL> <START_TIME> <END_TIME> [OUTPUT_FILENAME]
```

**Example:**
```powershell
python AudioTools/download_exact.py https://www.youtube.com/watch?v=Whsq6npp8rM 00:00 00:21 segment_1
```

**Current Reference Location:**
`SampleAudio/Exact_Reference`

## 3. Generating Audio
The main engine (`main.py`) has been refactored to focus exclusively on VibeVoice.

**Command:**
```powershell
python main.py <INPUT_TEXT_FILE> --ref_audio SampleAudio/Exact_Reference --engine vibevoice --nfe_step 30 --cfg_strength 3.0 --output <OUTPUT_FILE>
```

### Key Parameters:
- **`--ref_audio`**: Path to the folder containing your clean reference clips.
- **`--nfe_step 30`**: Number of inference steps. Higher = better quality (diminishing returns > 30).
- **`--cfg_strength 3.0`**: Classifier-Free Guidance.
    - `1.5`: Loose, more expressive, less stable.
    - `3.0`: Strict adherence to Kratos' voice tone. **Recommended**.
    - `4.0+`: Very rigid, potential artifacts.

## 4. Context Window & Memory
- **Limit**: The engine is configured to use a maximum of **100 seconds** of reference audio effectively.
- **Behavior**: If your `Exact_Reference` folder contains more than 100s of audio, the engine will **randomly shuffle** and pick a subset to fill the 100s window. This prevents "hallucinations" (repeating text) and GPU crashes.
- **VRAM**: This setting is tuned for 24GB VRAM.

## 5. Directory Structure
- `AudioTools/`: Scripts for sourcing audio.
- `SampleAudio/`: Stores reference material.
- `legacy_tts_tools/`: Archive of old F5-TTS/XTTS engines and scripts.
