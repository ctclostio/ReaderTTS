$VenvPath = "AudioTools\venv_processing"
$Requirements = "numpy>=2.0 pyannote.audio demucs ffmpeg-python webrtcvad-wheels torchaudio"

# 1. Create Venv if missing
if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating Processing Venv at $VenvPath..."
    python -m venv $VenvPath
}

# 2. Activate and Install
& "$VenvPath\Scripts\python" -m pip install --upgrade pip
# Install packages explicitly
& "$VenvPath\Scripts\python" -m pip install "numpy>=2.0" "pyannote.audio" "demucs" "ffmpeg-python" "webrtcvad-wheels" "torchaudio" "speechbrain" --extra-index-url https://download.pytorch.org/whl/cu121

# 3. Validation
$NpVer = & "$VenvPath\Scripts\python" -c "import numpy; print(numpy.__version__)"
Write-Host "Processing Environment Ready. NumPy Version: $NpVer"
