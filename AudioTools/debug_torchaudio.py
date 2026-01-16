import torch
import torchaudio

print(f"Torch: {torch.__version__}")
print(f"Torchaudio: {torchaudio.__version__}")

try:
    print("Available backends:", torchaudio.list_audio_backends())
except Exception as e:
    print(f"list_audio_backends failed: {e}")

try:
    import soundfile
    print(f"Soundfile version: {soundfile.__version__}")
except ImportError:
    print("Soundfile not installed")

try:
    from torchaudio.io import AudioDecoder
    print("AudioDecoder imported successfully")
except ImportError:
    print("AudioDecoder import failed")
except NameError:
    print("AudioDecoder NameError")
except Exception as e:
    print(f"AudioDecoder error: {e}")
