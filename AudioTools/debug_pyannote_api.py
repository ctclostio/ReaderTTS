import torch
import torchaudio
import sys

# --- Runtime Patch for Pyannote on Windows ---
try:
    import torchaudio.io
    if not hasattr(torchaudio.io, "AudioDecoder"):
        class MockAudioDecoder:
            pass
        torchaudio.io.AudioDecoder = MockAudioDecoder
    torchaudio.set_audio_backend("soundfile")
except:
    pass

from dataclasses import dataclass
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
        return MockMetadata(self.info.sample_rate, self.info.num_frames / self.info.sample_rate)
    def get_all_samples(self):
        wav, sr = torchaudio.load(self.path)
        return MockSamples(wav, sr)
    def get_samples_played_in_range(self, start, end):
        sr = self.info.sample_rate
        frame_start = int(start * sr)
        num_frames = int((end - start) * sr)
        wav, sr = torchaudio.load(self.path, frame_offset=frame_start, num_frames=num_frames)
        return MockSamples(wav, sr)

import pyannote.audio.core.io
pyannote.audio.core.io.AudioDecoder = MockAudioDecoder
pyannote.audio.core.io.AudioStreamMetadata = MockMetadata
if "torchcodec" not in sys.modules:
    import types
    sys.modules["torchcodec"] = types.ModuleType("torchcodec")
    sys.modules["torchcodec"].decoders = types.ModuleType("decoders")
    sys.modules["torchcodec"].decoders.AudioDecoder = MockAudioDecoder

# --- End Patch ---

from pyannote.audio import Pipeline
TOKEN = os.environ.get("HF_TOKEN")

def inspect_api():
    print("Loading pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=TOKEN)
    
    print("Running on short clip...")
    output = pipeline("SampleAudio/test_short.wav")
    
    print(f"Output Type: {type(output)}")
    print(f"Dir: {dir(output)}")
    
    try:
        if hasattr(output, "speaker_diarization"):
            print("\nFound speaker_diarization attribute. Inspecting...")
            annotation = output.speaker_diarization
            print(f"Type: {type(annotation)}")
            print(f"Dir: {dir(annotation)}")
            
            print("\nAttempting iteration on speaker_diarization:")
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                print(f"Speaker {speaker}: {turn.start}-{turn.end}")
        else:
            print("No speaker_diarization attribute found.")
            
    except Exception as e:
        print(f"Inspection failed: {e}")

    # Inspect if it has 'annotation' attribute (common in wrappers)
    if hasattr(output, "annotation"):
        print("\nHas .annotation attribute. Type:", type(output.annotation))
        
if __name__ == "__main__":
    inspect_api()
