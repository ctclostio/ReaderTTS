import os
import argparse
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm

def chunk_file(file_path, output_dir, chunk_length_ms=15000):
    """Splits an audio file into fixed-length chunks."""
    print(f"Loading {file_path}...")
    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    duration_ms = len(audio)
    print(f"Duration: {duration_ms/1000:.2f}s")
    
    base_name = file_path.stem
    
    chunks = []
    for i in range(0, duration_ms, chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        if len(chunk) < 2000: # Skip tiny chunks < 2s
            continue
        chunks.append(chunk)

    print(f"Exporting {len(chunks)} chunks to {output_dir}...")
    for i, chunk in enumerate(tqdm(chunks)):
        out_name = f"{base_name}_part{i:04d}.wav"
        chunk.export(output_dir / out_name, format="wav")

def main():
    parser = argparse.ArgumentParser(description="Chunk large audio files into smaller segments.")
    parser.add_argument("input_dir", help="Directory containing audio files to chunk")
    parser.add_argument("output_dir", help="Output directory for chunks")
    parser.add_argument("--length", type=int, default=15, help="Chunk length in seconds (default 15)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    if not output_path.exists():
        output_path.mkdir(parents=True)
        
    files = list(input_path.glob("*.wav")) + list(input_path.glob("*.mp3"))
    
    for f in files:
        chunk_file(f, output_path, args.length * 1000)
        
    print("Done.")

if __name__ == "__main__":
    main()
