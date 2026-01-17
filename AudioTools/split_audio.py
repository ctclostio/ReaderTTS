import argparse
import os
import math
import torchaudio
import torch

def split_audio(input_file, splits=10):
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        return

    print(f"Loading {input_file}...")
    wav, sr = torchaudio.load(input_file)
    
    total_frames = wav.shape[1]
    duration = total_frames / sr
    print(f"Total Duration: {duration:.2f}s")
    
    frames_per_split = math.ceil(total_frames / splits)
    print(f"Splitting into {splits} parts of ~{frames_per_split/sr:.2f}s each...")
    
    base_name = os.path.splitext(input_file)[0]
    output_files = []
    
    for i in range(splits):
        start = i * frames_per_split
        end = min((i + 1) * frames_per_split, total_frames)
        
        if start >= total_frames:
            break
            
        chunk = wav[:, start:end]
        output_path = f"{base_name}_part{i+1:02d}.wav"
        
        torchaudio.save(output_path, chunk, sr)
        print(f"Saved {output_path} ({chunk.shape[1]/sr:.2f}s)")
        output_files.append(output_path)
        
    return output_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input WAV file")
    parser.add_argument("--splits", type=int, default=10, help="Number of splits")
    args = parser.parse_args()
    
    split_audio(args.input_file, args.splits)
