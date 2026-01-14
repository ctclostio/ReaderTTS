import argparse
import sys
import os
import numpy as np
import soundfile as sf
# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text_parser import parse_text, chunk_text
from text_parser import parse_text, chunk_text
# Lazy imports to avoid heavy load if not used
# from f5_engine import F5Client 
# from xtts_engine import XTTSClient

def main():
    parser = argparse.ArgumentParser(description="AI Input Audio Book Reader (VibeVoice)")
    parser.add_argument("input_file", help="Path to the input text/pdf/epub file")
    parser.add_argument("--output", "-o", help="Output audio file path. Defaults to input name + .wav")
    parser.add_argument("--ref_audio", "-r", required=True, help="Reference audio file (wav/mp3) for voice cloning (5-15s recommended)")
    parser.add_argument("--ref_text", "-t", default="", help="Transcript of the reference audio (optional)")
    parser.add_argument("--engine", "-e", default="vibevoice", choices=["vibevoice"], help="TTS Engine to use: 'vibevoice'")
    parser.add_argument("--model", "-m", default="VibeVoice-1.5b", help="Model type")
    parser.add_argument("--language", "-l", default="en", help="Language code")
    
    # Generation Params
    parser.add_argument("--gen_speed", type=float, default=1.0, help="Speech speed (1.0 = normal)")
    parser.add_argument("--nfe_step", type=int, default=30, help="Number of inference steps (default 30)")
    parser.add_argument("--cfg_strength", type=float, default=1.5, help="Classifier Free Guidance strength (1.5 default)")
    parser.add_argument("--model_name", default="microsoft/VibeVoice-1.5b", help="HuggingFace model ID")
    parser.add_argument("--sway", type=float, default=-1, help="Sway sampling coefficient (unused in VibeVoice)")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    if not os.path.exists(args.ref_audio):
        print(f"Error: Reference audio '{args.ref_audio}' not found.")
        sys.exit(1)

    # --- Setup Output Path ---
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output = f"{base_name}_cloned.wav"

    # --- Parse Input Text ---
    print(f"Reading {args.input_file}...")
    try:
        full_text = parse_text(args.input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    print(f"Text length: {len(full_text)} characters.")
    print("Chunking text...")
    chunks = chunk_text(full_text, max_chars=400) # VibeVoice might handle longer, but 400 safe
    print(f"Created {len(chunks)} chunks.")

    # --- Handle Reference Audio ---
    # Supports passing a directory of clips
    ref_audios = []
    final_ref_text = "" 
    
    if os.path.isdir(args.ref_audio):
        # Gather all wav/mp3 files
        for root, dirs, files in os.walk(args.ref_audio):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3')):
                    ref_audios.append(os.path.join(root, file))
        
        # Look for metadata.json in the directory
        metadata_path = os.path.join(args.ref_audio, "metadata.json")
        if os.path.exists(metadata_path):
             print(f"Loading reference text from {metadata_path}...")
             import json
             with open(metadata_path, 'r', encoding='utf-8') as f:
                 final_ref_text = json.load(f) # Load as dict
        else:
            print("No metadata.json found. ASR usage is disabled in this VibeVoice-only version.")
            
    else:
        ref_audios.append(args.ref_audio)
        if args.ref_text:
            final_ref_text = args.ref_text

    if not ref_audios:
        print("Error: No valid reference audio files found.")
        return

    print(f"Found {len(ref_audios)} reference audio files. They will be combined.")

    # --- Initialize Engine ---
    client = None
    
    if args.engine == "vibevoice":
        print("Initializing VIBEVOICE Engine...")
        from vibevoice_engine import VibeVoiceClient
        client = VibeVoiceClient(model_name=args.model_name) # Model path internal default
    
    if client is None:
        print(f"Error: Engine {args.engine} failed to initialize.")
        return

    # --- Generate Audio ---
    audio_segments = []
    full_audio = []
    
    try:
        generator = client.generate_audio(
            chunks,
            ref_audios,
            final_ref_text,
            cfg_strength=args.cfg_strength,
            nfe_step=args.nfe_step
        )
            
        for i, (audio_chunk, sample_rate) in enumerate(generator):
             if audio_chunk is not None:
                print(f"Processing chunk {i+1}/{len(chunks)}")
                full_audio.append(audio_chunk)
             else:
                 print(f"Skipped empty audio for chunk {i+1}")
                 
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()

    # --- Stitch and Save ---
    if full_audio:
        print("Stitching audio...")
        import numpy as np
        final_wave = np.concatenate(full_audio)
        
        print(f"Saving to {args.output}...")
        # Ensure float32 for soundfile compatibility (7B model outputs float16)
        sf.write(args.output, final_wave.astype(np.float32), 24000) # VibeVoice defaults to 24k commonly
        print("Done!")
    else:
        print("No audio generated.")

if __name__ == "__main__":
    main()
