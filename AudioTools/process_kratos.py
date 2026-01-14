import os
import subprocess
import yt_dlp
import shutil
from pathlib import Path

# --- Configuration ---
OUTPUT_BASE = Path("AudioTools/Kratos_Vocals")
TEMP_DL = Path("AudioTools/temp_dl")
REFINED_DIR = Path("SampleAudio/Refined_Kratos")

# Categories and URLs (Round 4: The Truth Only)
SOURCES = {
    "Kratos_TheTruth": [
        "https://www.youtube.com/watch?v=Whsq6npp8rM", 
    ],
    "Kratos_Anchor": [
        "https://www.youtube.com/watch?v=GFkTTCnhSkg", # Read it Boy (Pure Kratos for verification)
    ]
}

def download_audio(url, output_path):
    """Downloads audio from YouTube using yt-dlp."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': str(output_path),
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def separate_and_move(audio_path, output_dir, final_name):
    """Separates vocals and moves result to Refined folder."""
    print(f"Separating vocals for: {audio_path.name}...")
    
    cmd = [
        "demucs",
        "-n", "htdemucs",
        "--two-stems", "vocals",
        "-o", str(output_dir),
        str(audio_path)
    ]
    subprocess.run(cmd, check=True)
    
    # Demucs output: output_dir/htdemucs/{audio_filename}/vocals.wav
    # Note: audio_filename is audio_path.stem
    demucs_out = output_dir / "htdemucs" / audio_path.stem / "vocals.wav"
    
    if demucs_out.exists():
        dest = REFINED_DIR / f"{final_name}.wav"
        print(f"Moving to {dest}...")
        shutil.copy(demucs_out, dest)
    else:
        print(f"Error: Expected output not found at {demucs_out}")

def main():
    if not OUTPUT_BASE.exists():
        OUTPUT_BASE.mkdir(parents=True)
    if not TEMP_DL.exists():
        TEMP_DL.mkdir(parents=True)
    if not REFINED_DIR.exists():
        REFINED_DIR.mkdir(parents=True)

    print("Starting processing (Round 2)...")

    for category, urls in SOURCES.items():
        print(f"\n--- Processing Category: {category} ---")
        category_dir = OUTPUT_BASE / category
        if not category_dir.exists():
            category_dir.mkdir()

        for i, url in enumerate(urls):
            filename = f"{category}_{i+1}"
            dl_path = TEMP_DL / filename # yt-dlp adds extension
            expected_wav = TEMP_DL / f"{filename}.wav"
            
            # Download
            if not expected_wav.exists():
                print(f"Downloading {url} as {filename}...")
                download_audio(url, dl_path)
            
            if expected_wav.exists():
                # Separate and Move
                separate_and_move(expected_wav, category_dir, filename)
            else:
                print(f"Failed to download {url}")

    print("\nProcessing complete. Files ready in SampleAudio/Refined_Kratos.")

if __name__ == "__main__":
    main()
