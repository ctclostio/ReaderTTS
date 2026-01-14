import sys
import yt_dlp
from pathlib import Path

OUTPUT_DIR = Path("SampleAudio/Exact_Reference")

def download_section(url, start_time, end_time, output_name="reference"):
    """
    Downloads a specific time slice from a YouTube video as wav.
    start_time/end_time format: "MM:SS" or "SS"
    """
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    print(f"Downloading {url} [{start_time}-{end_time}]...")
    
    # yt-dlp format for sections: "*start-end"
    # Example: "*00:30-01:00"
    section_arg = f"*{start_time}-{end_time}"
    
    output_path = OUTPUT_DIR / f"{output_name}.%(ext)s"
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'download_ranges': yt_dlp.utils.download_range_func(None, [(list(map(int, start_time.split(':')))[0]*60 + int(list(map(int, start_time.split(':')))[1]) if ':' in start_time else int(start_time), 
                                                                    list(map(int, end_time.split(':')))[0]*60 + int(list(map(int, end_time.split(':')))[1]) if ':' in end_time else int(end_time))]), 
        # Actually yt-dlp has a simpler way using 'download_ranges' callback but the command line arg --download-sections is easier to map.
        # Let's use the --download-sections arg equivalent which is passed via 'download_ranges' usually? 
        # No, 'download_ranges' expects a function. 
        # Simpler: Use 'external_downloader_args' to pass ffmpeg args? No.
        # Simplest: Use the 'download_ranges' callback helper.
        
        'outtmpl': str(output_path),
        'force_keyframes_at_cuts': True,
    }
    
    # Let's try to just run the command via subprocess to avoid complex callback logic if we want to be safe, 
    # BUT yt-dlp python embedding is cleaner. 
    # Wait, the simplest way to do sections in python `yt_dlp` is providing `download_ranges` callback.
    
    def parse_time(t_str):
        parts = list(map(int, t_str.split(':')))
        if len(parts) == 1: return parts[0]
        if len(parts) == 2: return parts[0]*60 + parts[1]
        return parts[0]*3600 + parts[1]*60 + parts[2]

    s = parse_time(start_time)
    e = parse_time(end_time)
    
    ydl_opts['download_ranges'] = yt_dlp.utils.download_range_func(None, [(s, e)])
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    print(f"Download complete: {OUTPUT_DIR}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python download_exact.py <URL> <START> <END> [OUTPUT_NAME]")
        print("Example: python download_exact.py https://... 00:30 01:15 my_clip")
        sys.exit(1)
        
    url = sys.argv[1]
    start = sys.argv[2]
    end = sys.argv[3]
    out_name = sys.argv[4] if len(sys.argv) > 4 else "reference"
    
    download_section(url, start, end, out_name)
