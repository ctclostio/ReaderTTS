import yt_dlp
import sys
import re
from pathlib import Path

def sanitize_filename(name):
    """Sanitize string to be safe for filenames."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def download_short(url, target_dir=None):
    """Downloads a YouTube short/video as wav.
    
    Args:
        url (str): The URL or query (ytsearch1:...) to download.
        target_dir (str, optional): The specific directory to save to. 
                                    If None, creates a new folder based on video title.
    """
    
    # Base output directory
    base_dir = Path("SampleAudio")
    if not base_dir.exists():
        base_dir.mkdir()

    print(f"Fetching metadata for {url}...")
    
    ydl_opts_meta = {
        'quiet': True,
        'extract_flat': True,
    }
    
    try:
        title = "Unknown_Video"
        # Only fetch metadata if we need to generate a folder name
        if target_dir is None:
            with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl:
                info = ydl.extract_info(url, download=False)
                # handle search results which return a 'entries' list
                if 'entries' in info:
                    info = info['entries'][0]
                title = info.get('title', 'Unknown_Video')
                clean_title = sanitize_filename(title).replace(' ', '_')
                output_dir = base_dir / clean_title
        else:
            output_dir = Path(target_dir)
            # We still need the title for the filename though, or we can use a generic one?
            # Better to get title for filename
            with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl:
                info = ydl.extract_info(url, download=False)
                if 'entries' in info:
                    info = info['entries'][0]
                title = info.get('title', 'Unknown_Video')
                clean_title = sanitize_filename(title).replace(' ', '_')

        if not output_dir.exists():
            output_dir.mkdir(parents=True)
            
        print(f"Target Directory: {output_dir}")
        
        # Download options
        output_template = str(output_dir / f"{clean_title}.%(ext)s")
        
        ydl_opts_download = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': output_template,
            'quiet': False,
            'overwrite': True,
        }
        
        print(f"Downloading '{title}'...")
        with yt_dlp.YoutubeDL(ydl_opts_download) as ydl_dl:
            ydl_dl.download([url])
            
        print(f"Successfully downloaded to: {output_dir}")
        return output_dir

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to the requested URL for this task if no argument provided
        default_url = "https://www.youtube.com/shorts/t136sPDbEv0"
        print(f"No URL provided, using default: {default_url}")
        download_short(default_url)
    else:
        url = sys.argv[1]
        target_dir = sys.argv[2] if len(sys.argv) > 2 else None
        download_short(url, target_dir)
