import yt_dlp
import os

def download_mp3(file: str, out_path: str):
    abs_out_path = os.path.abspath(out_path)
    os.makedirs(abs_out_path, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "paths": {
            "home": abs_out_path  
        },
        "outtmpl": "%(title)s.%(ext)s",        
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "extractor_args": {
            "youtube": {
                "player_client": ["android"]
            }
        },
        "noplaylist": True,
        "ignoreerrors": True,
        "quiet": False,
        "keepvideo": False, 
    }

    with open(file) as f:
        urls = [line.strip() for line in f if line.strip()]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)

    print("All mp3 downloaded")