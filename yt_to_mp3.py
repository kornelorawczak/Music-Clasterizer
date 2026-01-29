import yt_dlp
import os


def download_mp3(file: str, out_path: str):
    os.makedirs(out_path, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{out_path}/%(title)s.%(ext)s",
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
    }

    with open(file) as f:
        urls = [line.strip() for line in f if line.strip()]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)

    print("All mp3 downloaded")


download_mp3("urls.txt", "./data")
