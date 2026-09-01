import yt_dlp
import subprocess

def get_stream_url(url):
    ydl_opts = {
        "format": "bestvideo[height<=720][ext=mp4]/bestvideo[ext=mp4]/bestvideo",
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return info["url"]


def extract_frame(video_source, timestamp, output_path):
    subprocess.run(
        [
            "ffmpeg",
            "-ss", str(timestamp),
            "-i", video_source,
            "-frames:v", "1",
            "-q:v", "2",
            "-y",
            output_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

# with yt_dlp.YoutubeDL({"quiet": False}) as ydl:
#     info = ydl.extract_info("https://www.youtube.com/watch?v=zjkBMFhNj_g", download=False)
#     for f in info["formats"]:
#         print(f["format_id"], f.get("ext"), f.get("height"), f.get("vcodec"))