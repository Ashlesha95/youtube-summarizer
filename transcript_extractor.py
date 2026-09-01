from urllib.parse import urlparse
from urllib.parse import parse_qs
from youtube_transcript_api import YouTubeTranscriptApi

import re
# example url -

def extract_youtube_id(url):
    if urlparse(url).netloc == "www.youtube.com":
        return parse_qs(urlparse(url).query).get('v')[0]
    else:
        return urlparse(url).path.split('/')[-1]

def fetch_transcript(url):
    video_id = extract_youtube_id(url)

    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)
    text = []
    for snippet in fetched_transcript:
        text.append(snippet.text)

    transcript = " ".join(text)
    return transcript

def fetch_timestamped_transcript(url):
    video_id = extract_youtube_id(url)

    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)

    transcript = []

    for snippet in fetched_transcript:
        transcript.append({
            "text": snippet.text,
            "start": snippet.start,
            "duration": snippet.duration,
            "end": snippet.start + snippet.duration
        })

    return transcript












