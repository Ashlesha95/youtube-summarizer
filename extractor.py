from urllib.parse import urlparse
from urllib.parse import parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
import re
# example url -
url = "https://www.youtube.com/watch?v=ut-ZwrpPb0U"

def extract_youtube_id(url):
    if urlparse(url).netloc == "www.youtube.com":
        return parse_qs(urlparse(url).query).get('v')[0]
    else:
        return urlparse(url).path.split('/')[-1]


video_id = extract_youtube_id(url)

ytt_api = YouTubeTranscriptApi()
fetched_transcript = ytt_api.fetch(video_id)
text = []
for snippet in fetched_transcript:
    text.append(snippet.text)

transcript = " ".join(text)

def preprocess_transcript(transcript):
    transcript.replace("\n", " ")
    transcript.replace("\r", " ")
    transcript = re.sub(r"\[.*?\]", "", transcript)
    return transcript






