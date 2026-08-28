import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


# preoprocessing the transcript
def preprocess_transcript(transcript1):
    transcript1 = transcript1.replace("\n", " ")
    transcript1 = transcript1.replace("\r", " ")
    transcript1 = re.sub(r"\[.*?\]", "", transcript1)
    words = transcript1.split()
    transcript1 = " ".join(words)
    return transcript1

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)















