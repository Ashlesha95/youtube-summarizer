import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_semantic_chunker import YoutubeSemanticChunker
from langchain_community.embeddings import FastEmbedEmbeddings

from transcript_extractor import fetch_transcript


# preoprocessing the transcript
def preprocess_transcript(transcript1):
    transcript1 = transcript1.replace("\n", " ")
    transcript1 = transcript1.replace("\r", " ")
    transcript1 = re.sub(r"\[.*?\]", "", transcript1)
    words = transcript1.split()
    transcript1 = " ".join(words)
    return transcript1



def chuck_transcript(transcript):

    preprocessed_transcript  = preprocess_transcript(transcript)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20
    )

    chunks = text_splitter.split_text(preprocessed_transcript)

    return chunks


# transcript = fetch_transcript(url)
# chunks = chuck_transcript(transcript)
#
# print("Number of chunks:", len(chunks))
#
# for i, chunk in enumerate(chunks[:3]):
#     print(f"\n--- Chunk {i + 1} ---")
#     print(chunk)

def semantic_chunking(transcript1):
    preprocessed_transcript = preprocess_transcript(transcript1)
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    text_splitter = YoutubeSemanticChunker(embeddings,
                                           method="gradient")

    chunks_semantic = text_splitter.split_text(preprocessed_transcript)

    return chunks_semantic

def create_child_chunks(semantic_chunk, max_size=2000):
    sentences = re.split(r'(?<=[.!?])\s+', semantic_chunk)

    child_chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)

        if current_size + sentence_size <= max_size:
            current_chunk.append(sentence)
            current_size += sentence_size

        else:
            child_chunks.append(" ".join(current_chunk))

            current_chunk = [sentence]
            current_size = sentence_size

    if current_chunk:
        child_chunks.append(" ".join(current_chunk))

    return child_chunks


def get_final_chunks(transcript):
    preprocessed_transcript = preprocess_transcript(transcript)
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    chunker = YoutubeSemanticChunker(embeddings, min_chunk_size=50)

    sections = chunker.hierarchical_chunk(
        preprocessed_transcript,
        parent_method="percentile", parent_amount=95,
        child_method="gradient", child_amount=95,
        max_child_size=2000,
    )

    return sections

def pick_frame_timestamp(diagrams):

    for diagram in diagrams :
        start = diagram['start']
        end = diagram['end']

        if end-start>2:
            diagram['start'] = end-2


    return diagrams












# transcript = fetch_transcript(url)
# chunks = get_final_chunks(transcript)
# print("Number of chunks:", len(chunks))
#
# for section in chunks[:5]:
#     print(f"\n=== Section {section['section_id']} ({len(section['chunks'])} chunks) ===")
#
#     for i, chunk in enumerate(section["chunks"]):
#         print(f"\n--- Chunk {i+1} ---")
#         print(chunk)

















