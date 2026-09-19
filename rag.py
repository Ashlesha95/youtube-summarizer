# rag.py

import numpy as np
import streamlit as st

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from preprocess import get_final_chunks


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )


# ============================================================
# RERANKER
#
# A cross-encoder scores (question, chunk) pairs jointly, which is
# far more precise than comparing two independently-embedded
# vectors. We use it to re-score a broader candidate pool pulled
# back by the vector search, and keep only the best few.
# ============================================================

@st.cache_resource
def get_reranker():

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================
# PREPARE CHUNKS
# ============================================================

def prepare_chunks(transcript):

    sections = get_final_chunks(transcript)

    documents = []

    for section_index, section in enumerate(sections):

        section_id = section.get(
            "section_id",
            section_index
        )

        chunks = section.get(
            "chunks",
            []
        )

        for chunk_index, chunk in enumerate(chunks):

            # If chunk is just a string
            if isinstance(chunk, str):

                text = chunk
                chunk_id = chunk_index

            # If chunk is a dictionary
            else:

                text = chunk.get(
                    "text",
                    ""
                )

                chunk_id = chunk.get(
                    "chunk_id",
                    chunk_index
                )

            if not text:
                continue

            if not text.strip():
                continue

            documents.append(
                {
                    "text": text,
                    "section_id": section_id,
                    "chunk_id": chunk_id
                }
            )

    return documents


def prepare_notes_chunks(notes):
    """
    Splits the generated notes (with [VISUAL:n] markers stripped out)
    into a few chunks, tagged section_id="summary", so questions like
    "what is this video about" have something retrievable that
    actually represents the whole video — no single transcript chunk
    can do that, but the notes/summary can.
    """

    if not notes or not notes.strip():
        return []

    # Drop visual marker lines — they're layout instructions, not
    # content, and would just be noise in the embedding.
    cleaned_lines = [
        line for line in notes.splitlines()
        if "[VISUAL:" not in line
    ]

    cleaned_notes = "\n".join(cleaned_lines).strip()

    if not cleaned_notes:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50
    )

    chunks = splitter.split_text(cleaned_notes)

    return [
        {
            "text": chunk,
            "section_id": "summary",
            "chunk_id": index
        }
        for index, chunk in enumerate(chunks)
        if chunk.strip()
    ]


# ============================================================
# CREATE VECTOR INDEX
# ============================================================

def create_index(
        transcript,
        video_id,
        notes=None
):

    documents = prepare_chunks(
        transcript
    )

    # Add the notes/summary as extra retrievable documents, tagged
    # section_id="summary" so format_sources() shows where an answer
    # came from. Kept optional (notes=None default) so existing call
    # sites that only have a transcript keep working unchanged.
    documents.extend(
        prepare_notes_chunks(notes)
    )

    if not documents:
        raise ValueError(
            "No chunks were generated from the transcript."
        )

    texts = [
        document["text"]
        for document in documents
    ]

    metadatas = [
        {
            "section_id": str(
                document["section_id"]
            ),
            "chunk_id": str(
                document["chunk_id"]
            )
        }
        for document in documents
    ]

    embeddings = get_embeddings()

    collection_name = (
        f"youtube_{video_id}"
    )

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=collection_name,

        # IMPORTANT: without this, Chroma defaults to raw L2
        # distance, not cosine. Since bge-small embeddings are
        # normalized, that made the earlier 0.45 threshold in
        # has_sufficient_evidence roughly 3x stricter than intended
        # (it was effectively requiring ~0.775+ cosine similarity),
        # silently rejecting plenty of genuinely relevant chunks.
        collection_metadata={"hnsw:space": "cosine"}
    )

    return vectorstore


# ============================================================
# RETRIEVE RELEVANT CHUNKS (vector search + cross-encoder rerank)
# ============================================================

def retrieve(
        vectorstore,
        question,
        k=5,
        fetch_k=20
):
    """
    Two-stage retrieval:
      1. Vector search pulls back a broader candidate pool
         (fetch_k) — cheap, but only approximately relevant.
      2. A cross-encoder reranker rescores each candidate against
         the actual question text, jointly, and we keep the best k.
         This is what fixes borderline/imprecise vector matches.

    Each returned chunk carries:
      - "score": the reranker's relevance probability (0-1, higher
         = more relevant). This is what has_sufficient_evidence and
         downstream ranking should use.
      - "vector_score": the original cosine distance from the
         vector search (0 = identical, higher = less similar),
         kept only for debugging/inspection.
    """

    candidates = vectorstore.similarity_search_with_score(
        question,
        k=fetch_k
    )

    if not candidates:
        return []

    pairs = [
        (question, document.page_content)
        for document, _ in candidates
    ]

    reranker = get_reranker()
    raw_rerank_scores = reranker.predict(pairs)
    rerank_scores = _sigmoid(np.array(raw_rerank_scores))

    scored_chunks = []

    for (document, vector_score), rerank_score in zip(candidates, rerank_scores):

        scored_chunks.append(
            {
                "text": document.page_content,

                "section_id": document.metadata.get(
                    "section_id",
                    "Unknown"
                ),

                "chunk_id": document.metadata.get(
                    "chunk_id",
                    "Unknown"
                ),

                "score": float(rerank_score),
                "vector_score": float(vector_score)
            }
        )

    # Best reranked match first
    scored_chunks.sort(key=lambda chunk: chunk["score"], reverse=True)

    return scored_chunks[:k]


# ============================================================
# CHECK WHETHER RETRIEVED EVIDENCE IS GOOD ENOUGH
#
# NOTE: the meaning of "score" flipped with the rerank change above
# — it's now a 0-1 relevance probability where HIGHER is better, not
# a distance where lower is better. The threshold default here
# (0.3) is a reasonable starting point for
# cross-encoder/ms-marco-MiniLM-L-6-v2, but cross-encoder scores
# aren't perfectly calibrated across every domain — if you're still
# seeing too many false "not in the video" responses, or the
# opposite (confident wrong answers), print the scores for a few
# real questions and adjust this number accordingly.
#
# IMPORTANT: update the call in app.py from
#   has_sufficient_evidence(results, threshold=0.45)
# to something like
#   has_sufficient_evidence(results, threshold=0.3)
# — the old 0.45 was calibrated for the old (lower-is-better)
# distance scale and means something different now.
# ============================================================

def has_sufficient_evidence(
        retrieved_chunks,
        threshold=0.3
):

    if not retrieved_chunks:
        return False

    best_score = retrieved_chunks[0]["score"]

    return best_score >= threshold


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
        retrieved_chunks
):

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"""
Section {chunk['section_id']}
Chunk {chunk['chunk_id']}

{chunk['text']}
"""
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# FORMAT SOURCES
# ============================================================

def format_sources(
        retrieved_chunks
):

    sources = []

    seen = set()

    for chunk in retrieved_chunks:

        source = (
            str(chunk["section_id"]),
            str(chunk["chunk_id"])
        )

        if source in seen:
            continue

        seen.add(source)

        sources.append(
            f"Section {source[0]} "
            f"• Chunk {source[1]}"
        )

    return sources