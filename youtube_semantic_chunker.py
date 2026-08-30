"""
Semantic chunker for the YouTube summarizer / Q&A project.

Built specifically to handle YouTube transcripts, which often come with
little or no punctuation (auto-captions frequently have no periods/commas
at all). A plain regex sentence-splitter would treat a punctuation-less
transcript as a single giant "sentence" and produce zero useful splits --
so this falls back to a word-window split when that happens.

Supports hierarchical chunking with ONE embedding pass:
  1. Embed every sentence in the transcript once.
  2. Coarse pass ("percentile"): find major topic-section boundaries.
  3. Fine pass ("gradient"): within each section, reuse the SAME cached
     sentence embeddings (just sliced) to find sub-boundaries -- no
     re-embedding needed, since distance/gradient math is cheap and the
     embeddings themselves don't change.
"""

import re
import numpy as np


# ---------------------------------------------------------------------------
# Sentence splitting, with a fallback for punctuation-less transcripts
# ---------------------------------------------------------------------------

def split_into_sentences(
        text: str,
        fallback_window_words: int = 20,
        max_avg_words_per_sentence: int = 35,
) -> list[str]:
    """
    Try normal punctuation-based splitting first. If the resulting
    "sentences" are, on average, way too long (a sign the transcript has
    no/rare punctuation -- common with YouTube auto-captions -- so the
    regex found few or no split points), fall back to splitting into
    fixed-size word windows instead.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]

    words = text.split()
    if not words:
        return sentences

    avg_words_per_sentence = len(words) / max(1, len(sentences))

    if avg_words_per_sentence > max_avg_words_per_sentence:
        sentences = [
            " ".join(words[i:i + fallback_window_words])
            for i in range(0, len(words), fallback_window_words)
        ]

    return sentences


# ---------------------------------------------------------------------------
# Distance + threshold
# ---------------------------------------------------------------------------

def cosine_distance(vec1, vec2) -> float:
    vec1, vec2 = np.array(vec1), np.array(vec2)
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return 1 - similarity


def compute_distances(embeddings) -> list[float]:
    """Pairwise cosine distance between each consecutive embedding."""
    return [
        cosine_distance(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]


def compute_threshold(distances: list[float], method: str, amount: float) -> tuple[float, np.ndarray]:
    """
    Returns (threshold, split_signal) -- split_signal is what gets compared
    against the threshold (raw distances for percentile, the gradient of
    distances for gradient-based splitting).
    """
    distances = np.array(distances)

    if method == "percentile":
        threshold = float(np.percentile(distances, amount))
        return threshold, distances

    elif method == "gradient":
        gradient = np.gradient(distances)
        threshold = float(np.percentile(gradient, amount))
        return threshold, gradient

    else:
        raise ValueError(f"method must be 'percentile' or 'gradient', got: {method}")


def segment_by_indices(embeddings, method: str, amount: float) -> list[tuple[int, int]]:
    """
    Given a list of sentence embeddings, return (start, end) index ranges
    -- exclusive of end -- marking where each chunk should begin/end.
    Pure math on embeddings already in hand; no embedding calls made here.
    """
    n = len(embeddings)
    if n <= 1:
        return [(0, n)]

    distances = compute_distances(embeddings)

    # np.gradient needs at least 2 distance values to compute a rate of
    # change; with only 1 distance (n==2 sentences) there's nothing to
    # split on either way, so just return the whole thing as one chunk.
    if method == "gradient" and len(distances) < 2:
        return [(0, n)]

    threshold, split_signal = compute_threshold(distances, method, amount)

    ranges = []
    start = 0
    for i, signal_value in enumerate(split_signal):
        if signal_value > threshold:
            ranges.append((start, i + 1))
            start = i + 1
    ranges.append((start, n))

    return ranges


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

class YoutubeSemanticChunker:
    """
    Usage (single-level):
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        chunker = YoutubeSemanticChunker(embeddings, method="percentile", amount=95)
        chunks = chunker.split_text(preprocessed_transcript)

    Usage (hierarchical, ONE embedding pass reused for both levels):
        sections = chunker.hierarchical_chunk(
            preprocessed_transcript,
            parent_method="percentile", parent_amount=85,
            child_method="gradient", child_amount=85,
            max_child_size=2000,
        )
    """

    def __init__(
            self,
            embeddings_model,
            method: str = "percentile",   # "percentile" or "gradient"
            amount: float = 95,
            min_chunk_size: int = 50,     # merge tiny fragments into neighbors
    ):
        self.embeddings_model = embeddings_model
        self.method = method
        self.amount = amount
        self.min_chunk_size = min_chunk_size

    # -- single-level split (unchanged behavior) -----------------------

    def split_text(self, transcript: str) -> list[str]:
        sentences = split_into_sentences(transcript)

        if len(sentences) <= 1:
            return sentences

        sentence_embeddings = self.embeddings_model.embed_documents(sentences)
        ranges = segment_by_indices(sentence_embeddings, self.method, self.amount)

        chunks = [" ".join(sentences[start:end]) for start, end in ranges]

        if self.min_chunk_size > 0:
            chunks = self._merge_small_chunks(chunks)

        return chunks

    # -- hierarchical split, single embedding pass ----------------------

    def hierarchical_chunk(
            self,
            transcript: str,
            parent_method: str = "percentile",
            parent_amount: float = 85,
            child_method: str = "gradient",
            child_amount: float = 85,
            max_child_size: int = 2000,
    ) -> list[dict]:
        """
        Embeds sentences ONCE. Uses parent_method/amount to find coarse
        section boundaries, then reuses the SAME cached sentence embeddings
        (sliced per section) to find fine sub-boundaries with
        child_method/amount -- no second embedding call.

        Returns: [{"section_id": int, "chunks": [str, ...]}, ...]
        """
        sentences = split_into_sentences(transcript)

        if len(sentences) <= 1:
            return [{"section_id": 0, "chunks": sentences}]

        # single embedding call for the whole transcript
        sentence_embeddings = self.embeddings_model.embed_documents(sentences)

        parent_ranges = segment_by_indices(sentence_embeddings, parent_method, parent_amount)

        sections = []
        for section_id, (p_start, p_end) in enumerate(parent_ranges):
            section_sentences = sentences[p_start:p_end]
            section_embeddings = sentence_embeddings[p_start:p_end]  # reused, not recomputed

            if len(section_sentences) <= 1:
                child_texts = section_sentences
            else:
                child_ranges = segment_by_indices(section_embeddings, child_method, child_amount)
                child_texts = [
                    " ".join(section_sentences[start:end])
                    for start, end in child_ranges
                ]

            # size safety net: hard-cut any child chunk still over max_child_size
            final_chunks = []
            for chunk in child_texts:
                if len(chunk) <= max_child_size:
                    final_chunks.append(chunk)
                else:
                    for i in range(0, len(chunk), max_child_size):
                        final_chunks.append(chunk[i:i + max_child_size])

            if self.min_chunk_size > 0:
                final_chunks = self._merge_small_chunks(final_chunks)

            sections.append({"section_id": section_id, "chunks": final_chunks})

        return sections

    # -- helpers ----------------------------------------------------------

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        if not chunks:
            return chunks

        merged = [chunks[0]]
        for chunk in chunks[1:]:
            if len(chunk) < self.min_chunk_size:
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)

        if len(merged[0]) < self.min_chunk_size and len(merged) > 1:
            merged[1] = merged[0] + " " + merged[1]
            merged = merged[1:]

        return merged