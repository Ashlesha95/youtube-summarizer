import os
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types

from frame_extractor import (
    get_stream_url,
    extract_frame
)

from transcript_extractor import (
    fetch_transcript,
    fetch_timestamped_transcript
)

from preprocess import (
    pick_frame_timestamp
)


load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def format_timestamped_transcript(timestamped_transcript):

    lines = []

    for segment in timestamped_transcript:

        lines.append(
            f"[{segment['start']:.2f}s] "
            f"{segment['text']}"
        )

    return "\n".join(lines)


def correct_transcript(transcript_clean):

    prompt = f"""
You are a transcript correction tool.

Correct the transcript ONLY where there is strong evidence that the
transcription contains an error.

You may correct:
- obvious spelling errors
- obvious speech-to-text errors
- punctuation
- capitalization
- clearly misrecognized technical terms

IMPORTANT:
- Do NOT summarize.
- Do NOT paraphrase.
- Do NOT add information.
- Do NOT remove information.
- Do NOT rewrite sentences for style.
- Do NOT infer information that the speaker did not say.
- Preserve the original wording as much as possible.
- If you are uncertain whether a word is wrong, KEEP the original word.
- Technical terms should be corrected only when the surrounding context
  makes the intended term clear.

Return ONLY the corrected transcript.

Transcript:
{transcript_clean}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text


def generate_topic(transcript):

    prompt = f"""
Analyze the following YouTube transcript.

Identify the main topic being discussed.

Rules:
- Give a concise title of 5-12 words.
- The title should represent the main subject of the transcript.
- Do not invent information.
- Do not use phrases like "This video discusses".
- Return ONLY the title.

Transcript:
{transcript}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text.strip()


def get_diagram_timestamps(url):

    timestamped_transcript = (
        fetch_timestamped_transcript(url)
    )

    formatted_text = (
        format_timestamped_transcript(
            timestamped_transcript
        )
    )

    prompt = f"""
You are identifying sections of a video where an important
educational visual is likely being shown.

Use ONLY the timestamped transcript.

Select sections where the speaker is discussing something that
would likely be better understood with a visual, such as:

- diagrams
- architecture diagrams
- flowcharts
- graphs
- tables
- equations
- code
- informative slides
- technical illustrations

IMPORTANT:

- Do NOT select sections merely because a technical topic is mentioned.
- Do NOT assume a visual exists just because the speaker says
  "diagram" or "look at this".
- Select only sections that are strong candidates for visual inspection.
- At least one candidate should be selected if there is a meaningful
  visual concept in the transcript.
- Do not select introductions, greetings, or irrelevant content.
- Avoid repeated visuals where possible.
- if the whole screen have a person only then ignore it
- display diagrams, table like data which can be used to understand the notes.
For every candidate return:
- IMPORTANT - DO NOT SHOW ANYBODY LIKE IF ON THE WHOLE SCREEN ONLY SPEAKER IS PRESENT THOSE VISUALS SHOULD NOT BE CHOSEN

- start: start time in seconds
- end: end time in seconds
- reason: why this section is a strong candidate for a useful visual

Return ONLY valid JSON in this format:

[
    {{
        "start": 693.76,
        "end": 704.32,
        "reason": "The speaker explains the Transformer architecture, making this a strong candidate for an architecture diagram."
    }}
]

Timestamped transcript:

{formatted_text}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    return json.loads(
        response.text
    )

def get_diagrams(url):

    timestamps = get_diagram_timestamps(
        url
    )

    diagram_dic = pick_frame_timestamp(
        timestamps
    )

    stream_url = get_stream_url(
        url
    )

    os.makedirs(
        "static/frames",
        exist_ok=True
    )

    diagrams = []

    for diagram in diagram_dic:

        frame_ts = diagram["start"]

        disk_path = os.path.abspath(
            f"static/frames/frame_{int(frame_ts)}.jpg"
        )

        extract_frame(
            stream_url,
            frame_ts,
            disk_path
        )

        diagrams.append({
            "timestamp": frame_ts,
            "path": disk_path,
            "reason": diagram["reason"]
        })

    return diagrams


def generate_notes(transcript, diagrams):

    diagrams_block = "\n".join(
        f"""
VISUAL_ID: {i}
TIMESTAMP: {diagram["timestamp"]} seconds
REASON: {diagram["reason"]}
"""
        for i, diagram in enumerate(
            diagrams,
            start=1
        )
    )

    prompt = f"""
You are a professional note-taking assistant.

Your task is to convert the provided transcript into detailed,
well-organized study notes.

SOURCE-FAITHFULNESS IS THE HIGHEST PRIORITY.

Rules:

1. Use ONLY information explicitly present in the transcript.

2. Do NOT use your own knowledge to fill gaps.

3. Do NOT invent examples, filenames, commands, practices,
   concepts, tools, or facts.

4. Do NOT infer additional information unless the relationship
   is explicitly stated by the speaker.

5. Preserve the speaker's terminology and intended meaning.

6. Correct obvious transcription/spelling errors when the
   intended word is clear from the transcript.

7. Remove filler, repetition, greetings, and conversational noise.

8. Organize related information under headings and subheadings.

9. Use concise bullet points.

10. Preserve important explanations rather than reducing
    everything to short summary statements.

11. If the transcript mentions a list, preserve all items
    that are actually present.

12. Do not create additional items to complete a list.

13. Do not add information merely because it is commonly
    associated with the topic.

14. If something is unclear, leave it out rather than guessing.


VISUAL RULES:

Here think of yourself as a student or teacher, which visuals would you like to have in notes. select 
only those visuals.



CAPTIONS:

Do NOT create captions yourself in the Markdown.

The application will generate/display the caption using
the visual metadata.


OUTPUT:

Return ONLY the study notes in Markdown.

Use visual markers like:

[VISUAL:1]

Do not add any explanation about the markers.

AVAILABLE VISUALS:

{diagrams_block}


TRANSCRIPT:

{transcript}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text


def generate_notes_with_diagrams(url):

    transcript = fetch_transcript(
        url
    )

    transcript = correct_transcript(
        transcript
    )

    diagrams = get_diagrams(
        url
    )

    notes = generate_notes(
        transcript,
        diagrams
    )
    return notes, diagrams

def answer_from_video(
        question,
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

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a question-answering assistant
for a YouTube video.

Answer the user's question ONLY using
the provided transcript context.

If the context does not contain enough
information to answer the question,
say:

"I don't have enough information in the
video to answer this question."

Do NOT use your own general knowledge.

Do NOT invent facts.

Question:
{question}

Transcript context:
{context}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text

def answer_generally(question):
    """
    Fallback used when retrieval couldn't find enough relevant
    transcript context. This must NOT reference "the video",
    "the transcript", or any retrieval/internal-system language in
    its own reasoning — the previous version primed the model with
    that framing in the prompt itself, then separately told it not
    to mention it, which is a contradiction the model resolved by
    following the framing (hence answers like "the video transcript
    did not contain enough information").

    Note: a question like "what is this video about" is inherently
    unanswerable here even with a clean prompt, since general
    knowledge has no way to know which video you mean. That class of
    question needs to be answered from retrieval (i.e. actual video
    content), not this fallback — see the note in rag.py/app.py about
    indexing the generated notes/summary so retrieval can serve it.
    """

    prompt = f"""
You are a helpful, knowledgeable assistant.
Answer the following question directly, using your own general
knowledge.

IMPORTANT:
- Do not reference a transcript, a video's content, retrieval, or
  any internal system — just answer the question itself.
- Start your answer by briefly noting that this comes from general
  knowledge, not the video's own content.
- If the question cannot be meaningfully answered without knowing
  which specific video is being discussed (for example, a vague
  question like "what is this about" with no topic given), say so
  plainly in one sentence and ask the user to share the video's
  title or subject — do not guess or ramble about YouTube videos
  in general.
- Otherwise, give a useful, concise answer to the actual question.

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text


def classify_question_scope(question):
    """
    Routes a question before retrieval even happens.

    "broad": needs the video's overall structure/content — a
    summary, an overview, "what's this about", "what's in chapter/
    section N", "what topics does this cover". No single transcript
    chunk represents this; answer from the full notes instead
    (see answer_from_notes below), never through chunk retrieval.

    "specific": a narrow factual/detail question best answered from
    one or two specific transcript passages — keep using the
    existing retrieve() + has_sufficient_evidence() pipeline for
    these, since that's genuinely the right tool for them.

    Deliberately a separate, tiny classification call rather than a
    keyword list — phrasing for "give me an overview" varies too
    much ("what's this about", "summarize this", "what does this
    cover", "tell me about chapter 2"...) for fixed keywords to
    reliably catch, and this call is small/fast/cheap.
    """

    prompt = f"""
Classify the following question about a video into exactly one of
two categories: "broad" or "specific".

"broad" means the question is asking about the video's overall
content, summary, main topics, structure, or the contents of a
named chapter/section (even if the phrasing is roundabout or
casual). Examples: "what is this video about", "summarize this",
"what does this cover", "what's in chapter 1", "give me the gist",
"what are the main topics".

"specific" means the question asks about one narrow fact, detail,
example, or explanation that would appear in one specific part of
the video, not the whole thing. Examples: "what does the speaker
say about X", "what command did they run", "why did they choose Y
over Z".

Return ONLY the single word "broad" or "specific" — nothing else.

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    scope = response.text.strip().lower()

    return "broad" if "broad" in scope else "specific"


def answer_broad_question(question, notes, transcript):
    """
    Answers "broad" questions (see classify_question_scope) using
    BOTH the notes and the full transcript — not the notes alone.

    Notes are a deliberately trimmed, curated version of the video
    (your own generate_notes prompt explicitly removes filler,
    repetition, and reduces things to concise bullets) — anything
    the notes-writer judged less essential simply isn't there. The
    transcript is the actual complete source. Notes are still useful
    here for their headings/structure (helpful for "what's in
    chapter 1"-style questions), but the transcript is what makes
    sure nothing the video actually said is unreachable in chat just
    because it got trimmed out of the notes.

    No chunk retrieval, no similarity threshold — this reads
    everything at once, so it can't fail the way chunk-level
    matching can for summary/overview-style questions.
    """

    # Strip [VISUAL:n] markers — layout instructions, not content.
    cleaned_lines = [
        line for line in notes.splitlines()
        if "[VISUAL:" not in line
    ]

    cleaned_notes = "\n".join(cleaned_lines).strip()

    prompt = f"""
You are a teacher teaching the student with the examples which are easy to understand.


1. NOTES — a curated, organized summary of the video, useful mainly
   for understanding its structure (headings/sections/chapters).
   The notes leave out some content for brevity, so do NOT treat
   them as the complete picture.

2. FULL TRANSCRIPT — the complete, unfiltered content of the video.
   This is the authoritative source for anything the notes may have
   trimmed out. Prefer the transcript whenever it has more detail
   than the notes on the same point.

3. Explain everything with proper examples as if you are teaching it. And if the user asks to understand again, walk them step by step.
   JUST FOR EXAMPLES - YOU CAN USE RESOURCES FROM ANYWHERE BUT ANSWERS SHOULD BE GROUNDED TO NOTES AND TRANSCRIPT.
   

Answer the user's question using both sources together: use the
notes to understand structure/organization (e.g. which part of the
video a "chapter" or section refers to), and use the transcript to
make sure your answer reflects everything the video actually covers
on that topic, not just what made it into the notes.

If neither source contains enough information to answer the
question, say:

"I don't have enough information in the video to answer this
question."

Do NOT use your own general knowledge to fill gaps.
Do NOT invent facts, chapters, or sections that aren't in the video.

Question: 
{question}

NOTES:
{cleaned_notes}

FULL TRANSCRIPT:
{transcript}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text