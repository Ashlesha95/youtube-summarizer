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

For every candidate return:

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

15. The provided visuals are candidate diagrams, graphs,
    tables, slides, or other educational visuals.

16. Place a visual ONLY where it genuinely helps explain
    the surrounding concept.

17. Do NOT place a visual merely because it is available.

18. Do NOT repeat the same visual.

19. Use a visual when it adds meaningful understanding to
    the notes.

20. When you decide to use a visual, insert ONLY its marker.

    Example:

    [VISUAL:1]

21. The number must correspond to the VISUAL_ID provided below.

22. NEVER create Markdown links for visuals.

23. NEVER create HTML image tags.

24. NEVER include image file paths.

25. NEVER create URLs for visuals.

26. Do not mention unused visuals.

27. Keep the visual marker close to the explanation
    that it supports.

28. Do not put a visual at the beginning of the notes unless
    it genuinely belongs there.

29. Prefer visuals that explain:
    - architecture
    - relationships
    - processes
    - comparisons
    - graphs
    - tables
    - technical concepts


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