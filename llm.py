import os
from dotenv import load_dotenv
from google import genai
from extractor import fetch_transcript
from preprocess import get_final_chunks, preprocess_transcript

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def generate_notes(transcript):
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

The output should be detailed study notes, NOT a short summary.

Return ONLY the notes in Markdown.

TRANSCRIPT:
{transcript}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text



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

