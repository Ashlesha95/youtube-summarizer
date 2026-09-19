import os
import re

import streamlit as st
from fpdf import FPDF

from transcript_extractor import (
    fetch_transcript,
    extract_youtube_id
)

from llm import (
    generate_notes_with_diagrams,
    generate_topic,
    answer_from_video,
    answer_generally,
    classify_question_scope,
    answer_broad_question
)

from rag import (
    create_index,
    retrieve,
    has_sufficient_evidence,
    format_sources
)

from database import (
    initialize_database,
    save_notes,
    load_notes,
    load_diagrams,
    load_metadata,
    get_all_notes,
    get_versions,
    revert_notes,
    delete_video
)


# ============================================================
# INITIALIZE DATABASE
# ============================================================

initialize_database()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="YouTube Notes",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "selected_video" not in st.session_state:
    st.session_state["selected_video"] = None

if "rag_indexes" not in st.session_state:
    st.session_state["rag_indexes"] = {}

if "transcripts" not in st.session_state:
    st.session_state["transcripts"] = {}

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}

# Replaces the old @st.dialog-driven visibility. This flag is checked
# at the top level on every rerun (not just the run where a button was
# clicked), which is what keeps the panel open while you're chatting.
if "chat_open" not in st.session_state:
    st.session_state["chat_open"] = False

if "chat_maximized" not in st.session_state:
    st.session_state["chat_maximized"] = False

if "pending_general_question" not in st.session_state:
    st.session_state["pending_general_question"] = None

if "pending_general_video" not in st.session_state:
    st.session_state["pending_general_video"] = None

# A question just submitted, waiting to be answered. Keyed by
# video_id. Set immediately on submit (before any retrieval/LLM
# call) and rerun right away, so the question renders in the chat
# instantly instead of staying invisible until the answer arrives.
if "pending_questions" not in st.session_state:
    st.session_state["pending_questions"] = {}

# Holds the most recent retrieval debug info per video, so it can be
# displayed persistently (see render_chat_content) instead of
# flashing and disappearing on the rerun that follows each answer.
if "retrieval_debug" not in st.session_state:
    st.session_state["retrieval_debug"] = {}

# Read once, up front, so the CSS block below can add/remove the
# fixed-panel layout rules depending on whether chat is open.
chat_is_open = st.session_state.get("chat_open", False)


# ============================================================
# CUSTOM CSS
# ============================================================

CHAT_PANEL_WIDTH = 420

chat_panel_css = (
    f"""
    .block-container {{
        padding-right: {CHAT_PANEL_WIDTH + 40}px;
    }}

    .st-key-chat_panel_fixed {{
        position: fixed;
        top: 3.75rem;
        right: 0;
        width: {CHAT_PANEL_WIDTH}px;
        height: calc(100vh - 3.75rem);
        overflow-y: auto;
        z-index: 999;
        padding: 1.25rem;
        border-left: 1px solid rgba(128, 128, 128, 0.20);
    }}
    """
    if chat_is_open else ""
)

st.markdown(
    f"""
    <style>

    /* ======================================================
       GENERAL
       ====================================================== */

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}


    /* ======================================================
       LEFT SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {{
        border-right: 1px solid rgba(128, 128, 128, 0.20);
    }}

    section[data-testid="stSidebar"] .stButton button {{
        text-align: left;
    }}


    /* ======================================================
       RIGHT CHAT SIDEBAR (fixed to the viewport, same idea as
       Streamlit's own left sidebar, only added via CSS since
       Streamlit only ships one built-in sidebar)
       ====================================================== */

    .chat-panel-title {{
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1.3;
    }}

    .chat-panel-subtitle {{
        font-size: 0.8rem;
        opacity: 0.65;
        margin-top: -4px;
        margin-bottom: 6px;
    }}

    .max-chat-title {{
        font-size: 2rem;
        font-weight: 700;
    }}

    {chat_panel_css}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DISPLAY NOTES + VISUALS
# ============================================================

def display_notes_with_visuals(
        notes,
        diagrams
):

    if not notes:
        return

    for line in notes.splitlines():

        if "[VISUAL:" in line:

            try:

                start = (
                        line.index("[VISUAL:")
                        + len("[VISUAL:")
                )

                end = line.index(
                    "]",
                    start
                )

                visual_id = int(
                    line[start:end]
                )

                if 1 <= visual_id <= len(diagrams):

                    diagram = diagrams[visual_id - 1]
                    image_path = os.path.abspath(diagram["path"])

                    if os.path.exists(image_path):

                        st.image(
                            image_path,
                            caption=diagram.get("reason", ""),
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "Visual file not found:\n"
                            f"{image_path}"
                        )

            except (ValueError, KeyError, IndexError):
                pass

        else:

            st.markdown(line)


# ============================================================
# SAFE PDF TEXT
# ============================================================

def safe_pdf_text(text):
    return str(text).encode("latin-1", "replace").decode("latin-1")


# ============================================================
# PDF GENERATION
# ============================================================

def generate_pdf_bytes(title, notes, diagrams):

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, safe_pdf_text(title))
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 11)

    for line in notes.splitlines():

        if "[VISUAL:" in line:

            try:
                start = line.index("[VISUAL:") + len("[VISUAL:")
                end = line.index("]", start)
                visual_id = int(line[start:end])

                if 1 <= visual_id <= len(diagrams):

                    diagram = diagrams[visual_id - 1]
                    image_path = os.path.abspath(diagram["path"])

                    if os.path.exists(image_path):

                        pdf.ln(2)
                        page_width = pdf.w - pdf.l_margin - pdf.r_margin
                        pdf.image(image_path, w=page_width)
                        pdf.ln(2)

                        reason = diagram.get("reason", "")

                        if reason:
                            pdf.set_font("Helvetica", "I", 9)
                            pdf.multi_cell(0, 6, safe_pdf_text(reason))
                            pdf.set_font("Helvetica", "", 11)

            except (ValueError, KeyError, IndexError):
                pass

            continue

        stripped = line.strip()

        if stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 8, safe_pdf_text(stripped[4:]))
            pdf.set_font("Helvetica", "", 11)

        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 9, safe_pdf_text(stripped[3:]))
            pdf.set_font("Helvetica", "", 11)

        elif stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 15)
            pdf.multi_cell(0, 10, safe_pdf_text(stripped[2:]))
            pdf.set_font("Helvetica", "", 11)

        elif stripped.startswith("- ") or stripped.startswith("* "):
            pdf.multi_cell(0, 7, "- " + safe_pdf_text(stripped[2:]))

        elif stripped == "":
            pdf.ln(3)

        else:
            pdf.multi_cell(0, 7, safe_pdf_text(stripped))

    output = pdf.output(dest="S")

    if isinstance(output, str):
        return output.encode("latin-1")

    return bytes(output)


# ============================================================
# GET VIDEO INDEX
# ============================================================

def get_video_index(video_id, transcript):

    if video_id not in st.session_state["rag_indexes"]:

        with st.spinner("Preparing video for chat..."):

            # Pass the saved notes through too, so "what is this
            # video about"-style questions have something retrievable
            # even for videos that were indexed before this feature
            # existed (this fallback path rebuilds the index fresh).
            notes = load_notes(video_id)

            st.session_state["rag_indexes"][video_id] = create_index(
                transcript,
                video_id,
                notes
            )

    return st.session_state["rag_indexes"][video_id]


# ============================================================
# GET VIDEO TRANSCRIPT
# ============================================================

def get_video_transcript(video_id, url):

    if video_id not in st.session_state["transcripts"]:

        with st.spinner("Loading video transcript..."):
            st.session_state["transcripts"][video_id] = fetch_transcript(url)

    return st.session_state["transcripts"][video_id]


# ============================================================
# ADD CHAT MESSAGE
# ============================================================

def add_chat_message(video_id, role, content, sources=None):

    if video_id not in st.session_state["chat_history"]:
        st.session_state["chat_history"][video_id] = []

    message = {"role": role, "content": content}

    if sources:
        message["sources"] = sources

    st.session_state["chat_history"][video_id].append(message)


# ============================================================
# RENDER CHAT CONTENT
#
# Unchanged in behavior from before — this just draws the message
# history, the pending-general-answer prompt, and the chat input.
# It no longer cares whether it's being called from the docked panel
# or the maximized full-page view; both just call this.
# ============================================================

def render_chat_content(video_id, url, input_key_suffix=""):

    transcript = get_video_transcript(video_id, url)

    history = st.session_state["chat_history"].get(video_id, [])

    for message in history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            sources = message.get("sources")

            if sources:

                with st.expander("📌 Sources from video"):
                    for source in sources:
                        st.caption(source)

    # --------------------------------------------------------
    # RETRIEVAL DEBUG (persistent — stays visible until the next
    # question overwrites it). Temporary aid for tuning the
    # has_sufficient_evidence threshold against real numbers;
    # safe to remove once retrieval is behaving as expected.
    # --------------------------------------------------------

    debug_info = st.session_state["retrieval_debug"].get(video_id)

    if debug_info:

        with st.expander(
                f"🔍 Retrieval debug — \"{debug_info['question']}\"",
                expanded=False
        ):

            st.caption(f"scope = {debug_info.get('scope', 'specific')}")
            st.caption(f"sufficient = {debug_info['sufficient']}")

            for chunk in debug_info["results"]:
                st.write(
                    f"section={chunk['section_id']} "
                    f"chunk={chunk['chunk_id']} "
                    f"rerank_score={chunk['score']:.3f} "
                    f"vector_score={chunk['vector_score']:.3f}"
                )
                st.caption(chunk["text"][:200])

    # --------------------------------------------------------
    # ANSWER A QUESTION THAT WAS JUST SUBMITTED
    #
    # By the time we get here, the user's question is already in
    # `history` above (added on the previous run, before the
    # rerun that got us to this run) — so it's already visible.
    # This is where we actually do the retrieval + generation,
    # inside a visible "thinking" bubble.
    # --------------------------------------------------------

    pending = st.session_state["pending_questions"].get(video_id)

    if pending:

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    scope = classify_question_scope(pending)

                    if scope == "broad":

                        notes_text = load_notes(video_id)

                        if notes_text:

                            # `transcript` is already loaded above via
                            # get_video_transcript() — reused here so
                            # broad questions can draw on everything
                            # the video actually said, not just what
                            # made it into the (trimmed) notes.
                            answer = answer_broad_question(pending, notes_text, transcript)

                            add_chat_message(video_id, "assistant", answer)

                            st.session_state["retrieval_debug"][video_id] = {
                                "question": pending,
                                "scope": "broad (answered from notes + full transcript, no retrieval)",
                                "sufficient": True,
                                "results": []
                            }

                        else:
                            # No saved notes to fall back on for some
                            # reason — treat as specific so it still
                            # gets a real attempt via retrieval.
                            scope = "specific"

                    if scope == "specific":

                        index = get_video_index(video_id, transcript)

                        results = retrieve(index, pending, k=5)

                        sufficient = has_sufficient_evidence(results, threshold=0.3)

                        # Stored (not rendered here) since this whole
                        # block disappears on the rerun below — it's
                        # displayed persistently near the chat input
                        # instead, until the next question overwrites it.
                        st.session_state["retrieval_debug"][video_id] = {
                            "question": pending,
                            "scope": "specific",
                            "sufficient": sufficient,
                            "results": results
                        }

                        if sufficient:

                            answer = answer_from_video(pending, results)
                            sources = format_sources(results)

                            add_chat_message(video_id, "assistant", answer, sources)

                        else:

                            st.session_state["pending_general_question"] = pending
                            st.session_state["pending_general_video"] = video_id

                except Exception as e:
                    st.error(f"Chat error: {e}")

        st.session_state["pending_questions"][video_id] = None
        st.rerun()

    pending_question = st.session_state.get("pending_general_question")
    pending_video = st.session_state.get("pending_general_video")

    if pending_question and pending_video == video_id:

        st.info(
            "The video does not contain enough "
            "information to answer this question."
        )

        st.write("Would you like me to answer using general knowledge?")

        yes_col, no_col = st.columns(2)

        with yes_col:

            if st.button(
                    "Yes, answer generally",
                    key=f"general_yes_{video_id}{input_key_suffix}",
                    use_container_width=True
            ):

                with st.spinner("Generating answer..."):
                    answer = answer_generally(pending_question)

                add_chat_message(video_id, "assistant", answer)

                st.session_state["pending_general_question"] = None
                st.session_state["pending_general_video"] = None

                st.rerun()

        with no_col:

            if st.button(
                    "No",
                    key=f"general_no_{video_id}{input_key_suffix}",
                    use_container_width=True
            ):

                st.session_state["pending_general_question"] = None
                st.session_state["pending_general_video"] = None

                st.rerun()

    question = st.chat_input(
        "Ask anything about this video...",
        key=f"chat_input_{video_id}{input_key_suffix}"
    )

    if not question:
        return

    # Show the question immediately: add it to history and rerun
    # right away, before doing any retrieval or LLM work. The
    # "pending" block above picks it up and answers it on the very
    # next run.
    add_chat_message(video_id, "user", question)
    st.session_state["pending_questions"][video_id] = question
    st.rerun()


# ============================================================
# LEFT SIDEBAR — SAVED NOTES
# ============================================================

st.sidebar.title("📚 Saved Notes")

saved_notes = get_all_notes()

if not saved_notes:

    st.sidebar.info("No saved notes yet.")

else:

    for video in saved_notes:

        vid = video["video_id"]
        confirm_key = f"confirm_delete_{vid}"

        row_col1, row_col2 = st.sidebar.columns([5, 1])

        with row_col1:

            title = video.get("title", "Untitled Video")
            display_sidebar_title = title[:35] + "..." if len(title) > 35 else title
            is_selected = st.session_state.get("selected_video") == vid
            button_label = ("▶ " + display_sidebar_title) if is_selected else display_sidebar_title

            if st.button(button_label, key=f"saved_video_{vid}", use_container_width=True):

                st.session_state["selected_video"] = vid
                st.session_state["chat_open"] = False
                st.session_state["chat_maximized"] = False
                st.session_state["pending_general_question"] = None
                st.session_state["pending_general_video"] = None

                st.rerun()

        with row_col2:

            if st.button("🗑️", key=f"delete_video_{vid}", help="Delete notes", use_container_width=True):
                st.session_state[confirm_key] = True

        if st.session_state.get(confirm_key, False):

            st.sidebar.warning(f"Delete all notes for **{video['title']}**?")

            confirm_col1, confirm_col2 = st.sidebar.columns(2)

            with confirm_col1:

                if st.button("Yes", key=f"confirm_delete_yes_{vid}", use_container_width=True):

                    delete_video(vid)
                    st.session_state[confirm_key] = False

                    st.session_state["rag_indexes"].pop(vid, None)
                    st.session_state["transcripts"].pop(vid, None)
                    st.session_state["chat_history"].pop(vid, None)

                    if st.session_state.get("selected_video") == vid:
                        st.session_state["selected_video"] = None
                        st.session_state["chat_open"] = False

                    st.rerun()

            with confirm_col2:

                if st.button("Cancel", key=f"confirm_delete_no_{vid}", use_container_width=True):
                    st.session_state[confirm_key] = False
                    st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.title("📚 YouTube Notes Generator")
st.write("Generate detailed, structured notes from any YouTube video.")


# ============================================================
# SELECTED VIDEO
# ============================================================

selected_video = st.session_state.get("selected_video")


# ============================================================
# MAXIMIZED CHAT (full page — unchanged concept, still available
# via the ↗️ button inside the docked panel)
# ============================================================

if selected_video and st.session_state.get("chat_maximized"):

    metadata = load_metadata(selected_video)

    if metadata:
        selected_url = metadata.get("url")
        display_title = metadata.get("title", "Video Chat")
    else:
        selected_url = None
        display_title = "Video Chat"

    title_col, restore_col, close_col = st.columns([7, 1, 1])

    with title_col:
        st.markdown('<div class="max-chat-title">💬 Chat with this video</div>', unsafe_allow_html=True)
        st.caption(display_title)

    with restore_col:

        if st.button("↙️", key=f"restore_max_chat_{selected_video}", help="Restore panel view"):
            st.session_state["chat_maximized"] = False
            st.session_state["chat_open"] = True
            st.rerun()

    with close_col:

        if st.button("✕", key=f"close_max_chat_{selected_video}", help="Close chat"):
            st.session_state["chat_maximized"] = False
            st.session_state["chat_open"] = False
            st.rerun()

    st.divider()

    if selected_url:
        render_chat_content(selected_video, selected_url, input_key_suffix="_max")
    else:
        st.error("Could not find the YouTube URL for this video.")

    st.stop()


# ============================================================
# YOUTUBE URL
# ============================================================

url = st.text_input(
    "YouTube URL",
    placeholder="Paste a YouTube video URL here...",
    key="youtube_url_input"
)


# ============================================================
# GENERATE NOTES
# ============================================================

if st.button("Generate Notes", type="primary", key="generate_notes_button"):

    if not url:

        st.warning("Please enter a YouTube URL.")

    else:

        try:

            video_id = extract_youtube_id(url)

            if not video_id:
                st.error("Could not extract the YouTube video ID.")
                st.stop()

            with st.spinner("Generating notes (this can take a minute)..."):
                notes, diagrams = generate_notes_with_diagrams(url)

            with st.spinner("Preparing video information..."):
                transcript = fetch_transcript(url)
                topic = generate_topic(transcript)

            title = topic.strip()

            version = save_notes(video_id, title, url, notes, diagrams)

            st.session_state["transcripts"][video_id] = transcript

            if video_id not in st.session_state["rag_indexes"]:

                with st.spinner("Preparing video for chat..."):
                    st.session_state["rag_indexes"][video_id] = create_index(transcript, video_id, notes)

            st.session_state["selected_video"] = video_id
            st.session_state["chat_open"] = False
            st.session_state["chat_maximized"] = False
            st.session_state["pending_general_question"] = None
            st.session_state["pending_general_video"] = None

            st.rerun()

        except Exception as e:
            st.error(f"Something went wrong: {e}")


# ============================================================
# DISPLAY SELECTED NOTES + DOCKED CHAT PANEL
# ============================================================

if selected_video:

    current_notes = load_notes(selected_video)

    if current_notes:

        current_diagrams = load_diagrams(selected_video)
        metadata = load_metadata(selected_video)

        if metadata:
            display_title = metadata.get("title", "Notes")
            selected_url = metadata.get("url")
        else:
            display_title = "Notes"
            selected_url = None

        # --------------------------------------------------------
        # LAYOUT: notes always render full-width now. The chat
        # panel is a position:fixed overlay (see CSS above), not a
        # grid column, so it doesn't need to steal space here — the
        # CSS block-container padding-right does that instead.
        # --------------------------------------------------------

        notes_col = st.container()

        with notes_col:

            title_col, pdf_col, chat_btn_col = st.columns([6, 1.2, 1.6])

            with title_col:
                st.header(display_title)

            with pdf_col:

                pdf_bytes = generate_pdf_bytes(display_title, current_notes, current_diagrams)
                safe_filename = re.sub(r'[\\/*?:"<>|]', "", display_title)
                safe_filename = safe_filename[:50] or selected_video

                st.download_button(
                    label="⬇️ PDF",
                    data=pdf_bytes,
                    file_name=f"{safe_filename}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"download_pdf_{selected_video}"
                )

            with chat_btn_col:

                chat_button_label = "✕ Close chat" if chat_is_open else "💬 Chat"

                if st.button(
                        chat_button_label,
                        type="primary",
                        use_container_width=True,
                        key=f"toggle_chat_{selected_video}"
                ):

                    st.session_state["chat_open"] = not chat_is_open
                    st.rerun()

            st.divider()

            display_notes_with_visuals(current_notes, current_diagrams)

            st.divider()

            with st.expander("🕘 Version History"):

                versions = get_versions(selected_video)

                if not versions:

                    st.info("No versions available.")

                else:

                    for version in versions:

                        version_col1, version_col2 = st.columns([5, 1])

                        with version_col1:

                            st.write(f"**Version {version['version']}**")
                            st.caption(version["created_at"])

                            if version["is_current"]:
                                st.caption("Current version")

                        with version_col2:

                            if not version["is_current"]:

                                if st.button(
                                        "Revert",
                                        key=f"revert_{selected_video}_{version['id']}",
                                        use_container_width=True
                                ):

                                    success = revert_notes(selected_video, version["version"])

                                    if success:
                                        st.success(f"Reverted to version {version['version']}")
                                        st.rerun()
                                    else:
                                        st.error("Could not revert this version.")

                        st.divider()

        # --------------------------------------------------------
        # FIXED RIGHT CHAT SIDEBAR
        #
        # key="chat_panel_fixed" gives this container the CSS class
        # ".st-key-chat_panel_fixed", which the CSS block above pins
        # to the right edge of the viewport at full height — the
        # same technique Streamlit's own left sidebar effectively
        # uses. It renders outside the notes_col flow entirely, so
        # its screen position comes purely from the fixed CSS, not
        # from where this code happens to sit in the script.
        # --------------------------------------------------------

        if chat_is_open:

            with st.container(key="chat_panel_fixed"):

                header_col, maximize_col = st.columns([5, 1])

                with header_col:
                    st.markdown('<div class="chat-panel-title">💬 Chat with this video</div>', unsafe_allow_html=True)
                    st.markdown('<div class="chat-panel-subtitle">Ask questions based on the video</div>', unsafe_allow_html=True)

                with maximize_col:

                    if st.button("↗️", key=f"maximize_chat_{selected_video}", help="Maximize chat"):
                        st.session_state["chat_maximized"] = True
                        st.rerun()

                if selected_url:
                    render_chat_content(selected_video, selected_url, input_key_suffix="_panel")
                else:
                    st.error("Could not find the YouTube URL for this video.")