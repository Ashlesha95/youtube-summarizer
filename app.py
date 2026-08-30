import streamlit as st

from extractor import fetch_transcript
from preprocess import preprocess_transcript
from llm import correct_transcript, generate_notes
from database import (
    initialize_database,
    save_notes,
    load_notes,
    get_all_notes,
    get_versions,
    revert_notes
)


# --------------------------------------------------
# INITIALIZE DATABASE
# --------------------------------------------------

initialize_database()


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="YouTube Notes",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📚 YouTube Notes Generator")

st.write(
    "Generate detailed, structured notes from any YouTube video."
)


# --------------------------------------------------
# SIDEBAR - SAVED NOTES
# --------------------------------------------------

st.sidebar.title("📚 Saved Notes")

saved_notes = get_all_notes()

if not saved_notes:

    st.sidebar.info("No saved notes yet.")

else:

    for video in saved_notes:

        if st.sidebar.button(
                video["title"],
                key=f"video_{video['video_id']}"
        ):

            st.session_state["selected_video"] = video["video_id"]


# --------------------------------------------------
# YOUTUBE URL
# --------------------------------------------------

url = st.text_input(
    "YouTube URL",
    placeholder="Paste a YouTube video URL here..."
)


# --------------------------------------------------
# GENERATE NOTES
# --------------------------------------------------

if st.button("Generate Notes", type="primary"):

    if not url:

        st.warning("Please enter a YouTube URL.")

    else:

        try:

            with st.spinner("Extracting transcript..."):

                transcript = fetch_transcript(url)

            with st.spinner("Cleaning transcript..."):

                clean_transcript = preprocess_transcript(
                    transcript
                )

            with st.spinner("Correcting transcript..."):

                corrected_transcript = correct_transcript(
                    clean_transcript
                )

            with st.spinner("Generating notes..."):

                notes = generate_notes(
                    corrected_transcript
                )

            # ------------------------------------------
            # GET VIDEO ID
            # ------------------------------------------

            from extractor import extract_youtube_id

            video_id = extract_youtube_id(url)

            # ------------------------------------------
            # TITLE
            # ------------------------------------------

            title = f"YouTube Video - {video_id}"

            # ------------------------------------------
            # SAVE
            # ------------------------------------------

            version = save_notes(
                video_id,
                title,
                url,
                notes
            )

            st.session_state["selected_video"] = video_id

            st.success(
                f"Notes generated successfully! Version {version}"
            )

        except Exception as e:

            st.error(
                f"Something went wrong: {e}"
            )


# --------------------------------------------------
# DISPLAY SELECTED NOTES
# --------------------------------------------------

selected_video = st.session_state.get(
    "selected_video"
)


if selected_video:

    current_notes = load_notes(
        selected_video
    )

    if current_notes:

        st.divider()

        st.markdown(
            current_notes
        )


        # ------------------------------------------
        # VERSION HISTORY
        # ------------------------------------------

        st.divider()

        st.subheader("Version History")

        versions = get_versions(
            selected_video
        )

        for version in versions:

            col1, col2, col3 = st.columns(
                [2, 2, 1]
            )

            with col1:

                st.write(
                    f"Version {version['version']}"
                )

            with col2:

                st.write(
                    version["created_at"]
                )

            with col3:

                if version["is_current"]:

                    st.write("Current")

                else:

                    if st.button(
                            "Revert",
                            key=f"revert_{version['id']}"
                    ):

                        success = revert_notes(
                            selected_video,
                            version["version"]
                        )

                        if success:

                            st.success(
                                f"Reverted to version "
                                f"{version['version']}"
                            )

                            st.rerun()

                        else:

                            st.error(
                                "Could not revert this version."
                            )