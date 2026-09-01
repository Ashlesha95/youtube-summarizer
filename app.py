import os

import streamlit as st

from transcript_extractor import (
    fetch_transcript,
    extract_youtube_id
)

from llm import (
    generate_notes_with_diagrams,
    generate_topic
)

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

st.title(
    "📚 YouTube Notes Generator"
)

st.write(
    "Generate detailed, structured notes from any YouTube video."
)


# --------------------------------------------------
# DISPLAY NOTES + VISUALS
# --------------------------------------------------

def display_notes_with_visuals(
        notes,
        diagrams
):

    for line in notes.splitlines():

        # ------------------------------------------
        # VISUAL MARKER
        # ------------------------------------------

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

                # ----------------------------------
                # VALID VISUAL ID
                # ----------------------------------

                if (
                        1 <= visual_id
                        <= len(diagrams)
                ):

                    diagram = diagrams[
                        visual_id - 1
                        ]

                    image_path = diagram[
                        "path"
                    ]

                    # ----------------------------------
                    # MAKE ABSOLUTE PATH
                    # ----------------------------------

                    image_path = os.path.abspath(
                        image_path
                    )

                    # ----------------------------------
                    # CHECK FILE
                    # ----------------------------------

                    if os.path.exists(
                            image_path
                    ):

                        st.image(
                            image_path,
                            caption=diagram.get(
                                "reason",
                                ""
                            ),
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "Visual file not found:\n"
                            f"{image_path}"
                        )

            except (
                    ValueError,
                    KeyError,
                    IndexError
            ):

                pass

        else:

            # --------------------------------------
            # NORMAL MARKDOWN
            # --------------------------------------

            st.markdown(
                line
            )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title(
    "📚 Saved Notes"
)

saved_notes = get_all_notes()


if not saved_notes:

    st.sidebar.info(
        "No saved notes yet."
    )

else:

    for video in saved_notes:

        if st.sidebar.button(
                video["title"],
                key=f"video_{video['video_id']}"
        ):

            st.session_state[
                "selected_video"
            ] = video["video_id"]

            st.session_state[
                "just_generated"
            ] = False


# --------------------------------------------------
# YOUTUBE URL
# --------------------------------------------------

url = st.text_input(
    "YouTube URL",
    placeholder=(
        "Paste a YouTube video URL here..."
    )
)


# --------------------------------------------------
# GENERATE NOTES
# --------------------------------------------------

if st.button(
        "Generate Notes",
        type="primary"
):

    if not url:

        st.warning(
            "Please enter a YouTube URL."
        )

    else:

        try:

            with st.spinner(
                    "Generating notes (this can take a minute)..."
            ):

                # ----------------------------------
                # GENERATE NOTES + DIAGRAMS
                # ----------------------------------

                notes, diagrams = (
                    generate_notes_with_diagrams(
                        url
                    )
                )

            # ----------------------------------
            # STORE DIAGRAMS
            # ----------------------------------

            st.session_state[
                "diagrams"
            ] = diagrams

            # ----------------------------------
            # GET TOPIC
            # ----------------------------------

            transcript = fetch_transcript(
                url
            )

            topic = generate_topic(
                transcript
            )

            title = topic

            # ----------------------------------
            # VIDEO ID
            # ----------------------------------

            video_id = extract_youtube_id(
                url
            )

            # ----------------------------------
            # SAVE NOTES
            # ----------------------------------

            version = save_notes(
                video_id,
                title,
                url,
                notes
            )

            # ----------------------------------
            # SESSION STATE
            # ----------------------------------

            st.session_state[
                "selected_video"
            ] = video_id

            st.session_state[
                "just_generated"
            ] = True

            # ----------------------------------
            # SUCCESS
            # ----------------------------------

            st.success(
                f"Notes generated successfully! "
                f"Version {version}"
            )

            # ----------------------------------
            # DISPLAY IMMEDIATELY
            # ----------------------------------

            st.divider()

            display_notes_with_visuals(
                notes,
                diagrams
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

        # ------------------------------------------
        # ONLY DISPLAY HERE IF NOT JUST GENERATED
        # ------------------------------------------

        if not st.session_state.get(
                "just_generated",
                False
        ):

            st.divider()

            diagrams = st.session_state.get(
                "diagrams",
                []
            )

            display_notes_with_visuals(
                current_notes,
                diagrams
            )


        # ------------------------------------------
        # RESET FLAG
        # ------------------------------------------

        st.session_state[
            "just_generated"
        ] = False


        # ------------------------------------------
        # VERSION HISTORY
        # ------------------------------------------

        st.divider()

        st.subheader(
            "Version History"
        )

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

                    st.write(
                        "Current"
                    )

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