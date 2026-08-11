
import streamlit as st

from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.whisper_transcription import transcribe_all
from core.summarize import summarize, generate_title
from core.rag_engine import build_rag_chain, question
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions
)

load_dotenv()


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎥",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #888;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: #1e1e1e;
        border: 1px solid #333;
        margin-bottom: 15px;
    }

    .chat-user {
        background-color: #2563eb;
        color: white;
        padding: 12px 16px;
        border-radius: 15px;
        margin: 8px 0;
    }

    .chat-assistant {
        background-color: #262626;
        padding: 12px 16px;
        border-radius: 15px;
        margin: 8px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎥 AI Video Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Summarize videos, extract meeting insights and chat with your video'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("🎬 Video Input")

source = st.text_input(
    "YouTube URL or local file path",
    placeholder="https://youtube.com/watch?v=..."
)

language = st.selectbox(
    "Language",
    ["english", "hinglish"]
)


# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

if st.button(
    "🚀 Analyze Video",
    type="primary",
    use_container_width=True
):

    if not source:
        st.warning("Please enter a YouTube URL or local file path.")

    else:

        try:

            # ------------------------------------------
            # STEP 1 - DOWNLOAD / PROCESS AUDIO
            # ------------------------------------------

            with st.status(
                "Processing video...",
                expanded=True
            ) as status:

                st.write("🎵 Processing audio...")

                chunks = process_input(source)

                st.write(
                    f"✅ Audio ready - {len(chunks)} chunks created."
                )


                # ------------------------------------------
                # STEP 2 - TRANSCRIPTION
                # ------------------------------------------

                st.write("🎙️ Transcribing audio...")

                transcript = transcribe_all(chunks)

                st.write("✅ Transcription complete.")


                # ------------------------------------------
                # STEP 3 - TITLE
                # ------------------------------------------

                st.write("🏷️ Generating title...")

                title = generate_title(transcript)

                st.write("✅ Title generated.")


                # ------------------------------------------
                # STEP 4 - SUMMARY
                # ------------------------------------------

                st.write("📋 Generating summary...")

                summary = summarize(transcript)

                st.write("✅ Summary generated.")


                # ------------------------------------------
                # STEP 5 - EXTRACT INFORMATION
                # ------------------------------------------

                st.write("🔎 Extracting meeting insights...")

                action_items = extract_action_items(transcript)

                decisions = extract_key_decisions(transcript)

                open_questions = extract_questions(transcript)

                st.write("✅ Meeting insights extracted.")


                # ------------------------------------------
                # STEP 6 - BUILD RAG
                # ------------------------------------------

                st.write("🧠 Building RAG system...")

                rag_chain = build_rag_chain(transcript)

                st.write("✅ RAG system ready.")


                # ------------------------------------------
                # SAVE RESULT
                # ------------------------------------------

                st.session_state.result = {
                    "title": title,
                    "transcript": transcript,
                    "summary": summary,
                    "action_items": action_items,
                    "key_decisions": decisions,
                    "open_questions": open_questions
                }

                st.session_state.rag_chain = rag_chain

                st.session_state.chat_history = []

                status.update(
                    label="🎉 Video analysis complete!",
                    state="complete"
                )


        except Exception as e:

            st.error(
                f"❌ Something went wrong:\n\n{str(e)}"
            )


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if st.session_state.result:

    result = st.session_state.result

    st.divider()

    # ----------------------------------------------
    # TITLE
    # ----------------------------------------------

    st.header("📌 " + result["title"])


    # ----------------------------------------------
    # SUMMARY
    # ----------------------------------------------

    st.subheader("📋 Summary")

    st.markdown(
        f"""
        <div class="card">
        {result["summary"]}
        </div>
        """,
        unsafe_allow_html=True
    )


    # ----------------------------------------------
    # THREE COLUMNS
    # ----------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.subheader("✅ Action Items")

        st.markdown(
            f"""
            <div class="card">
            {result["action_items"]}
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.subheader("🔑 Key Decisions")

        st.markdown(
            f"""
            <div class="card">
            {result["key_decisions"]}
            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.subheader("❓ Open Questions")

        st.markdown(
            f"""
            <div class="card">
            {result["open_questions"]}
            </div>
            """,
            unsafe_allow_html=True
        )


    # ----------------------------------------------
    # TRANSCRIPT
    # ----------------------------------------------

    with st.expander("📄 View Full Transcript"):

        st.text_area(
            "Transcript",
            result["transcript"],
            height=400
        )


    # ----------------------------------------------
    # CHAT
    # ----------------------------------------------

    st.divider()

    st.header("💬 Chat with your meeting")

    st.caption(
        "Ask questions about the video transcription."
    )


    # ----------------------------------------------
    # CHAT HISTORY
    # ----------------------------------------------

    for chat in st.session_state.chat_history:

        if chat["role"] == "user":

            st.markdown(
                f"""
                <div class="chat-user">
                <b>You:</b> {chat["message"]}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="chat-assistant">
                <b>🤖 Assistant:</b> {chat["message"]}
                </div>
                """,
                unsafe_allow_html=True
            )


    # ----------------------------------------------
    # CHAT INPUT
    # ----------------------------------------------

    user_question = st.chat_input(
        "Ask something about the meeting..."
    )


    if user_question:

        # Save user message

        st.session_state.chat_history.append(
            {
                "role": "user",
                "message": user_question
            }
        )


        try:

            # Get answer from RAG

            answer = question(
                st.session_state.rag_chain,
                user_question
            )


            # Save assistant response

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "message": answer
                }
            )


            # Refresh UI

            st.rerun()


        except Exception as e:

            st.error(
                f"❌ Chat error: {str(e)}"
            )

