from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.whisper_transcription import transcribe_all
from core.summarize import summarize ,generate_title
from core.rag_engine import build_rag_chain , question
from core.extractor import extract_action_items ,extract_key_decisions, extract_questions

load_dotenv()

def run_pipeline(source:str) ->dict:
    print("starting AI Video Assistant")

    chunks = process_input(source)

    transcript = transcribe_all(chunks)
    print(f"raw transcription (first 300 characters ) {transcript[:300]}")

    title = generate_title(transcript)

    summary = summarize(transcript)

    action_item = extract_action_items(transcript)

    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }

if __name__ == "__main__":

    source = input("Enter YouTube URL or local file path: ").strip()
    result = run_pipeline(source)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)


    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        user_question = input("You: ").strip()
        if user_question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not user_question:
            continue
        answer = question(rag_chain, user_question)
        print(f"\n🤖 Assistant: {answer}\n")


