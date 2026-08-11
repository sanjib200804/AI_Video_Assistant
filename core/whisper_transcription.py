import os
import sys

venv = sys.prefix

cublas_path = os.path.join(
    venv,
    "Lib",
    "site-packages",
    "nvidia",
    "cublas",
    "bin"
)

cudnn_path = os.path.join(
    venv,
    "Lib",
    "site-packages",
    "nvidia",
    "cudnn",
    "bin"
)

os.add_dll_directory(cublas_path)
os.add_dll_directory(cudnn_path)

os.environ["PATH"] += os.pathsep + cublas_path
os.environ["PATH"] += os.pathsep + cudnn_path

from faster_whisper import WhisperModel

_model = None


def load_model():
    global _model
    if _model is None:
        print("faster whisper model is loaded...")
        _model = WhisperModel(
                    "small",
                    device="cuda",
                    compute_type="int8_float16"
)


        print('model loading')


    return _model



def transcribe_audio (path : str , translate : bool = False) ->str:

    model = load_model()

    task = 'translate' if translate else 'transcribe'

    segments,info = model.transcribe(
        path,
        task= task
    )

    transcript = ""


    for segment in segments:
        transcript += segment.text + " "

    return transcript.strip()



def transcribe_all(chunks: list[str], translate: bool = False) -> str:
    full_transcript = ""

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}...")

        text = transcribe_audio(chunk, translate)

        full_transcript += text + "\n"

    print("Transcription complete!")

    return full_transcript