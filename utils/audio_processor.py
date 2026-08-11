import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = "downloads"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_yt_video(url: str) -> str:

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(id)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",

        "outtmpl": output_path,

        "noplaylist": True,

        # YouTube JS runtime
        "js_runtimes": {
            "deno": {}
        },

        # Retry settings
        "retries": 10,
        "fragment_retries": 10,

        # Don't download playlist
        "noplaylist": True,

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],

        "quiet": False,
    }

    try:

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info = ydl.extract_info(
                url,
                download=True
            )

            video_id = info["id"]

            wav_path = os.path.join(
                DOWNLOAD_DIR,
                f"{video_id}.wav"
            )

        if not os.path.exists(wav_path):
            raise FileNotFoundError(
                f"WAV file was not created: {wav_path}"
            )

        print(f"Audio downloaded: {wav_path}")

        return wav_path

    except yt_dlp.utils.DownloadError as e:

        print(f"yt-dlp error: {e}")

        raise

    except Exception as e:

        print(f"Unexpected error: {e}")

        raise






def convert_to_wav( path : str) -> str :
    try:
        output_path = os.path.splitext(path)[0] + ".wav"

        audio = AudioSegment.from_file(path)

        audio = (
        audio
        .set_channels(1)      # Mono
        .set_frame_rate(16000) # 16 kHz
    )

        audio.export(output_path, format="wav")

        return output_path

    except Exception as e:
        print('error:',e)



def audio_chunk(path :str ,chunk_min : float = 10) -> str:   
    audio = AudioSegment.from_wav(path)     
    chunks_times = chunk_min * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunks_times)):
        chunk = audio[start : start + chunks_times]
        chunk_path = f'{path}_chunk_{i}.wav'
        chunk.export(chunk_path,format='wav')

        chunks.append(chunk_path)


    return chunks    
    
        

def process_input(source :str) -> list:
    if source.startswith('http://') or source.startswith('https://'):
        print('Detected URL. Downloading audio...')
        wav_path = download_yt_video(source)


    else:
        print("Detected local file .Converting to WAV...")
        wav_path = convert_to_wav(source)


    print('Chunking Audio....')
    chunks = audio_chunk(wav_path)
    print(f'Audio ready - {len(chunks)} chunk(s) created.')

    return chunks        




  