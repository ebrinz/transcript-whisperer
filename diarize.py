import os
import json
import glob
import tempfile
import subprocess
from dotenv import load_dotenv
from pyannote.audio.pipelines import SpeechSeparation
from pyannote.audio.pipelines.utils.hook import ProgressHook
from loguru import logger
import scipy.io.wavfile
import torchaudio
import argparse

def timestamp_to_seconds(ts):
    h, m, s_ms = ts.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

# Load environment variables from .env
load_dotenv()

# Parse command line arguments for test mode
parser = argparse.ArgumentParser(description="Diarization script")
parser.add_argument("--test", action="store_true", help="Run in test mode (use test/input and test/output)")
args = parser.parse_args()

INPUT_DIR = "test/input" if args.test else "input"
OUTPUT_DIR = "test/output" if args.test else "output"

HF_TOKEN = os.environ["HF_TOKEN"]
WHISPER_CPP_BIN = os.getenv("WHISPER_CPP_BIN", os.path.join("whisper.cpp", "build", "bin", "whisper-cli"))
WHISPER_CPP_MODEL = os.getenv("WHISPER_CPP_MODEL", os.path.join("whisper.cpp", "models", "ggml-large-v3-turbo.bin"))

# Instantiate pipeline once
pipeline = SpeechSeparation.from_pretrained(
    "pyannote/speech-separation-ami-1.0",
    use_auth_token=HF_TOKEN
)

def run_whisper_cpp(wav_path, out_dir, name_no_ext):
    # Output JSON and TXT will be named {name_no_ext}.json and {name_no_ext}.txt in OUT_DIR
    whisper_cmd = [
        WHISPER_CPP_BIN,
        "-m", WHISPER_CPP_MODEL,
        "-f", wav_path,
        # "-tdrz",  # diarization output (includes word-level by default)
        "-oj",
        "-of", os.path.join(out_dir, name_no_ext),
        "-otxt",  # output plain text transcript
        "--max-len", "15", 
        "-pp"     # print progress (optional, for more stdout info)
        # removed --word_timestamps, not supported in your build
    ]
    logger.info(f"Running whisper.cpp: {' '.join(whisper_cmd)}")
    result = subprocess.run(whisper_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logger.error(f"whisper.cpp failed: {result.stderr.decode()}")
        raise RuntimeError("whisper.cpp failed")

def run_pyannote_diarization_with_progress(wav_path):
    with ProgressHook() as hook:
        diarization, sources = pipeline(wav_path, hook=hook)
    return diarization, sources

def batch_process():
    for video_path in glob.glob(os.path.join(INPUT_DIR, "*.mp4")):
        base = os.path.basename(video_path)
        name_no_ext = os.path.splitext(base)[0]
        # Use a temporary wav file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
            temp_wav = temp_wav_file.name
        whisper_json = os.path.join(OUTPUT_DIR, f"{name_no_ext}.json")
        # Extract audio to temp wav
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", temp_wav
        ]
        logger.info(f"Extracting audio: {' '.join(ffmpeg_cmd)}")
        subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        try:
            # Run whisper.cpp at word level
            run_whisper_cpp(temp_wav, OUTPUT_DIR, name_no_ext)
            if not os.path.exists(whisper_json):
                logger.error(f"Whisper output not found: {whisper_json}")
                continue
            # Run diarization
            diarization, sources = run_pyannote_diarization_with_progress(temp_wav)
            # --- Build speaker-by-speaker text script (as before) ---
            with open(whisper_json, 'r') as f:
                transcription = json.load(f)
            segments = transcription["transcription"] if "transcription" in transcription else transcription.get("segments", [])
            result = []
            for segment in segments:
                start_time = timestamp_to_seconds(segment["timestamps"]["from"])
                end_time = timestamp_to_seconds(segment["timestamps"]["to"])
                text = segment["text"]
                speaker_times = {}
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    overlap_start = max(turn.start, start_time)
                    overlap_end = min(turn.end, end_time)
                    if overlap_end > overlap_start:
                        duration = overlap_end - overlap_start
                        speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
                if speaker_times:
                    dominant_speaker = max(speaker_times, key=speaker_times.get)
                else:
                    dominant_speaker = "Unknown"
                result.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "speaker": dominant_speaker
                })
            # Save the diarized segments as a new JSON, conflating diarization and cpp output
            diarized_json_path = os.path.join(OUTPUT_DIR, f"{name_no_ext}.diarized.json")
            with open(diarized_json_path, 'w') as f:
                json.dump(result, f, indent=2)
            # Write script-style text file with adjacent lines of the same speaker concatenated
            txt_path = os.path.join(OUTPUT_DIR, f"{name_no_ext}.script.txt")
            with open(txt_path, 'w') as f:
                last_speaker = None
                buffer = []
                for seg in result:
                    speaker = seg['speaker']
                    text = seg['text']
                    if speaker != last_speaker and last_speaker is not None:
                        # Write previous speaker's block
                        f.write(f"{last_speaker}: {' '.join(buffer).strip()}\n\n")
                        buffer = []
                    buffer.append(text)
                    last_speaker = speaker
                # Write the last block
                if last_speaker is not None and buffer:
                    f.write(f"{last_speaker}: {' '.join(buffer).strip()}\n")
        finally:
            # Delete the temporary wav file after processing
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

if __name__ == "__main__":
    batch_process()