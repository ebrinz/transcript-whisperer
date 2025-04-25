# transcript-whisperer

A Python pipeline for transcribing and diarizing MP4 video files using whisper.cpp for transcription and pyannote.audio for speaker diarization. Outputs include diarized JSON, plain transcripts, and script-style text files.

## Features
- **Transcription**: Uses whisper.cpp for fast, accurate, word-level transcription.
- **Speaker Diarization**: Uses pyannote.audio to identify and separate speakers.
- **Temporary Audio Handling**: Only a temporary WAV file is created per video, which is deleted after processing.
- **Flexible Output**: Produces diarized JSON, plain text, and script-style text files.
- **Test Mode**: Easily test the pipeline with separate input/output folders using a command-line flag.

## Installation

1. **Clone the repository**
   ```sh
   git clone <your-repo-url>
   cd transcript-whisperer
   ```
2. **Set up environment variables**
   - Copy `sample.env` to `.env` and fill in the required paths and Hugging Face token.
   ```sh
   cp sample.env .env
   # Edit .env and set HF_TOKEN, WHISPER_CPP_BIN, and WHISPER_CPP_MODEL
   ```
3. **Install dependencies**
   ```sh
   pipenv install
   # or
   pip install -r requirements.txt
   ```
4. **Build whisper.cpp and download model**
   ```sh
   make all
   ```

## Usage

### Standard Mode (Production)
- Place your `.mp4` files in the `input/` directory.
- Run the pipeline:
  ```sh
  pipenv run python diarize.py
  # or
  python diarize.py
  ```
- Outputs will be saved in the `output/` directory.

### Test Mode
- Place your test `.mp4` files in the `test/input/` directory.
- Run the pipeline in test mode:
  ```sh
  pipenv run python diarize.py --test
  # or
  python diarize.py --test
  ```
- Outputs will be saved in the `test/output/` directory.

## Output Files
For each video, the following files are created in the output directory:

| File Name                      | Description                                            |
|------------------------------- |--------------------------------------------------------|
| `{name}.json`                  | Raw whisper.cpp JSON output (no speaker info)           |
| `{name}.txt`                   | Plain text transcript from whisper.cpp                  |
| `{name}.diarized.json`         | Diarization + transcription, with speaker attribution   |
| `{name}.script.txt`            | Script-style text, speaker-prefixed, grouped by turn    |

## Notes
- No intermediate or debug files are kept; only the output files above are produced.
- The `.gitkeep` files ensure empty input/output directories are tracked by git.
- Requires a valid Hugging Face token for pyannote.audio.
- Make sure the paths in your `.env` are correct for your system.

## License
MIT