# transcript-whisperer
diarization of dialogue


instructions for:
x-code
homebrew


## Installation

1. Install [Homebrew](https://brew.sh/) (if not already installed).
2. Run the full setup with Make (recommended):

    ```sh
    make all
    ```

    This will install ffmpeg, set up Python dependencies using pipenv, and clone whisper.cpp.

    Or, install Python dependencies manually:

    ```sh
    pip install -r requirements.txt
    ```

3. Accept user conditions for models:
    - Accept pyannote/separation-ami-1.0 user conditions
    - Accept pyannote/speech-separation-ami-1.0 user conditions


4. Copy `.env` from `sample.env` and add your HuggingFace token.

    ```sh
    cp sample.env .env
    # Edit .env and add your HF_TOKEN
    ```

## Running the diarization script

To run diarize.py from inside the pipenv environment, use:

```sh
pipenv run python diarize.py
```

This ensures all dependencies are available and your environment variables are loaded.