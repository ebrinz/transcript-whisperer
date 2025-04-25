# Makefile for transcript-whisperer setup

.PHONY: all ffmpeg whisper.cpp pipenv whisper.cpp-build whisper.cpp-model whisper.cpp-full

all: ffmpeg whisper.cpp pipenv
	@echo "Setup complete."

# Install ffmpeg using Homebrew (macOS)
ffmpeg:
	@if ! command -v ffmpeg >/dev/null 2>&1; then \
		echo "ffmpeg not found. Installing with Homebrew..."; \
		brew install ffmpeg; \
	else \
		echo "ffmpeg already installed."; \
	fi

# Clone whisper.cpp if not already present
whisper.cpp:
	@if [ ! -d "whisper.cpp" ]; then \
		echo "Cloning whisper.cpp..."; \
		git clone https://github.com/ggerganov/whisper.cpp.git; \
	else \
		echo "whisper.cpp already exists."; \
	fi

# Build whisper.cpp
whisper.cpp-build: whisper.cpp
	@cd whisper.cpp && make

# Download whisper.cpp large-v3-turbo model
whisper.cpp-model: whisper.cpp
	@mkdir -p whisper.cpp/models
	@if [ ! -f "whisper.cpp/models/ggml-large-v3-turbo.bin" ]; then \
		echo "Downloading ggml-large-v3-turbo.bin..."; \
		curl -L -o whisper.cpp/models/ggml-large-v3-turbo.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin; \
	else \
		echo "Model ggml-large-v3-turbo.bin already exists."; \
	fi

# Full setup for whisper.cpp (clone, build, download model)
whisper.cpp-full: whisper.cpp whisper.cpp-build whisper.cpp-model
	@echo "whisper.cpp is ready."

# Create a pipenv environment and install dependencies
pipenv:
	@pip install --user pipenv
	@pipenv install --skip-lock --python $(shell which python3)
	@pipenv install -r requirements.txt
