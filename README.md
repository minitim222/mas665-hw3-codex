# MAS.665 Voice-Enabled Homework Agent

This repository extends the MAS.665 HW1–HW3 assistant with speech capabilities. The project delivers:

* A modular speech pipeline built on Whisper (STT) and Kokoro (TTS).
* A FastAPI server for browser or mobile integrations.
* A CLI for rapid offline experimentation.
* Documentation and a video walkthrough (see `docs/`).

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the desired Whisper and Kokoro models on first run; the wrappers will handle caching.

## Command Line Workflow

```bash
python scripts/run_voice_cli.py input.wav output.wav
```

Key flags:

* `--whisper-model`: choose Whisper size (e.g., `small`, `medium`).
* `--kokoro-model`: pick a Kokoro checkpoint.
* `--voice`: voice preset (`af_sky`, `af_aria`, etc.).

The CLI prints the transcript and saves synthesized speech to `output.wav`.

## FastAPI Server

```bash
uvicorn src.server:app --reload
```

Endpoints:

* `GET /healthz`: readiness probe.
* `POST /voice-chat`: multipart audio upload (`file`). Returns transcript, agent response, base64-encoded audio, and the conversation history.

## Tests

Run the unit tests (mocked STT/TTS) with:

```bash
pytest
```

## Documentation

* `docs/writeup.md`: 1–2 page implementation summary and example run.
* `docs/video_link.md`: add the unlisted YouTube link for the demonstration.

## Repository Layout

```
.
├── data/knowledge_base.json    # FAQ source for the text agent
├── docs/                       # Write-up and deliverables
├── scripts/run_voice_cli.py    # Command line interface
├── src/server.py               # FastAPI server entry point
└── src/voice_agent/            # Core multimodal pipeline
```
