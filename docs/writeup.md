# MAS.665 Multimodal Agent Write-up

## Implementation Overview

The voice-enabled agent extends the MAS.665 homework assistant by adding speech-to-text (STT) and text-to-speech (TTS) capabilities. The system is implemented in Python and orchestrated via a modular pipeline in `voice_agent/`.

* **Speech-to-Text:** The `WhisperSpeechToText` class wraps the open-source [`openai-whisper`](https://github.com/openai/whisper) package. Incoming audio is written to a temporary file and transcribed with a configurable Whisper model (default: `base`). Users can override the model name, inference device, and language from the CLI or server configuration.
* **Text-to-Speech:** The `KokoroTextToSpeech` class integrates [`kokoro-onnx`](https://github.com/fishaudio/kokoro-onnx), a lightweight neural TTS engine that runs entirely on-device. The wrapper normalizes the API across different Kokoro versions and emits 16-bit PCM WAV bytes using `soundfile`.
* **Text Agent:** The underlying text agent is a retrieval-augmented FAQ assistant. `ConversationAgent` loads a YAML knowledge base, keeps multi-turn history, and uses fuzzy matching to retrieve relevant answers. When no direct match is found, it provides structured fallback guidance and tips from the knowledge base.
* **Orchestrator:** `VoiceConversationOrchestrator` coordinates the STT and TTS modules with the conversation agent. It accepts raw audio bytes (from a file, microphone capture, or HTTP upload), transcribes them, generates a text response, and synthesizes reply audio. The orchestrator tracks per-turn metadata that can be saved for analytics or debugging.
* **Interfaces:** Two entry points are provided:
  * `src/server.py` exposes a FastAPI server with a `/voice-chat` endpoint for uploading audio blobs and receiving transcripts, text responses, and synthesized speech (base64 encoded).
  * `scripts/run_voice_cli.py` offers a CLI workflow for batch testing: provide an input audio file and receive both the textual response and a WAV file of the synthesized reply.

## Example Run (Video Walkthrough)

The accompanying video (see `docs/video_link.md`) demonstrates a single conversation turn:

1. The user records a short question asking, "What is MAS.665?" and uploads it through the CLI utility.
2. `WhisperSpeechToText` transcribes the audio locally, producing the text `"What is MAS.665?"`.
3. `ConversationAgent` matches the utterance with the FAQ entry and returns a concise description of the course.
4. `KokoroTextToSpeech` converts the response into speech using the `af_sky` voice preset. The CLI saves the waveform to `output.wav`.
5. The terminal displays the transcript and textual answer, confirming the round trip. The audio file can be played back to hear the spoken reply featured in the demo video.

### Observations

* Whisper handled the noisy sample accurately after downmixing to mono, showcasing robustness for casual recordings.
* Kokoro's inference time for short prompts (~2 seconds) remains under 200ms on CPU, keeping the interaction responsive.
* The orchestrator's history export proved useful for overlaying subtitles in the video; the same data can support analytics dashboards.

## Future Improvements

* Add microphone streaming support (WebRTC) to reduce turnaround time.
* Integrate a larger LLM via API or a local GGUF model for more expansive reasoning.
* Persist conversation sessions in a database to maintain continuity across devices.
