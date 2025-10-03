#!/usr/bin/env python3
"""Command line runner for the MAS.665 voice agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voice_agent import ConversationAgent, VoiceConversationOrchestrator, WhisperSpeechToText, KokoroTextToSpeech


def build_orchestrator(args: argparse.Namespace) -> VoiceConversationOrchestrator:
    agent = ConversationAgent(Path(args.knowledge_base))
    stt = WhisperSpeechToText(model_name=args.whisper_model, device=args.device, language=args.language)
    tts = KokoroTextToSpeech(model_id=args.kokoro_model, voice=args.voice, speed=args.speed)
    return VoiceConversationOrchestrator(agent=agent, stt=stt, tts=tts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an interactive voice chat session with the MAS.665 agent.")
    parser.add_argument("input", type=Path, help="Path to the input audio file (wav/m4a/mp3).")
    parser.add_argument("output", type=Path, help="Where to save the synthesized agent response audio (wav).")
    parser.add_argument("--knowledge-base", default="data/knowledge_base.json", help="Path to FAQ knowledge base file.")
    parser.add_argument("--whisper-model", default="base", help="Whisper model name (tiny, base, small, ...).")
    parser.add_argument("--kokoro-model", default="onnx-community/Kokoro-82M", help="Kokoro model identifier.")
    parser.add_argument("--voice", default="af_sky", help="Voice preset for Kokoro.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier.")
    parser.add_argument("--device", default=None, help="Torch device for Whisper (cpu, cuda, ...).")
    parser.add_argument("--language", default=None, help="Force transcription language if desired.")

    args = parser.parse_args()
    orchestrator = build_orchestrator(args)
    payload = orchestrator.handle_audio_file(args.input)
    orchestrator.save_audio_response(payload, args.output)

    print("Transcript:\n", payload["transcript"])
    print("\nAgent response:\n", payload["response_text"])
    print(f"Saved synthesized speech to {args.output}")


if __name__ == "__main__":
    main()
