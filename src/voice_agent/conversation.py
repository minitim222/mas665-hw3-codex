"""High-level orchestration for the multimodal conversation."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from .agent import ConversationAgent
from .stt import SpeechToTextProtocol
from .tts import TextToSpeechProtocol


@dataclass
class VoiceConversationOrchestrator:
    """Glue logic that turns audio input into audio responses."""

    agent: ConversationAgent
    stt: SpeechToTextProtocol
    tts: TextToSpeechProtocol
    transcription_sample_rate: int = 16_000
    response_history: list = field(default_factory=list)

    def handle_audio_bytes(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> Dict[str, object]:
        transcript = self.stt.transcribe(audio_bytes, sample_rate=sample_rate)
        response_text = self.agent.respond(transcript)
        speech_audio, sr = self.tts.synthesize(response_text)
        payload = {
            "transcript": transcript,
            "response_text": response_text,
            "audio": speech_audio,
            "sample_rate": sr,
            "history": self.agent.export_history(),
        }
        self.response_history.append(payload)
        return payload

    def handle_audio_file(self, audio_path: Path) -> Dict[str, object]:
        with audio_path.open("rb") as handle:
            audio_bytes = handle.read()
        return self.handle_audio_bytes(audio_bytes)

    def save_audio_response(self, response_payload: Dict[str, object], output_path: Path) -> None:
        audio_bytes = response_payload["audio"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            handle.write(audio_bytes)

    def summarize_session(self) -> str:
        if not self.response_history:
            return "No conversation has taken place yet."
        turns = [item["response_text"] for item in self.response_history]
        return "\n".join(f"Turn {idx + 1}: {text}" for idx, text in enumerate(turns))
