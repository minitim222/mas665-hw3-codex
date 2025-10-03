"""Speech-to-text helpers."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class SpeechToTextProtocol(Protocol):
    """Minimal interface for STT backends."""

    def transcribe(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> str:
        ...

    def transcribe_file(self, audio_path: Path) -> str:
        ...


@dataclass
class WhisperSpeechToText:
    """Wraps OpenAI Whisper for transcription."""

    model_name: str = "base"
    device: Optional[str] = None
    language: Optional[str] = None

    def __post_init__(self) -> None:
        try:
            import whisper  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "WhisperSpeechToText requires the 'whisper' package. Install it with `pip install openai-whisper`."
            ) from exc
        self._whisper = whisper.load_model(self.model_name, device=self.device)

    def transcribe(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            return self.transcribe_file(Path(tmp.name))

    def transcribe_file(self, audio_path: Path) -> str:
        result = self._whisper.transcribe(str(audio_path), language=self.language, fp16=False)
        return result.get("text", "").strip()
