"""Text-to-speech helpers."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class TextToSpeechProtocol(Protocol):
    """Minimal interface for TTS backends."""

    def synthesize(self, text: str) -> Tuple[bytes, int]:
        ...


@dataclass
class KokoroTextToSpeech:
    """Kokoro ONNX text-to-speech wrapper."""

    model_id: str = "onnx-community/Kokoro-82M"
    voice: str = "af_sky"
    speed: float = 1.0

    def __post_init__(self) -> None:
        try:
            from kokoro_onnx import Kokoro  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "KokoroTextToSpeech requires the 'kokoro-onnx' package. Install it with `pip install kokoro-onnx`."
            ) from exc
        self._kokoro = Kokoro.from_pretrained(self.model_id)

    def synthesize(self, text: str) -> Tuple[bytes, int]:
        if not text.strip():
            raise ValueError("Text to synthesize cannot be empty.")
        import soundfile as sf  # type: ignore

        if hasattr(self._kokoro, "generate"):
            audio, sample_rate = self._kokoro.generate(  # type: ignore[attr-defined]
                text, voice=self.voice, speed=self.speed
            )
        elif hasattr(self._kokoro, "__call__"):
            audio, sample_rate = self._kokoro(  # type: ignore[operator]
                text, voice=self.voice, speed=self.speed
            )
        else:  # pragma: no cover - defensive branch
            raise RuntimeError("Unsupported Kokoro interface; expected `generate` or callable API.")
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        return buffer.getvalue(), sample_rate
