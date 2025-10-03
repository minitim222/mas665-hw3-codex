from __future__ import annotations

from pathlib import Path

from voice_agent.agent import ConversationAgent
from voice_agent.conversation import VoiceConversationOrchestrator
from voice_agent.stt import SpeechToTextProtocol
from voice_agent.tts import TextToSpeechProtocol


class DummySTT(SpeechToTextProtocol):
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def transcribe(self, audio_bytes: bytes, sample_rate=None) -> str:  # type: ignore[override]
        return self.transcript

    def transcribe_file(self, audio_path: Path) -> str:  # type: ignore[override]
        return self.transcript


class DummyTTS(TextToSpeechProtocol):
    def __init__(self, audio: bytes, sample_rate: int = 16_000) -> None:
        self.audio = audio
        self.sample_rate = sample_rate

    def synthesize(self, text: str):  # type: ignore[override]
        return self.audio, self.sample_rate


def test_orchestrator_round_trip(tmp_path):
    kb_path = Path("data/knowledge_base.json")
    agent = ConversationAgent(kb_path)
    stt = DummySTT("What is MAS.665?")
    tts = DummyTTS(b"audio-bytes")
    orchestrator = VoiceConversationOrchestrator(agent=agent, stt=stt, tts=tts)

    payload = orchestrator.handle_audio_bytes(b"fake-audio")

    assert "MAS.665 is a course" in payload["response_text"]
    assert payload["audio"] == b"audio-bytes"
    assert payload["sample_rate"] == 16_000
    assert payload["history"][-2]["content"] == "What is MAS.665?"

    output_path = tmp_path / "response.wav"
    orchestrator.save_audio_response(payload, output_path)
    assert output_path.exists()
    assert output_path.read_bytes() == b"audio-bytes"
