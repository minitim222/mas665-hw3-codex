"""FastAPI server exposing the multimodal agent."""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .voice_agent import (
    ConversationAgent,
    VoiceConversationOrchestrator,
    WhisperSpeechToText,
    KokoroTextToSpeech,
)


app = FastAPI(title="MAS.665 Voice Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_orchestrator() -> VoiceConversationOrchestrator:
    knowledge_base = Path("data/knowledge_base.json")
    agent = ConversationAgent(knowledge_base)
    stt = WhisperSpeechToText()
    tts = KokoroTextToSpeech()
    return VoiceConversationOrchestrator(agent=agent, stt=stt, tts=tts)


@app.get("/healthz")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...)) -> dict:
    orchestrator = get_orchestrator()
    audio_bytes = await file.read()
    payload = orchestrator.handle_audio_bytes(audio_bytes)
    audio_base64 = base64.b64encode(payload["audio"]).decode("utf-8")
    return {
        "transcript": payload["transcript"],
        "response_text": payload["response_text"],
        "audio_base64": audio_base64,
        "sample_rate": payload["sample_rate"],
        "history": payload["history"],
    }
