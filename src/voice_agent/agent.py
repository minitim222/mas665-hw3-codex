"""Text agent that powers the multimodal interface."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency for YAML
    import yaml
except ImportError:  # pragma: no cover - allow JSON-only deployments
    yaml = None


@dataclass
class ConversationTurn:
    """Represents a single exchange in the conversation history."""

    role: str
    content: str


@dataclass
class ConversationAgent:
    """Lightweight conversational agent backed by a FAQ knowledge base."""

    knowledge_base_path: Path
    similarity_threshold: float = 0.55
    system_prompt: str = (
        "You are the MAS.665 homework assistant. Answer questions using the knowledge base when "
        "possible. Be concise and provide actionable suggestions."
    )
    history: List[ConversationTurn] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._knowledge_base = self._load_knowledge_base(self.knowledge_base_path)
        self.history.append(ConversationTurn("system", self.system_prompt))

    @staticmethod
    def _load_knowledge_base(path: Path) -> Dict[str, Iterable[Dict[str, str]]]:
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix in {".yaml", ".yml"}:
                if yaml is None:
                    raise ImportError(
                        "Loading YAML knowledge bases requires the optional 'pyyaml' dependency."
                    )
                return yaml.safe_load(handle)
            return json.load(handle)

    def add_user_message(self, message: str) -> None:
        self.history.append(ConversationTurn("user", message))

    def add_agent_message(self, message: str) -> None:
        self.history.append(ConversationTurn("assistant", message))

    def _search_faq(self, query: str) -> Optional[str]:
        faqs = self._knowledge_base.get("faqs", [])
        questions = [faq["question"] for faq in faqs]
        matches = difflib.get_close_matches(query, questions, n=1, cutoff=self.similarity_threshold)
        if not matches:
            return None
        match = matches[0]
        for faq in faqs:
            if faq["question"] == match:
                return faq["answer"].strip()
        return None

    def respond(self, message: str) -> str:
        self.add_user_message(message)
        faq_answer = self._search_faq(message)
        if faq_answer:
            response = faq_answer
        else:
            response = self._fallback_response(message)
        self.add_agent_message(response)
        return response

    def _fallback_response(self, message: str) -> str:
        tips = self._knowledge_base.get("tips", [])
        tip_text = tips[0] if tips else ""
        history_excerpt = " ".join(turn.content for turn in self.history[-4:])
        response = (
            "I do not have an exact FAQ entry for that yet. "
            "Based on the conversation so far, here is how you might proceed:\n"
            f"- Re-read your prompt: '{message}'.\n"
            "- Summarize the key intent before acting.\n"
        )
        if tip_text:
            response += f"- Tip: {tip_text}\n"
        response += f"- Context recap: {history_excerpt[:280]}"
        return response

    def export_history(self) -> List[Dict[str, str]]:
        return [{"role": turn.role, "content": turn.content} for turn in self.history]
