"""STAND-IN embedder and chat model for offline tests and demos (DOCQA_STANDINS=1).

Everything here is a STAND-IN: the embedder counts keywords and the chat model
returns scripted replies. Neither is a real provider and neither touches the network.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

STANDIN_NOTICE = "[STAND-IN model and embedder in use (DOCQA_STANDINS=1); answers are scripted]"

# Concept order fixes the vector layout (spec section 7.1).
CONCEPTS = [
    ("SUN", ["sun", "sunblocking", "sunblock", "upf", "uv"]),
    ("SHIRT", ["shirt", "shirts", "tee"]),
    ("POCKET", ["pocket", "pockets"]),
    ("JACKET", ["jacket", "jackets"]),
    ("DOWN", ["down"]),
    ("RAIN", ["rain", "waterproof"]),
    ("PANTS", ["pants"]),
    ("WOOL", ["wool", "merino"]),
    ("SOCKS", ["socks"]),
    ("HAT", ["hat"]),
    ("FLEECE", ["fleece", "pullover"]),
    ("COLLECTION", ["collection"]),
    ("KAYAK", ["kayak", "kayaks"]),
    ("HIKING", ["hiking"]),
]
TOKEN_TO_CONCEPT = {token: index for index, (_, tokens) in enumerate(CONCEPTS) for token in tokens}


class StandInEmbedder(Embeddings):
    """STAND-IN for the embedding service: 14-number keyword count vector."""

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * len(CONCEPTS)
        for token in re.split(r"[^a-z0-9]+", text.lower()):
            index = TOKEN_TO_CONCEPT.get(token)
            if index is not None:
                vector[index] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


@dataclass
class CallRecord:
    role: str
    prompt: str
    temperature: float | None


ANSWERER_SCRIPT = [
    ("sunblocking", "The Breezeway Sun Shirt has UPF 50+ sun protection and the Trailmark Hiking Shirt has UPF 30."),
    ("side pockets", "The Harbor Fleece Pullover does have side pockets."),
    ("What collection", "The Summit Down Jacket is from the Alpine collection."),
    ("fabric is the Breezeway", "The Breezeway Sun Shirt is made of 100% recycled polyester."),
    ("UPF rating does the Trailmark", "The Trailmark Hiking Shirt has UPF 30 sun protection."),
    ("weigh", "The Harbor Fleece Pullover weighs 12 oz."),
    ("fill does the Summit", "The Summit Down Jacket uses 800-fill goose down."),
    ("seams do the Riverbend", "The Riverbend Rain Pants have welded seams."),  # planted wrong answer
]
ANSWERER_DEFAULT = "I don't know."

WRITER_SCRIPT = [
    ("Breezeway Sun Shirt", {"query": "What fabric is the Breezeway Sun Shirt made of?", "answer": "100% recycled polyester"}),
    ("Trailmark Hiking Shirt", {"query": "What UPF rating does the Trailmark Hiking Shirt have?", "answer": "UPF 30"}),
    ("Harbor Fleece Pullover", {"query": "How much does the Harbor Fleece Pullover weigh?", "answer": "12 oz"}),
    ("Summit Down Jacket", {"query": "What fill does the Summit Down Jacket use?", "answer": "800-fill goose down"}),
    ("Riverbend Rain Pants", {"query": "What seams do the Riverbend Rain Pants have?", "answer": "Taped seams"}),
]


class StandInChatModel(BaseChatModel):
    """STAND-IN for the chat model: scripted replies chosen by the first matching row."""

    role: Literal["answerer", "writer", "grader"]
    temperature: float | None = None
    calls: list[CallRecord] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "docqa-standin"

    def _reply(self, prompt: str) -> str:
        if self.role == "answerer":
            question = prompt.rsplit("Question:", 1)[-1]
            for needle, reply in ANSWERER_SCRIPT:
                if needle in question:
                    return reply
            return ANSWERER_DEFAULT
        if self.role == "writer":
            for needle, reply in WRITER_SCRIPT:
                if needle in prompt:
                    return json.dumps(reply)
            return json.dumps({"query": "What is this document about?", "answer": "Unknown"})
        return "INCORRECT" if "seams do the Riverbend" in prompt else "CORRECT"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(message.content) for message in messages)
        self.calls.append(CallRecord(role=self.role, prompt=prompt, temperature=self.temperature))
        message = AIMessage(content=self._reply(prompt))
        return ChatResult(generations=[ChatGeneration(message=message)])
