"""Fixed-field records shared across the program (spec section 5)."""

from dataclasses import dataclass
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


@dataclass(frozen=True)
class Source:
    file: str
    row: int | None


@dataclass(frozen=True)
class RawDocument:
    piece_id: str
    text: str
    source: Source


@dataclass(frozen=True)
class Piece:
    piece_id: str
    text: str
    vector: list[float]
    source: Source


@dataclass(frozen=True)
class Retrieved:
    piece_id: str
    score: float


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class AnswerRecord:
    question: str
    answer: str
    retrieved: list[Retrieved]
    # Extras used only by --debug.
    prompt: str = ""
    token_usage: TokenUsage | None = None


@dataclass(frozen=True)
class EvalExample:
    query: str
    answer: str
    origin: Literal["hand", "generated"]


@dataclass(frozen=True)
class EvalResult:
    query: str
    real_answer: str
    predicted_answer: str
    grade: Literal["CORRECT", "INCORRECT"]


@dataclass
class Services:
    embedder: Embeddings
    answerer: BaseChatModel
    writer: BaseChatModel
    grader: BaseChatModel
    standins: bool
