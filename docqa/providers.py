"""Build the embedder and the three chat-model roles (stand-ins or Ollama)."""

import httpx
import ollama
from langchain_ollama import ChatOllama, OllamaEmbeddings

from docqa.config import Settings
from docqa.records import Services
from docqa.standins import StandInChatModel, StandInEmbedder


class OllamaUnavailableError(RuntimeError):
    """The Ollama server cannot be reached or a needed model is not pulled."""


def _same_model(wanted: str, installed: str) -> bool:
    def norm(name: str) -> str:
        return name if ":" in name else f"{name}:latest"

    return norm(wanted) == norm(installed)


def check_ollama(settings: Settings) -> None:
    """Raise OllamaUnavailableError unless the server answers and both models are present."""
    pull = f"ollama pull {settings.chat_model} && ollama pull {settings.embed_model}"
    try:
        listing = ollama.Client(host=settings.ollama_host, timeout=3).list()
    except (ConnectionError, httpx.HTTPError, OSError) as error:
        raise OllamaUnavailableError(
            f"Cannot reach the Ollama server at {settings.ollama_host} ({error.__class__.__name__}). "
            f"Start it with: ollama serve\nThen pull the models: {pull}"
        ) from error
    installed = [item.model or "" for item in listing.models]
    for wanted in (settings.chat_model, settings.embed_model):
        if not any(_same_model(wanted, name) for name in installed):
            raise OllamaUnavailableError(
                f"The Ollama model '{wanted}' is not installed. Pull it with: ollama pull {wanted}"
            )


def build_services(settings: Settings) -> Services:
    if settings.standins:
        return Services(
            embedder=StandInEmbedder(),
            answerer=StandInChatModel(role="answerer", temperature=settings.answer_temperature),
            writer=StandInChatModel(role="writer", temperature=settings.example_writer_temperature),
            grader=StandInChatModel(role="grader", temperature=settings.grader_temperature),
            standins=True,
        )
    check_ollama(settings)
    host = settings.ollama_host
    return Services(
        embedder=OllamaEmbeddings(model=settings.embed_model, base_url=host),
        answerer=ChatOllama(model=settings.chat_model, base_url=host, temperature=settings.answer_temperature),
        # Example writer: no temperature argument (framework default, spec section 5).
        writer=ChatOllama(model=settings.chat_model, base_url=host, format="json"),
        grader=ChatOllama(model=settings.chat_model, base_url=host, temperature=settings.grader_temperature),
        standins=False,
    )
