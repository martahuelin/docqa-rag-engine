import pytest

from docqa.config import DEFAULT_DATA_DIR, Settings
from docqa.documents import load_catalog, load_examples
from docqa.providers import build_services, check_ollama, OllamaUnavailableError
from docqa.store import InMemoryPieceStore


@pytest.fixture
def settings() -> Settings:
    return Settings(standins=True, data_dir=DEFAULT_DATA_DIR)


@pytest.fixture
def services(settings):
    return build_services(settings)


@pytest.fixture
def raw_docs(settings):
    return load_catalog(settings.data_dir / "catalog.csv")


@pytest.fixture
def hand_examples(settings):
    return load_examples(settings.data_dir / "examples.json")


@pytest.fixture
def store(raw_docs, services):
    store = InMemoryPieceStore()
    store.index(raw_docs, services.embedder)
    return store


def ollama_reachable(settings: Settings) -> bool:
    try:
        check_ollama(settings)
    except OllamaUnavailableError:
        return False
    return True
