import os
import re

import pytest

from docqa.config import Settings
from docqa.documents import load_catalog
from docqa.providers import build_services
from docqa.qa import ask
from docqa.store import InMemoryPieceStore
from tests.conftest import ollama_reachable

pytestmark = pytest.mark.live

LIVE_SETTINGS = Settings.from_env({k: v for k, v in os.environ.items() if k != "DOCQA_STANDINS"})

skip_without_ollama = pytest.mark.skipif(
    not ollama_reachable(LIVE_SETTINGS), reason="Ollama server/models not available"
)


@pytest.fixture
def live():
    services = build_services(LIVE_SETTINGS)
    store = InMemoryPieceStore()
    store.index(load_catalog(LIVE_SETTINGS.data_dir / "catalog.csv"), services.embedder)
    return services, store


@skip_without_ollama
def test_t12_live_sunblocking(live):
    services, store = live
    record = ask("Please suggest a shirt with sunblocking", store, services, LIVE_SETTINGS)
    ids = [r.piece_id for r in record.retrieved]
    assert "P1" in ids and "P2" in ids
    assert "Breezeway" in record.answer


@skip_without_ollama
def test_t13_live_unanswerable(live):
    services, store = live
    record = ask("Do you sell kayaks?", store, services, LIVE_SETTINGS)
    answer = record.answer.lower()
    assert re.search(r"don't know|do not know|no kayak|not (listed|available|sell)|n't (sell|have|list)|not aware", answer)
    assert not re.search(r"(is|are) (a )?kayaks?\b", answer)
