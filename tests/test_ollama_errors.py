import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from docqa import providers
from docqa.cli import main
from docqa.config import Settings
from docqa.providers import OllamaUnavailableError, build_services

ROOT = Path(__file__).resolve().parent.parent
Q = "Please suggest a shirt with sunblocking"


def run_cli(*args: str, **env: str) -> subprocess.CompletedProcess:
    base = {k: v for k, v in os.environ.items() if k not in {"DOCQA_STANDINS", "OPENAI_API_KEY", "OLLAMA_HOST"}}
    return subprocess.run(
        [sys.executable, "-m", "docqa", *args],
        cwd=ROOT,
        env={**base, **env},
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("args", [("ask", Q), ("eval",)])
def test_t11_unreachable_ollama_exits_nonzero_with_clear_message(args):
    result = run_cli(*args, OLLAMA_HOST="http://127.0.0.1:9", DOCQA_DATA_DIR=str(ROOT / "sample_data"))
    assert result.returncode != 0
    assert "ollama serve" in result.stderr
    assert "ollama pull" in result.stderr
    assert "OPENAI_API_KEY" not in result.stderr + result.stdout
    assert "Traceback" not in result.stderr


def test_standins_need_no_key_and_succeed():
    result = run_cli("ask", Q, DOCQA_STANDINS="1")
    assert result.returncode == 0
    assert result.stdout == "The Breezeway Sun Shirt has UPF 50+ sun protection and the Trailmark Hiking Shirt has UPF 30.\n"
    assert "STAND-IN" in result.stderr


def test_no_network_in_standin_mode(monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise AssertionError("network call attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    env = {"DOCQA_STANDINS": "1"}
    assert main(["ask", Q], env=env) == 0
    assert main(["eval"], env=env) == 0
    assert "CORRECT: 6  INCORRECT: 1" in capsys.readouterr().out


def test_settings_env_overrides_and_defaults():
    defaults = Settings.from_env({})
    assert (defaults.chat_model, defaults.embed_model, defaults.ollama_host) == (
        "llama3.1:8b",
        "nomic-embed-text",
        "http://localhost:11434",
    )
    assert not defaults.standins
    custom = Settings.from_env(
        {"DOCQA_CHAT_MODEL": "m", "DOCQA_EMBED_MODEL": "e", "OLLAMA_HOST": "http://h:1", "DOCQA_STANDINS": "1"}
    )
    assert (custom.chat_model, custom.embed_model, custom.ollama_host, custom.standins) == ("m", "e", "http://h:1", True)


class FakeModel:
    def __init__(self, model: str) -> None:
        self.model = model


class FakeListing:
    def __init__(self, names: list[str]) -> None:
        self.models = [FakeModel(n) for n in names]


def test_missing_model_names_the_pull_command(monkeypatch):
    class FakeClient:
        def __init__(self, host=None, **kwargs):
            pass

        def list(self):
            return FakeListing(["nomic-embed-text:latest"])

    monkeypatch.setattr(providers.ollama, "Client", FakeClient)
    with pytest.raises(OllamaUnavailableError, match="ollama pull llama3.1:8b"):
        build_services(Settings())


def test_latest_tag_is_treated_as_equal(monkeypatch):
    class FakeClient:
        def __init__(self, host=None, **kwargs):
            pass

        def list(self):
            return FakeListing(["llama3.1:8b", "nomic-embed-text:latest"])

    monkeypatch.setattr(providers.ollama, "Client", FakeClient)
    services = build_services(Settings())
    assert not services.standins
