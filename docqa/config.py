"""Settings (spec section 5) read from the environment."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"


@dataclass(frozen=True)
class Settings:
    standins: bool = False
    top_k: int = 4
    answer_temperature: float = 0.0
    grader_temperature: float = 0.0
    example_writer_temperature: float | None = None
    piece_separator: str = "<<<<>>>>>"
    examples_from_first_n_docs: int = 5
    debug: bool = False
    chat_model: str = "llama3.1:8b"
    embed_model: str = "nomic-embed-text"
    ollama_host: str = "http://localhost:11434"
    data_dir: Path = DEFAULT_DATA_DIR

    @classmethod
    def from_env(cls, env: Mapping[str, str], debug: bool = False) -> "Settings":
        defaults = cls()
        return cls(
            standins=env.get("DOCQA_STANDINS") == "1",
            debug=debug,
            chat_model=env.get("DOCQA_CHAT_MODEL") or defaults.chat_model,
            embed_model=env.get("DOCQA_EMBED_MODEL") or defaults.embed_model,
            ollama_host=env.get("OLLAMA_HOST") or defaults.ollama_host,
            data_dir=Path(env["DOCQA_DATA_DIR"]) if env.get("DOCQA_DATA_DIR") else defaults.data_dir,
        )
