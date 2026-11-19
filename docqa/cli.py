"""Command line: python -m docqa ask "<question>" [--debug] | python -m docqa eval."""

import argparse
import os
import sys
from collections.abc import Mapping
from typing import TextIO

import httpx
import ollama
from dotenv import load_dotenv

from docqa.config import Settings
from docqa.documents import load_catalog, load_examples
from docqa.evaluation import format_report, run_eval
from docqa.providers import OllamaUnavailableError, build_services
from docqa.qa import ask, format_debug
from docqa.records import Services
from docqa.standins import STANDIN_NOTICE
from docqa.store import InMemoryPieceStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docqa", description="Ask your documents.")
    commands = parser.add_subparsers(dest="command", required=True)
    ask_parser = commands.add_parser("ask", help="answer one question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--debug", action="store_true", help="print every step")
    commands.add_parser("eval", help="grade the program on example questions")
    return parser


def main(
    argv: list[str] | None = None,
    services: Services | None = None,
    env: Mapping[str, str] | None = None,
    out: TextIO | None = None,
) -> int:
    out = out if out is not None else sys.stdout
    args = _parser().parse_args(argv)
    if env is None:
        load_dotenv()  # does not override variables already set
        env = os.environ
    settings = Settings.from_env(env, debug=getattr(args, "debug", False))
    try:
        if services is None:
            services = build_services(settings)
            if services.standins:
                print(STANDIN_NOTICE, file=sys.stderr)
        documents = load_catalog(settings.data_dir / "catalog.csv")
        store = InMemoryPieceStore()
        store.index(documents, services.embedder)
        if args.command == "ask":
            record = ask(args.question, store, services, settings)
            if settings.debug:
                print(format_debug(record, store), file=out)
            else:
                print(record.answer, file=out)
        else:
            hand = load_examples(settings.data_dir / "examples.json")
            print(format_report(run_eval(store, services, settings, documents, hand)), file=out)
    except OllamaUnavailableError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except (ConnectionError, httpx.HTTPError, ollama.ResponseError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
