"""Load the catalog and the hand-written evaluation examples."""

import csv
import json
from pathlib import Path

from docqa.records import EvalExample, RawDocument, Source


def load_catalog(path: Path) -> list[RawDocument]:
    """One document per CSV row; text is each column as 'column: value', in header order."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        RawDocument(
            piece_id=row["id"],
            text="\n".join(f"{column}: {value}" for column, value in row.items()),
            source=Source(file=path.name, row=index),
        )
        for index, row in enumerate(rows)
    ]


def load_examples(path: Path) -> list[EvalExample]:
    with Path(path).open(encoding="utf-8") as handle:
        items = json.load(handle)
    examples = []
    for item in items:
        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{path}: every example needs a non-empty 'query'")
        examples.append(EvalExample(query=query, answer=str(item["answer"]), origin="hand"))
    return examples
