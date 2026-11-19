"""In-memory piece store (C7a): rebuilt from the documents at every start."""

import math

from langchain_core.embeddings import Embeddings

from docqa.records import Piece, RawDocument


def cosine(a: list[float], b: list[float]) -> float:
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


class InMemoryPieceStore:
    def __init__(self) -> None:
        self._pieces: list[Piece] = []

    def __len__(self) -> int:
        return len(self._pieces)

    @property
    def pieces(self) -> list[Piece]:
        return list(self._pieces)

    def add(self, pieces: list[Piece]) -> None:
        self._pieces.extend(pieces)

    def index(self, documents: list[RawDocument], embedder: Embeddings) -> None:
        vectors = embedder.embed_documents([doc.text for doc in documents])
        self.add(
            [
                Piece(piece_id=doc.piece_id, text=doc.text, vector=list(vector), source=doc.source)
                for doc, vector in zip(documents, vectors)
            ]
        )

    def get(self, piece_id: str) -> Piece:
        for piece in self._pieces:
            if piece.piece_id == piece_id:
                return piece
        raise KeyError(piece_id)

    def similarity_search(self, query_vector: list[float], k: int) -> list[tuple[Piece, float]]:
        scored = [(piece, cosine(query_vector, piece.vector)) for piece in self._pieces]
        # sorted() is stable: equal scores keep catalog order.
        scored.sort(key=lambda item: -item[1])
        return scored[:k]
