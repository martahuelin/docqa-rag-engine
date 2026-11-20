from docqa.records import Source
from docqa.store import InMemoryPieceStore, cosine

SUN, SHIRT, POCKET, JACKET, DOWN, RAIN, PANTS, WOOL, SOCKS, HAT, FLEECE, COLLECTION, KAYAK, HIKING = range(14)

EXPECTED = {
    "P1": {SUN: 3, SHIRT: 2, POCKET: 1},
    "P2": {SUN: 2, SHIRT: 1, HIKING: 1},
    "P3": {FLEECE: 5, POCKET: 1},
    "P4": {JACKET: 2, DOWN: 2, COLLECTION: 1},
    "P5": {RAIN: 3, PANTS: 2},
    "P6": {WOOL: 3, SOCKS: 2, HIKING: 1},
    "P7": {SUN: 3, HAT: 2},
    "P8": {SHIRT: 2},
}


def test_t1_store_starts_empty_then_holds_eight_pieces(raw_docs, services):
    store = InMemoryPieceStore()
    assert len(store) == 0
    store.index(raw_docs, services.embedder)
    assert len(store) == 8
    assert [p.piece_id for p in store.pieces] == [f"P{i}" for i in range(1, 9)]


def test_t2_piece_p1(store):
    piece = store.get("P1")
    assert piece.text == (
        "id: P1\nname: Breezeway Sun Shirt\n"
        "description: Lightweight long-sleeve shirt with UPF 50+ sun protection. "
        "Fabric: 100% recycled polyester. Has two chest pockets."
    )
    assert len(piece.vector) == 14
    assert piece.vector == [3, 2, 1] + [0] * 11
    assert piece.source == Source("catalog.csv", 0)


def test_stand_in_vectors_match_spec_for_every_piece(store):
    for piece_id, counts in EXPECTED.items():
        expected = [float(counts.get(i, 0)) for i in range(14)]
        assert store.get(piece_id).vector == expected, piece_id


def test_source_rows_are_zero_based(store):
    assert [p.source.row for p in store.pieces] == list(range(8))


def test_zero_vector_scores_zero():
    assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine([1.0, 2.0], [0.0, 0.0]) == 0.0
