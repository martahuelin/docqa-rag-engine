import pytest

from docqa.qa import INSTRUCTION, ask

SUN_Q = "Please suggest a shirt with sunblocking"
SUN_ANSWER = "The Breezeway Sun Shirt has UPF 50+ sun protection and the Trailmark Hiking Shirt has UPF 30."


def test_t3_retrieval_order_scores_and_answer(store, services, settings):
    record = ask(SUN_Q, store, services, settings)
    assert [r.piece_id for r in record.retrieved] == ["P1", "P2", "P8", "P7"]
    assert [r.score for r in record.retrieved] == pytest.approx([0.945, 0.866, 0.707, 0.588], abs=0.001)
    assert record.answer == SUN_ANSWER


def test_t4_prompt_has_only_retrieved_pieces(store, services, settings):
    ask(SUN_Q, store, services, settings)
    prompt = services.answerer.calls[0].prompt
    for piece_id in ("P1", "P2", "P8", "P7"):
        assert store.get(piece_id).text in prompt
    for name in ("Harbor", "Summit", "Riverbend", "Canyon"):
        assert name not in prompt


def test_t5_single_call_prompt_shape_and_temperature(store, services, settings):
    record = ask(SUN_Q, store, services, settings)
    calls = services.answerer.calls
    assert len(calls) == 1
    prompt = calls[0].prompt
    assert prompt.startswith(INSTRUCTION)
    assert prompt.count("<<<<>>>>>") == 3
    texts = [store.get(i).text for i in ("P1", "P2", "P8", "P7")]
    assert "<<<<>>>>>".join(texts) in prompt
    assert prompt.index(texts[-1]) < prompt.rindex(SUN_Q)
    assert prompt.endswith(SUN_Q)
    assert calls[0].temperature == 0.0
    assert record.answer == SUN_ANSWER


def test_t6_unanswerable_question_ties_keep_catalog_order(store, services, settings):
    record = ask("Do you sell kayaks?", store, services, settings)
    assert [r.piece_id for r in record.retrieved] == ["P1", "P2", "P3", "P4"]
    assert all(r.score == 0 for r in record.retrieved)
    assert "If you don't know the answer, just say that you don't know" in services.answerer.calls[0].prompt
    assert record.answer == "I don't know."


def test_blank_question_is_rejected(store, services, settings):
    with pytest.raises(ValueError):
        ask("   ", store, services, settings)
    assert services.answerer.calls == []
