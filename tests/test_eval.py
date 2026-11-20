import io

import pytest

from docqa.cli import main
from docqa.evaluation import build_examples, grade, normalise_grade, run_eval
from docqa.records import EvalExample

E1 = EvalExample("Does the Harbor Fleece Pullover have side pockets?", "Yes", "hand")


def test_t7_example_set(raw_docs, hand_examples, services, settings):
    examples = build_examples(raw_docs, hand_examples, services, settings)
    assert [(e.query, e.answer, e.origin) for e in examples] == [
        ("Does the Harbor Fleece Pullover have side pockets?", "Yes", "hand"),
        ("What collection is the Summit Down Jacket from?", "The Alpine collection", "hand"),
        ("What fabric is the Breezeway Sun Shirt made of?", "100% recycled polyester", "generated"),
        ("What UPF rating does the Trailmark Hiking Shirt have?", "UPF 30", "generated"),
        ("How much does the Harbor Fleece Pullover weigh?", "12 oz", "generated"),
        ("What fill does the Summit Down Jacket use?", "800-fill goose down", "generated"),
        ("What seams do the Riverbend Rain Pants have?", "Taped seams", "generated"),
    ]
    calls = services.writer.calls
    assert len(calls) == 5
    for call, name in zip(calls, ["Breezeway", "Trailmark", "Harbor", "Summit", "Riverbend"]):
        assert name in call.prompt
        assert call.temperature is None


def test_t8_eval_command(services, settings):
    out = io.StringIO()
    assert main(["eval"], services=services, env={}, out=out) == 0
    lines = out.getvalue().strip().splitlines()
    assert lines[-1] == "CORRECT: 6  INCORRECT: 1"
    grades = [line.split(": ", 1)[1] for line in lines if line.startswith("Grade: ")]
    assert grades == ["CORRECT"] * 6 + ["INCORRECT"]
    predicted = [line.split(": ", 1)[1] for line in lines if line.startswith("Predicted answer: ")]
    assert predicted[0] == "The Harbor Fleece Pullover does have side pockets."
    assert predicted[-1] == "The Riverbend Rain Pants have welded seams."
    assert len(services.grader.calls) == 7
    assert all(c.temperature == 0.0 for c in services.grader.calls)


def test_t9_grading_is_by_meaning_not_string_match(store, services, settings, raw_docs, hand_examples):
    results = run_eval(store, services, settings, raw_docs, hand_examples)
    first = results[0]
    assert first.real_answer == "Yes"
    assert first.predicted_answer == "The Harbor Fleece Pullover does have side pockets."
    assert first.real_answer != first.predicted_answer
    prompt = services.grader.calls[0].prompt
    assert first.real_answer in prompt and first.predicted_answer in prompt
    assert first.grade == "CORRECT"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("CORRECT", "CORRECT"),
        ("CORRECT.", "CORRECT"),
        ("correct, same meaning", "CORRECT"),
        ("incorrect", "INCORRECT"),
        ("Incorrect, because", "INCORRECT"),
        ("I think correct", "INCORRECT"),
        ("", "INCORRECT"),
    ],
)
def test_grade_normalisation(reply, expected):
    assert normalise_grade(reply) == expected


def test_grade_uses_grader_model(services):
    assert grade(E1, "The Harbor Fleece Pullover does have side pockets.", services.grader) == "CORRECT"
