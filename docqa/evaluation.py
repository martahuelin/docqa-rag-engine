"""Evaluation: hand-written plus model-written examples, graded by a model (C9a)."""

import re
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from docqa.config import Settings
from docqa.qa import ask
from docqa.records import EvalExample, EvalResult, RawDocument, Services
from docqa.store import InMemoryPieceStore

WRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Write one question and its answer that can be answered from the document below. "
            "The question must name the product it asks about and must make sense without seeing the document. "
            'Reply with JSON with exactly the keys "query" and "answer".\n\nDocument:\n{doc}',
        )
    ]
)

GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "You are a teacher grading a quiz. Compare the student's answer with the true answer by "
            "meaning, not by wording. Reply with CORRECT or INCORRECT as the first word, then a short reason.\n\n"
            "Question: {query}\nTrue answer: {real}\nStudent answer: {predicted}",
        )
    ]
)


def build_examples(
    documents: list[RawDocument],
    hand_examples: list[EvalExample],
    services: Services,
    settings: Settings,
) -> list[EvalExample]:
    chain = WRITER_PROMPT | services.writer | JsonOutputParser()
    generated = []
    for document in documents[: settings.examples_from_first_n_docs]:
        parsed = chain.invoke({"doc": document.text})
        query = parsed.get("query") if isinstance(parsed, dict) else None
        answer = parsed.get("answer") if isinstance(parsed, dict) else None
        if not (isinstance(query, str) and query.strip() and isinstance(answer, str) and answer.strip()):
            raise ValueError(f"The example writer did not return a usable query and answer for {document.piece_id}.")
        generated.append(EvalExample(query=query, answer=answer, origin="generated"))
    return [*hand_examples, *generated]


def normalise_grade(reply: str) -> Literal["CORRECT", "INCORRECT"]:
    words = reply.split()
    first = re.sub(r"[^A-Za-z]", "", words[0]).upper() if words else ""
    return "CORRECT" if first == "CORRECT" else "INCORRECT"


def grade(example: EvalExample, predicted: str, grader: BaseChatModel) -> Literal["CORRECT", "INCORRECT"]:
    chain = GRADER_PROMPT | grader | StrOutputParser()
    reply = chain.invoke({"query": example.query, "real": example.answer, "predicted": predicted})
    return normalise_grade(reply)


def run_eval(
    store: InMemoryPieceStore,
    services: Services,
    settings: Settings,
    documents: list[RawDocument],
    hand_examples: list[EvalExample],
) -> list[EvalResult]:
    results = []
    for example in build_examples(documents, hand_examples, services, settings):
        predicted = ask(example.query, store, services, settings).answer
        results.append(
            EvalResult(
                query=example.query,
                real_answer=example.answer,
                predicted_answer=predicted,
                grade=grade(example, predicted, services.grader),
            )
        )
    return results


def format_report(results: list[EvalResult]) -> str:
    blocks = [
        f"Question: {r.query}\nReal answer: {r.real_answer}\nPredicted answer: {r.predicted_answer}\nGrade: {r.grade}"
        for r in results
    ]
    correct = sum(r.grade == "CORRECT" for r in results)
    totals = f"CORRECT: {correct}  INCORRECT: {len(results) - correct}"
    return "\n\n".join([*blocks, totals])
