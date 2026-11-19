"""Retrieve the closest pieces and answer from them in one prompt (C8a, "stuff")."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from docqa.config import Settings
from docqa.records import AnswerRecord, Retrieved, Services, TokenUsage
from docqa.store import InMemoryPieceStore

INSTRUCTION = (
    "Use the following pieces of context to answer the user's question. "
    "If you don't know the answer, just say that you don't know, don't try to make up an answer."
)

PROMPT = ChatPromptTemplate.from_messages([("human", INSTRUCTION + "\n\n{context}\n\nQuestion: {question}")])


def ask(question: str, store: InMemoryPieceStore, services: Services, settings: Settings) -> AnswerRecord:
    if not question.strip():
        raise ValueError("The question must not be empty.")
    vector = services.embedder.embed_query(question)
    hits = store.similarity_search(vector, settings.top_k)
    context = settings.piece_separator.join(piece.text for piece, _ in hits)
    messages = PROMPT.format_messages(context=context, question=question)
    reply = services.answerer.invoke(messages)
    usage = getattr(reply, "usage_metadata", None)
    return AnswerRecord(
        question=question,
        answer=StrOutputParser().invoke(reply),
        retrieved=[Retrieved(piece_id=piece.piece_id, score=score) for piece, score in hits],
        prompt=str(messages[0].content),
        token_usage=TokenUsage(usage["input_tokens"], usage["output_tokens"], usage["total_tokens"])
        if usage
        else None,
    )


def format_debug(record: AnswerRecord, store: InMemoryPieceStore) -> str:
    lines = ["=== QUESTION ===", record.question, "", "=== RETRIEVED ==="]
    for item in record.retrieved:
        lines += [f"id: {item.piece_id}  score: {item.score:.3f}", store.get(item.piece_id).text, ""]
    lines += ["=== PROMPT ===", record.prompt, "", "=== REPLY ===", record.answer, "", "=== TOKENS ==="]
    usage = record.token_usage
    if usage is None:
        lines.append("not reported by provider")
    else:
        lines.append(f"input: {usage.input_tokens}  output: {usage.output_tokens}  total: {usage.total_tokens}")
    return "\n".join(lines)
