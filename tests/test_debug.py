import io

from langchain_core.messages import AIMessage
from langchain_core.language_models import FakeListChatModel

from docqa.cli import main
from docqa.qa import INSTRUCTION, ask, format_debug

Q = "Please suggest a shirt with sunblocking"
ANSWER = "The Breezeway Sun Shirt has UPF 50+ sun protection and the Trailmark Hiking Shirt has UPF 30."


def test_t10_without_and_with_debug(services, store):
    plain = io.StringIO()
    assert main(["ask", Q], services=services, env={}, out=plain) == 0
    assert plain.getvalue() == ANSWER + "\n"

    debug = io.StringIO()
    assert main(["ask", Q, "--debug"], services=services, env={}, out=debug) == 0
    text = debug.getvalue()
    assert Q in text
    for piece_id, score in [("P1", "0.945"), ("P8", "0.707"), ("P2", "0.866"), ("P7", "0.588")]:
        assert f"id: {piece_id}  score: {score}" in text
        assert store.get(piece_id).text in text
    assert INSTRUCTION in text
    assert ANSWER in text
    assert "not reported by provider" in text


class UsageModel(FakeListChatModel):
    """Test double whose reply carries token counts."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop, run_manager, **kwargs)
        message = result.generations[0].message
        result.generations[0].message = AIMessage(
            content=message.content,
            usage_metadata={"input_tokens": 11, "output_tokens": 5, "total_tokens": 16},
        )
        return result


def test_debug_prints_token_counts_when_reported(services, store, settings):
    services.answerer = UsageModel(responses=["fine"])
    record = ask(Q, store, services, settings)
    assert record.token_usage is not None
    text = format_debug(record, store)
    assert "input: 11  output: 5  total: 16" in text
