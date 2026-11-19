# Resolved decisions

Confirmed by the user on 2026-11-16. Build path: **choose** (recommended options except where noted).

| Question | Chosen option | Status |
|---|---|---|
| C1 Collection | C1a (Recommended - course default): The sample product catalog | = course choice |
| C2 Document form | not asked (C1 is not C1b) | not asked (C1 is not C1b) |
| C3 Answer model provider | C3c: A local model on your machine (Ollama) | ≠ course choice (course used: C3a OpenAI chat model) |
| C4 How to ask questions | C4a (Recommended): From the command line | no course choice |
| C5 Cutting long documents | C5a (Recommended - course default): No cutting; each document is one piece | = course choice |
| C6 Embeddings | C6b: A local embedding model (Ollama) | ≠ course choice (course used: C6a OpenAI embeddings) |
| C7 Store | C7a (Recommended - course default): In memory, rebuilt at each start | = course choice |
| C8 Combine method | C8a (Recommended - course default): All in one prompt ("stuff") | = course choice |
| C9 Evaluation | C9a (Recommended - course default): Hand-written plus model-written examples, graded by a model | = course choice |
| C10 Seeing inside an answer | C10a (Recommended - course default): A debug switch that prints every step | = course choice |

## Model roles

Ollama server at `http://localhost:11434`.

- Answerer: `llama3.1:8b` (Ollama)
- Example writer: `llama3.1:8b` (Ollama)
- Grader: `llama3.1:8b` (Ollama). Weaker than the course's OpenAI grader, so some grades may be wrong.
- Embedder: `nomic-embed-text` (Ollama)

## What to check (non-recommended options)

- C3c: with no hosted-provider key set, the sunblocking sample question is answered by the local model and names the Breezeway Sun Shirt.
- C6b: with no OpenAI key set, indexing the sample catalog and asking the sunblocking question works, and the four retrieved pieces include both the Breezeway Sun Shirt and the Trailmark Hiking Shirt.

## Departures from the course on this path

- D-local-1: answers, example writing and grading use a local `llama3.1:8b` instead of an OpenAI chat model, so no OpenAI key is needed.
- D-local-2: embeddings use local `nomic-embed-text` instead of OpenAI embeddings.

## Adaptations for the Ollama build

### R10 adapted (keys)

- With `DOCQA_STANDINS=1`: no key is read and no network call is made (tested with sockets disabled).
- Otherwise the provider is Ollama. `OPENAI_API_KEY` is never read and nothing needs it.
- If the Ollama server is unreachable, or a model is not pulled, `ask` and `eval` print a clear message on stderr and exit with status 2: start the server with `ollama serve`, then `ollama pull <model>`. Other model or parsing errors exit with status 1.
- T11 is re-targeted to this: an unreachable `OLLAMA_HOST` gives a non-zero exit and the `ollama serve` / `ollama pull` message. Live tests T12 and T13 run against Ollama and are skipped when it is not reachable.

### Environment variables

| Variable | Default |
|---|---|
| `DOCQA_CHAT_MODEL` | `llama3.1:8b` |
| `DOCQA_EMBED_MODEL` | `nomic-embed-text` |
| `OLLAMA_HOST` | `http://localhost:11434` |
| `DOCQA_STANDINS` | unset (`1` = stand-ins) |
| `DOCQA_DATA_DIR` | `sample_data/` (addition, used by tests) |

### Framework-left behaviours (spec section 4) in the installed releases

Installed: langchain 1.5.0, langchain-core 1.8.2, langchain-ollama 1.1.0, ollama 0.6.3, python-dotenv 1.2.4, pytest 9.2.0 (Python 3.12.16). Nothing is pinned.

- Prompt formatting: `ChatPromptTemplate` (langchain-core) in `docqa/qa.py`, one human message: instruction, pieces joined by `<<<<>>>>>`, then `Question: ...`. The old `RetrievalQA` "stuff" chain does not exist in these releases, so the same behaviour is built from the prompt template and the chat model.
- Reply parsing: `StrOutputParser` for answers and grades (reply text unchanged), `JsonOutputParser` for generated examples (the writer model also uses Ollama `format="json"`).
- Retries: checked `ChatOllama.model_fields` and `BaseChatModel.model_fields` in the installed packages; neither has a retry field, so there are no automatic retries. Each question makes exactly one answerer call; a failure surfaces as exit status 1 (or 2 for an unreachable server).
- Temperature: Ollama accepts it, so it is passed (0.0 for answerer and grader). The example writer gets no temperature argument (framework default).
- Token counts come from `AIMessage.usage_metadata`, which `ChatOllama` fills; stand-ins report none.

### Other choices

- CSV loading uses the standard-library `csv` module, not LangChain's `CSVLoader` (that lives in `langchain-community`, which is not a dependency). The text is the same `column: value` lines.
- The store is a small in-memory class with plain-Python cosine similarity. LangChain's `InMemoryVectorStore` needs numpy and does not guarantee the spec's tie order.
- Stand-in notice goes to stderr so stdout stays answer-only (T10).
- `pyproject.toml` has both a `dev` and a `test` extra (the spec's install line says `test`).
- Replies are grouped as spec R3 says: pieces joined literally by the separator, no extra newlines.

### Checks for the non-recommended options

- C3c: with no OpenAI key, the sunblocking question was answered by `llama3.1:8b` and named the Breezeway Sun Shirt (live test T12 passed; manual `ask --debug` run, 199 input / 50 output tokens).
- C6b: with no OpenAI key, retrieval with `nomic-embed-text` returned P1 and P2 among the four pieces (live test T12 passed).
