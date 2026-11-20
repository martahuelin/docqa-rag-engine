# docqa: ask your documents

Answers plain-language questions about a small outdoor-clothing catalog. It embeds the catalog, retrieves the four entries closest in meaning to the question, and has a local model write the answer from only those entries. It says it doesn't know when they lack the answer. It can also grade itself on example questions (`eval`) and print every step of one answer (`--debug`).

Everything runs on the local machine through [Ollama](https://ollama.com). No OpenAI key is needed.

## Requirements

- Python 3.11 or newer (the project was built on 3.12)
- Ollama, for example `brew install ollama`

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Start Ollama and pull the models

```bash
ollama serve                 # skip if the Ollama app is already running
ollama pull llama3.1:8b      # chat model (answers, example writing, grading), about 5 GB
ollama pull nomic-embed-text # embedding model, about 270 MB
```

## Commands

Run from the project folder with the virtual environment active:

```bash
python -m docqa ask "Please suggest a shirt with sunblocking"
python -m docqa ask "Please suggest a shirt with sunblocking" --debug
python -m docqa eval
```

- `ask` prints only the answer. With `--debug` it also prints the question, each retrieved piece (id, score, text), the full prompt, the reply and the token counts.
- `eval` runs 2 hand-written examples plus 5 examples the model writes from the first 5 catalog entries, grades each answer by meaning with the model, and ends with `CORRECT: <n>  INCORRECT: <m>`. The grader is a local 8B model, weaker than a hosted one, so some grades can be wrong.
- If Ollama is not running or a model is missing, both commands exit with a non-zero status and tell you to run `ollama serve` and `ollama pull ...`.

## Settings (environment variables or a local `.env` file)

| Variable | Default | Meaning |
|---|---|---|
| `DOCQA_CHAT_MODEL` | `llama3.1:8b` | Ollama chat model |
| `DOCQA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `DOCQA_STANDINS` | unset | `1` uses the scripted stand-ins (see below) |
| `DOCQA_DATA_DIR` | `sample_data/` | folder with `catalog.csv` and `examples.json` |

## Tests

```bash
python -m pytest -m "not live"   # offline tests, scripted stand-ins, no network
python -m pytest -m live         # needs a running Ollama with both models; skipped otherwise
python -m pytest                 # everything
```

Skipped live tests are not failures.

## Stand-ins (offline demo)

`DOCQA_STANDINS=1` replaces the embedder and the chat model with scripted stand-ins: no network, no Ollama. Answers are fixed for the sample questions, and a `[STAND-IN ...]` notice is printed on stderr.

```bash
DOCQA_STANDINS=1 python -m docqa ask "Please suggest a shirt with sunblocking" --debug
DOCQA_STANDINS=1 python -m docqa eval
```

## Freeing disk space afterwards

`brew uninstall ollama` removes the program. The downloaded models live in `~/.ollama`; delete that folder (or run `ollama rm llama3.1:8b nomic-embed-text`) to get the space back.

## Structure

This is the infra/structure diagram of this app

```
python -m docqa ask|eval [--debug]
        |
   cli.py --> config.py (Settings: env vars, .env)
        |
   documents.py: sample_data/catalog.csv  -> one document per row
        |
   embedder: OllamaEmbeddings (nomic-embed-text)   [stand-in with DOCQA_STANDINS=1]
        |
   store.py: in-memory store, rebuilt at each start
        |
   question -> embed -> cosine top 4 -> qa.py prompt ("stuff") -> answerer: ChatOllama (llama3.1:8b)
        |
   eval only: evaluation.py
        example writer (ChatOllama, JSON) -> ask -> grader (ChatOllama) -> report
```

## Known limitations and production roadmap

This is a course exercise on 8 documents. Two choices would change at larger scale.

- **The same 8B model answers, writes examples and grades.** A model grading its own kind of output tends to favour it (self-preference bias) and reasons poorly on edge cases. A 7/7 result from `eval` should be read with suspicion, not as proof of quality. Generated questions can also be vague: one earlier run produced "What is the name of the article?", which the program rightly failed. In production, use a stronger judge model, grading rubrics with reference answers, or human spot-checks of a sample of results.
- **Retrieval is an exact scan.** `store.py` computes cosine similarity against every stored vector, which costs O(N·d) per question for N vectors of dimension d. Below a few thousand pieces this is fine, because it always finds the true nearest neighbours and needs no index. At tens of thousands of pieces or more, switch to an approximate nearest-neighbour index (for example HNSW or IVF in FAISS, or a vector database such as Qdrant). That trades a small loss of recall for much lower latency.

Other simplifications: the store is rebuilt in memory at every start (C7a), documents are not split into smaller pieces (C5a), and all pieces go into a single prompt (C8a).
