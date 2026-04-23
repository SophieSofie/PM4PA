# Agentic RAG System

## Overview


This repository implements an **agentic retrieval-augmented generation (RAG) pipeline** that supports **document-grounded BPMN 2.0 process modeling**. Given heterogeneous text sources (e.g. regulations, handbooks, PDF or Office documents), the system **retrieves** relevant passages, **drafts** a process description, **generates** a nested BPMN JSON representation, and **validates** (and optionally **revises**) the model against the query and sources. The design targets **public administration** knowledge bases but is applicable wherever processes must be modelled from different sources.

**Source annotation** on generated process elements (e.g. links to documents, pages, or retrieved chunks in model metadata) **increases transparency and explainability**, so user can trace model content back to the underlying corpus.

Interaction is supported through a **Streamlit** web interface and through **command-line** entrypoints for single runs and for **quantitative evaluation** against gold BPMN artifacts (see [src/eval/evaluation.md](src/eval/evaluation.md)). The Streamlit app can be run **in Docker** for a reproducible environment; see [Docker](#docker).

## Table of contents

- [Web interface](#web-interface)
- [Setup](#setup)
- [Pipeline](#pipeline)
- [Repository structure](#repository-structure)
- [Installation and usage](#installation-and-usage)
- [Docker](#docker)
- [License](#license)

## Web interface

**Document ingestion.** Users can upload or select source documents (PDF, Word, text, etc.); the app ingests them into the configured Chroma collection so retrieval is grounded in the corpus.

![Document Upload](images/Document%20Upload.png)

**Process modeling view.** After the documents are uploaded, the user inserts a query about the process which should be modelled based on the uploaded documents. The user can also select, if open source or closed source models should be used for generation.

![Process modeler](images/Process%20modeller.png)

**Resulting process model.** The final generated process model is displayed and can be edited and downloaded in .bpmn, .xml, .pdf, or .png format. When a modeled process element is selected, the corresponding reference aka the source document from which the process element was extracted is listed on the right, increasing transparency.

![Modelled process](images/Modelled%20process.png)


## Setup

### Environment

- **Python** 3.11 or newer (aligned with the provided `Dockerfile`).
- **Working directory:** repository root (the directory containing `src/`) for CLI modules and relative paths.
- **Network:** the runtime host (or container) must reach whichever **LLM** and **embedding** endpoints are configured; local `localhost` URLs in `.env` may need to be rewritten for containerized runs (e.g. `host.docker.internal` on Docker Desktop).

### Dependencies

Install Python packages from the lockfile-equivalent list:

```bash
pip install -r requirements.txt
```

Heavy optional stacks (e.g. local embedding models via `sentence-transformers`) follow the same file; GPU is not assumed.

### Configuration

1. Copy `.env.example` to `.env`.
2. Set at least: API base URLs and keys for your **Open WebUI / Ollama** and/or **Azure OpenAI** setup, embedding behavior, timeouts, and **`CHROMA_DB_PATH`** / **`CHROMA_COLLECTION_NAME`** as needed.

### Optional system tools (host only)

For **scanned PDFs** or text embedded in images, **Tesseract** and **Poppler** improve extraction quality when running **outside** the default Docker image (the slim image does not ship these packages).

## Pipeline

End-to-end behavior, summarized:

1. **Ingestion** — Documents are parsed, chunked, embedded, and stored in **ChromaDB**.
2. **Retrieval** — The user query drives **hybrid** retrieval (e.g. dense + lexical components)
3. **Drafting** — An intermediate **process draft** is produced from retrieved context.
4. **BPMN generation** — Structured **nested BPMN JSON** is generated and is automatically submitted to an external service for XML layout.
5. **Validation and revision** — Configurable validators compare the model to sources and the query; the graph may loop for **revision** within iteration limits.

## Repository structure

Application code is packaged under **`src/`**:

```
.
├── src/
│   ├── app/                    # CLI entrypoints and orchestration
│   │   ├── pipeline.py         # GraphRAGSystem: wiring, graph execution
│   │   ├── ingestion.py        # Document → Chroma ingestion CLI
│   │   └── run_request.py      # Single full pipeline run from the terminal
│   ├── agents/                 # LLM agents (draft, BPMN, retrieval, relevance, validation, judge)
│   ├── bpmn_service/           # HTTP client helpers for external BPMN layout / conversion
│   ├── eval/                   # Evaluation CLI, metrics, evaluation.md, run artifacts
│   ├── graphs/
│   │   └── pipeline_graphs.py  # Pydantic Graph nodes and get_graph_for_setting
│   ├── infrastructure/         # API clients, ingestion, retrieval, Chroma store
│   ├── models/                 # Pydantic domain models
│   ├── web/                    # Streamlit UI and .streamlit theme
│   └── config.py               # Environment-backed settings
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

**Runtime artifacts** (commonly gitignored): vector store at `CHROMA_DB_PATH` (default `./chroma_db`), evaluation outputs under `eval_output/`, optional local corpora (e.g. `data/`).

## Docker

Compose injects `.env` and persists Chroma under a named volume (`chroma_data` → `/data/chroma`); see `docker-compose.yml`. The image is listed in `.dockerignore` so secrets are not copied into layers. Remote services referenced in `.env` must be reachable **from inside the container**.

**Build and run:**

```bash
docker compose up --build
```

**Image only** (no container):

```bash
docker compose build
```

**Run image without Compose** (example):

```bash
docker build -t agentic-rag .
docker run --rm -p 8501:8501 --env-file .env \
  -v agentic_chroma:/data/chroma \
  -e CHROMA_DB_PATH=/data/chroma \
  agentic-rag
```


## Installation and usage

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```
**1. Edit .env** (endpoints, keys, CHROMA_DB_PATH, etc.)

**2. Run the system**
**Option 1: Web interface** — theme: `src/web/.streamlit/config.toml`. From `src/web` so Streamlit resolves config:

```bash
cd src/web
streamlit run streamlit_app.py
```

Then open **http://localhost:8501** (default). Browser uploads are ephemeral unless your deployment persists them; the durable index lives under `CHROMA_DB_PATH`.

**Option 2: Single CLI run** (default graph matches the main web flow unless `--setting` overrides):
**1. Ingest documents** (after configuring paths/collection as needed):

```bash
python -m src.app.ingestion
python -m src.app.ingestion --skip-existing   # skip already embedded files
```
**2. Run BPMN Generation**

```bash
python -m src.app.run_request --query "The process to be modelled"
```

**Information about the evaluation of this project:** final evaluation metrics & instructions on how to run the evaluation pipeline [src/eval/evaluation.md](src/eval/evaluation.md).


## License

This project is licensed under the [MIT License](LICENSE).
