# HARC-RAG

Hallucination-Aware Retrieval-Augmented Generation (HARC-RAG) is a local PDF question-answering system. It retrieves evidence from uploaded documents, generates an answer constrained by that evidence, estimates confidence, and routes low-confidence answers to a local verifier.

## Problem Statement

Retrieval-augmented generation can produce answers that sound plausible even when the retrieved document does not support them. HARC-RAG is designed to detect low-confidence answers, verify them against retrieved context, and avoid exposing unsupported information as fact.

## Objectives

- Answer questions using uploaded PDF content.
- Combine dense and lexical retrieval.
- Estimate retrieval, generation, and evidence confidence.
- Route low-confidence answers to verification.
- Preserve supported claims, correct unsupported claims where possible, and safely refuse unsupported questions.
- Maintain conversation context for chat requests.

## Key Features

- PDF upload and indexing through FastAPI.
- Character-based document chunking.
- MiniLM sentence embeddings.
- FAISS dense retrieval and BM25 lexical retrieval combined with reciprocal-rank fusion.
- Ollama-backed answer generation and local verification.
- Joint confidence metadata and adaptive verification thresholds.
- Claim-level `SUPPORTED`, `CORRECTED`, and `UNSUPPORTED` verification results.
- Source chunks and routing reasons in API responses and the frontend.
- SQLite-backed conversation memory.

## System Architecture

```text
PDF upload
		-> DocumentLoader
		-> TextSplitter
		-> MiniLM embeddings
		-> FAISS vector store + BM25 index

Question
		-> HybridRetriever
		-> PromptBuilder
		-> Ollama RAG generator
		-> JointEstimator
		-> AdaptiveRouter
			 -> high confidence: return RAG answer
			 -> low confidence: LocalVerifier
		-> answer, confidence metadata, verdict, reason, and sources
```

## End-to-End RAG Workflow

1. The client uploads a PDF to `POST /documents/upload`.
2. The backend loads the PDF with `pypdf`.
3. Text is split into overlapping character chunks.
4. Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2`.
5. Embeddings are added to FAISS and chunks are added to BM25.
6. A question is sent to `POST /chat`.
7. Dense and BM25 results are fused and inserted into a context-only prompt.
8. Ollama generates the initial RAG answer.
9. Confidence is calculated from retrieval, generation, and evidence signals.
10. The adaptive router either returns the answer or sends it to verification.

## Hallucination-Aware Verification

Low-confidence answers are checked by `LocalVerifier` using the question, generated answer, and up to the first three retrieved chunks. The verifier is instructed to use only retrieved context and return structured claim-level output. The pipeline returns the verified or corrected answer, its verdict, and the verification reason.

## Claim-Level Logic

The verifier reports one `CLAIM`, `STATUS`, and `REASON` block per requested claim, followed by an `ANSWER`. Python aggregates the claim statuses:

- `SUPPORTED`: every requested claim is supported; retain the generated answer.
- `CORRECTED`: at least one claim is supported and at least one is missing or unsupported; return only supported information and identify unavailable details.
- `UNSUPPORTED`: no requested claim is supported; return the canonical safe refusal.

The system does not use outside knowledge to fill missing numbers, countries, percentages, effect sizes, causal claims, or comparisons.

## Technology Stack

- Python 3.12 or newer
- FastAPI and Uvicorn
- Pydantic and pydantic-settings
- `pypdf` for PDF extraction
- NumPy and FAISS for vector search
- `sentence-transformers` for embeddings and semantic similarity
- `rank-bm25` for lexical retrieval
- Ollama Python client with the configured `qwen2.5:3b` model
- SQLite for conversation memory
- HTML, CSS, and browser JavaScript frontend
- pytest for testing

## Project Structure

```text
data/uploads/                 Uploaded PDFs
docs/                         Project documentation
frontend/                     Browser client: index.html, app.js, styles.css
src/harc_rag/
	api/                        FastAPI app, routes, request/response models
	chunking/                   Chunk models, strategies, and splitter
	document/                   PDF loader and document models
	embedding/                  MiniLM embedding implementation
	evaluation/                 Retrieval evaluation utilities
	generation/                 RAG generation and prompt building
	llm/                        Ollama client
	memory/                     Conversation memory and SQLite store
	pipeline/                   Main HARC-RAG pipeline
	retrieval/                  Dense, BM25, hybrid, and fusion retrieval
	routing/                    Adaptive routing and cost utilities
	uncertainty/                Confidence estimators and thresholds
	verification/               Local verifier and verification models
	vectorstore/                FAISS vector store
tests/                        Unit, API, pipeline, and evaluation tests
```

## Installation and Setup

Create or use the project virtual environment and install the project with its development dependencies. The repository declares runtime dependencies in `pyproject.toml` and pytest in the `dev` dependency group.

```powershell
cd C:\harc-rag
uv sync --dev
```

Install and start Ollama separately, then make sure the configured model is available:

```powershell
ollama pull qwen2.5:3b
```

The default Ollama endpoint is `http://127.0.0.1:11434`. It can be changed with `HARC_RAG_OLLAMA_HOST`; the model can be changed with `HARC_RAG_MODEL`.

## Start the Backend

From the project root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m uvicorn harc_rag.api.main:app --host 127.0.0.1 --port 8000
```

The API is available at `http://127.0.0.1:8000`. Health checking is available at `GET /health`. The backend also mounts the frontend at `/app` when the frontend directory exists.

## Start the Frontend

The frontend is plain static HTML, CSS, and JavaScript. With the backend running, serve it from the frontend directory:

```powershell
cd C:\harc-rag\frontend
python -m http.server 5500
```

Open `http://127.0.0.1:5500`. The frontend uses `http://127.0.0.1:8000` as its API base URL.

## Live Demo

For a local demonstration, start Ollama, the backend, and the static frontend using the commands above. Open the frontend at `http://127.0.0.1:5500`, upload a PDF, and ask a question. Each assistant response shows the final answer, confidence components, verification status, routing reason, retrieved chunk count, and source text. The backend-mounted alternative is `http://127.0.0.1:8000/app/` when accessing the FastAPI application directly.

## Upload and Query a PDF

1. Open the frontend.
2. Select **Upload PDF** and choose a PDF file.
3. Wait for the indexing confirmation.
4. Enter a question and select **Send**.
5. Review the answer, confidence values, verification status, routing reason, retrieved chunk count, and sources.

Equivalent API requests can be made with any HTTP client:

```powershell
curl.exe -X POST http://127.0.0.1:8000/documents/upload -F "file=@path\to\document.pdf"
curl.exe -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\":\"What is discussed in this document?\"}"
```

The chat response includes `answer`, `original_answer`, confidence components, `verified`, `verification_verdict`, `verification_reason`, `retrieved_chunks`, and `sources`.

## Evaluation Methodology

The evaluation runner loads the Development Studies PDF at:

```text
data/uploads/Towards a  Development Humanities   widening the multi-disciplinary field of development studies.pdf
```

It builds a real document index and `HARCRAGPipeline`, then executes all 20 cases from `tests/evaluation/harc_test_cases.json` through retrieval, generation, confidence estimation, routing, and verification. Results are written to `tests/evaluation/results.json`, including the original answer, final answer, confidence components, threshold, verification flag, and verdict.

The cases contain 10 supported, 5 partially supported, and 5 unsupported questions.

## Current Evaluation Results

The completed 20-question evaluation recorded:

| Metric | Result |
| --- | ---: |
| Total questions | 20 |
| Verification rate | 0.550 |
| Unsupported detection rate | 0.600 |
| Correction rate | 0.000 |
| Safe refusal rate | 0.600 |
| High-confidence no-verification | 9 |
| Low-confidence verification | 11 |
| Unsupported verification | 5 |

The final saved artifact contained 3 `SUPPORTED`, 0 `CORRECTED`, and 8 `UNSUPPORTED` verification outcomes. The local 3B model remains conservative when a question contains partial evidence: it often safely refuses instead of producing an evidence-only correction.

## Known Limitations

- The FAISS and BM25 indexes are in memory and are built during the process lifetime; uploaded documents are not automatically restored into a new process.
- Ollama must be running locally with a compatible model.
- The local `qwen2.5:3b` verifier can classify partially supported compound questions as `UNSUPPORTED` even when some evidence is available.
- Confidence and semantic similarity depend on the embedding model and retrieved chunk quality.
- The current evaluation includes intentionally compound partial-evidence questions, so correction rate measures both verifier classification and answer correction behavior.
- FastAPI emits deprecation warnings for the current startup event API.

## Future Improvements

- Improve claim decomposition and evidence alignment while retaining deterministic, evidence-only safeguards.
- Add persistence for document indexes and metadata.
- Add stronger evaluation instrumentation for raw verifier output and per-claim decisions.
- Add deterministic integration tests for API metadata and frontend status rendering.
- Evaluate additional local models and embedding/retrieval configurations.

## Example Questions

For the included Development Studies paper:

- What new sub-field does the paper propose within Development Studies?
- Which source materials does Development Humanities use?
- What debate did C. P. Snow initiate during the 1950s?
- How can community arts contribute to reconciliation in post-conflict settings?
- What risks of establishing Development Humanities does the conclusion identify?

Unsupported questions should receive a safe refusal when the uploaded document provides no useful evidence, for example:

- What is the GDP of Japan in 2026?
- What is the chemical formula of water?

## Testing

Run the complete test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The final verified test run passed **54 tests**, with 3 deprecation warnings.

Run the complete evaluation:

```powershell
.\.venv\Scripts\python.exe tests\evaluation\run_evaluation.py
```

Run focused pipeline tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_pipeline.py
```
