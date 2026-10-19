# Multilingual Semantic Search API

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Service](#running-the-service)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Performance & Scaling](#performance--scaling)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview
The **Multilingual Semantic Search API** provides a RESTful interface for performing semantic similarity search across documents in multiple languages. It leverages state‑of‑the‑art transformer models from the 🤗 Transformers library to embed text, and FAISS (Facebook AI Similarity Search) for fast, scalable nearest‑neighbor retrieval.

The service is built with **Python 3.11**, **Flask**, **Transformers**, **FAISS**, and **PyTest** for testing. It is designed to be container‑friendly and can be deployed behind a reverse proxy or as a part of a larger micro‑services architecture.

---

## Features
- **Multilingual embeddings** using `sentence-transformers` models (e.g., `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
- **GPU‑accelerated inference** when a CUDA‑enabled device is available.
- **Efficient similarity search** with FAISS IndexFlatIP (inner product) and optional IVF/PQ quantization for large corpora.
- **Dynamic index updates**: add, delete, or replace documents without downtime.
- **REST API** with JSON payloads for indexing and searching.
- **Docker support** (Dockerfile included).
- **Comprehensive test suite** with PyTest and coverage.

---

## Architecture
```
┌─────────────────────┐
│   Flask Web Server  │
│  (app.py, routes)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Embedding Service │   ← sentence‑transformers
│   (embedder.py)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   FAISS Index       │   ← faiss.IndexFlatIP
│   (index_manager.py)│
└─────────────────────┘
```

- **app.py** – Flask entry point, request validation, and routing.
- **embedder.py** – Handles model loading, tokenization, and batch embedding.
- **index_manager.py** – Wraps FAISS index operations (add, search, delete, persist).
- **config.py** – Central configuration (environment variables, defaults).
- **tests/** – PyTest suite covering all public functions and API endpoints.

---

## Prerequisites
- Python **3.11** or newer
- `pip` (>=23.0)
- Optional: NVIDIA GPU with CUDA 11.8+ for accelerated inference
- (Optional) Docker Engine 20.10+ for containerized deployment

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/multilingual-semantic-search-api.git
cd multilingual-semantic-search-api
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install Python dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Install the GPU version of PyTorch
If you have a CUDA‑capable GPU, install the matching PyTorch wheel:
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```
Otherwise, the CPU‑only wheel will be installed automatically.

### 5. Verify installation
```bash
python -c "import transformers, faiss, flask; print('All imports succeeded')"
```

---

## Configuration
Configuration is driven by environment variables. Default values are defined in `config.py`. Override them as needed.

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Flask host address | `0.0.0.0` |
| `PORT` | Flask listening port | `5000` |
| `MODEL_NAME` | HuggingFace model identifier | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `FAISS_INDEX_PATH` | Path to persisted FAISS index file | `faiss_index.bin` |
| `MAX_BATCH_SIZE` | Max documents per embedding batch | `64` |
| `EMBEDDING_DIM` | Dimensionality of the model embeddings (auto‑detected) | `-` |
| `LOG_LEVEL` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |

Example (Unix):
```bash
export MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
export LOG_LEVEL=DEBUG
```

---

## Running the Service

### Development mode
```bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host $HOST --port $PORT
```

### Production mode (Gunicorn)
```bash
gunicorn --workers 4 --bind ${HOST}:${PORT} app:app
```

### Docker
A ready‑to‑use Dockerfile is provided.

```bash
docker build -t multilingual-semantic-search .
docker run -d -p 5000:5000 \
  -e MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  multilingual-semantic-search
```

---

## API Endpoints

### 1. **Health Check**
- **URL**: `GET /health`
- **Response**: `200 OK` with JSON `{ "status": "healthy" }`

### 2. **Add Documents**
- **URL**: `POST /documents`
- **Payload**:
```json
{
  "documents": [
    { "id": "doc-1", "text": "Your text here" },
    { "id": "doc-2", "text": "Another piece of text" }
  ]
}
```
- **Response**: `201 Created` with `{ "added": 2 }`

### 3. **Delete Document**
- **URL**: `DELETE /documents/<doc_id>`
- **Response**: `200 OK` with `{ "deleted": true }`

### 4. **Search**
- **URL**: `POST /search`
- **Payload**:
```json
{
  "query": "Find similar content in any language",
  "k": 5,
  "filter_ids": ["doc-1", "doc-3"]   // optional
}
```
- **Response**: `200 OK` with:
```json
{
  "results": [
    { "id": "doc-7", "score": 0.92 },
    { "id": "doc-3", "score": 0.88 }
  ]
}
```

### 5. **Persist Index**
- **URL**: `POST /index/persist`
- **Response**: `200 OK` with `{ "path": "faiss_index.bin" }`

---

## Testing

### Run the full test suite
```bash
pytest -v --cov=.
```

### Test coverage report
```bash
pytest --cov=./ --cov-report=html
# Open htmlcov/index.html in a browser
```

### Sample test command (quick sanity check)
```bash
python - <<'PY'
from embedder import Embedder
e = Embedder()
vec = e.embed_text("Hello world")
assert len(vec) == e.dim, "Embedding dimension mismatch"
print("Embedding test passed")
PY
```

All tests are located in the `tests/` directory and cover:
- Model loading and batching
- FAISS index CRUD operations
- Flask request validation
- End‑to‑end API integration using the Flask test client

---

## Performance & Scaling

| Concern | Recommendation |
|---------|----------------|
| **Large corpora (>1 M docs)** | Use `faiss.IndexIVFFlat` or `faiss.IndexIVFPQ` with training step; adjust `nlist` and `nprobe` for speed/accuracy trade‑off. |
| **High QPS** | Deploy multiple Gunicorn workers behind a load balancer; enable model inference on GPU; cache recent query embeddings. |
| **Cold start** | Persist the FAISS index to disk and load it at startup; keep the transformer model in memory (singleton pattern). |
| **Horizontal scaling** | Store the FAISS index in a shared volume (e.g., NFS) or use a distributed FAISS variant; keep the Flask stateless. |

---

## License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

## Acknowledgements
- 🤗 Transformers & Sentence‑Transformers teams for the multilingual models.
- Facebook AI Research for FAISS.
- The Flask community for the lightweight web framework.
- All contributors who helped shape this project.