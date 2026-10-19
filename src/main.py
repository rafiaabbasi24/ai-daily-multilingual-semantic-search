import os
import logging
from typing import List, Dict, Any

import torch
import faiss
from flask import Flask, request, jsonify, abort
from transformers import AutoModel, AutoTokenizer

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MODEL_NAME = os.getenv(
    "SEMANTIC_SEARCH_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_DIM = 384  # This is the output size of the above model
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "")
MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", "100000"))

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Perform mean pooling on token embeddings.

    Args:
        model_output: Tensor of shape (batch_size, seq_len, hidden_size).
        attention_mask: Tensor of shape (batch_size, seq_len).

    Returns:
        Tensor of shape (batch_size, hidden_size) with pooled embeddings.
    """
    token_embeddings = model_output  # (batch_size, seq_len, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


# ----------------------------------------------------------------------
# Core semantic search engine
# ----------------------------------------------------------------------


class SemanticSearchEngine:
    """
    Encapsulates a multilingual transformer model and a FAISS index for
    fast semantic similarity search.
    """

    def __init__(self, model_name: str, embedding_dim: int, max_documents: int = 100000):
        """
        Initialise the engine.

        Args:
            model_name: HuggingFace model identifier.
            embedding_dim: Dimensionality of the embedding vectors.
            max_documents: Maximum number of vectors the FAISS index can hold.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading model %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.embedding_dim = embedding_dim
        self.max_documents = max_documents

        # Initialise a flat (inner product) index; we will normalise vectors for cosine similarity.
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.id_to_doc: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0

        if FAISS_INDEX_PATH and os.path.isfile(FAISS_INDEX_PATH):
            self._load_faiss_index(FAISS_INDEX_PATH)

    # ------------------------------------------------------------------
    # Embedding utilities
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> torch.Tensor:
        """
        Compute normalized embeddings for a list of texts.

        Args:
            texts: List of raw strings.

        Returns:
            Tensor of shape (len(texts), embedding_dim) with L2‑normalised embeddings.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            model_output = self.model(**encoded)
            embeddings = mean_pooling(model_output.last_hidden_state, encoded["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu()

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def add_documents(self, docs: List[Dict[str, Any]]) -> List[int]:
        """
        Add a batch of documents to the index.

        Each document dict must contain at least a ``text`` field. Additional
        metadata can be supplied and will be returned unchanged in search results.

        Args:
            docs: List of document dictionaries.

        Returns:
            List of internal integer IDs assigned to the documents.
        """
        texts = [doc["text"] for doc in docs]
        embeddings = self._embed(texts).numpy()
        ids = []
        for doc, vec in zip(docs, embeddings):
            if self.next_id >= self.max_documents:
                raise RuntimeError("Maximum document capacity reached.")
            self.index.add(vec.reshape(1, -1))
            self.id_to_doc[self.next_id] = doc
            ids.append(self.next_id)
            self.next_id += 1
        logger.info("Added %d documents (total=%d)", len(ids), len(self.id_to_doc))
        return ids

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a semantic similarity search.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of document dicts ordered by relevance. Each dict includes a
            ``score`` field (cosine similarity) and the original document fields.
        """
        if self.index.ntotal == 0:
            return []

        query_emb = self._embed([query]).numpy()
        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            doc = self.id_to_doc.get(idx, {})
            result = {"score": float(score), **doc}
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist the FAISS index and document mapping to ``path``.
        """
        os.makedirs(path, exist_ok=True)
        faiss_path = os.path.join(path, "index.faiss")
        meta_path = os.path.join(path, "metadata.pt")
        faiss.write_index(self.index, faiss_path)
        torch.save({"id_to_doc": self.id_to_doc, "next_id": self.next_id}, meta_path)
        logger.info("Saved FAISS index to %s and metadata to %s", faiss_path, meta_path)

    def _load_faiss_index(self, path: str) -> None:
        """
        Load a previously saved FAISS index and metadata.
        """
        faiss_path = os.path.join(path, "index.faiss")
        meta_path = os.path.join(path, "metadata.pt")
        if not os.path.isfile(faiss_path) or not os.path.isfile(meta_path):
            logger.warning("FAISS index or metadata not found at %s; starting empty.", path)
            return
        self.index = faiss.read_index(faiss_path)
        data = torch.load(meta_path, map_location=self.device)
        self.id_to_doc = data.get("id_to_doc", {})
        self.next_id = data.get("next_id", 0)
        logger.info("Loaded FAISS index with %d vectors.", self.index.ntotal)


# ----------------------------------------------------------------------
# Flask application
# ----------------------------------------------------------------------


def create_app() -> Flask:
    """
    Factory that creates and configures the Flask app.
    """
    app = Flask(__name__)
    engine = SemanticSearchEngine(MODEL_NAME, EMBEDDING_DIM, max_documents=MAX_DOCUMENTS)

    @app.route("/healthz", methods=["GET"])
    def health_check():
        """Simple health check endpoint."""
        return jsonify({"status": "ok"}), 200

    @app.route("/documents", methods=["POST"])
    def add_documents():
        """
        Add documents to the semantic store.

        Expected JSON payload:
        {
            "documents": [
                {"text": "...", "metadata_key": "..."},
                ...
            ]
        }
        """
        if not request.is_json:
            abort(400, description="Request body must be JSON.")
        payload = request.get_json()
        docs = payload.get("documents")
        if not isinstance(docs, list) or not docs:
            abort(400, description="`documents` must be a non‑empty list.")
        try:
            ids = engine.add_documents(docs)
        except Exception as exc:
            logger.exception("Failed to add documents")
            abort(500, description=str(exc))
        return jsonify({"added_ids": ids}), 201

    @app.route("/search", methods=["POST"])
    def search():
        """
        Perform a semantic search.

        Expected JSON payload:
        {
            "query": "text to search",
            "top_k": 5   // optional
        }
        """
        if not request.is_json:
            abort(400, description="Request body must be JSON.")
        payload = request.get_json()
        query = payload.get("query")
        top_k = int(payload.get("top_k", 5))
        if not query or not isinstance(query, str):
            abort(400, description="`query` must be a non‑empty string.")
        results = engine.search(query, top_k=top_k)
        return jsonify({"results": results}), 200

    @app.route("/save", methods=["POST"])
    def save_index():
        """
        Persist the current index to disk.

        Expected JSON payload:
        {
            "path": "/optional/custom/path"
        }
        If ``path`` is omitted, the environment variable ``FAISS_INDEX_PATH`` is used.
        """
        if not request.is_json:
            abort(400, description="Request body must be JSON.")
        payload = request.get_json()
        path = payload.get("path") or FAISS_INDEX_PATH
        if not path:
            abort(400, description="No path supplied and FAISS_INDEX_PATH not set.")
        try:
            engine.save(path)
        except Exception as exc:
            logger.exception("Failed to save index")
            abort(500, description=str(exc))
        return jsonify({"saved_to": path}), 200

    return app


app = create_app()

if __name__ == "__main__":
    # When executed directly, run the Flask development server.
    # In production, a WSGI server (gunicorn, uWSGI, etc.) should be used.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)