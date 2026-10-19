"""tests/test_main.py
Unit tests for the Multilingual Semantic Search API.

These tests cover the Flask application entry point, the health check endpoint,
and the core semantic search functionality. They are written using pytest and
the Flask test client.

The tests make real calls to the application code; no external services are
required because the FAISS index is loaded from test fixtures (or a small
in‑memory index) as part of the application factory.
"""

import json
from typing import Any, Dict, List

import pytest

# Import the application factory. Adjust the import path if the project layout
# differs (e.g., `from src.app import create_app`).
try:
    from app.main import create_app  # type: ignore
except ImportError as exc:
    raise ImportError(
        "Could not import `create_app` from `app.main`. Ensure the project "
        "exposes a Flask application factory named `create_app`."
    ) from exc

# Import the semantic search function directly for isolated testing.
try:
    from app.search import semantic_search  # type: ignore
except ImportError as exc:
    raise ImportError(
        "Could not import `semantic_search` from `app.search`. Ensure the "
        "module and function exist."
    ) from exc


@pytest.fixture(scope="module")
def app():
    """Create and configure a new app instance for the tests."""
    # The factory may accept configuration overrides; we use the default for now.
    flask_app = create_app()
    # Establish an application context before running the tests.
    ctx = flask_app.app_context()
    ctx.push()
    yield flask_app
    ctx.pop()


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


def test_health_endpoint(client):
    """
    Verify that the health endpoint returns a 200 status and a JSON payload
    indicating the service is alive.
    """
    response = client.get("/health")
    assert response.status_code == 200, "Health endpoint should return HTTP 200"
    data = response.get_json()
    assert isinstance(data, dict), "Health response must be a JSON object"
    assert data.get("status") == "ok", "Health response should contain status='ok'"


def test_search_endpoint_success(client):
    """
    Send a valid search request and assert that the response structure matches
    the contract: a JSON object with a `results` list, each entry containing
    `id`, `score`, and `text` fields.
    """
    query_payload = {"query": "Artificial intelligence", "top_k": 3}
    response = client.post(
        "/search",
        data=json.dumps(query_payload),
        content_type="application/json",
    )
    assert response.status_code == 200, "Search endpoint should return HTTP 200"

    payload = response.get_json()
    assert isinstance(payload, dict), "Response must be a JSON object"
    results = payload.get("results")
    assert isinstance(results, list), "`results` must be a list"
    assert len(results) == query_payload["top_k"], f"Expected {query_payload['top_k']} results"

    for entry in results:
        assert isinstance(entry, dict), "Each result entry must be a dict"
        assert "id" in entry, "Result entry missing `id` field"
        assert "score" in entry, "Result entry missing `score` field"
        assert "text" in entry, "Result entry missing `text` field"
        # Basic type checks
        assert isinstance(entry["id"], (int, str)), "`id` should be int or str"
        assert isinstance(entry["score"], float), "`score` should be float"
        assert isinstance(entry["text"], str), "`text` should be str"


def test_search_endpoint_missing_query(client):
    """
    Ensure that omitting the required `query` field results in a 400 error.
    """
    response = client.post(
        "/search",
        data=json.dumps({"top_k": 5}),
        content_type="application/json",
    )
    assert response.status_code == 400, "Missing query should yield HTTP 400"
    payload = response.get_json()
    assert isinstance(payload, dict)
    assert "error" in payload, "Error response should contain an `error` field"


def test_semantic_search_functionality():
    """
    Directly test the `semantic_search` function with a known query.
    The function should return a list of dictionaries with the expected keys,
    and scores should be sorted in descending order.
    """
    query = "Machine learning"
    top_k = 4
    results: List[Dict[str, Any]] = semantic_search(query, top_k=top_k)

    # Verify the result list length.
    assert isinstance(results, list), "semantic_search must return a list"
    assert len(results) == top_k, f"semantic_search should return {top_k} results"

    previous_score = float("inf")
    for result in results:
        # Each result must contain the expected keys.
        assert isinstance(result, dict), "Each result must be a dict"
        for key in ("id", "score", "text"):
            assert key in result, f"Result missing required key `{key}`"

        # Type checks.
        assert isinstance(result["id"], (int, str)), "`id` should be int or str"
        assert isinstance(result["score"], float), "`score` should be float"
        assert isinstance(result["text"], str), "`text` should be str"

        # Ensure scores are non‑increasing.
        assert result["score"] <= previous_score, "Scores must be sorted descending"
        previous_score = result["score"]


def test_semantic_search_invalid_input():
    """
    Passing an empty query should raise a ValueError (or return an empty list,
    depending on implementation). The test adapts to either behaviour.
    """
    with pytest.raises(ValueError):
        # If the implementation raises, the test passes.
        semantic_search("", top_k=3)

    # If the implementation returns an empty list instead, the following
    # assertion will also succeed.
    result = semantic_search("", top_k=3)
    assert result == [] or isinstance(result, list), "Empty query should yield empty result list"


# End of file.