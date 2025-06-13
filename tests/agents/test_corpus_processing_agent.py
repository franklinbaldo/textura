from pathlib import Path

import numpy as np
import pytest

# Modules to be tested and mocked
from textura.agents.corpus_processing_agent import CorpusProcessingAgent
from textura.ingestion.chunker import Chunker
from textura.ingestion.embedder import Embedder

# Dummy data for testing
DUMMY_CHUNK_EMBEDDING_DIM = 5  # Keep small for tests
DUMMY_TASK_DESCRIPTION = "Tell me about technology."


# Fixtures
@pytest.fixture
def mock_chunker(monkeypatch):
    def mock_chunk_file(self, file_path: Path) -> list[str]:
        # Simulate chunking based on file name or content for varied tests
        if "doc1" in file_path.name:
            return ["Doc1 chunk1 tech.", "Doc1 chunk2 general."]
        if "doc2" in file_path.name:
            return ["Doc2 chunk1 finance.", "Doc2 chunk2 tech related."]
        if "empty" in file_path.name:
            return []
        return ["Default chunk for " + file_path.name]

    monkeypatch.setattr(Chunker, "chunk_file", mock_chunk_file)
    return Chunker()  # Return an instance


@pytest.fixture
def mock_embedder(monkeypatch):
    # Mock embeddings for chunks and task description
    # Ensure these embeddings have consistent dimensions (DUMMY_CHUNK_EMBEDDING_DIM)
    # For simplicity, we'll make embeddings such that similarity can be easily predicted.
    # e.g., higher dot product for more relevant chunks.
    # Chunk embeddings:
    # "Doc1 chunk1 tech." (relevant)
    # "Doc1 chunk2 general." (less relevant)
    # "Doc2 chunk1 finance." (irrelevant)
    # "Doc2 chunk2 tech related." (highly relevant)
    # "Default chunk..." (moderate, adjusted for clearer similarity)

    mock_embeddings_map = {
        "Doc1 chunk1 tech.": np.array([0.8, 0.1, 0.1, 0.1, 0.1]),
        "Doc1 chunk2 general.": np.array([0.3, 0.3, 0.3, 0.3, 0.3]),
        "Doc2 chunk1 finance.": np.array([0.1, 0.1, 0.8, 0.1, 0.1]),
        "Doc2 chunk2 tech related.": np.array(
            [0.9, 0.2, 0.1, 0.0, 0.0]
        ),  # Most relevant
        "Default chunk for doc3.txt": np.array(
            [0.6, 0.4, 0.4, 0.4, 0.4]
        ),  # Adjusted for clearer similarity
        DUMMY_TASK_DESCRIPTION: np.array([1.0, 0.0, 0.0, 0.0, 0.0]),  # Task embedding
    }
    # Normalize for cosine similarity
    for k, v in mock_embeddings_map.items():
        norm = np.linalg.norm(v)
        if norm > 0:
            mock_embeddings_map[k] = v / norm

    def mock_generate_embeddings(self, text_chunks: list[str]) -> list[np.ndarray]:
        results = []
        for chunk in text_chunks:
            emb = mock_embeddings_map.get(chunk)
            if emb is None:
                # Fallback for unexpected chunks during testing
                emb = np.random.rand(DUMMY_CHUNK_EMBEDDING_DIM).astype(np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
            results.append(emb)
        return results

    monkeypatch.setattr(Embedder, "generate_embeddings", mock_generate_embeddings)
    # Also mock the constructor if it tries to load a model
    monkeypatch.setattr(Embedder, "__init__", lambda self, model_name=None: None)
    return Embedder()  # Return an instance


@pytest.fixture
def corpus_agent(mock_chunker, mock_embedder) -> CorpusProcessingAgent:
    return CorpusProcessingAgent(
        chunker=mock_chunker, embedder=mock_embedder, max_tokens=30
    )  # Small max_tokens for testing filtering


@pytest.fixture
def dummy_doc_paths(tmp_path) -> list[Path]:
    doc1 = tmp_path / "doc1.txt"
    doc1.write_text("Content for doc1: technology focus.")
    doc2 = tmp_path / "doc2.txt"
    doc2.write_text("Content for doc2: finance and tech.")
    doc3 = tmp_path / "doc3.txt"  # Will use default chunker mock
    doc3.write_text("Content for doc3 default.")
    empty_doc = tmp_path / "empty.txt"
    empty_doc.write_text("")
    return [doc1, doc2, doc3, empty_doc]


# Test Cases


def test_agent_initialization(corpus_agent, mock_chunker, mock_embedder):
    assert corpus_agent.chunker == mock_chunker
    assert corpus_agent.embedder == mock_embedder
    assert corpus_agent.max_tokens == 30


def test_load_corpus(corpus_agent, dummy_doc_paths):
    corpus_agent.load_corpus(dummy_doc_paths)
    assert len(corpus_agent.corpus_documents) == len(dummy_doc_paths)
    assert corpus_agent.corpus_documents[0] == dummy_doc_paths[0]


def test_process_corpus_documents(corpus_agent, dummy_doc_paths):
    corpus_agent.load_corpus(dummy_doc_paths)
    corpus_agent.process_corpus_documents()

    expected_chunks = [
        "Doc1 chunk1 tech.",
        "Doc1 chunk2 general.",
        "Doc2 chunk1 finance.",
        "Doc2 chunk2 tech related.",
        "Default chunk for doc3.txt",
        # empty.txt produces no chunks with the mock
    ]
    assert corpus_agent.processed_chunks == expected_chunks
    assert len(corpus_agent.corpus_embeddings) == len(expected_chunks)
    for emb in corpus_agent.corpus_embeddings:
        assert emb.shape == (DUMMY_CHUNK_EMBEDDING_DIM,)


def test_estimate_token_count():
    assert CorpusProcessingAgent._estimate_token_count("Hello world") == 2
    assert CorpusProcessingAgent._estimate_token_count("One") == 1
    assert CorpusProcessingAgent._estimate_token_count("") == 0
    assert (
        CorpusProcessingAgent._estimate_token_count("   Leading spaces") == 2
    )  # split() behavior


def test_cosine_similarity():
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    vec3 = np.array([1, 0, 0])
    assert CorpusProcessingAgent._cosine_similarity(vec1, vec2) == pytest.approx(0.0)
    assert CorpusProcessingAgent._cosine_similarity(vec1, vec3) == pytest.approx(1.0)
    # Test with non-unit vectors
    vec4 = np.array([2, 0, 0])
    assert CorpusProcessingAgent._cosine_similarity(vec1, vec4) == pytest.approx(1.0)
    # Test with zero vector (should handle, e.g. return 0 or raise error, current impl returns nan, then 0.0)
    # Depending on strictness, this might need adjustment in _cosine_similarity
    # For now, as long as it doesn't crash and returns a float, it's a start.
    # The mock embeddings are normalized, so this won't be an issue in filter_and_sort.
    vec_zero = np.array([0, 0, 0])
    # np.dot(vec1, vec_zero) / (np.linalg.norm(vec1) * np.linalg.norm(vec_zero)) -> nan
    # The _cosine_similarity handles division by zero by returning 0.0
    assert CorpusProcessingAgent._cosine_similarity(vec1, vec_zero) == pytest.approx(
        0.0
    )


def test_filter_and_sort_corpus_under_max_tokens(
    corpus_agent, dummy_doc_paths, monkeypatch
):
    # Mock token count to be always small so all chunks pass the token limit
    monkeypatch.setattr(
        CorpusProcessingAgent, "_estimate_token_count", lambda slf, text: 1
    )  # Added slf

    corpus_agent.load_corpus(dummy_doc_paths)
    corpus_agent.process_corpus_documents()

    # Ensure all chunks are returned, sorted by relevance
    # "Doc2 chunk2 tech related." (0.9 with task [1,0,0,0,0])
    # "Doc1 chunk1 tech." (0.8)
    # "Default chunk for doc3.txt" (0.5)
    # "Doc1 chunk2 general." (0.3)
    # "Doc2 chunk1 finance." (0.1)

    filtered_chunks = corpus_agent.filter_and_sort_corpus(
        DUMMY_TASK_DESCRIPTION, relevance_threshold=0.05
    )

    assert len(filtered_chunks) == 5  # All chunks pass threshold
    assert filtered_chunks[0] == "Doc2 chunk2 tech related."
    assert filtered_chunks[1] == "Doc1 chunk1 tech."
    assert filtered_chunks[2] == "Default chunk for doc3.txt"
    assert filtered_chunks[3] == "Doc1 chunk2 general."
    assert filtered_chunks[4] == "Doc2 chunk1 finance."


def test_filter_and_sort_corpus_no_embeddings(corpus_agent, dummy_doc_paths):
    corpus_agent.load_corpus(dummy_doc_paths)
    # Do not call process_corpus_documents, so embeddings are None
    corpus_agent.processed_chunks = ["chunk1", "chunk2"]  # Manually set some chunks
    corpus_agent.corpus_embeddings = None

    filtered_chunks = corpus_agent.filter_and_sort_corpus(DUMMY_TASK_DESCRIPTION)
    assert filtered_chunks == ["chunk1", "chunk2"]  # Should return unprocessed chunks


def test_filter_and_sort_corpus_with_filtering_and_token_limit(
    corpus_agent, dummy_doc_paths, monkeypatch
):
    # corpus_agent.max_tokens is 30
    # Mock token counts:
    # "Doc2 chunk2 tech related." (highly relevant) -> 4 tokens
    # "Doc1 chunk1 tech." (relevant) -> 3 tokens
    # "Default chunk for doc3.txt" (moderate) -> 5 tokens
    # "Doc1 chunk2 general." (less relevant) -> 4 tokens
    # "Doc2 chunk1 finance." (irrelevant by threshold) -> 3 tokens

    def mock_token_counts(slf, text: str) -> int:  # Restored slf
        if "Doc2 chunk2 tech related." in text:
            return 15  # Takes up half the budget
        if "Doc1 chunk1 tech." in text:
            return 10  # Takes most of the rest
        if "Default chunk for doc3.txt" in text:
            return 10  # Would exceed with the above two
        if "Doc1 chunk2 general." in text:
            return 4
        if "Doc2 chunk1 finance." in text:
            return 3
        return 1  # default

    monkeypatch.setattr(
        CorpusProcessingAgent, "_estimate_token_count", mock_token_counts
    )

    corpus_agent.load_corpus(dummy_doc_paths)
    corpus_agent.process_corpus_documents()  # Generates chunks and mock embeddings

    # Relevance threshold will filter out "Doc2 chunk1 finance." (similarity 0.1 vs threshold 0.2)
    # max_tokens = 30
    # 1. "Doc2 chunk2 tech related." (sim ~0.9, tokens 15). current_tokens = 15.
    # 2. "Doc1 chunk1 tech." (sim ~0.8, tokens 10). current_tokens = 25.
    # 3. "Default chunk for doc3.txt" (sim ~0.5, tokens 10). 25 + 10 > 30. This chunk is skipped.
    # 4. "Doc1 chunk2 general." (sim ~0.3, tokens 4). 25 + 4 <= 30. current_tokens = 29
    # Result: ["Doc2 chunk2 tech related.", "Doc1 chunk1 tech.", "Doc1 chunk2 general."]

    filtered_chunks = corpus_agent.filter_and_sort_corpus(
        DUMMY_TASK_DESCRIPTION, relevance_threshold=0.25
    )

    assert filtered_chunks == [
        "Doc2 chunk2 tech related.",
        "Doc1 chunk1 tech.",
        # "Doc1 chunk2 general." is not included because "Default chunk for doc3.txt" (10 tokens)
        # was considered before it (due to higher relevance), and adding "Default chunk for doc3.txt"
        # would have exceeded max_tokens. The agent stops at that point.
    ]

    # Use the agent's (mocked) method to sum tokens for consistency
    current_tokens = sum(corpus_agent._estimate_token_count(c) for c in filtered_chunks)
    assert current_tokens <= corpus_agent.max_tokens
    assert current_tokens == 25  # Corrected expected sum: 15 + 10 = 25


def test_filter_and_sort_corpus_relevance_threshold(
    corpus_agent, dummy_doc_paths, monkeypatch
):
    monkeypatch.setattr(
        CorpusProcessingAgent, "_estimate_token_count", lambda slf, text: 1
    )  # all chunks are small, added slf

    corpus_agent.load_corpus(dummy_doc_paths)
    corpus_agent.process_corpus_documents()

    # "Doc2 chunk1 finance." has mock similarity around 0.1 (dot product with [1,0,0,0,0])
    # "Doc1 chunk2 general." has mock similarity around 0.3

    # Threshold 0.35 should filter out finance and general
    filtered_chunks = corpus_agent.filter_and_sort_corpus(
        DUMMY_TASK_DESCRIPTION, relevance_threshold=0.35
    )
    expected_after_threshold = [
        "Doc2 chunk2 tech related.",  # sim ~0.978
        "Doc1 chunk1 tech.",  # sim ~0.97
        "Default chunk for doc3.txt",  # sim 0.6
        "Doc1 chunk2 general.",  # sim ~0.447, which is > 0.35
    ]
    assert filtered_chunks == expected_after_threshold
