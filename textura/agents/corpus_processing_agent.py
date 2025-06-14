from __future__ import annotations

from pathlib import Path

import numpy as np  # Added for cosine similarity and array handling

from textura.ingestion.chunker import Chunker
from textura.ingestion.embedder import Embedder

# We'll need a vector store later, but let's not import it yet to keep things simple for now.
# from textura.ingestion.vector_store import FAISSVectorStore # Or MilvusVectorStoreWrapper


class CorpusProcessingAgent:
    """
    Agent responsible for processing a large corpus of documents,
    including embedding-based filtering and sorting when the corpus
    exceeds a certain token limit.
    """

    def __init__(self, chunker: Chunker, embedder: Embedder, max_tokens: int = 100000):
        """
        Initializes the CorpusProcessingAgent.

        Args:
            chunker: An instance of the Chunker to split documents.
            embedder: An instance of the Embedder to generate document embeddings.
            max_tokens: The maximum number of tokens the corpus should be reduced to
                        if it exceeds this limit.

        """
        self.chunker = chunker
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.corpus_documents: list[
            Path
        ] = []  # Stores paths to documents in the corpus
        self.processed_chunks: list[str] = []
        self.corpus_embeddings = None  # Will hold embeddings if generated

    def load_corpus(self, document_paths: list[str | Path]) -> None:
        """
        Loads the paths of the documents that form the corpus.

        Args:
            document_paths: A list of paths (strings or Path objects) to the documents.

        """
        self.corpus_documents = [Path(p) for p in document_paths]
        print(f"Corpus loaded with {len(self.corpus_documents)} documents.")

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        """Estimates token count by splitting by space. A simple heuristic."""
        return len(text.split())

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates cosine similarity between two numpy vectors."""
        if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
            return 0.0  # Or raise error
        if vec1.shape != vec2.shape:
            # This can happen if one of the embeddings is zero-dimensional or mismatched
            print(
                f"Warning: Shape mismatch in cosine similarity: {vec1.shape} vs {vec2.shape}"
            )
            return 0.0

        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0  # Avoid division by zero
        similarity = np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)
        return float(similarity)

    def process_corpus_documents(self) -> None:
        """
        Processes all documents in the loaded corpus:
        1. Chunks each document.
        2. Generates embeddings for all accumulated chunks.
        """
        all_chunks: list[str] = []
        if not self.corpus_documents:
            print("No documents loaded in the corpus to process.")
            return

        for i, doc_path in enumerate(self.corpus_documents):
            print(
                f"Chunking document {i + 1}/{len(self.corpus_documents)}: {doc_path.name}..."
            )
            try:
                chunks = self.chunker.chunk_file(doc_path)
                all_chunks.extend(chunks)
                print(f"Found {len(chunks)} chunks in {doc_path.name}.")
            except Exception as e:
                print(f"Error chunking document {doc_path.name}: {e}")

        self.processed_chunks = all_chunks

        if self.processed_chunks:
            print(
                f"Generating embeddings for {len(self.processed_chunks)} total chunks..."
            )
            try:
                # Ensure embedder is not None and generate_embeddings exists
                if hasattr(self.embedder, "generate_embeddings"):
                    self.corpus_embeddings = self.embedder.generate_embeddings(
                        self.processed_chunks
                    )
                    print("Embeddings generated successfully.")
                else:
                    print(
                        "Embedder or embedding generation method not available. Skipping embedding generation."
                    )
                    self.corpus_embeddings = None
            except Exception as e:
                print(f"Error generating embeddings: {e}")
                self.corpus_embeddings = None  # Ensure it's None on error
        else:
            print("No chunks were processed, skipping embedding generation.")
            self.corpus_embeddings = None

    def filter_and_sort_corpus(
        self, task_description: str, relevance_threshold: float = 0.7
    ) -> list[str]:
        """
        Filters and sorts the processed corpus chunks based on relevance to a task description.
        """
        print("\nStarting corpus filtering and sorting...")
        if not self.processed_chunks:
            print("No processed chunks available to filter and sort.")
            return []

        current_total_tokens = sum(
            self._estimate_token_count(chunk) for chunk in self.processed_chunks
        )
        print(f"Total estimated tokens before filtering: {current_total_tokens}")

        if self.corpus_embeddings is None or not hasattr(
            self.embedder, "generate_embeddings"
        ):
            print(
                "Corpus embeddings not available or embedder misconfigured. Returning all processed chunks."
            )
            if current_total_tokens > self.max_tokens:
                print(
                    f"Warning: Unfiltered chunks ({current_total_tokens} tokens) exceed max_tokens ({self.max_tokens})."
                )
            return self.processed_chunks

        if not self.corpus_embeddings:  # Handles case where list is empty but not None
            print("Corpus embeddings list is empty. Returning all processed chunks.")
            if current_total_tokens > self.max_tokens:
                print(
                    f"Warning: Unfiltered chunks ({current_total_tokens} tokens) exceed max_tokens ({self.max_tokens})."
                )
            return self.processed_chunks

        # Generate task embedding first, as it's needed for sorting even if no token filtering occurs.
        print(f"Generating embedding for task description: '{task_description}'")
        try:
            task_embedding_list = self.embedder.generate_embeddings([task_description])
            if not task_embedding_list or task_embedding_list[0] is None:
                print(
                    "Failed to generate task embedding. Returning unprocessed chunks (sorted by original order if applicable)."
                )
                # Sort self.processed_chunks by some default if desired, or return as is.
                # For now, returning as is, as similarity sort isn't possible.
                return self.processed_chunks
            task_embedding = np.array(task_embedding_list[0])
            print("Task embedding generated.")
        except Exception as e:
            print(
                f"Error generating task embedding: {e}. Returning unprocessed chunks (sorted by original order if applicable)."
            )
            return self.processed_chunks

        # Ensure corpus_embeddings is a list of numpy arrays
        corpus_embeddings_np = [np.array(emb) for emb in self.corpus_embeddings]

        similarities: list[tuple[str, float]] = []
        for chunk, embedding in zip(self.processed_chunks, corpus_embeddings_np):
            if (
                embedding is None or task_embedding.shape != embedding.shape
            ):  # Check shape compatibility
                similarity = (
                    0.0  # or some other default for failed/mismatched embeddings
                )
                print(
                    f"Warning: Skipping similarity calculation for a chunk due to embedding issue (shape: {embedding.shape if embedding is not None else 'None'})."
                )
            else:
                similarity = self._cosine_similarity(task_embedding, embedding)
            similarities.append((chunk, similarity))

        similarities.sort(key=lambda item: item[1], reverse=True)
        print("Chunks sorted by relevance to the task.")

        # If total tokens are already within limits, return all chunks sorted by relevance
        # AND filtered by the relevance threshold.
        if current_total_tokens <= self.max_tokens:
            print(
                f"Total tokens ({current_total_tokens}) are within the max limit ({self.max_tokens}). Returning chunks sorted by relevance and meeting threshold."
            )
            return [
                chunk for chunk, score in similarities if score >= relevance_threshold
            ]

        # Proceed with filtering by token limit and relevance threshold (already sorted by relevance)
        filtered_chunks: list[str] = []
        current_filtered_token_count = 0
        for chunk, score in similarities:
            if score >= relevance_threshold:
                chunk_token_count = self._estimate_token_count(chunk)
                if current_filtered_token_count + chunk_token_count <= self.max_tokens:
                    filtered_chunks.append(chunk)
                    current_filtered_token_count += chunk_token_count
                else:
                    print(
                        f"Reached max token limit ({self.max_tokens}). Stopping filtering."
                    )
                    break
            else:
                # Since chunks are sorted, once we hit a chunk below threshold, the rest will also be.
                print(
                    f"Chunks below relevance threshold ({relevance_threshold}). Stopping filtering."
                )
                break

        print(f"Filtered chunks count: {len(filtered_chunks)}")
        print(f"Total estimated tokens after filtering: {current_filtered_token_count}")
        return filtered_chunks

    def build_hierarchical_clusters(
        self,
        linkage: str = "average",
        distance_threshold: float | None = None,
        n_clusters: int | None = None,
    ) -> dict[int, list[int]]:
        """Cluster corpus embeddings hierarchically."""

        if not self.corpus_embeddings:
            raise ValueError("Corpus embeddings are not available for clustering")

        from .cluster_agent import ClusterAgent

        agent = ClusterAgent(list(self.corpus_embeddings))
        clusters = agent.hierarchical(
            linkage_method=linkage,
            distance_threshold=distance_threshold,
            n_clusters=n_clusters,
        )
        self.cluster_labels = agent.labels
        self.cluster_linkage = agent.linkage_matrix
        return clusters


if __name__ == "__main__":
    # Example Usage (very basic for now)
    # This requires Chunker and Embedder to be instantiable
    # and may require model downloads on first run for Embedder.
    try:
        print("Initializing Chunker...")
        default_chunker = Chunker()
        print("Chunker initialized.")

        print("Initializing Embedder...")
        default_embedder = Embedder()  # Uses default model 'BAAI/bge-small-en-v1.5'
        print("Embedder initialized.")

        print("Initializing CorpusProcessingAgent...")
        agent = CorpusProcessingAgent(
            chunker=default_chunker, embedder=default_embedder
        )
        print("CorpusProcessingAgent initialized.")

        # Create some dummy files for testing
        dummy_corpus_dir = Path("_test_corpus")
        dummy_corpus_dir.mkdir(exist_ok=True)
        (dummy_corpus_dir / "doc1.txt").write_text("This is the first test document.")
        (dummy_corpus_dir / "doc2.txt").write_text(
            "This is a second document for testing corpus mode."
        )
        (dummy_corpus_dir / "doc3.pdf").write_text(
            "This is a dummy PDF content (not a real PDF)."
        )  # Chunker will treat as .txt

        doc_paths = [
            dummy_corpus_dir / "doc1.txt",
            dummy_corpus_dir / "doc2.txt",
            dummy_corpus_dir
            / "doc3.pdf",  # Chunker will handle this as plain text due to its fallback
        ]

        agent.load_corpus(doc_paths)
        print(f"Agent has {len(agent.corpus_documents)} documents loaded.")

        # Conditional execution to avoid disk space errors from embedder
        can_run_embedding_dependent_code = False
        try:
            # Attempt to run the part that might fail due to environment (e.g. model download)
            # This is a proxy; actual embedding generation is in process_corpus_documents
            if default_embedder is not None:  # Check if embedder was initialized
                print("Embedder seems available, proceeding with corpus processing.")
                agent.process_corpus_documents()
                can_run_embedding_dependent_code = (
                    agent.corpus_embeddings is not None
                    and len(agent.corpus_embeddings) > 0
                )

            print(f"Agent processed {len(agent.processed_chunks)} chunks.")
            if agent.corpus_embeddings is not None:  # Check if list itself is None
                print(f"Agent generated {len(agent.corpus_embeddings)} embeddings.")
            else:  # This case handles if self.corpus_embeddings was set to None
                print("Agent did not generate embeddings (corpus_embeddings is None).")
                can_run_embedding_dependent_code = False  # Ensure it's false

        except Exception as e:
            print(f"Error during corpus processing (likely embedder-related): {e}")
            print("Skipping embedding-dependent parts of the example.")
            can_run_embedding_dependent_code = False

        if can_run_embedding_dependent_code:
            print(
                f"Agent generated {len(agent.corpus_embeddings)} embeddings for {len(agent.processed_chunks)} chunks."
            )
            task_desc = "A document about testing software and code."
            print(f"Filtering and sorting based on task: '{task_desc}'")
            filtered = agent.filter_and_sort_corpus(
                task_description=task_desc, relevance_threshold=0.1
            )  # Low threshold for dummy data
            print(f"\n--- Filtered Chunks ({len(filtered)}) ---")
            for i, chunk_text in enumerate(filtered):
                print(
                    f"Chunk {i + 1} (tokens: {CorpusProcessingAgent._estimate_token_count(chunk_text)}):\n{chunk_text}\n"
                )
        else:
            print(
                "\nSkipping filtering and sorting example as embeddings were not generated or an error occurred."
            )
            # Optionally, show processed chunks if they exist, even if not filtered
            if agent.processed_chunks:
                print(
                    f"\n--- Unfiltered Processed Chunks ({len(agent.processed_chunks)}) ---"
                )
                for i, chunk_text in enumerate(agent.processed_chunks):
                    print(
                        f"Chunk {i + 1} (tokens: {CorpusProcessingAgent._estimate_token_count(chunk_text)}):\n{chunk_text}\n"
                    )

        # Clean up dummy files
        # import shutil
        # shutil.rmtree(dummy_corpus_dir)
        print(
            "Dummy files created for example. Manual cleanup of '_test_corpus' directory may be needed."
        )

    except Exception as e:
        print(f"Error during CorpusProcessingAgent example: {e}")
        print(
            "This might be due to issues with initializing Chunker/Embedder, "
            "model downloads, or file operations."
        )
