import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss

class FAISSVectorStore:
    """
    Manages a FAISS index for vector storage and retrieval, along with
    associated document metadata.
    """
    def __init__(self, workspace_path: str, index_subdirectory: str = "index"):
        self.workspace_path = Path(workspace_path)
        self.index_dir = self.workspace_path / index_subdirectory
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = self.index_dir / "faiss.index"
        self.docs_file = self.index_dir / "docs.jsonl"

        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict[str, Any]] = [] # Stores metadata and original text
        self._next_doc_id: int = 0

        self._load_if_exists()

    def _load_if_exists(self):
        """Loads the index and documents if they exist on disk."""
        if self.index_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                print(f"FAISS index loaded from {self.index_file}. Index size: {self.index.ntotal if self.index else 0}")
            except Exception as e:
                print(f"Error loading FAISS index: {e}. A new index will be created if data is added.")
                self.index = None # Ensure index is None if loading failed

        if self.docs_file.exists():
            try:
                with open(self.docs_file, 'r') as f:
                    for line in f:
                        self.documents.append(json.loads(line))
                self._next_doc_id = len(self.documents)
                print(f"Document metadata loaded from {self.docs_file}. Number of documents: {len(self.documents)}")
            except Exception as e:
                print(f"Error loading document metadata: {e}. Document store will be considered empty.")
                self.documents = []
                self._next_doc_id = 0

        if self.index is not None and self.index.ntotal != len(self.documents):
            print(f"Warning: FAISS index size ({self.index.ntotal if self.index else 0}) "
                  f"does not match document store size ({len(self.documents)}). "
                  "This might indicate corruption or an interrupted save. Consider re-indexing.")
            # Decide on a recovery strategy: e.g., rebuild index, clear documents, or flag as corrupted.
            # For now, we'll allow it but a more robust solution might be needed.

    def add_embeddings(self, text_chunks: List[str], embeddings: List[np.ndarray], source_file: str):
        """
        Adds embeddings and their corresponding text chunks to the store.

        Args:
            text_chunks: List of original text chunks.
            embeddings: List of numpy array embeddings for the chunks.
            source_file: The name of the file from which these chunks originated.
        """
        if not embeddings or len(text_chunks) != len(embeddings):
            print("Error: Embeddings list is empty or does not match text_chunks length.")
            return

        embeddings_np = np.array(embeddings, dtype=np.float32)
        if embeddings_np.ndim == 1: # Single embedding
             embeddings_np = np.expand_dims(embeddings_np, axis=0)

        if self.index is None:
            dimension = embeddings_np.shape[1]
            # Using IndexFlatL2 as a basic, widely compatible index type.
            # For larger datasets, more complex factories like "IDMap,Flat" or "IDMap,IVFxxx,Flat" might be better.
            self.index = faiss.IndexFlatL2(dimension)
            print(f"Created new FAISS index with dimension {dimension}.")

        self.index.add(embeddings_np)

        for i, text_chunk in enumerate(text_chunks):
            doc_metadata = {
                "id": self._next_doc_id,
                "text": text_chunk,
                "source_file": source_file,
                "embedding_index": self.index.ntotal - len(embeddings) + i # FAISS index of this chunk
            }
            self.documents.append(doc_metadata)
            self._next_doc_id += 1

        print(f"Added {len(embeddings)} embeddings. Index size: {self.index.ntotal}. Documents count: {len(self.documents)}")


    def save(self):
        """Saves the FAISS index and document metadata to disk."""
        if self.index is not None:
            try:
                faiss.write_index(self.index, str(self.index_file))
                print(f"FAISS index saved to {self.index_file}")
            except Exception as e:
                print(f"Error saving FAISS index: {e}")

        try:
            with open(self.docs_file, 'w') as f:
                for doc in self.documents:
                    f.write(json.dumps(doc) + '\n')
            print(f"Document metadata saved to {self.docs_file}")
        except Exception as e:
            print(f"Error saving document metadata: {e}")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the index for the k nearest neighbors to the query_embedding.

        Args:
            query_embedding: A numpy array representing the query embedding.
            k: The number of nearest neighbors to retrieve.

        Returns:
            A list of document metadata for the nearest neighbors.
        """
        if self.index is None or self.index.ntotal == 0:
            print("Index is empty or not initialized.")
            return []

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        distances, indices = self.index.search(query_embedding.astype(np.float32), k)

        results = []
        for i in range(indices.shape[1]):
            idx = indices[0][i]
            if 0 <= idx < len(self.documents):
                # Find the document corresponding to the FAISS index ID.
                # This assumes FAISS index IDs correspond to the order in self.documents
                # IF using IDMap, this lookup would be different.
                # For IndexFlatL2, the indices returned are direct row numbers.
                doc = next((d for d in self.documents if d.get("embedding_index") == idx), None)
                if doc:
                    results.append({**doc, "distance": float(distances[0][i])})
                else:
                    # This case should ideally not happen if using IndexFlatL2 and documents are added sequentially
                    print(f"Warning: No document found for FAISS index {idx}")
            else:
                print(f"Warning: Invalid index {idx} from FAISS search results.")
        return results

if __name__ == '__main__':
    # Example Usage
    test_workspace_vs = Path("_test_workspace_vs")
    test_workspace_vs.mkdir(exist_ok=True)

    # Clean up previous test files if they exist
    if (test_workspace_vs / "index" / "faiss.index").exists():
        (test_workspace_vs / "index" / "faiss.index").unlink()
    if (test_workspace_vs / "index" / "docs.jsonl").exists():
        (test_workspace_vs / "index" / "docs.jsonl").unlink()

    vector_store = FAISSVectorStore(workspace_path=str(test_workspace_vs))

    # Simulate some embeddings (e.g., from the Embedder)
    # In a real scenario, these would come from the Embedder class
    try:
        from sentence_transformers import SentenceTransformer # For example
        model = SentenceTransformer('BAAI/bge-small-en-v1.5') # ensure this is installed

        sample_texts = ["Hello world", "FAISS is cool", "Another document here"]
        sample_embeddings = model.encode(sample_texts)

        vector_store.add_embeddings(sample_texts, list(sample_embeddings), source_file="example.txt")
        vector_store.save()

        # Test loading
        print("\n--- Reloading Vector Store ---")
        vector_store_loaded = FAISSVectorStore(workspace_path=str(test_workspace_vs))
        print(f"Loaded index size: {vector_store_loaded.index.ntotal if vector_store_loaded.index else 0}")
        print(f"Loaded documents count: {len(vector_store_loaded.documents)}")

        if vector_store_loaded.index and vector_store_loaded.index.ntotal > 0:
            print("\n--- Searching ---")
            query_text = "A document about FAISS"
            query_embedding = model.encode([query_text])[0]

            search_results = vector_store_loaded.search(query_embedding, k=2)
            print(f"Search results for '{query_text}':")
            for res in search_results:
                print(f"  ID: {res['id']}, Source: {res['source_file']}, Text: '{res['text']}', Distance: {res['distance']:.4f}")
        else:
            print("Skipping search test as index is empty after loading.")

    except ImportError:
        print("SentenceTransformer not installed. Skipping FAISSVectorStore example with live embeddings.")
    except Exception as e:
        print(f"An error occurred in FAISSVectorStore example: {e}")

    # Clean up
    # import shutil
    # shutil.rmtree(test_workspace_vs)
