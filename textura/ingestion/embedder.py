from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    """
    Generates text embeddings using SentenceTransformer models.
    """
    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5'):
        """
        Initializes the Embedder.

        Args:
            model_name: The name of the SentenceTransformer model to use.
                        Defaults to 'BAAI/bge-small-en-v1.5'.
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error initializing SentenceTransformer model '{model_name}': {e}")
            print("Please ensure the model name is correct and you have an internet connection "
                  "if the model needs to be downloaded.")
            # Fallback or raise error, for now, let it raise if model cannot be loaded
            raise

    def generate_embeddings(self, text_chunks: List[str]) -> List[np.ndarray]:
        """
        Generates embeddings for a list of text chunks.

        Args:
            text_chunks: A list of strings, where each string is a text chunk.

        Returns:
            A list of numpy arrays, where each array is the embedding for a chunk.
            Returns an empty list if an error occurs or input is empty.
        """
        if not text_chunks:
            return []

        try:
            embeddings = self.model.encode(text_chunks, convert_to_numpy=True, show_progress_bar=False)
            return [emb for emb in embeddings] # Ensure it's a list of arrays
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []

if __name__ == '__main__':
    # Example Usage
    # This will download the model if you don't have it cached.
    try:
        embedder = Embedder()
        sample_chunks = [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?"
        ]

        print(f"Generating embeddings for {len(sample_chunks)} chunks...")
        embeddings_list = embedder.generate_embeddings(sample_chunks)

        if embeddings_list:
            print(f"Successfully generated {len(embeddings_list)} embeddings.")
            for i, emb in enumerate(embeddings_list):
                print(f"Embedding {i+1} (shape: {emb.shape}):\n{emb[:5]}...\n---") # Print first 5 elements

            # Example: Check similarity (optional)
            # from sentence_transformers.util import cos_sim
            # if len(embeddings_list) >= 2:
            #     similarity = cos_sim(embeddings_list[0], embeddings_list[3])
            #     print(f"Similarity between first and fourth chunk: {similarity.item()}")

    except Exception as e:
        print(f"An error occurred during the Embedder example: {e}")
        print("This might be due to model download issues or other runtime problems.")
