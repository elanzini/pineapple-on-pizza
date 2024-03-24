import os
import pickle
import logging


class EmbeddingCache:
    def __init__(self, cache_dir: str, model_name: str, corpus_name: str):
        self.cache = {}
        self.cache_dir: str = cache_dir
        self.model_name: str = model_name
        self.corpus_name: str = corpus_name

    def get_embedding(self, model, word):
        """
        Retrieve the embedding for a word using a given model.
        If the embedding is not cached, generate it and cache it.

        Parameters:
        - model: The model to generate embeddings with.
        - word: The word to generate an embedding for.

        Returns:
        The embedding vector for the word.
        """
        # Check if the embedding is already in the cache
        if word in self.cache:
            return self.cache[word]
        else:
            # Generate the embedding and cache it
            embedding = model.compute_embeddings(word)
            self.cache[word] = embedding
            return embedding

    def _get_cache_path(self):
        cache_file_name = f"{self.model_name}_{self.corpus_name}_cache.pkl"
        cache_path = os.path.join(self.cache_dir, cache_file_name)
        return cache_path

    def save_cache(self):
        """
        Save the cache to a file for persistent storage.

        Parameters:
        - filepath: The path to the file where the cache should be saved.
        """
        # Implement saving logic, e.g., using pickle or json
        if self.cache_dir is None:
            return
        cache_path = self._get_cache_path()
        logging.info(f"Saving embeddings cache to {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(self.cache, f)

    def load_cache(self):
        """
        Load the cache from a file.

        Parameters:
        - filepath: The path to the file from which to load the cache.
        """
        # Implement loading logic, e.g., using pickle or json
        if self.cache_dir is None:
            return

        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            logging.info(f"Loading embeddings cache from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.cache = pickle.load(f)
        else:
            logging.info("No existing cache found. Starting fresh...")
