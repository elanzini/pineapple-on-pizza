import os
import pickle
import logging


class EmbeddingCache:
    def __init__(self, cache_dir: str, model_name: str, corpus_name: str):
        self.cache = {}
        self.cache_dir: str = cache_dir
        self.model_name: str = model_name
        self.corpus_name: str = corpus_name

    def get_embeddings(self):
        return self.cache

    def _get_cache_path(self):
        cache_file_name = f"{self.model_name}_{self.corpus_name}_cache.pkl"
        cache_path = os.path.join(self.cache_dir, cache_file_name)
        return cache_path

    def save_cache(self, embeddings):
        self.cache = embeddings
        if self.cache_dir is None:
            return
        cache_path = self._get_cache_path()
        logging.info(f"Saving embeddings cache to {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(self.cache, f)

    def load_cache(self):
        if self.cache_dir is None:
            return False

        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            logging.info(f"Loading embeddings cache from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.cache = pickle.load(f)
            return True
        else:
            logging.info("No existing cache found. Starting fresh...")
            return False
