import pickle


class EmbeddingCache:
    def __init__(self):
        self.cache = {}

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
        # Model name or identifier to distinguish embeddings from different models
        model_name = model.__class__.__name__

        # Check if the embedding is already in the cache
        if (model_name, word) in self.cache:
            return self.cache[(model_name, word)]
        else:
            # Generate the embedding and cache it
            embedding = model.compute_embeddings(word)
            self.cache[(model_name, word)] = embedding
            return embedding

    def save_cache(self, filepath):
        """
        Save the cache to a file for persistent storage.

        Parameters:
        - filepath: The path to the file where the cache should be saved.
        """
        # Implement saving logic, e.g., using pickle or json
        with open(filepath, 'wb') as f:
            pickle.dump(self.cache, f)

    def load_cache(self, filepath):
        """
        Load the cache from a file.

        Parameters:
        - filepath: The path to the file from which to load the cache.
        """
        # Implement loading logic, e.g., using pickle or json
        with open(filepath, 'rb') as f:
            self.cache = pickle.load(f)
