import os
import logging
from tqdm import tqdm
from embeddings_explorer.corpus.corpus_provider import CorpusProvider
from embeddings_explorer.graph.graph_constructor import GraphConstructor
from embeddings_explorer.graph.traverser import Traverser
from embeddings_explorer.utils.cache import EmbeddingCache
from embeddings_explorer.models.generator import Generator


class EmbeddingsExplorer:
    def __init__(self, corpus_provider, embedding_generator, graph_constructor, traverser, cache_dir):
        """
        Initializes the EmbeddingsExplorer with the necessary components.

        Parameters:
        - corpus_provider: An instance of a corpus provider.
        - embedding_generator: An instance of an embedding generator.
        - graph_constructor: An instance of a graph constructor.
        - traverser: An instance of a traverser.
        - cache_dir: The directory where the embedding cache will be saved.
        """
        self.corpus_provider: CorpusProvider = corpus_provider
        self.embedding_generator: Generator = embedding_generator
        self.graph_constructor: GraphConstructor = graph_constructor
        self.traverser: Traverser = traverser
        self.cache_dir = cache_dir
        self.cache = EmbeddingCache()
        self._load_cache()

    def _cache_file_name(self):
        """
        Generates a cache file name based on the generator's name.
        """
        model_name = self.embedding_generator.get_name()
        return f"{model_name}_embeddings_cache.pkl"

    def _load_cache(self):
        """
        Attempts to load the embedding cache from disk.
        """
        cache_path = os.path.join(self.cache_dir, self._cache_file_name())
        if os.path.exists(cache_path):
            logging.info("Loading embeddings cache from disk...")
            self.cache.load_cache(cache_path)
        else:
            logging.info("No existing cache found. Starting fresh...")

    def _save_cache(self):
        """
        Saves the embedding cache to disk, naming the file after the generator.
        """
        cache_path = os.path.join(self.cache_dir, self._cache_file_name())
        logging.info("Saving embeddings cache to disk...")
        self.cache.save_cache(cache_path)

    def generate_embeddings(self, words):
        """
        Generates embeddings for the provided words, utilizing cache.
        """
        embeddings = {}
        for word in tqdm(words, desc='Generating Embeddings'):
            embeddings[word] = self.cache.get_embedding(
                self.embedding_generator, word)
        self._save_cache()  # Save cache after embeddings generation
        return embeddings

    def explore(self, start_node, end_node):
        """
        Orchestrates the exploration workflow.

        Parameters:
        - start_node: The starting node for traversal.
        - end_node: The ending node for traversal.
        """
        logging.info("Initializing the model...")
        self.embedding_generator.initialize_model()

        logging.info("Loading the corpus...")
        words = self.corpus_provider.get_words()

        logging.info("Generating embeddings...")
        embeddings = self.generate_embeddings(words)

        logging.info("Constructing the graph...")
        G = self.graph_constructor.construct_graph(embeddings)

        logging.info("Traversing the graph...")
        path, total_distance = self.traverser.traverse(
            G, start_node, end_node)
        if path:
            logging.info(f"Path from {start_node} to {end_node}: {path}")
            logging.info(f"Total distance traveled: {total_distance}")
        else:
            logging.warning(
                f"Failed to find a path from '{start_node}' to '{end_node}'.")
