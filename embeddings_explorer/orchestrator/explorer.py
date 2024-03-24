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
        self.cache = EmbeddingCache(
            cache_dir, self.embedding_generator.get_name(), self.corpus_provider.get_name())
        self.cache_available = self.cache.load_cache()

    def generate_embeddings(self, words):
        if self.cache_available:
            return self.cache.get_embeddings()

        embeddings = self.embedding_generator.compute_embeddings(words)

        self.cache.save_cache(embeddings)
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
            return path, total_distance
        else:
            logging.warning(
                f"Failed to find a path from '{start_node}' to '{end_node}'.")
