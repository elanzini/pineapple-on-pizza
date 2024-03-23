import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from embeddings_explorer.corpus.corpus_provider import CorpusProvider
from embeddings_explorer.graph.graph_constructor import GraphConstructor
from embeddings_explorer.graph.traverser import Traverser

from embeddings_explorer.models.generator import Generator


class EmbeddingsExplorer:
    def __init__(self, corpus_provider, embedding_generator, graph_constructor, traverser):
        """
        Initializes the EmbeddingsExplorer with the necessary components.

        Parameters:
        - corpus_provider: An instance of a corpus provider.
        - embedding_generator: An instance of an embedding generator.
        - graph_constructor: An instance of a graph constructor.
        - traverser: An instance of a traverser.
        """
        self.corpus_provider: CorpusProvider = corpus_provider
        self.embedding_generator: Generator = embedding_generator
        self.graph_constructor: GraphConstructor = graph_constructor
        self.traverser: Traverser = traverser

    def generate_embeddings(self, words):
        """
        Generates embeddings for the provided words in parallel.

        Parameters:
        - words: A list of words to generate embeddings for.

        Returns:
        A dictionary mapping words to their embeddings.
        """
        embeddings = {}
        for word in tqdm(words, desc='Generating Embeddings'):
            embeddings[word] = self.embedding_generator.compute_embeddings(
                word)
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

        logging.info("Generating embeddings in parallel...")
        embeddings = self.generate_embeddings(words)

        logging.info("Constructing the graph...")
        G = self.graph_constructor.construct_graph(embeddings)

        logging.info("Traversing the graph...")
        path = self.traverser.traverse(G, start_node, end_node)
        logging.info(f"Path from {start_node} to {end_node}: {path}")
