import logging
from embeddings_explorer.graph.bfs_traverser import BfsTraverser
from embeddings_explorer.orchestrator.explorer import EmbeddingsExplorer
from embeddings_explorer.corpus.brown_corpus_provider import BrownCorpusProvider
from embeddings_explorer.corpus.food_corpus_provider import FoodCorpusProvider, Language
from embeddings_explorer.models.sentence_bert import SentenceBertGenerator
from embeddings_explorer.models.bert import BertGenerator
from embeddings_explorer.models.openai import OpenAIGenerator
from embeddings_explorer.graph.knn_graph import KnnGraphConstructor
from embeddings_explorer.graph.weighted_traverser import WeightedTraverser


def main():
    # Initialize logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Instantiate components
    corpus_provider = FoodCorpusProvider(language=Language.EN)
    embedding_generator = BertGenerator()
    graph_constructor = KnnGraphConstructor(
        k=5, metric='cosine', weighted=True)
    traverser = WeightedTraverser()

    # Initialize EmbeddingsExplorer
    explorer = EmbeddingsExplorer(
        corpus_provider, embedding_generator, graph_constructor, traverser,
        "/tmp/embeddings_cache/")

    # Start exploration
    explorer.explore(start_node='Pizza', end_node='Pineapple')


if __name__ == "__main__":
    main()
