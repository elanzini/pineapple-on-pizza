from embeddings_explorer.corpus.brown_corpus_provider import BrownCorpusProvider
from embeddings_explorer.models.sentence_bert import SentenceBertGenerator
from embeddings_explorer.graph.knn_graph import KnnGraphConstructor
from embeddings_explorer.graph.bfs_traverser import BfsTraverser


def main():
    # Initialize your corpus provider and embedding generator
    corpus_provider = BrownCorpusProvider()
    embedding_generator = SentenceBertGenerator()

    # Initialize KNN Graph Constructor
    graph_constructor = KnnGraphConstructor(k=5)

    # Load the corpus and model
    words = corpus_provider.get_words()[:100]  # Limit for quick demonstration
    embedding_generator.initialize_model()
    embeddings = {word: embedding_generator.compute_embeddings(
        word) for word in words}

    # Construct the graph
    G = graph_constructor.construct_graph(embeddings)
    print(
        f"Constructed a graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")

    traverser = BfsTraverser()
    start_node = 'have'
    end_node = 'court'
    path = traverser.traverse(G, start_node, end_node)
    print(f"Path from {start_node} to {end_node}: {path}")


if __name__ == "__main__":
    main()
