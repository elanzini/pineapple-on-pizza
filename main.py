import pandas as pd
import logging
from embeddings_explorer.corpus.food_corpus_provider import FoodCorpusProvider, Language
from embeddings_explorer.models.openai import OpenAIGenerator
from embeddings_explorer.models.voyage import VoyageGenerator
from embeddings_explorer.graph.knn_graph import KnnGraphConstructor
from embeddings_explorer.graph.weighted_traverser import WeightedTraverser
from embeddings_explorer.orchestrator.explorer import EmbeddingsExplorer


def run_experiment(model_size, k, language, start_node, end_node):
    corpus_provider = FoodCorpusProvider(language=language)
    embedding_generator = VoyageGenerator(model_size=model_size)
    graph_constructor = KnnGraphConstructor(
        k=k, metric='cosine', weighted=True)
    traverser = WeightedTraverser()

    explorer = EmbeddingsExplorer(
        corpus_provider, embedding_generator, graph_constructor, traverser, "/tmp/embeddings_cache/")
    path, total_distance = explorer.explore(
        start_node=start_node, end_node=end_node)

    result = {
        'distance': total_distance,
        'path': path
    }
    return result


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    languages = [Language.EN, Language.IT]
    model_sizes = ['small', 'large']
    knn_values = range(2, 11)

    results = []

    for language in languages:
        start_node = 'Pizza'
        end_node = 'Ananas' if language == Language.IT else 'Pineapple'

        for model_size in model_sizes:
            for k in knn_values:
                logging.info(
                    f"Running experiment with model_size={model_size}, k={k}, language={language.name}")
                result = run_experiment(
                    model_size, k, language, start_node, end_node)
                # Assuming result contains distance and path, adjust as necessary
                results.append({
                    'language': language.name,
                    'model_size': model_size,
                    'k': k,
                    # Adjust based on actual result structure
                    'distance': result['distance'],
                    # Adjust based on actual result structure
                    'path': ' -> '.join(result['path'])
                })

    df = pd.DataFrame(results)
    # Save to disk
    df.to_csv('./voyage_results.csv', index=False)
    logging.info("Results saved")


if __name__ == "__main__":
    main()
