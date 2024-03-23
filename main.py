from models.bert import BertGenerator
from corpus.brown_corpus_provider import BrownCorpusProvider


def main():
    # Initialize and use a corpus provider
    corpus_provider = BrownCorpusProvider()
    words = corpus_provider.get_words()
    print(f"Sample words from Brown Corpus: {words[:10]}")

    # Example of using a model generator (adjust as needed)
    bert_gen = BertGenerator()
    bert_gen.initialize_model()
    embeddings = bert_gen.compute_embeddings("This is a test sentence.")
    print("BERT Embeddings computed.")


if __name__ == "__main__":
    main()
