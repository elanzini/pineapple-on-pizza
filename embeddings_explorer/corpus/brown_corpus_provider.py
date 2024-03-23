from .corpus_provider import CorpusProvider
import nltk
from nltk.corpus import brown


class BrownCorpusProvider(CorpusProvider):
    def get_words(self):
        # Ensure necessary NLTK data is available
        nltk.download('brown')
        nltk.download('punkt')

        # Retrieve words from the Brown corpus
        raw_words = brown.words()

        # Filter out non-words and convert to lowercase
        clean_words = [w.lower() for w in raw_words if w.isalpha()]

        # Generate a frequency distribution of the cleaned words
        freq_dist = nltk.FreqDist(clean_words)

        # Get unique words as a list
        unique_words = list(freq_dist.keys())

        return unique_words[:100]
