from .corpus_provider import CorpusProvider
import nltk
from nltk.corpus import brown

"""
If you have not downloaded the brown corpus you are going to have to run this first
import nltk

nltk.download('brown')
nltk.download('punkt')
"""


class BrownCorpusProvider(CorpusProvider):
    def get_name(self):
        return 'nltk_brown'

    def get_words(self):
        # Retrieve words from the Brown corpus
        raw_words = brown.words()

        # Filter out non-words and convert to lowercase
        clean_words = [w.lower() for w in raw_words if w.isalpha()]

        # Generate a frequency distribution of the cleaned words
        freq_dist = nltk.FreqDist(clean_words)

        # Get unique words as a list
        unique_words = list(freq_dist.keys())

        return unique_words
