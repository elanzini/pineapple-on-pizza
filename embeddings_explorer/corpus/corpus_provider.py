from abc import ABC, abstractmethod


class CorpusProvider(ABC):
    @abstractmethod
    def get_name(self):
        """
        Return a unique name or identifier for the corpus.

        Returns:
        str: The unique name of the corpus.
        """
        pass

    @abstractmethod
    def get_words(self):
        """
        Retrieves a list of unique words from a corpus.

        Returns:
            list: A list of unique words.
        """
        pass
