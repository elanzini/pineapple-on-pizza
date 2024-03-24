from abc import ABC, abstractmethod


class Generator(ABC):
    @abstractmethod
    def get_name(self):
        """
        Return a unique name or identifier for the generator.

        Returns:
        str: The unique name of the generator.
        """
        pass

    @abstractmethod
    def initialize_model(self):
        """
        Initialize and load the model.
        """
        pass

    @abstractmethod
    def compute_embeddings(self, words):
        """
        Compute embeddings for a given input string.

        Parameters:
        words (str): Words to compute the embeddings for

        Returns:
        The computed embeddings in a dictionary where each key is the word
        and the value the embedding computed for the word.
        """
        pass
