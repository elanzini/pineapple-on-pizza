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
    def compute_embeddings(self, input_string):
        """
        Compute embeddings for a given input string.

        Parameters:
        input_string (str): The input string for which embeddings are to be computed.

        Returns:
        The computed embeddings.
        """
        pass
