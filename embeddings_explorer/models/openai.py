from embeddings_explorer.models.generator import Generator
import openai
from dotenv import load_dotenv
import os

# Load the API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIEmbeddingGenerator(Generator):
    def __init__(self, cache, model_size="small"):
        super().__init__(cache)
        # Map the model_size parameter to actual OpenAI model identifiers
        self.model = self._model_identifier(model_size)

    def _model_identifier(self, model_size):
        """
        Returns the OpenAI model identifier based on the specified size.

        Parameters:
        - model_size (str): The size of the model ("small" or "large").

        Returns:
        - str: The OpenAI model identifier.
        """
        model_identifiers = {
            "small": "text-embedding-3-small",
            "large": "text-embedding-3-large"
        }
        # Default to "small"
        return model_identifiers.get(model_size, "text-embedding-3-small")

    def initialize_model(self):
        # For the OpenAI API, initialization might not be necessary,
        # but you could set up any required configuration here.
        pass

    def compute_embeddings(self, input_string):
        # Check cache first
        cached_embedding = self.cache.get_embedding(self, input_string)
        if cached_embedding is not None:
            return cached_embedding

        # Call the OpenAI embeddings endpoint
        response = openai.Embedding.create(
            input=input_string,
            model=self.model
        )
        embedding = response["data"][0]["embedding"]

        # Cache the newly fetched embedding
        self.cache.cache[(self.get_name(), input_string)] = embedding
        return embedding

    def get_name(self):
        # Return a unique name for this generator
        return f"openai_text_embedding_3_{self.model}"
