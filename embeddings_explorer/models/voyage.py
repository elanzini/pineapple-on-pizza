from embeddings_explorer.models.generator import Generator
import voyageai
from dotenv import load_dotenv
import os

# Load the API key from the environment
load_dotenv()
voyage_api_key = os.getenv("VOYAGE_API_KEY")


class VoyageGenerator(Generator):
    def __init__(self, model_size="small"):
        self.client = voyageai.Client(api_key=voyage_api_key)
        self.model = self._model_identifier(model_size)

    def _model_identifier(self, model_size):
        model_identifiers = {
            "small": "voyage-2",
            "large": "voyage-large-2"
        }
        return model_identifiers.get(model_size, "voyage-2")

    def initialize_model(self):
        # Initialization for Voyage API client if needed
        pass

    def compute_embeddings(self, words):
        embeddings = {}
        # Process in chunks of 128 texts to comply with API limits
        # Docs: https://docs.voyageai.com/docs/embeddings
        for i in range(0, len(words), 128):
            chunk = words[i:i+128]
            result = self.client.embed(chunk, model=self.model)
            # Map each word in the chunk to its corresponding embedding
            for word, embedding in zip(chunk, result.embeddings):
                embeddings[word] = embedding
        return embeddings

    def get_name(self):
        return f"voyage_{self.model}"
