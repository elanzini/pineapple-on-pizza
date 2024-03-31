from embeddings_explorer.models.generator import Generator
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
from tqdm import tqdm
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIGenerator(Generator):
    def __init__(self, model_size="small"):
        self.model = self._model_identifier(model_size)

    def get_name(self):
        return f"openai_{self.model}"

    def _model_identifier(self, model_size):
        model_identifiers = {
            "small": "text-embedding-3-small",
            "large": "text-embedding-3-large"
        }
        return model_identifiers.get(model_size, "text-embedding-3-small")

    def initialize_model(self):
        pass

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def compute_embedding(self, input_string):
        response = openai.embeddings.create(
            input=input_string,
            model=self.model
        )
        return response.data[0].embedding

    def compute_embeddings(self, words):
        embeddings = {}
        for word in tqdm(words, desc='Generating Embeddings'):
            embeddings[word] = self.compute_embedding(word)
        return embeddings
