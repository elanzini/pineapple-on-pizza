from .generator import Generator
from sentence_transformers import SentenceTransformer


class SentenceBertGenerator(Generator):
    def __init__(self):
        self.model = None

    def initialize_model(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def compute_embeddings(self, input_string):
        return self.model.encode(input_string)
