from .generator import Generator
from sentence_transformers import SentenceTransformer


class SentenceBertGenerator(Generator):
    def __init__(self):
        self.model = None

    def get_name(self):
        return "sentence_bert"

    def initialize_model(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def compute_embeddings(self, input_string):
        return self.model.encode(input_string, show_progress_bar=False)
