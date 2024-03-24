from .generator import Generator
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import torch


class BertGenerator(Generator):
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def get_name(self):
        return "bert"

    def initialize_model(self):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model.eval()  # Set the model to evaluation mode

    def compute_embedding(self, input_string):
        inputs = self.tokenizer(input_string, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract the embeddings for the [CLS] token
        return outputs.last_hidden_state[:, 0, :][0]

    def compute_embeddings(self, words):
        embeddings = {}
        for word in tqdm(words, desc='Generating Embeddings'):
            embeddings[word] = self.compute_embedding(word)
        return embeddings
