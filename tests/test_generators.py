import unittest
from embeddings_explorer.models.sentence_bert import SentenceBertGenerator


class TestSentenceBertGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = SentenceBertGenerator()
        self.generator.initialize_model()

    def test_embeddings_shape(self):
        sentence = "This is a test."
        embedding = self.generator.compute_embeddings(sentence)
        # Assuming using default model
        self.assertEqual(embedding.shape[0], 768)


if __name__ == '__main__':
    unittest.main()
