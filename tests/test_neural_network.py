import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from rdf_graph import RDFGraph
from neural_network import RelationalGraphNeuralNetwork


class TestRelationalGraphNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.graph = RDFGraph()
        triples = [
            ('x', 'likes', 'y'),
            ('y', 'likes', 'z'),
            ('z', 'friend', 'x')
        ]
        self.graph.add_triples_batch(triples)
        self.model = RelationalGraphNeuralNetwork(self.graph, embedding_dim=16)

    def test_embeddings_initialized(self):
        # entity embeddings should exist
        entities = list(self.graph.get_entities())
        for e in entities:
            emb = self.model.get_entity_embedding(e)
            self.assertIsNotNone(emb)
            self.assertEqual(len(emb), 16)

    def test_training_and_prediction(self):
        # train for 1 epoch
        self.model.train_on_triples(num_epochs=1, learning_rate=0.01)
        prob = self.model.predict_link('x', 'likes', 'y')
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)


if __name__ == '__main__':
    unittest.main()
