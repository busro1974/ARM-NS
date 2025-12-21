import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from rdf_graph import RDFGraph
from neural_lp import NeuralLPHybrid


class TestNeuralLPHybrid(unittest.TestCase):
    def setUp(self):
        self.graph = RDFGraph()
        triples = [
            ('c1', 'buys', 'p1'),
            ('c2', 'buys', 'p2'),
            ('p1', 'has_category', 'electronics')
        ]
        self.graph.add_triples_batch(triples)
        self.hybrid = NeuralLPHybrid(self.graph, embedding_dim=16, alpha=0.5)

    def test_training_and_stats(self):
        # train with minimal epochs for smoke test
        self.hybrid.train(num_epochs=1, learning_rate=0.01, mine_rules=False)
        stats = self.hybrid.get_statistics()
        self.assertIn('num_entities', stats)
        self.assertIn('embedding_dim', stats)

    def test_predict_and_rank(self):
        pred = self.hybrid.predict_link('c1', 'buys', 'p3')
        self.assertTrue(0.0 <= pred.confidence <= 1.0)
        ranked = self.hybrid.rank_candidates('c1', 'buys', ['p1', 'p2', 'p3'], top_k=2)
        self.assertLessEqual(len(ranked), 2)


if __name__ == '__main__':
    unittest.main()
