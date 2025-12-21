import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from rdf_graph import RDFGraph


class TestRDFGraph(unittest.TestCase):
    def setUp(self):
        self.graph = RDFGraph()
        self.triples = [
            ('a', 'rel', 'b'),
            ('b', 'rel2', 'c'),
            ('a', 'rel3', 'd')
        ]
        for s, p, o in self.triples:
            self.graph.add_triple(s, p, o)

    def test_triple_count_and_entities(self):
        self.assertEqual(self.graph.get_triple_count(), 3)
        entities = self.graph.get_entities()
        self.assertTrue({'a', 'b', 'c', 'd'}.issubset(entities))

    def test_predicates_and_stats(self):
        preds = self.graph.get_predicates()
        self.assertTrue('rel' in preds and 'rel2' in preds)
        stats = self.graph.get_graph_statistics()
        self.assertIn('num_entities', stats)
        self.assertIn('num_triples', stats)

    def test_triple_exists_and_neighbors(self):
        self.assertTrue(self.graph.triple_exists('a', 'rel', 'b'))
        neigh = self.graph.get_neighbors('a')
        self.assertIn('b', neigh)


if __name__ == '__main__':
    unittest.main()
