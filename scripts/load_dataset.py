"""
Simple loader/demo script to load sample dataset and run a short training & prediction
"""
import os
import sys
# Ensure project root is on sys.path so `from src.*` works when running this script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rdf_graph import RDFGraph
from src.neural_lp import NeuralLPHybrid

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_dataset')


def load_triples_from_file(path: str, sep: str = '\t'):
    triples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(sep)
            if len(parts) >= 3:
                s, p, o = parts[0].strip(), parts[1].strip(), parts[2].strip()
                triples.append((s, p, o))
    return triples


if __name__ == '__main__':
    train_path = os.path.join(DATA_DIR, 'train.txt')
    valid_path = os.path.join(DATA_DIR, 'valid.txt')
    test_path = os.path.join(DATA_DIR, 'test.txt')

    graph = RDFGraph()
    # Load train triples
    train_triples = load_triples_from_file(train_path)
    graph.add_triples_batch(train_triples)

    print("Loaded graph stats:")
    print(graph.get_graph_statistics())

    # Initialize model and do a short training run
    model = NeuralLPHybrid(graph, embedding_dim=32, alpha=0.5)
    model.train(num_epochs=5, learning_rate=0.01, mine_rules=True)

    # Example prediction
    subj = 'customer_1'
    rel = 'buys'
    candidates = ['product_3', 'product_4', 'product_5', 'product_6']
    ranked = model.rank_candidates(subj, rel, candidates, top_k=3)

    print('\nTop predictions:')
    for i, pred in enumerate(ranked, 1):
        print(f"{i}. {pred.target} - confidence: {pred.confidence:.2%}")

    # Optionally evaluate on test set
    test_triples = load_triples_from_file(test_path)
    print(f"\nTest triples: {len(test_triples)}")
    for s, p, o in test_triples:
        pred = model.predict_link(s, p, o)
        print(f"Test: {s}-{p}-{o} -> confidence {pred.confidence:.2%}")
