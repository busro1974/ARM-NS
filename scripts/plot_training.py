"""
Plot training loss and confidence distribution using the sample dataset
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for script
import matplotlib.pyplot as plt

# ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rdf_graph import RDFGraph
from src.neural_lp import NeuralLPHybrid

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_dataset')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)


def load_triples_from_file(path: str, sep: str = '\t'):
    triples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(sep)
            if len(parts) >= 3:
                triples.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return triples


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train & plot model metrics from sample dataset')
    parser.add_argument('--train-path', default=os.path.join(DATA_DIR, 'train.txt'), help='Path to train file')
    parser.add_argument('--test-path', default=os.path.join(DATA_DIR, 'test.txt'), help='Path to test file')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--embedding-dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--no-mine-rules', action='store_true', help='Disable rule mining step')
    parser.add_argument('--out-dir', default=OUT_DIR, help='Directory to write output images')

    args = parser.parse_args()

    train_path = args.train_path
    test_path = args.test_path
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    graph = RDFGraph()
    loaded = graph.load_from_file(train_path)
    print(f"Loaded {loaded} triples from {train_path}")

    model = NeuralLPHybrid(graph, embedding_dim=args.embedding_dim, alpha=0.5)

    print(f'Training model for {args.epochs} epochs...')
    model.train(num_epochs=args.epochs, learning_rate=0.01, mine_rules=not args.no_mine_rules)

    # Plot training loss
    loss = model.training_history.get('loss', [])
    epochs = list(range(1, len(loss)+1))

    plt.figure()
    plt.plot(epochs, loss, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    loss_png = os.path.join(OUT_DIR, 'training_loss.png')
    plt.savefig(loss_png)
    print(f"Saved training loss plot: {loss_png}")

    # Evaluate confidence distribution on test set
    test_triples = load_triples_from_file(test_path)
    confidences = []
    for s, p, o in test_triples:
        pred = model.predict_link(s, p, o)
        confidences.append(pred.confidence)

    if confidences:
        plt.figure()
        plt.hist(confidences, bins=10, range=(0,1))
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution on Test Set')
        hist_png = os.path.join(OUT_DIR, 'confidence_histogram.png')
        plt.savefig(hist_png)
        print(f"Saved confidence histogram: {hist_png}")

    # Bar chart: triples count per predicate
    pred_counts = {}
    for triple in graph.triples:
        pred_counts[triple.predicate] = pred_counts.get(triple.predicate, 0) + 1

    if pred_counts:
        plt.figure(figsize=(8,4))
        names = list(pred_counts.keys())
        counts = [pred_counts[n] for n in names]
        plt.bar(names, counts)
        plt.xlabel('Predicate')
        plt.ylabel('Triple Count')
        plt.title('Triples per Predicate (train set)')
        plt.xticks(rotation=45, ha='right')
        preds_png = os.path.join(OUT_DIR, 'triples_per_predicate.png')
        plt.tight_layout()
        plt.savefig(preds_png)
        print(f"Saved triples-per-predicate chart: {preds_png}")

    print('Plotting completed.')