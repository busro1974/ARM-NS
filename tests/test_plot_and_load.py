import os
import sys
import subprocess


def test_load_from_file():
    from src.rdf_graph import RDFGraph

    train_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_dataset', 'train.txt'))
    g = RDFGraph()
    loaded = g.load_from_file(train_path)
    assert loaded > 0
    assert g.get_triple_count() == loaded
    preds = g.get_predicates()
    assert 'buys' in preds


def test_plot_training_smoke(tmp_path):
    script = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'plot_training.py'))
    out_dir = tmp_path / 'out'
    out_dir_str = str(out_dir)

    cmd = [sys.executable, script, '--epochs', '1', '--out-dir', out_dir_str]
    res = subprocess.run(cmd, capture_output=True, text=True)

    # Print output to help debugging if the test fails
    print(res.stdout)
    print(res.stderr, file=sys.stderr)

    assert res.returncode == 0, f"Plot script failed: {res.stderr}"

    expected_files = ['training_loss.png', 'confidence_histogram.png', 'triples_per_predicate.png']
    for fname in expected_files:
        path = out_dir / fname
        assert path.exists(), f"Expected output not found: {path}"
