# Neural-LP System

[![CI](https://github.com/busro1974/ARM-NS/actions/workflows/ci.yml/badge.svg)](https://github.com/busro1974/ARM-NS/actions/workflows/ci.yml)

Hybrid Neural Network + Rule Mining system for link prediction (TransE + rule mining).

## Quick start

1. Clone or open the repository and change to project root:

```powershell
cd C:\Users\USER\Documents\Neural
```

2. Create & activate a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -r dev-requirements.txt
```

3. Run demo (recommended):

```powershell
python run_demo.py
# or
python scripts/load_dataset.py
```

4. Generate plots (training loss, confidence histogram, predicate counts):

```powershell
python scripts/plot_training.py --epochs 20 --out-dir outputs
```

5. Run unit tests:

```powershell
python -m pytest -q
```

## Files of interest
- `scripts/load_dataset.py` — demo loader that loads data/sample_dataset and runs a short training.
- `scripts/plot_training.py` — CLI plotting script (options: `--epochs`, `--embedding-dim`, `--out-dir`, `--no-mine-rules`).
- `data/sample_dataset/` — example `train.txt`, `valid.txt`, `test.txt` (tab-separated triples).
- `src/` — source code (RDF graph, neural network, rule mining, hybrid model).
- `tests/` — unit tests including a smoke test for plotting and loader.

## CI
This repository includes a GitHub Actions workflow `.github/workflows/ci.yml` that runs linting, type checks, and the test suite. The badge above points to `busro1974/ARM-NS` (auto-filled from your `origin` remote). Replace the badge URL with your repo if needed.

## Troubleshooting
- If you see `ModuleNotFoundError: No module named 'src'` when running scripts, run from project root or ensure `run_demo.py` / scripts are run from the root so imports resolve.
- If PowerShell blocks activation, run: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` (with caution).

## License & Contact
See `IMPLEMENTATION_SUMMARY.txt` and `README.py` for more details and documentation.

Happy experimenting! 🚀
