"""
Wrapper to run the demo script from project root: `python run_demo.py`
"""
import subprocess
import sys
import os

script = os.path.join(os.path.dirname(__file__), 'scripts', 'load_dataset.py')
if not os.path.exists(script):
    raise SystemExit(f"Demo script not found: {script}")

print(f"Running demo script: {script}\n")
ret = subprocess.call([sys.executable, script])
if ret != 0:
    raise SystemExit(f"Demo script exited with status {ret}")
