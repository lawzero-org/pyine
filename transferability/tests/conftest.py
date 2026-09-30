"""Pytest config for the transferability study tests.

Adds the study's scripts/ and cueflip/ directories to sys.path so test
files can import the helper functions directly. Tests must run from a
Python environment where the pyine package is installed editable
(e.g., the project's .venv) -- analysis_d.py and analysis_g.py import
pyine.utils.metrics.confidence at module level.
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]  # transferability/
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "cueflip"))
