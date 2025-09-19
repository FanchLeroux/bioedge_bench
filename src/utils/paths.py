"""
paths.py

This script define the directories of the project

"""

# %%

import sys

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
