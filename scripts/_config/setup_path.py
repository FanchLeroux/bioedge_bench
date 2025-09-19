"""
setup_path.py

Run this script by adding
"from config import setup_path"
at the begining of a script allow to use any module inside src folder

"""

# %%

import sys

from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[2])

sys.path.append(ROOT_DIR)
