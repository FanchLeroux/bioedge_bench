"""
paths.py

This script define the directories of the project

"""

# %%

import sys

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

SLM_SDK_DIR = ROOT_DIR / "third_party" / "meadowlark" / "slm1920"
SLM_LUT_DIR = ROOT_DIR / "configs" / "slm" / "lut"
SLM_WFC_DIR = ROOT_DIR / "configs" / "slm" / "wfc"

THORCAM_SDK_DIR = ROOT_DIR / "third_party" / "thorlabs" / "zelux_camera"
