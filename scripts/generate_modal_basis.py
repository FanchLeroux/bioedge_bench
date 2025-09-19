"""
This script aims at generating modal basis
to be displayed by SLM during calibration
"""

# %%

import numpy as np

from _config import setup_path  # add project root to path
from src.utils.paths import DATA_DIR  # set data directory

from src.modal_basis import compute_KL_modes
from src.utils.miscellaneous import get_utc_now
from src.utils.save_utils import save

# %% Parameters

n_subapertures = 8
n_pixels = 100

# %% Compute KL modes

KL_modes = compute_KL_modes(n_subapertures=n_subapertures, n_pixels=n_pixels)
utc_now = get_utc_now()

# %% Save KL modes

filename = (
    utc_now + f"_KL_modes_{n_pixels}x{n_pixels}_pixels_{n_subapertures}_subap.npy"
)

save(KL_modes, DATA_DIR / "numerical" / "raw" / "modal_basis" / filename)
