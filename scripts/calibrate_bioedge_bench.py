# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 16:41:01 2025

@author: lgs
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from src.hardware.slm import MeadowlarkSLM

from src.hardware.camera import OrcaCamera

from src.calibration import measure_interaction_matrix

from src.utils.miscellaneous import get_utc_now

from src.utils.save_utils import save

from src.utils.paths import DATA_DIR, SLM_LUT_DIR

# %% Parameters

# lut filename
lut_path = str(SLM_LUT_DIR / "slm5758_at675.lut")

exposure_time = 200e-3  # [s]

# %% Load modal basis

# Load KL modes
filename = (
    "KL_modes_slm_units_600_pixels_in_slm_pupil_30_subapertures_"
    "no_wrapping_required.npy"
)
KL_modes = np.load(
    DATA_DIR / "raw" / "modal_basis" / filename,
    mmap_mode="r",
)

# chose how many modes are used to calibrate
n_calibration_modes = KL_modes.shape[0]
KL_modes = KL_modes[:n_calibration_modes]

# %% Connect hardware

slm = MeadowlarkSLM(lut_path=lut_path)
orca = OrcaCamera(serial="S/N: 002369")

# %% Create folder to save results

utc_now = get_utc_now()

dirc_interaction_matrix = (
    DATA_DIR / "experimental" / "raw" / "interaction_matrix" / utc_now
)

pathlib.Path(dirc_interaction_matrix).mkdir(parents=True, exist_ok=True)

# %% Measure interaction matrix

interaction_matrix = measure_interaction_matrix(
    slm,
    orca,
    KL_modes,
    n_frames=3,
    exposure_time=exposure_time,
    stroke=1.0,
    display=False,
    dark=None,
)

# %% Save interaction matrix

save(
    interaction_matrix,
    dirc_interaction_matrix
    / (
        utc_now + f"_push_pull_measurements_orca_inline"
        f"_{KL_modes.shape[0]}_modes.npy"
    ),
)

# %% Display

plt.figure()
plt.imshow(interaction_matrix[-1])

# %% Disconnect harware

slm.close()
orca.close()
