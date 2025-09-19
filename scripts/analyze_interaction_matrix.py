# %%-*- coding: utf-8 -*-
"""
Created on Thu Sep 18 11:23:09 2025

@author: fleroux
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from src.analysis import make_gif, select_valid_pixels, flatten_interaction_matrix

from src.utils.paths import DATA_DIR

# %% parameters

utc_acquisition = "utc_2025-09-17_16-05-18"

# directories and filename
dirc_interaction_matrix = (
    DATA_DIR / "experimental" / "raw" / "interaction_matrix" / utc_acquisition
)

dirc_analysis = DATA_DIR / "experimental" / "processed" / "analysis" / utc_acquisition

filename_interaction_matrix = (
    utc_acquisition + "_push_pull_measurements_orca_inline_743_modes.npy"
)

filename_gif_interaction_matrix = dirc_analysis / (
    utc_acquisition + "_interaction_matrix.gif"
)

filename_gif_eigenmodes = dirc_analysis / (
    utc_acquisition + "_eigen_modes_measurements_space.gif"
)

# post processing parameters
threshold_valid_pixels = 0.2

n_eigen_mode = 1

# %% create folder to save analysis results

pathlib.Path(dirc_analysis).mkdir(parents=True, exist_ok=True)


# %% import data

interaction_matrix = np.load(
    dirc_interaction_matrix / filename_interaction_matrix, mmap_mode="r"
)

interaction_matrix_flat = flatten_interaction_matrix(interaction_matrix)


# %% select valid pixels

valid_pixels = select_valid_pixels(interaction_matrix, threshold_valid_pixels)
valid_pixels_flat = select_valid_pixels(interaction_matrix_flat, threshold_valid_pixels)

plt.figure()
plt.imshow(valid_pixels)

plt.figure()
plt.imshow(np.reshape(valid_pixels_flat, (2048, 2048)))

plt.show()

# %% compute SVD

interaction_matrix_flat_valid_measures = interaction_matrix_flat[valid_pixels_flat]

U, s, VT = np.linalg.svd(interaction_matrix_flat_valid_measures, full_matrices=False)

condition_number = s.max() / s.min()

# %% eigen modes measurement space extraction

eigen_modes_measurement_space = np.full(interaction_matrix.shape, np.nan)

for n_eigen_mode in range(interaction_matrix.shape[0]):
    eigen_modes_measurement_space[n_eigen_mode, :, :][valid_pixels] = U[:, n_eigen_mode]

# %%

plt.figure()
plt.plot(s)
plt.title("Interaction matrix eigenvalues")

# %%

slicer_x = np.r_[np.s_[0:500], np.s_[850:1150], np.s_[1500:2048]]
slicer_y = np.r_[np.s_[0:400], np.s_[800:1350], np.s_[1650:2048]]

# %% make_gif interaction matrix

interaction_matrix_gif = np.delete(
    np.delete(interaction_matrix, slicer_x, 2), slicer_y, 1
)

make_gif(
    interaction_matrix_gif,
    filename_gif_interaction_matrix,
    title="Interaction matrix KL mode ",
)

# %% make_gif eigen_modes_measurement_space

eigen_modes_measurement_space_gif = np.delete(
    np.delete(eigen_modes_measurement_space, slicer_x, 2), slicer_y, 1
)

make_gif(
    eigen_modes_measurement_space_gif,
    filename_gif_eigenmodes,
    title="Measurement space eigen mode ",
)

# %%
