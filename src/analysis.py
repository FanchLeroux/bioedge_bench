# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 11:08:15 2025

@author: lgs
"""

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import gif

# %%


def make_gif(array: np.ndarray, filename: str, title: str = ""):

    plt.close('all')

    @gif.frame
    def plot(i):
        plt.imshow(array[i, :, :])
        plt.title(title + f"{i}")

    frames = [plot(i) for i in tqdm(range(array.shape[0]))]
    gif.save(frames, str(filename), duration=200)

    return
# %%


def set_binning(array, binning_factor, mode='sum'):

    if array.shape[0] % binning_factor == 0:
        if array.ndim == 2:
            new_shape = [int(np.round(array.shape[0]/binning_factor)),
                         int(np.round(array.shape[1]/binning_factor))]
            shape = (new_shape[0], array.shape[0] // new_shape[0],
                     new_shape[1], array.shape[1] // new_shape[1])
            if mode == 'sum':
                return array.reshape(shape).sum(-1).sum(1)
            else:
                return array.reshape(shape).mean(-1).mean(1)
        else:
            new_shape = [int(np.round(array.shape[0]/binning_factor)),
                         int(np.round(array.shape[1]/binning_factor)),
                         array.shape[2]]
            shape = (new_shape[0], array.shape[0] // new_shape[0],
                     new_shape[1], array.shape[1] // new_shape[1],
                     new_shape[2])
            if mode == 'sum':
                return array.reshape(shape).sum(-2).sum(1)
            else:
                return array.reshape(shape).mean(-2).mean(1)
    else:
        raise ValueError("""Binning factor {binning_factor} not compatible with
                         the array size""")

# %%


def select_valid_pixels(array, threshold):

    axis = 1 if len(array.shape) == 2 else 0
    std = array.std(axis=axis)
    normalized_std = std / std.max()
    valid_pixels = normalized_std > threshold

    return valid_pixels

# %%

def flatten_interaction_matrix(interaction_matrix):
    return interaction_matrix.reshape(interaction_matrix.shape[0], -1).T

