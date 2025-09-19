# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 16:30:43 2025

@author: lgs
"""

from tqdm import tqdm
from time import sleep

import numpy as np
import matplotlib.pyplot as plt

# %%


def measure_interaction_matrix(
        slm, camera, modal_basis, n_frames=3, exposure_time=None,
        stroke=1., display=False, dark=None
):
    """
    Measures an interaction matrix between an SLM and a camera.

    Parameters:
        slm: MeadowlarkSLM object
        camera: OrcaCamera object
        modal_basis: numpy array of SLM patterns (N_modes x H x W)
        repetitions: int, number of repetitions per modes for averaging

    Returns:
        imat: Interaction matrix (N_pixels x N_modes)
    """
    n_modes = modal_basis.shape[0]
    image_shape = [camera.roi[3], camera.roi[2]]

    imat = np.full([n_modes]+image_shape, np.nan, dtype=np.float32)

    if dark is None:

        input("Aquire dark\nTurn Off light source\nPress Enter to continue...")
        dark = np.mean(camera.acquire(n_frames=11, exp_time=exposure_time),
                       axis=0)
        input("Turn On Laser\nPress Enter to continue...")

    if display:

        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(nrows=1, ncols=2)
        im1 = ax[0].imshow(np.zeros((modal_basis.shape[1],
                                     modal_basis.shape[2])), cmap='viridis')
        im2 = ax[1].imshow(np.zeros(imat.shape[1:]), cmap='viridis')
        ax[0].set_title("SLM Command")
        ax[1].set_title("Detector Irradiance")
        plt.tight_layout()
        plt.show()

    sleep(10)

    for n_mode in tqdm(range(n_modes), desc="Measuring interaction matrix"):

        phase = stroke*modal_basis[n_mode]
        slm.display_phase(phase)
        push = np.mean(camera.acquire(
            n_frames=n_frames, exp_time=exposure_time) - dark, axis=0)
        slm.display_phase(-phase)
        pull = np.mean(camera.acquire(
            n_frames=n_frames, exp_time=exposure_time) - dark, axis=0)

        imat[n_mode, :, :] = push/push.sum() - pull/pull.sum()

        del push, pull

        if display:

            im1.set_data(modal_basis[n_mode, :, :])
            im1.set_clim(vmin=np.min(modal_basis[n_mode, :, :]),
                         vmax=np.max(modal_basis[n_mode, :, :]))
            im2.set_data(imat[n_mode, :, :])
            im2.set_clim(vmin=np.min(imat[n_mode, :, :]),
                         vmax=np.max(imat[n_mode, :, :]))
            plt.pause(0.01)

    return imat
