# %% Imports

import pathlib

import numpy as np

from OOPAO.Telescope import Telescope
from OOPAO.Source import Source
from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.tools.interpolateGeometricalTransformation import interpolate_cube

# %% Compute Karhunen–Loeve basis


def compute_KL_modes(n_subapertures, r0=0.15, L0=25, n_pixels=None):

    fractionalR0 = [0.45, 0.1, 0.1, 0.25, 0.1]  # Cn2 Profile
    windSpeed = [10, 12, 11, 15, 20]  # Wind Speed [m/s]
    windDirection = [0, 72, 144, 216, 288]  # Wind Direction [deg]
    altitude = [0, 1000, 5000, 10000, 12000]  # Altitude Layers [m]

    # %% Telescope

    tel = Telescope(
        resolution=8 * n_subapertures,
        diameter=8,
        samplingTime=1 / 1000,
        centralObstruction=0.0,
        display_optical_path=False,
        fov=10,
    )

    if n_pixels is not None:

        tel_hr = Telescope(
            resolution=n_pixels,
            diameter=8,
            samplingTime=1 / 1000,
            centralObstruction=0.0,
            display_optical_path=False,
            fov=10,
        )

    # %% Natural Guide Star

    ngs = Source(optBand="I", magnitude=8, coordinates=[0, 0])

    ngs * tel

    # %% Atmosphere

    atm = Atmosphere(
        telescope=tel,
        r0=r0,
        L0=L0,
        fractionalR0=fractionalR0,
        windSpeed=windSpeed,
        windDirection=windDirection,
        altitude=altitude,
    )

    # %% Deformable mirror

    nAct = n_subapertures + 1
    dm = DeformableMirror(telescope=tel, nSubap=nAct - 1)

    # %% Coupling tel and atm

    atm.initializeAtmosphere(tel)
    tel + atm

    # %% Compute M2C

    M2C_KL = compute_KL_basis(tel, atm, dm)

    # %% Compute KL modes

    tel.resetOPD()

    dm.coefs = M2C_KL

    # propagate through the DM
    ngs * tel * dm
    KL_modes = tel.OPD

    # reshape data cube as [n_images, n_pixels, n_pixels]
    KL_modes = np.transpose(KL_modes, (2, 0, 1))

    # %% If needed, interpolation to high resolution

    if n_pixels is not None:

        KL_modes_hr = interpolate_cube(
            KL_modes, tel.pixelSize, tel_hr.pixelSize, tel_hr.resolution
        )
        return KL_modes_hr
    else:
        return KL_modes
