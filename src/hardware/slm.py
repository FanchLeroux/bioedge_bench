# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 15:38:21 2025

@author: lgs
"""

import ctypes as ct
import numpy as np
import pathlib

import matplotlib.pyplot as plt

from src.utils.miscellaneous import custom_warning
from src.utils.paths import SLM_SDK_DIR

# %%


class MeadowlarkSLM:
    """
    Interface for controlling a Meadowlark SLM using the Blink SDK.
    """

    def __init__(
        self,
        board_number=1,
        slm_shape=np.array([1152, 1920]),
        lut_path=None,
        wfc_path=None,
        sdk_path=str(SLM_SDK_DIR),
    ):

        self.board_number = ct.c_uint(board_number)
        self.slm_shape = slm_shape
        self.sdk_path = sdk_path
        self.lut_path = lut_path
        self.wfc_path = wfc_path

        self.bit_depth = ct.c_uint(12)
        self.num_boards_found = ct.c_uint(0)
        self.constructed_okay = ct.c_uint(-1)
        self.is_nematic_type = ct.c_bool(1)
        self.RAM_write_enable = ct.c_bool(1)
        self.use_GPU = ct.c_bool(1)
        self.max_transients = ct.c_uint(20)

        self.wait_For_Trigger = ct.c_uint(0)
        self.timeout_ms = ct.c_uint(5000)
        self.OutputPulseImageFlip = ct.c_uint(0)
        self.OutputPulseImageRefresh = ct.c_uint(0)

        if wfc_path is not None:
            self.wfc = np.load(wfc_path)
        else:
            self.wfc = None

        self._load_sdk()
        self._initialize_sdk()
        self.load_lut()
        self.display_phase(np.zeros(self.slm_shape))

    def _load_sdk(self):
        """
        Load Blink SDK and image generation libraries.
        """
        ct.cdll.LoadLibrary(str(pathlib.Path(self.sdk_path) / "Blink_C_wrapper"))
        ct.cdll.LoadLibrary(str(pathlib.Path(self.sdk_path) / "ImageGen"))
        self.slm_lib = ct.CDLL("Blink_C_wrapper")

    def _initialize_sdk(self):
        """
        Initialize SDK and check for successful setup.
        """
        self.slm_lib.Create_SDK(
            self.bit_depth,
            ct.byref(self.num_boards_found),
            ct.byref(self.constructed_okay),
            self.is_nematic_type,
            self.RAM_write_enable,
            self.use_GPU,
            self.max_transients,
            0,
        )

        if self.constructed_okay.value == 0:
            self.slm_lib.Get_last_error_message.restype = ct.c_char_p
            raise RuntimeError(
                "Blink SDK failed to initialize: "
                f"{self.slm_lib.Get_last_error_message().decode()}"
            )

    def load_lut(self):
        """
        Load a LUT file to the SLM.

        Parameters
        ----------
        lut_file_path : str or Path, optional
            Path to the LUT file. If None, use self.lut_path.
        """
        if self.lut_path is None:
            raise ValueError("LUT path must be provided.")

        self.slm_lib.Load_LUT_file(
            self.board_number, str(self.lut_path).encode("utf-8")
        )

    def display_phase(self, phase, display=False):
        """
        Display a phase pattern on the SLM.

        Parameters
        ----------
        phase : ndarray
            Phase map to be displayed (2D).
        slm_flat : ndarray, optional
            Flat phase offset (2D, same shape as SLM).
        return_command_vector : bool
            If True, return the vector sent to the SLM.

        Returns
        -------
        phase : ndarray or None
            The command vector if return_command_vector is True.
        """

        phase = phase - phase.mean() + 127.5  # ensure mean is half SLM dynamic

        phase_full_slm = np.full(self.slm_shape, 127.5, dtype=np.float32)
        center = self.slm_shape // 2
        phase_full_slm[
            center[0] - phase.shape[0] // 2 : center[0] + phase.shape[0] // 2,
            center[1] - phase.shape[1] // 2 : center[1] + phase.shape[1] // 2,
        ] = phase

        if self.wfc is not None:
            phase_full_slm = phase_full_slm + self.wfc

        if np.any((phase_full_slm < 0) | (phase_full_slm >= 256)):
            phase_full_slm = np.mod(phase_full_slm, 256)
            custom_warning("SLM is wrapping")

        phase_full_slm = phase_full_slm.astype(np.uint8)
        phase_full_slm = np.reshape(phase_full_slm, np.prod(self.slm_shape))

        self.slm_lib.Write_image(
            self.board_number,
            phase_full_slm.ctypes.data_as(ct.POINTER(ct.c_ubyte)),
            ct.c_uint(np.prod(self.slm_shape)),
            self.wait_For_Trigger,
            self.OutputPulseImageFlip,
            self.OutputPulseImageRefresh,
            self.timeout_ms,
        )

        self.slm_lib.ImageWriteComplete(self.board_number, self.timeout_ms)

        if display:
            plt.figure()
            plt.imshow(np.reshape(phase_full_slm, self.slm_shape))
            plt.title("Command")
            return None
        return None

    def close(self):
        """
        Properly close the SDK connection.
        """
        self.wfc = None
        self.display_phase(np.zeros(self.slm_shape))
        self.slm_lib.Delete_SDK()
