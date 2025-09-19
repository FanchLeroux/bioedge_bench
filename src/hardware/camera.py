# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 13:40:34 2025

@author: lgs
"""

import datetime
import pathlib
from typing import Union, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from astropy.io import fits
from pylablib.devices import DCAM

import os
import sys
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

from src.utils.paths import THORCAM_SDK_DIR

# %%


class OrcaCamera:
    """
    Interface for controlling an ORCA HAMMAMATSU Camera using the pylablib
    package.
    """

    def __init__(
        self,
        idx: Optional[int] = 0,
        serial: Optional[str] = None,
        exposure_time: Optional[float] = 0.1,
        n_frames: Optional[int] = 1,
        roi=[1024, 1024, 2048, 2048],
    ):
        """
        ORCA camera interface.

        Parameters:
            idx    : Index of the camera (default: 0).
            serial : Serial number of the desired camera.
        """
        if serial is not None:
            self.cam = self._find_camera_by_serial(serial)
        else:
            self.cam = DCAM.DCAMCamera(idx=idx)

        self.serial = self.cam.get_device_info().serial_number

        self.exp_time: float = exposure_time
        self.n_frames: int = n_frames
        self.roi = roi
        self.ID: Union[int, str] = 0

    # %%

    def _reset_cam(self):
        self.cam.exp_time = self.exp_time
        self.cam.n_frames = self.n_frames
        self.cam.roi = self.roi
        self.cam.ID = self.ID

    # %%

    def _find_camera_by_serial(self, target_serial: str) -> DCAM.DCAMCamera:
        """
        Search for camera matching the provided serial number.
        """
        n_cams = DCAM.get_cameras_number()

        for idx in range(n_cams):
            cam = DCAM.DCAMCamera(idx=idx)
            info = cam.get_device_info()
            if info.serial_number == target_serial:
                return cam
            cam.close()

        raise RuntimeError(f"Camera with serial '{target_serial}' not found.")

    # %%

    def acquire(
        self,
        n_frames: Optional[int] = None,
        exp_time: Optional[float] = None,
        roi: Union[List[int], bool] = False,
        dirc: Union[pathlib.Path, bool] = False,
        overwrite: bool = False,
    ) -> npt.NDArray[np.float64]:
        """
        Acquire image frames from the ORCA camera.
        """
        n_frames = n_frames or self.n_frames
        exp_time = exp_time or self.exp_time
        roi = roi or self.roi

        self.cam.set_exposure(exp_time)
        timestamp = str(datetime.datetime.now())

        image = np.double(self.cam.grab(n_frames))

        if roi:
            x, y, w, h = roi
            image = image[:, y - h // 2 : y + h // 2, x - w // 2 : x + w // 2]

        if dirc:
            hdu = fits.PrimaryHDU(data=image)
            hdr = hdu.header

            acq = self.cam.get_acquisition_parameters()
            val = self.cam.get_all_attribute_values()

            hdr["NFRAMES"] = (acq["nframes"], "Size of data cube")
            hdr["EXP_TIME"] = (val["exposure_time"], "Exposure time (s)")
            hdr["FPS"] = (val["internal_frame_rate"], "Frame rate (Hz)")
            hdr["INTERVAL"] = (val["internal_frame_interval"], "Frame delay (s)")
            hdr["HPOS"] = (val["subarray_hpos"], "ROI X-pos")
            hdr["YPOS"] = (val["subarray_vpos"], "ROI Y-pos")
            hdr["TIME"] = (timestamp, "Acquisition time")

            fname = (
                f"{self.ID}_exp_"
                f"{np.round(self.cam.get_exposure() * 1000, 3)}"
                f"_nframes_{n_frames}.fits"
            )

            file_path = pathlib.Path(dirc) / fname
            hdu.writeto(file_path, overwrite=overwrite)

        self._reset_cam()

        return image

    # %%

    def live_view(
        self,
        exp_time: Optional[float] = None,
        roi: Optional[List[int]] = None,
        interval: float = 0.005,
    ) -> None:
        """
        Live view display from the camera.
        """
        exp_time = exp_time or self.exp_time
        roi = roi or self.roi

        plt.ion()

        frame = self.acquire(n_frames=1, exp_time=exp_time, roi=roi)[0]
        serial = self.cam.get_device_info().serial_number

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(frame, cmap="viridis")
        plt.colorbar(im, ax=ax)

        title = ax.set_title(
            f"Serial: {serial} -\n"
            f"Max: {np.max(frame):.2f}  Mean: {np.mean(frame):.2f}"
        )

        while plt.fignum_exists(fig.number):
            frame = self.acquire(n_frames=1, exp_time=exp_time, roi=roi)[0]
            im.set_data(frame)
            im.set_clim(np.min(frame), np.max(frame))

            title.set_text(
                f"Serial: {serial} -\n"
                f"Max: {np.max(frame):.2f}  Mean: {np.mean(frame):.2f}"
            )

            plt.pause(interval)

    # %%

    def close(self):
        """
        Properly close the SDK connection.
        """
        self.cam.close()


# %%


class ThorlabsZeluxCamera:
    """
    Interface for controlling a Thorlabs camera using the TSI SDK.
    """

    def __init__(
        self,
        cam_id: str = "",
        exposure_time: float = 0.1,
        n_frames: int = 1,
        sdk_path: str = str(THORCAM_SDK_DIR),
    ):

        self.sdk_path = sdk_path
        self.cam_id = cam_id
        self.exposure_time = exposure_time
        self.n_frames = n_frames

        self._configure_dll_path()
        self.sdk = TLCameraSDK()
        self.available_cameras = self.sdk.discover_available_cameras()

        if len(self.available_cameras) == 0:
            raise RuntimeError("No Thorlabs cameras detected.")

        if not self.cam_id:
            self.cam_id = self.available_cameras[0]
        elif self.cam_id not in self.available_cameras:
            raise ValueError(
                f"Camera ID '{self.cam_id}' not found. "
                f"Available: {self.available_cameras}"
            )

        self.camera = self.sdk.open_camera(self.cam_id)

    # %%

    def _configure_dll_path(self):
        """
        Configure the DLL path for the Thorlabs SDK.
        """
        is_64bits = sys.maxsize > 2**32
        dll_subfolder = "64_lib" if is_64bits else "32_lib"

        dll_path = pathlib.Path(self.sdk_path) / "dlls" / dll_subfolder
        dll_path = dll_path.resolve()

        os.environ["PATH"] = str(dll_path) + os.pathsep + os.environ.get("PATH", "")

        try:
            os.add_dll_directory(str(dll_path))
        except AttributeError:
            pass

    # %%

    def acquire_frames(self):
        """
        Acquire a stack of frames from the camera.

        Returns
        -------
        frames : ndarray
            Array of shape (n_frames, height, width) with float64 values.
        """
        cam = self.camera
        cam.exposure_time_us = int(self.exposure_time * 1e6)
        cam.frames_per_trigger_zero_for_unlimited = 0
        cam.image_poll_timeout_ms = 1000

        cam.arm(self.n_frames)

        shape = (self.n_frames, cam.image_height_pixels, cam.image_width_pixels)

        frames = np.full(shape, np.nan, dtype=np.float64)

        for i in range(self.n_frames):
            cam.issue_software_trigger()
            frame = cam.get_pending_frame_or_null()

            if frame is None:
                raise RuntimeError("Frame acquisition failed (null frame).")

            buffer = np.copy(frame.image_buffer)
            image = buffer.reshape(
                cam.image_height_pixels, cam.image_width_pixels
            ).astype(np.float64)

            frames[i, :, :] = image

        cam.disarm()
        return frames

    # %%

    def live_view(self, interval: float = 0.01):
        """
        Display a real-time live view from the camera.

        Parameters
        ----------
        interval : float
            Pause time between frame updates in seconds.
        """
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 5))

        # Temporarily override n_frames to 1 for live view
        original_n_frames = self.n_frames
        self.n_frames = 2

        try:
            frame = self.acquire_frames()[0]
        except RuntimeError as e:
            raise RuntimeError("Initial frame acquisition failed.") from e

        im = ax.imshow(frame, cmap="viridis")
        plt.colorbar(im, ax=ax)

        title = ax.set_title(
            f"Cam ID: {self.cam_id}\n"
            f"Max: {np.max(frame):.2f}  Mean: {np.mean(frame):.2f}"
        )

        while plt.fignum_exists(fig.number):
            try:
                frame = self.acquire_frames()[0]
            except RuntimeError:
                continue  # Skip failed frame

            im.set_data(frame)
            im.set_clim(vmin=np.min(frame), vmax=np.max(frame))

            title.set_text(
                f"Cam ID: {self.cam_id}\n"
                f"Max: {np.max(frame):.2f}  Mean: {np.mean(frame):.2f}"
            )

            plt.pause(interval)

        plt.ioff()
        self.n_frames = original_n_frames  # Restore user-set value

    # %%

    def save_to_fits(
        self, frames: np.ndarray, output_path: pathlib.Path, overwrite: bool = False
    ):
        """
        Save acquired frames to a FITS file with camera metadata.

        Parameters
        ----------
        frames : ndarray
            Image stack to save.
        output_path : Path
            Destination path for the FITS file.
        overwrite : bool
            Overwrite existing file if True.
        """
        hdu = fits.PrimaryHDU(data=frames)
        hdr = hdu.header

        hdr["NFRAMES"] = (self.n_frames, "Number of frames")
        hdr["EXPTIME"] = (self.exposure_time, "Exposure time (s)")
        hdr["CAMERAID"] = (self.cam_id, "Camera Serial Number")
        hdr["ACQTIME"] = (str(datetime.datetime.now()), "Acquisition timestamp")

        hdu.writeto(output_path, overwrite=overwrite)

    # %%

    def close(self):
        """
        Close the camera and release the SDK.
        """
        if hasattr(self, "camera") and self.camera:
            self.camera.dispose()

        if hasattr(self, "sdk") and self.sdk:
            self.sdk.dispose()

        self.camera = None
        self.sdk = None

    # %%

    def __del__(self):
        self.close()


# %%

# ========================
# Example usage orca
# ========================


if __name__ == "__main__":
    # Option 1: Use first available camera (index 0)
    # orca = OrcaCamera()

    # Option 2: Use specific serial number
    orca = OrcaCamera(serial="S/N: 002369")

    orca.exp_time = 0.1
    orca.n_frames = 3
    orca.ID = 0

    # Acquire images
    images = orca.acquire()

    # Optionally start live view
    # orca.live_view(interval=0.01)

    plt.figure()
    plt.imshow(images[-1, :, :])
    plt.title("ORCA frame")

    orca.close()

# ========================
# Example usage Thorcam
# ========================


if __name__ == "__main__":

    sdk_dir = r"D:\Francois_Leroux\code\project\phd_bioedge\manip\thorcam_SDK"

    thorcam = ThorlabsZeluxCamera(
        sdk_path=sdk_dir, cam_id="18263", exposure_time=0.01, n_frames=2
    )

    # Optionally start live view
    # cam.live_view()

    # Acquire frames and save
    data = thorcam.acquire_frames()
    save_path = pathlib.Path("thorcam_output.fits")
    thorcam.save_to_fits(data, save_path, overwrite=True)

    plt.figure()
    plt.imshow(data[-1, :, :])
    plt.title("Thorcam frame")

    thorcam.close()
