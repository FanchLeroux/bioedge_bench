from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from astropy.io import fits


def save(obj, filepath):
    """
    Save object to file based on file extension.

    Supported:
    - NumPy arrays: .npy, .txt
    - Matplotlib figures: .png, .pdf, .svg, .jpg, .jpeg
    - PIL Images: .gif, .png, .bmp, .jpeg, etc.
    - Text files: .txt, .md
    - FITS files: .fits (via astropy.io.fits)

    Args:
        obj: Object to save.
        filepath: str or Path. Must include appropriate file extension.
        overwrite: bool. Only affects FITS files (default: False).
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    # Ensure parent directories exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # NumPy array saving
    if suffix == ".npy" and isinstance(obj, np.ndarray):
        np.save(filepath, obj)

    elif suffix == ".txt" and isinstance(obj, np.ndarray):
        np.savetxt(filepath, obj)

    # Matplotlib figure saving
    elif suffix in [".png", ".pdf", ".svg", ".jpg", ".jpeg"]:
        if isinstance(obj, Figure):
            obj.savefig(filepath)
        elif obj == plt:
            plt.savefig(filepath)
        else:
            raise TypeError("Expected a matplotlib Figure or plt for image saving.")

    # PIL image saving
    elif suffix in [".gif", ".tiff", ".bmp", ".png", ".jpeg", ".jpg"]:
        if isinstance(obj, Image.Image):
            obj.save(filepath)
        else:
            raise TypeError("Expected a PIL Image object for image saving.")

    # Text saving
    elif suffix in [".txt", ".md"]:
        if isinstance(obj, str):
            filepath.write_text(obj)
        else:
            raise TypeError("Expected a string for saving to a text file.")

    # FITS file saving
    elif suffix == ".fits":
        if isinstance(obj, (np.ndarray, list)):
            hdu = fits.PrimaryHDU(obj)
            hdul = fits.HDUList([hdu])
            hdul.writeto(filepath, overwrite=False)
        elif isinstance(obj, fits.HDUList):
            obj.writeto(filepath, overwrite=False)
        elif isinstance(obj, fits.PrimaryHDU):
            fits.HDUList([obj]).writeto(filepath, overwrite=False)
        else:
            raise TypeError("Expected a NumPy array or HDUList for FITS saving.")

    else:
        raise ValueError(
            f"Unsupported file extension: '{suffix}' for object type {type(obj)}"
        )
