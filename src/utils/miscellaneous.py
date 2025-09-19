# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 13:59:58 2025

@author: lgs
"""

import datetime
import tqdm


def get_utc_now():
    return datetime.datetime.utcnow().strftime("utc_%Y-%m-%d_%H-%M-%S")


def custom_warning(message: str) -> None:
    """
    Print a custom warning message in bold yellow text.
    """
    bold_yellow = '\033[1m\033[33m'
    reset = '\033[0m'
    tqdm.tqdm.write(f"\r{bold_yellow}Custom Warning: {message}{reset}")
