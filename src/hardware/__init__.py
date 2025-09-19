# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 16:38:04 2025

@author: fleroux
"""

# hardware/__init__.py

"""
SLM control package initialization.

This package provides an interface to control the Meadowlark SLM.
"""

from .slm import MeadowlarkSLM

__all__ = ["MeadowlarkSLM"]
