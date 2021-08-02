#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 13:26:54 2021

@author: u1876024
"""

import numpy as np

from staintools.stain_extraction.abc_stain_extractor import ABCStainExtractor
from staintools.utils.miscellaneous_functions import normalize_matrix_rows
from staintools.utils.optical_density_conversion import convert_RGB_to_OD
from staintools.tissue_masks.luminosity_threshold_tissue_locator import LuminosityThresholdTissueLocator
from staintools.preprocessing.input_validation import is_uint8_image

I = ihc_rgb0
luminosity_threshold=0.8
angular_percentile=99
# Convert to OD and ignore background
tissue_mask = LuminosityThresholdTissueLocator.get_tissue_mask(I, luminosity_threshold=luminosity_threshold).reshape((-1,))
OD = convert_RGB_to_OD(I).reshape((-1, 3))
OD = OD[tissue_mask]

# Eigenvectors of cov in OD space (orthogonal as cov symmetric)
ee, V = np.linalg.eigh(np.cov(OD, rowvar=False))


# The two principle eigenvectors
V = V[:, [2, 1, 0]]

# Make sure vectors are pointing the right way
if V[0, 0] < 0: V[:, 0] *= -1
if V[0, 1] < 0: V[:, 1] *= -1
if V[0, 2] < 0: V[:, 1] *= -1

# Project on this basis.
That = np.dot(OD, V)
1/0
# Angular coordinates with repect to the prinicple, orthogonal eigenvectors
phi = np.arctan2(That[:, 1], That[:, 0])

# Min and max angles
minPhi = np.percentile(phi, 100 - angular_percentile)
maxPhi = np.percentile(phi, angular_percentile)

# the two principle colors
v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

# Order of H and E.
# H first row.
if v1[0] > v2[0]:
    HE = np.array([v1, v2])
else:
    HE = np.array([v2, v1])

SM =  normalize_matrix_rows(HE)

