# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:09:35 2021

@author: fayya
"""


import functools
import numpy as np
from warnings import warn
from scipy import linalg
import dtype
def _prepare_colorarray(arr, force_copy=False):
    """Check the shape of the array and convert it to
    floating point representation.
    """
    arr = np.asanyarray(arr)

    if arr.shape[-1] != 3:
        raise ValueError("Input array must have a shape == (..., 3)), "
                         f"got {arr.shape}")

    return dtype.img_as_float(arr, force_copy=force_copy)
# Haematoxylin-Eosin-DAB colorspace
# From original Ruifrok's paper: A. C. Ruifrok and D. A. Johnston,
# "Quantification of histochemical staining by color deconvolution.,"
# Analytical and quantitative cytology and histology / the International
# Academy of Cytology [and] American Society of Cytology, vol. 23, no. 4,
# pp. 291-9, Aug. 2001.
rgb_from_hed = np.array([[0.65, 0.70, 0.29],
                         [0.07, 0.99, 0.11],
                         [0.27, 0.57, 0.78]])
hed_from_rgb = linalg.inv(rgb_from_hed)

# Following matrices are adapted form the Java code written by G.Landini.
# The original code is available at:
# https://web.archive.org/web/20160624145052/http://www.mecourse.com/landinig/software/cdeconv/cdeconv.html

# Hematoxylin + DAB
rgb_from_hdx = np.array([[0.650, 0.704, 0.286],
                         [0.268, 0.570, 0.776],
                         [0.0, 0.0, 0.0]])
rgb_from_hdx[2, :] = np.cross(rgb_from_hdx[0, :], rgb_from_hdx[1, :])
hdx_from_rgb = linalg.inv(rgb_from_hdx)

# Feulgen + Light Green
rgb_from_fgx = np.array([[0.46420921, 0.83008335, 0.30827187],
                         [0.94705542, 0.25373821, 0.19650764],
                         [0.0, 0.0, 0.0]])
rgb_from_fgx[2, :] = np.cross(rgb_from_fgx[0, :], rgb_from_fgx[1, :])
fgx_from_rgb = linalg.inv(rgb_from_fgx)

# Giemsa: Methyl Blue + Eosin
rgb_from_bex = np.array([[0.834750233, 0.513556283, 0.196330403],
                         [0.092789, 0.954111, 0.283111],
                         [0.0, 0.0, 0.0]])
rgb_from_bex[2, :] = np.cross(rgb_from_bex[0, :], rgb_from_bex[1, :])
bex_from_rgb = linalg.inv(rgb_from_bex)

# FastRed + FastBlue +  DAB
rgb_from_rbd = np.array([[0.21393921, 0.85112669, 0.47794022],
                         [0.74890292, 0.60624161, 0.26731082],
                         [0.268, 0.570, 0.776]])
rbd_from_rgb = linalg.inv(rgb_from_rbd)

# Methyl Green + DAB
rgb_from_gdx = np.array([[0.98003, 0.144316, 0.133146],
                         [0.268, 0.570, 0.776],
                         [0.0, 0.0, 0.0]])
rgb_from_gdx[2, :] = np.cross(rgb_from_gdx[0, :], rgb_from_gdx[1, :])
gdx_from_rgb = linalg.inv(rgb_from_gdx)

# Hematoxylin + AEC
rgb_from_hax = np.array([[0.650, 0.704, 0.286],
                         [0.2743, 0.6796, 0.6803],
                         [0.0, 0.0, 0.0]])
rgb_from_hax[2, :] = np.cross(rgb_from_hax[0, :], rgb_from_hax[1, :])
hax_from_rgb = linalg.inv(rgb_from_hax)

# Blue matrix Anilline Blue + Red matrix Azocarmine + Orange matrix Orange-G
rgb_from_bro = np.array([[0.853033, 0.508733, 0.112656],
                         [0.09289875, 0.8662008, 0.49098468],
                         [0.10732849, 0.36765403, 0.9237484]])
bro_from_rgb = linalg.inv(rgb_from_bro)

# Methyl Blue + Ponceau Fuchsin
rgb_from_bpx = np.array([[0.7995107, 0.5913521, 0.10528667],
                         [0.09997159, 0.73738605, 0.6680326],
                         [0.0, 0.0, 0.0]])
rgb_from_bpx[2, :] = np.cross(rgb_from_bpx[0, :], rgb_from_bpx[1, :])
bpx_from_rgb = linalg.inv(rgb_from_bpx)

# Alcian Blue + Hematoxylin
rgb_from_ahx = np.array([[0.874622, 0.457711, 0.158256],
                         [0.552556, 0.7544, 0.353744],
                         [0.0, 0.0, 0.0]])
rgb_from_ahx[2, :] = np.cross(rgb_from_ahx[0, :], rgb_from_ahx[1, :])
ahx_from_rgb = linalg.inv(rgb_from_ahx)

# Hematoxylin + PAS
rgb_from_hpx = np.array([[0.644211, 0.716556, 0.266844],
                         [0.175411, 0.972178, 0.154589],
                         [0.0, 0.0, 0.0]])
rgb_from_hpx[2, :] = np.cross(rgb_from_hpx[0, :], rgb_from_hpx[1, :])
hpx_from_rgb = linalg.inv(rgb_from_hpx)

def rgb2hed(rgb):
    """RGB to Haematoxylin-Eosin-DAB (HED) color space conversion.

    Parameters
    ----------
    rgb : (..., 3) array_like
        The image in RGB format. Final dimension denotes channels.

    Returns
    -------
    out : (..., 3) ndarray
        The image in HED format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3).

    References
    ----------
    .. [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical
           staining by color deconvolution.," Analytical and quantitative
           cytology and histology / the International Academy of Cytology [and]
           American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2hed
    >>> ihc = data.immunohistochemistry()
    >>> ihc_hed = rgb2hed(ihc)
    """
    return separate_stains(rgb, hed_from_rgb)


def hed2rgb(hed):
    """Haematoxylin-Eosin-DAB (HED) to RGB color space conversion.

    Parameters
    ----------
    hed : (..., 3) array_like
        The image in the HED color space. Final dimension denotes channels.

    Returns
    -------
    out : (..., 3) ndarray
        The image in RGB. Same dimensions as input.

    Raises
    ------
    ValueError
        If `hed` is not at least 2-D with shape (..., 3).

    References
    ----------
    .. [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical
           staining by color deconvolution.," Analytical and quantitative
           cytology and histology / the International Academy of Cytology [and]
           American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2hed, hed2rgb
    >>> ihc = data.immunohistochemistry()
    >>> ihc_hed = rgb2hed(ihc)
    >>> ihc_rgb = hed2rgb(ihc_hed)
    """
    return combine_stains(hed, rgb_from_hed)


def separate_stains(rgb, conv_matrix):
    """RGB to stain color space conversion.

    Parameters
    ----------
    rgb : (..., 3) array_like
        The image in RGB format. Final dimension denotes channels.
    conv_matrix: ndarray
        The stain separation matrix as described by G. Landini [1]_.

    Returns
    -------
    out : (..., 3) ndarray
        The image in stain color space. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3).

    Notes
    -----
    Stain separation matrices available in the ``color`` module and their
    respective colorspace:

    * ``hed_from_rgb``: Hematoxylin + Eosin + DAB
    * ``hdx_from_rgb``: Hematoxylin + DAB
    * ``fgx_from_rgb``: Feulgen + Light Green
    * ``bex_from_rgb``: Giemsa stain : Methyl Blue + Eosin
    * ``rbd_from_rgb``: FastRed + FastBlue +  DAB
    * ``gdx_from_rgb``: Methyl Green + DAB
    * ``hax_from_rgb``: Hematoxylin + AEC
    * ``bro_from_rgb``: Blue matrix Anilline Blue + Red matrix Azocarmine\
                        + Orange matrix Orange-G
    * ``bpx_from_rgb``: Methyl Blue + Ponceau Fuchsin
    * ``ahx_from_rgb``: Alcian Blue + Hematoxylin
    * ``hpx_from_rgb``: Hematoxylin + PAS

    This implementation borrows some ideas from DIPlib [2]_, e.g. the
    compensation using a small value to avoid log artifacts when
    calculating the Beer-Lambert law.

    References
    ----------
    .. [1] https://web.archive.org/web/20160624145052/http://www.mecourse.com/landinig/software/cdeconv/cdeconv.html
    .. [2] https://github.com/DIPlib/diplib/
    .. [3] A. C. Ruifrok and D. A. Johnston, “Quantification of histochemical
           staining by color deconvolution,” Anal. Quant. Cytol. Histol., vol.
           23, no. 4, pp. 291–299, Aug. 2001.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import separate_stains, hdx_from_rgb
    >>> ihc = data.immunohistochemistry()
    >>> ihc_hdx = separate_stains(ihc, hdx_from_rgb)
    """
    rgb = _prepare_colorarray(rgb, force_copy=True)
    np.maximum(rgb, 1E-6, out=rgb)  # avoiding log artifacts
    log_adjust = np.log(1E-6)  # used to compensate the sum above

    stains = (np.log(rgb) / log_adjust) @ conv_matrix

    return stains


def combine_stains(stains, conv_matrix):
    """Stain to RGB color space conversion.

    Parameters
    ----------
    stains : (..., 3) array_like
        The image in stain color space. Final dimension denotes channels.
    conv_matrix: ndarray
        The stain separation matrix as described by G. Landini [1]_.

    Returns
    -------
    out : (..., 3) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `stains` is not at least 2-D with shape (..., 3).

    Notes
    -----
    Stain combination matrices available in the ``color`` module and their
    respective colorspace:

    * ``rgb_from_hed``: Hematoxylin + Eosin + DAB
    * ``rgb_from_hdx``: Hematoxylin + DAB
    * ``rgb_from_fgx``: Feulgen + Light Green
    * ``rgb_from_bex``: Giemsa stain : Methyl Blue + Eosin
    * ``rgb_from_rbd``: FastRed + FastBlue +  DAB
    * ``rgb_from_gdx``: Methyl Green + DAB
    * ``rgb_from_hax``: Hematoxylin + AEC
    * ``rgb_from_bro``: Blue matrix Anilline Blue + Red matrix Azocarmine\
                        + Orange matrix Orange-G
    * ``rgb_from_bpx``: Methyl Blue + Ponceau Fuchsin
    * ``rgb_from_ahx``: Alcian Blue + Hematoxylin
    * ``rgb_from_hpx``: Hematoxylin + PAS

    References
    ----------
    .. [1] https://web.archive.org/web/20160624145052/http://www.mecourse.com/landinig/software/cdeconv/cdeconv.html
    .. [2] A. C. Ruifrok and D. A. Johnston, “Quantification of histochemical
           staining by color deconvolution,” Anal. Quant. Cytol. Histol., vol.
           23, no. 4, pp. 291–299, Aug. 2001.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import (separate_stains, combine_stains,
    ...                            hdx_from_rgb, rgb_from_hdx)
    >>> ihc = data.immunohistochemistry()
    >>> ihc_hdx = separate_stains(ihc, hdx_from_rgb)
    >>> ihc_rgb = combine_stains(ihc_hdx, rgb_from_hdx)
    """
    stains = _prepare_colorarray(stains)

    # log_adjust here is used to compensate the sum within separate_stains().
    log_adjust = -np.log(1E-6)
    log_rgb = -(stains * log_adjust) @ conv_matrix
    rgb = np.exp(log_rgb)

    return np.clip(rgb, a_min=0, a_max=1)