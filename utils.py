from collections.abc import Iterable
import itertools

from PIL import Image
import numpy as np
import openslide


def std_normalise(x):
    """Normalse x to be 0 centred with a standard deviation of 1"""
    x = np.array(x)
    x -= x.mean()
    x /= x.std()
    return x


def normalise(x):
    """Normalise x to be between 0 and 1"""
    x = np.array(x)
    x -= x.min()
    x /= x.max()
    x = np.nan_to_num(x)
    return x


def sumto1(x):
    """Rescale an array of values to sum to 1."""
    x = normalise(x)
    x /= x.sum()
    x = np.nan_to_num(x)
    return x


def alpha_composite(rgba_image, fill=255):
    """Paste the RGBA region over a background fill (white by default)"""
    bg = Image.fromarray(
        np.full(list(rgba_image.size[::-1]) + [4], fill, dtype=np.uint8)
    )
    bg.alpha_composite(rgba_image)
    bg = bg.convert("RGB")
    return bg


def size_scale_factor(from_size, to_size):
    """"Determine the scale difference between two sizes ([width, height] pairs)"""
    return np.array(to_size) / np.array(from_size)


def rescale_coords(coord, from_size, to_size, flip_axis=None):
    """Rescale coordinates relative to one (width, height) size to be relative to another size"""
    rescaled = np.array(coord) * size_scale_factor(from_size, to_size)
    if flip_axis is not None:
        rescaled[..., flip_axis] = to_size[flip_axis] - rescaled[:, flip_axis]
    return rescaled


def centre_to_origin(centre, size):
    """Translate a coordinate which is at the centre of a region to be the at the region's origin"""
    return np.array(centre) - (np.array(size) / 2)


def ppcm2mpp(ppcm):
    """Pixels per cm to microns per pixel."""
    return 1 / ppcm * 10000


def slide_mpp(slide_handle):
    """Determine the numer of microns per pixel (MPP) in x and y"""
    slide = slide_handle
    props = slide.properties
    # Get microns per pixel from OpenSlide
    mppx = props.get(openslide.PROPERTY_NAME_MPP_X)
    mppy = props.get(openslide.PROPERTY_NAME_MPP_Y)
    # Fallback to standard TIFF resolution tags
    try:
        unit = props["tiff.ResolutionUnit"]
        unit_per_micron = {
            "centimeter": 1e4,  # 10k
            "inch": 25400,
        }
        if mppx is None:
            x_res = float(props["tiff.XResolution"])
            mppx = 1 / x_res * unit_per_micron[unit]
        if mppy is None:
            y_res = float(props["tiff.YResolution"])
            mppy = 1 / y_res * unit_per_micron[unit]
    except KeyError:
        KeyError("Could not determine microns per pixel")
    return np.array([float(mppx), float(mppy)])


def xy_choice(value_map, n=1):
    """Choose n (x, y) coordinates weighted using the input value map"""
    pmf = sumto1(value_map.astype(float))
    index = np.ndindex(value_map.shape)
    coord_range = np.arange(np.prod(value_map.shape))
    choices = np.random.choice(coord_range, size=n, replace=False, p=pmf.reshape(-1),)
    xys = np.array(list(index))[choices]
    return xys


def iterable(obj):
    """Return true if object (obj) is iterable"""
    return isinstance(obj, Iterable)


def batch(lst, n):
    """Yield successive n sized batches from an list (lst)"""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def join_lists(lists):
    return list(itertools.chain.from_iterable(lists))
