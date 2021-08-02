""""Methods of detecting tissue and background."""
from abc import ABC
import math

import numpy as np
from numpy import logical_and as land, logical_not as lnot, logical_or as lor
from openslide import OpenSlide
from skimage.color import rgb2hsv
from skimage.filters.rank import entropy
from skimage.morphology import (
    disk,
    binary_opening,
    binary_closing,
    remove_small_holes,
    remove_small_objects,
)


import utils


class Segmenter(ABC):
    def __init__(
        self, level=0, thumbnail_mpp=64, alpha_composite=True, debug=False,
    ):
        super().__init__()
        self.debug = debug
        self.level = level
        self.thumbnail_mpp = thumbnail_mpp
        self.alpha_composite = alpha_composite
        self.slide = None

    def thumbnail(self, force=False):
        """Get a thmbnail at a certain microns per pixel (mpp)

        Args:
          mpp (int): Desired pixels per micron (MPP). Defaults to 64.
                This is roughly equal to 450 pixels per inch.

        Returns: A Pillow (PIL) RGB Image.
        """
        if hasattr(self, "_m_thumb") and not force:
            return self._m_thumb
        slide = self.slide
        slide_mpp = utils.slide_mpp(slide)
        scale_factor = self.thumbnail_mpp / slide_mpp

        slide_dims = np.array(self.slide.dimensions)
        thumb_size = np.round(slide_dims / scale_factor)
        thumb_size = [int(x) for x in thumb_size]
        thumb = self.slide.get_thumbnail(tuple(thumb_size))
        self._m_thumb = thumb
        return self._m_thumb

    def simple_tissue_mask(
        self, thumb, threshold=225, dilation=1, min_object=10, max_hole=3
    ):
        """Simple tissue masking using a fixed threshold"""
        from skimage.morphology import (
            disk,
            remove_small_objects,
            remove_small_holes,
            binary_dilation,
        )

        thumb = thumb.convert("L")
        binary = np.array(thumb) < threshold
        mask = binary_dilation(binary, disk(dilation))
        mask = remove_small_objects(mask, min_object)
        mask = remove_small_holes(mask, max_hole)
        return mask

    @property
    def level_dimensions(self):
        if self.slide is None:
            raise Exception("Missing slide file handle")
        return self.slide.level_dimensions[self.level]

    def __call__(
        self, slide_path, n=1,
    ):
        """Return a number of sample regions"""
        self.slide = OpenSlide(str(slide_path))
        self.thumbnail(force=True)
        return self.get_mask()


class MultiThresholdSegmenter(Segmenter):
    """"Segments a slide based on hue, saturation and value of the slide thumbnail"""

    def __init__(
        self, thumbnail_mpp=32, morphology=True, alpha_composite=True, debug=False,
    ):
        super().__init__(thumbnail_mpp=thumbnail_mpp, debug=debug)
        self.morphology = morphology

    def get_mask(self, slide_path):
        """Return n xy coordinates in thumbnail space"""

        self.slide = OpenSlide(slide_path)
        thumb = self.thumbnail(force=True)

        if self.debug:
            from matplotlib import pyplot as plt

            plt.imshow(thumb)
            plt.title("Input Thumbnail")
            plt.show()

        np_thumb = np.array(thumb)

        hsv = rgb2hsv(np_thumb)
        hue, sat, val = (hsv[..., i] for i in range(3))

        if self.debug:
            plt.imshow(hue, cmap="hsv", vmin=0, vmax=1)
            plt.colorbar()
            plt.title("Hue")
            plt.show()

            plt.imshow(sat, vmin=0, vmax=1)
            plt.title("Saturation")
            plt.colorbar()
            plt.show()

            plt.imshow(val, cmap="gray", vmin=0, vmax=1)
            plt.title("Value")
            plt.colorbar()
            plt.show()

        white_val = lor(land(val > 0.88, sat < 0.2), land(val > 0.5, sat < 0.05))
        if self.debug:
            plt.imshow(white_val)
            plt.title("White Areas Val")
            plt.show()

        white = lor(np_thumb.min(axis=2) > 225, white_val)
        if self.debug:
            plt.imshow(white)
            plt.title("White Areas")
            plt.show()

            plt.imshow(lnot(white))
            plt.title("Not White Areas")
            plt.show()

        dark = lor(
            land(val < 0.25, hue < 0.775),
            land(val < 0.33, land(val <= np.percentile(val, 5), sat < 0.6)),
        )
        reds = land(lor(hue < 0.1, hue > 0.9), sat > 0.6)
        blues = land(land(hue > 0.45, hue < 0.75), sat > 0.6)
        if self.debug:
            plt.imshow(dark)
            plt.title("Dark Areas")
            plt.show()

            plt.imshow(reds)
            plt.title("Red Areas")
            plt.show()

            plt.imshow(blues)
            plt.title("Blue Areas")
            plt.show()

            plt.imshow(land(land(hue > 0.2, hue < 0.5), sat > 0.6))
            plt.title("Green Areas")
            plt.show()

        # scale = max(1, int(64 / self.thumbnail_mpp))
        # entropy_map = entropy(val, np.ones((scale * 8, scale * 8)))
        # entropy_mask = entropy_map > 4.7
        # entropy_hue = land(land(entropy_mask, hue > 0.6), hue < 0.95)

        # if self.debug:
        #     plt.imshow(entropy_map)
        #     plt.title(f"Entropy (Scale={scale})")
        #     plt.colorbar()
        #     plt.show()

        #     plt.imshow(entropy_mask)
        #     plt.title(f"Entropy Mask")
        #     plt.show()

        #     plt.imshow(entropy_hue)
        #     plt.title(f"Entropy Hue")
        #     plt.show()

        magenta = land(hue > 0.65, hue < 0.97)
        if self.debug:
            plt.imshow(magenta)
            plt.title("Magenta Hue")
            plt.show()

        # combined = land(
        #     land(
        #         land(lor(land(magenta, lnot(white)), entropy_hue), lnot(blues)),
        #         lnot(reds),
        #     ),
        #     lnot(dark),
        # )
        combined = land(
            land(land(land(magenta, lnot(white)), lnot(blues)), lnot(reds),),
            lnot(dark),
        )
        if self.debug:
            plt.imshow(combined)
            plt.title("Magenta Hue ∧ ¬White ∧ ¬Dark")
            plt.show()

        if self.morphology:
            scale = max(1, int(64 / self.thumbnail_mpp))
            combined = remove_small_holes(combined, scale)
            combined = binary_closing(combined, disk(scale // 2))
            combined = binary_opening(combined, disk(scale))
            combined = remove_small_holes(combined, scale)
            combined = remove_small_objects(combined, scale)

        return combined,thumb
