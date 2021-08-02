import numpy as np
from openslide import OpenSlide

from segmenters import MultiThresholdSegmenter
import utils


class SimplePatchExtractor:
    def __init__(
        self,
        slide_path: str,
        patch_size: int,
        mpp=0.5,
        mask_mpp=32,
        tissue_threshold=0.5,
        stride=1.0,
    ):
        super().__init__()
        self.slide_path = slide_path
        self.slide = OpenSlide(str(self.slide_path))
        self.patch_size = patch_size
        self.mpp = mpp
        self.stride = stride
        self.index = np.ndindex(tuple(self.shape))
        self.mask_mpp = mask_mpp
        segmenter = MultiThresholdSegmenter(thumbnail_mpp=mask_mpp)
        
        self.mask,self.thumb = segmenter.get_mask(str(self.slide_path))
        self.threshold = tissue_threshold

    @property
    def scale_factor(self):
        """Scaling from mpp to level 0"""
        return self.mpp / utils.slide_mpp(self.slide)

    @property
    def stride_px(self):
        """Step between patches in x and y in pixels at level 0"""
        stride_px = self.patch_size * self.scale_factor
        stride_px *= self.stride
        stride_px = tuple(np.round(stride_px).astype(int))
        return stride_px

    @property
    def shape(self):
        return (np.array(self.slide.dimensions) // self.stride_px).astype(int)

    def __len__(self):
        return np.prod(self.shape)

    def __iter__(self):
        self.index = np.ndindex(tuple(self.shape))
        return self
        
    def getPatch(self,i,j):
        # x0, y0 is the origin of the region at level 0, x1, y1 is the opposite corner
        x0, y0 = np.array([i, j]) * self.stride_px
        x1, y1 = np.array([i + 1, j + 1]) * self.stride_px
        # ti0, tj0 etc. are the corresponding points in mask indexes
        ti0, tj0 = utils.rescale_coords(
            (x0, y0), self.slide.dimensions, self.mask.shape[::-1],
        ).astype(int)
        ti1, tj1 = utils.rescale_coords(
            (x1, y1), self.slide.dimensions, self.mask.shape[::-1],
        ).astype(int)

        tissue_area = np.mean(self.mask[tj0:tj1, ti0:ti1])
        if tissue_area < self.threshold:
            region = None
            return (x0, y0), tissue_area, region

        # TODO: Could be enhanced by finding the best level for downsampling
        region = self.slide.read_region(location=(x0, y0), level=0, size=self.stride_px)
        region = utils.alpha_composite(region)
        region = region.resize([self.patch_size] * 2)
        return (x0, y0), tissue_area, region
    
        
    def getRandomTissuePatch(self,attempts=10):
        """
        Returns a randomly selected tissue patch         

        Parameters
        ----------
        attempts : Positive INT, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        (x0,y0) : TYPE
            DESCRIPTION: Coordinates in the image.
        TYPE
            DESCRIPTION.
        tissue_area : TYPE
            DESCRIPTION.
        region : TYPE
            DESCRIPTION.

        """
        region = None
        n = 0
        while region is None and n<attempts:
            i,j = np.random.randint(0,self.shape[0]),np.random.randint(0,self.shape[1])
            (x0, y0), tissue_area, region = self.getPatch(i,j)
        return (x0, y0), tissue_area, region
         

    def __next__(self):
        i, j = next(self.index)
        return self.getPatch(i,j)

