from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor, PatchExtractor
import numpy as np
from sklearn.decomposition import DictionaryLearning
from tiatoolbox.utils.misc import contrast_enhancer
from scipy import linalg
from joblib import Parallel, delayed
from stainseparator import combine_stains, separate_stains
from some_staintools import LuminosityStandardizer
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
import pickle
from tiatoolbox.wsicore.wsireader import WSIReader
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
# vipshome = r'C:\Users\meast\vips-dev-8.12\bin'
import os
# os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
import pyvips as vips

def get_luminosity_tissue_mask(img, threshold):
    """Get tissue mask based on the luminosity of the input image.

    Args:
        img (:class:`numpy.ndarray`):
            Input image used to obtain tissue mask.
        threshold (float):
            Luminosity threshold used to determine tissue area.

    Returns:
        tissue_mask (:class:`numpy.ndarray`):
            Binary tissue mask.

    Examples:
        >>> from tiatoolbox import utils
        >>> tissue_mask = utils.misc.get_luminosity_tissue_mask(img, threshold=0.8)

    """
    img = img.astype("uint8")  # ensure input image is uint8
    #img = contrast_enhancer(img, low_p=2, high_p=98)  # Contrast  enhancement
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_lab = img_lab[:, :, 0] / 255.0  # Convert to range [0,1].
    tissue_mask = l_lab < threshold

    # check it's not empty
    #if tissue_mask.sum() == 0:
    #    raise ValueError("Empty tissue mask computed.")

    return tissue_mask

def rgb2od(img):
    r"""Convert from RGB to optical density (:math:`OD_{RGB}`) space.

    .. math::
        RGB = 255 * exp^{-1*OD_{RGB}}

    Args:
        img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
            RGB image.

    Returns:
        :class:`numpy.ndarray`:
            Optical density (OD) RGB image.

    Examples:
        >>> from tiatoolbox.utils import transforms, misc
        >>> rgb_img = misc.imread('path/to/image')
        >>> od_img = transforms.rgb2od(rgb_img)

    """
    mask = img < 5
    img[mask] = 5
    return np.maximum(-1 * np.log(img / 255), 1e-6)

def is_image(I):
    """
    Is I an image.
    """
    if not isinstance(I, np.ndarray):
        return False
    if not I.ndim == 3:
        return False
    return True

def is_uint8_image(I):
    """
    Is I a uint8 image.
    """
    if not is_image(I):
        return False
    if I.dtype != np.uint8:
        return False
    return True

def dl_output_for_hed(dictionary):
    """Return correct value for H, E and D from dictionary learning output.

    Args:
        dictionary (:class:`numpy.ndarray`):
            :class:`sklearn.decomposition.DictionaryLearning` output

    Returns:
        :class:`numpy.ndarray`:
            With correctly ordered values for H, E and D.

    """
    hed_to_rgb = np.array([[0.458,0.814,0.356],[0.259,0.866,0.428],[0.269,0.568,0.778]])
    # return dictionary with rows sorted in order of H, E and D by similarity to
    # the rows of the above matrix
    
    # first, find the closest row to each dictionary row
    norms = np.zeros((3,3))
    for i in range(3):
        norms[i,:] = np.linalg.norm(dictionary[i,:] - hed_to_rgb, axis=1)
    # find which permutation of rows would give the smallest total norm
    # (i.e. the closest match to the hed_to_rgb matrix). Must use each row of
    # the dictionary exactly once.
    # the possible permutations are:
    perms = np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]])
    # find the permutation that gives the smallest total norm
    best_val = np.inf
    best_perm = None
    for perm in perms:
        val = norms[0,perm[0]] + norms[1,perm[1]] + norms[2,perm[2]]
        if val < best_val:
            best_val = val
            best_perm = perm
    # return the dictionary rows in the correct order
    #best_perm = np.argsort(dictionary[:,2])
    
    return dictionary[best_perm,:]

def hist2percentile(n,p,bins=None):
    return_scalar = False
    if not hasattr(p, "__len__"):
        return_scalar = True
        p = [p]
    z = np.cumsum(n)/np.sum(n)    
    def search(px):
        if px==1.0:
            px-=1e-16
        idx = np.searchsorted(z,px, side = 'right')#np.where(z>p)[0][0]    
        if bins is not None:
            B = (bins[:-1]+bins[1:])/2.0
            idx = B[idx]
        return idx
    r = np.array(list(map(search,p)))
    if return_scalar: r = r[0]
    return r

class GlobalHistMatcher:
    def __init__(self, src_p=None, tgt_p=None):
        self.src_p = src_p
        self.tgt_p = tgt_p
        return None
    def fit(self,src_c,src_b,tgt_c,tgt_b):
        self.src_p = hist2percentile(src_c,np.linspace(0,1,255),src_b)
        self.tgt_p = hist2percentile(tgt_c,np.linspace(0,1,255),tgt_b)        
        return self
    def transform(self,src):
        return np.interp(src.ravel(),self.src_p,self.tgt_p).reshape(src.shape)
    def get_stats(self):
        return [self.src_p, self.tgt_p] 

class VahadaneExtractor3d:
    """Vahadane stain extractor.

    Get the stain matrix as defined in:

    Vahadane, Abhishek, et al. "Structure-preserving color normalization
    and sparse stain separation for histological images."
    IEEE transactions on medical imaging 35.8 (2016): 1962-1971.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Args:
        luminosity_threshold (float):
            Threshold used for tissue area selection.
        regularizer (float):
            Regularizer used in dictionary learning.

    Examples:
        >>> from tiatoolbox.tools.stainextract import VahadaneExtractor
        >>> from tiatoolbox.utils.misc import imread
        >>> extractor = VahadaneExtractor()
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    def __init__(self, luminosity_threshold=0.6, regularizer=0.1):
        self.__luminosity_threshold = luminosity_threshold
        self.__regularizer = regularizer

    def get_stain_matrix(self, img):
        """Stain matrix estimation.

        Args:
            img (:class:`numpy.ndarray`):
                Input image used for stain matrix estimation

        Returns:
            :class:`numpy.ndarray`:
                Estimated stain matrix.

        """
        img = img.astype("uint8")  # ensure input image is uint8
        luminosity_threshold = self.__luminosity_threshold
        regularizer = self.__regularizer
        # convert to OD and ignore background
        tissue_mask = get_luminosity_tissue_mask(
            img, threshold=luminosity_threshold
        ).reshape((-1,))
        img_od = rgb2od(img).reshape((-1, 3))
        img_od = img_od[tissue_mask]

        # do the dictionary learning
        dl = DictionaryLearning(
            n_components=3,
            alpha=regularizer,
            transform_alpha=regularizer,
            fit_algorithm="lars",
            transform_algorithm="lasso_lars",
            positive_dict=True,
            max_iter=10,
            transform_max_iter=500,
        )
        dictionary = dl.fit_transform(X=img_od.T).T

        # order H, E and D.
        # H on first row.
        dictionary /= (np.linalg.norm(dictionary, axis=1)[:, None])
        dictionary = dl_output_for_hed(dictionary)

        return dictionary

class VirtualRestainer:
    """This class defines methods to virtually restain a WSI that has been 
    triple stained with H, E and an IHC , to give the appearance of a
    WSI stained with H and E only, or H and IHC only. The class estimates the stain
    matrix of the input image as an average of the stain matrices of large patches of
    the image. A dict of coupling coefficients between stains is provided that describes the
    extent to which two stains bind to the same tissue components. Uses histogram matching
    and these coupling matrices to virtually restain the image.
    """
    def __init__(self, img=None, coupling_coeffs={"wdh":0.7, "wde":0.2, "weh":0.2, "wed":0.5}, patch_size=4096, stride=4096, stain_extractor=VahadaneExtractor3d(), load_path=None, stains="HE"):
        self.img = img
        self.coupling_coeffs = coupling_coeffs
        self.stain_extractor = stain_extractor
        self.stains = stains
        self.patch_extractor = None
        self.patch_size = patch_size
        self.stride = stride
        if img:
            self.patch_extractor = SlidingWindowPatchExtractor(
                input_img=self.img,
                patch_size=(patch_size, patch_size),
                stride=(stride, stride),
                input_mask='otsu', #'morphological',
                min_mask_ratio=0.75,
                within_bound=True,
            )
        # default stain mat and inverse
        # self.stain_mat = np.array([[0.458,0.814,0.356],[0.259,0.866,0.428],[0.269,0.568,0.778]])
        
        # self.stain_mat = np.array([[0.65, 0.70, 0.29],
        #                   [0.07, 0.99, 0.11],
        #                   [0.27, 0.57, 0.78]])
        # self.stain_mat = np.array([[0.590, 0.73, 0.35],  #PHH3, Qupath
        #                   [0.19, 0.9, 0.41],
        #                   [0.5, 0.615, 0.6]])
        self.stain_mat = np.array([[0.5207, 0.7412, 0.3322],
                            [0.1816, 0.9181, 0.4228],
                            [0.2083, 0.5305, 0.8307]])  #PHH3 stain learner
        self.stain_mat_inv = np.linalg.inv(self.stain_mat)
        if load_path is not None:
            # load learned matchers, sm, etc from file
            self.load_info(load_path)

    def _process_patch(self, patch):
        """Process a patch of the input image.

        Args:
            patch (:class:`numpy.ndarray`):
                Input patch.

        Returns:
            :class:`numpy.ndarray`:
                Stain matrix of the input patch.
            float: luminosity standardisation level.

        """
        stain_mat = self.stain_extractor.get_stain_matrix(patch)
        assert is_uint8_image(patch), "Image should be RGB uint8."
        I_LAB = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
        L_float = I_LAB[:, :, 0].astype(float)
        p = np.percentile(L_float, 98)
        return stain_mat, p

    def separate_stains(self, img, mat):
        """Estimate concentration matrix given an image and stain matrix.
        Args:
            img (:class:`numpy.ndarray`):
                Input image.
            stain_matrix (:class:`numpy.ndarray`):
                Stain matrix for haematoxylin and eosin stains.
        Returns:
            numpy.ndarray:
                Stain concentrations of input image.
        """
        od = rgb2od(img).reshape((-1, 3))
        x, _, _, _ = np.linalg.lstsq(self.stain_mat.T, od.T, rcond=-1)
        return x.T.reshape(img.shape)

    def combine_stains(self, img, mat):
        """Transform an image.

        Args:
            img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                stain space source image.

        Returns:
            :class:`numpy.ndarray`:
                RGB stain normalized image.

        """
        im_shape = img.shape
        img = img.reshape((-1, 3))
        max_c_source = np.percentile(img, 99, axis=0).reshape((1, 3))
        img *= np.array((self.hrange[1], self.erange[1], self.drange[1])) / np.clip(max_c_source, 1E-6, None)
        trans = 255 * np.exp(
            -1 * np.dot(img, self.stain_mat)
        )

        # ensure between 0 and 255
        trans[trans > 255] = 255
        trans[trans < 0] = 0

        return trans.reshape(im_shape).astype(np.uint8)

    def get_stain_matrix(self, n_jobs=1):
        """Estimate the stain matrix of the input image as an average of the stain matrices of
        large patches of the image.

        Args:
            patch_size (int):
                Size of the patch to extract.
            stride (int):
                Stride of the patch extraction.
            n_jobs (int):
                Number of jobs to use for patch extraction.

        Returns:
            :class:`numpy.ndarray`:
                Estimated stain matrix.

        """
        res = Parallel(n_jobs=n_jobs)(delayed(self._process_patch)(patch) for patch in tqdm(self.patch_extractor, total=len(self.patch_extractor)))
        p = np.mean([r[1] for r in res])
        stain_mats = np.concatenate([r[0][:, :, None] for r in res], axis=2)
        #stain_mats = stain_mats[~np.isnan(stain_mats).any(axis=(1, 2))]
        stain_mat = np.mean(stain_mats, axis=2)
        stain_mat /= (np.linalg.norm(stain_mat, axis=1)[:, None])
        self.stain_mat = stain_mat
        self.stain_mat_inv = np.linalg.inv(stain_mat)
        self.lum_p = p
        return stain_mat, p

    def _get_hist(self, patch, n_bins=1000, lum_norm=False):
        """Transform the patch into HED space and return the histograms of each channel."""
        bins = np.linspace(-2,2,num=1000)
        if lum_norm:
            patch = LuminosityStandardizer.standardize(patch)   #should we do this?
        
        hed = self.separate_stains(patch, self.stain_mat_inv).reshape((-1, 3))
        tissue_mask = get_luminosity_tissue_mask(
            patch, threshold=0.75
        ).reshape((-1,))
        nh=np.histogram(hed[tissue_mask, 0].flatten(),bins)[0]
        ne=np.histogram(hed[tissue_mask, 1].flatten(),bins)[0]
        nd=np.histogram(hed[tissue_mask, 2].flatten(),bins)[0]
        return (nh,ne,nd)       

    def get_histogram_matchers(self, n_jobs=1, n_bins=1000):
        """Get the histograms for each stain summed over all patches and calculate
        histogram matchers between pairs of stains."""
        #Z = Parallel(n_jobs=n_jobs)(delayed(self._get_hist)(f) for f in tqdm(self.patch_extractor, total=len(self.patch_extractor)))
        Z = [self._get_hist(f.copy()) for f in tqdm(self.patch_extractor, total=len(self.patch_extractor))]
        Zs = np.sum(Z,axis=0)
        nh,ne,nd = Zs[0],Zs[1],Zs[2]

        bins = np.linspace(-2,2,num=n_bins)

        self.hrange = (hist2percentile(nh, 0.01,bins),hist2percentile(nh, 0.99,bins))
        self.erange = (hist2percentile(ne, 0.01,bins),hist2percentile(ne, 0.99,bins))
        self.drange = (hist2percentile(nd, 0.01,bins),hist2percentile(nd, 0.99,bins))
        self.D2H = GlobalHistMatcher().fit(nd,bins,nh,bins)
        self.D2E = GlobalHistMatcher().fit(nd,bins,ne,bins)
        self.E2D = GlobalHistMatcher().fit(ne,bins,nd,bins)
        self.E2H = GlobalHistMatcher().fit(ne,bins,nh,bins)
        self.H2D = GlobalHistMatcher().fit(nh,bins,nd,bins)

    def __call__(self, tile):
        return self.restain_tile(tile, stains=self.stains, sigma=5, naive=False)

    def restain_tile(self, tile, sigma=5.0, stains='HE', lum_norm=False, naive=False):
        """Restain a tile of the input image. The tile is transformed into HED space and
        the unwanted stain is removed, adjusting the remaining stains according to the 
        coupling coeffs and histogram matchers.
        """
        if stains == "HED":
            return tile
        tile = tile[:,:,0:3]
        if lum_norm:
            tile = LuminosityStandardizer.standardize(tile, 98)

        tissue_mask = get_luminosity_tissue_mask(
            tile, threshold=0.8
        )
        if np.sum(tissue_mask) == 0:
            return tile

        tile=tile.copy() #stop it being read only
        tile_hed = self.separate_stains(tile, self.stain_mat_inv)
            
        h0 = tile_hed[:, :, 0]
        e0 = tile_hed[:, :, 1]
        d0 = tile_hed[:, :, 2]
        #null = np.zeros_like(h0)
        wdh = self.coupling_coeffs['wdh']
        wde = self.coupling_coeffs['wde']
        weh = self.coupling_coeffs['weh']
        wed = self.coupling_coeffs['wed']

        # h0 = rescale_intensity(h0, out_range=self.hrange,
        #                     in_range=self.hrange)
        # e0 = rescale_intensity(e0, out_range=self.erange,
        #                     in_range=self.erange)
        # d0 = rescale_intensity(d0, out_range=self.drange,
        #                     in_range=self.drange)

        if not naive:
            # in naive method, set all couplings to zero. We just throw away the stain 
            # we dont want and reconstruct the rgb, so below not needed     
            h = rescale_intensity(h0, out_range=(0, 1),
                                in_range=self.hrange)
            e = rescale_intensity(e0, out_range=(0, 1),
                                in_range=self.erange)
            d = rescale_intensity(d0, out_range=(0, 1),
                                in_range=self.drange)

            if sigma>0:
                h1 = gaussian_filter(h, sigma=sigma)[tissue_mask]
                e1 = gaussian_filter(e, sigma=sigma)[tissue_mask]
                d1 = gaussian_filter(d, sigma=sigma)[tissue_mask]
            else:
                h1=h[tissue_mask]
                e1=e[tissue_mask]
                d1=d[tissue_mask]

        d0 = d0[tissue_mask]
        h0 = h0[tissue_mask]
        e0 = e0[tissue_mask]
        blank = np.zeros_like(h0)

        if stains=='HE':
            if naive:
                rec = self.combine_stains(np.stack((h0, e0, blank), axis=-1), self.stain_mat)
            else:
                d2h = self.D2H.transform(d0)
                d2e = self.D2E.transform(d0)
                # hrec = (h0+wdh*h1[tissue_mask]*d2h)/(1+wdh)#(h0+wh*h1*d0)/(1+wh)
                # erec = (e0+wde*e1[tissue_mask]*d2e)/(1+wde)#(e0+we*e1*d0)/(1+we)

                smask_h = h1 + wdh * 1.5 * (np.abs(d1- np.percentile(d1, 60)))
                smask_e = e1 + wde * 1.5 * (np.abs(d1- np.percentile(d1, 60)))

                hrec = np.clip((h0+wdh*smask_h*d2h), 0, 1)#/(1+wdh)#(h0+wh*h1*d0)/(1+wh)
                erec = np.clip((e0+wde*smask_e*d2e), 0, 1)#/(1+wde)#(e0+we*e1*d0)/(1+we)
                # hrec = gaussian_filter(hrec, sigma=1.0)
                # erec = gaussian_filter(erec, sigma=1.0)
                rec = self.combine_stains(np.stack((hrec, erec, blank), axis=-1), self.stain_mat)
        elif stains=='HD':    
            #hdh,hdd = h0,d0
            if naive:
                rec = self.combine_stains(np.stack((h0, blank, d0), axis=-1), self.stain_mat)           
            else:
                e2h = self.E2H.transform(e0)
                e2d = self.E2D.transform(e0)
                # hrec = (h0+weh*h1[tissue_mask]*e2h)/(1+weh)
                # drec = (d0+wed*d1[tissue_mask]*e2d)/(1+wed)

                smask_h = h1 + weh * 1.5 * (np.abs(e1- np.percentile(e1, 60)))
                smask_d = d1 + wed * 1.5 * (np.abs(e1- np.percentile(e1, 60)))

                hrec = np.clip((h0+weh*smask_h*e2h), 0, 1)#/(1+weh)
                drec = np.clip((d0+wed*smask_d*e2d), 0, 1)#/(1+wed)
                rec = self.combine_stains(np.stack((hrec, blank, drec), axis=-1), self.stain_mat)
        elif stains=='D':
            # experimental
            if naive:
                rec = self.combine_stains(np.stack((0.15*h0, blank, d0), axis=-1), self.stain_mat)
            else:
                whd=0.2
                e2d = self.E2D.transform(e0)
                h2d = self.H2D.transform(h0)
                drec=(d0+wed*d1[tissue_mask]*e2d+whd*d1[tissue_mask]*h2d)/(1+wed+whd)
                odsum=(e0+d0+h0)/5
                drec=drec+odsum
                rec = self.combine_stains(np.stack((0.15*h0, blank, drec), axis=-1), self.stain_mat)
        else:
            print('unknown stain')
        
        #rec = (rec*255).astype('uint8')
        tile[tissue_mask,:] = rec
        return tile

    def save_info(self, filename):
        """Save the stain matrix, histogram matchers, ranges and coupling coeffs to a pickle file."""
        matcher_stats = {"D2H": self.D2H.get_stats(), "D2E": self.D2E.get_stats(), "E2D": self.E2D.get_stats(), "E2H": self.E2H.get_stats()}
        with open(filename, 'wb') as f:
            pickle.dump((self.stain_mat, self.stain_mat_inv, matcher_stats, self.hrange, self.erange, self.drange, self.coupling_coeffs), f)

    def load_info(self, filename):
        """Load the stain matrix, histogram matchers, ranges and coupling coeffs from a pickle file."""
        with open(filename, 'rb') as f:
            self.stain_mat, self.stain_mat_inv, matcher_stats, self.hrange, self.erange, self.drange, self.coupling_coeffs = pickle.load(f)
        self.D2H = GlobalHistMatcher(*matcher_stats["D2H"])
        self.D2E = GlobalHistMatcher(*matcher_stats["D2E"])
        self.E2D = GlobalHistMatcher(*matcher_stats["E2D"])
        self.E2H = GlobalHistMatcher(*matcher_stats["E2H"])

    def save_restained_WSI(self, filename, sigma=5.0, stains=['HE'], lum_norm=False):
        """Restain the whole slide image patch by patch, assemble the whole slide image 
        on disk as it is too large to fit in memory, and use pyvips to save the image
        as a pyramidal .tiff file."""
        filename = Path(filename)
        for stain in stains:
            tmp_path = filename.parent / (filename.stem + '_tmp_.npy')
            wsi = WSIReader.open(self.img)
            canvas_shape = wsi.info.slide_dimensions[::-1]
            mpp = wsi.info.mpp
            out_ch = 3
            self.patch_extractor = SlidingWindowPatchExtractor(
                input_img=self.img,
                patch_size=(self.patch_size, self.patch_size),
                stride=(self.stride, self.stride),
                input_mask='otsu', #'morphological',
                min_mask_ratio=0.1, # only discard patches with very low tissue content
                within_bound=True,
            ) 

            locs = self.patch_extractor.coordinate_list[:, :2]

            cum_canvas = np.lib.format.open_memmap(
                tmp_path,
                mode="w+",
                shape=tuple(canvas_shape) + (out_ch,),
                dtype=np.uint8,
            )
            cum_canvas[:] = 240

            for i, tile in tqdm(enumerate(self.patch_extractor), total=len(self.patch_extractor)):
                rec = self.restain_tile(tile, sigma=sigma, stains=stain, lum_norm=lum_norm)
                x, y = locs[i]
                if y+rec.shape[0] > canvas_shape[0] or x+rec.shape[1] > canvas_shape[1]:
                    print("meep")
                cum_canvas[y:y + self.patch_size, x:x + self.patch_size, :] = rec

            # make a vips image and save it as a pyramidal tiff
            #height, width, bands = cum_canvas.shape
            #linear = cum_canvas.reshape(width * height * bands)
            vips_img = vips.Image.new_from_memory(
                cum_canvas.tobytes(),
                canvas_shape[1],
                canvas_shape[0],
                out_ch,
                "uchar"
            )
            # set resolution metadata - tiffsave expects res in pixels per mm regardless of resunit
            vips_img.tiffsave(filename.with_stem(filename.stem + f"_{stain}"), tile=True, pyramid=True, compression="jpeg", Q=85, bigtiff=True, xres=1000/mpp[0], yres=1000/mpp[1], resunit="cm", tile_width=512, tile_height=512)
            # close memmap and clean up
            cum_canvas._mmap.close()
            del cum_canvas
            os.remove(tmp_path)

if __name__ == '__main__':
    slides_path = Path(r"/media/u2071810/Data/Multiplexstaining/Asmaa Multiplex Staining")
    slides = list(slides_path.glob('*PHH3_HE.svs'))
    for slide in slides[:5]:
        if "only" in slide.stem or (slide.parent / (slide.stem + '_info3.pkl')).exists():
            continue
        try:
            restainer = VirtualRestainer(slide)
            #restainer.get_stain_matrix(n_jobs=1)
            restainer.get_histogram_matchers(n_jobs=4)
            restainer.save_info(slide.parent / (slide.stem + '_info3.pkl'))
        except:
            print(f"Failed to restain {slide.stem}")
            continue
        if False:
            # restain a few patches to check the results
            for i, patch in enumerate(restainer.patch_extractor):
                rec = restainer.restain_tile(patch, sigma=5.0, stains='HD', lum_norm=False)
                naive_rec = restainer.restain_tile(patch, sigma=5.0, stains='HD', lum_norm=False, naive=True)
                # plot patch and both reconstructed patches side by side
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(patch)
                ax[1].imshow(rec)
                ax[2].imshow(naive_rec)
                plt.show()
                if i == 5:
                    break





        








