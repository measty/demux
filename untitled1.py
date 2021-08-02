#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 13:45:16 2021

@author: u1876024
"""


# -*- coding: utf-8 -*-
"""
Created on Tue May 25 12:27:53 2021

@author: fayya
"""


import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2hed, hed2rgb
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
# Example IHC image

import staintools
import numpy as np

csum = lambda z: np.cumsum(z)[:-1]
dsum = lambda z: np.cumsum(z[::-1])[-2::-1]
argmax = lambda x, f: np.mean(x[:-1][f == np.max(f)])  # Use the mean for ties.
clip = lambda z: np.maximum(1e-30, z)

def preliminaries(n, x):
  """Some math that is shared across multiple algorithms."""
  assert np.all(n >= 0)
  x = np.arange(len(n), dtype=n.dtype) if x is None else x
  assert np.all(x[1:] >= x[:-1])
  w0 = clip(csum(n))
  w1 = clip(dsum(n))
  p0 = w0 / (w0 + w1)
  p1 = w1 / (w0 + w1)
  mu0 = csum(n * x) / w0
  mu1 = dsum(n * x) / w1
  d0 = csum(n * x**2) - w0 * mu0**2
  d1 = dsum(n * x**2) - w1 * mu1**2
  return x, w0, w1, p0, p1, mu0, mu1, d0, d1

def GHT(n, x=None, nu=0, tau=0, kappa=0, omega=0.5):
  assert nu >= 0
  assert tau >= 0
  assert kappa >= 0
  assert omega >= 0 and omega <= 1
  x, w0, w1, p0, p1, _, _, d0, d1 = preliminaries(n, x)
  v0 = clip((p0 * nu * tau**2 + d0) / (p0 * nu + w0))
  v1 = clip((p1 * nu * tau**2 + d1) / (p1 * nu + w1))
  f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa *      omega)  * np.log(w0)
  f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
  return argmax(x, f0 + f1), f0 + f1

def Otsu(n, x=None):
  """Otsu's method."""
  x, w0, w1, _, _, mu0, mu1, _, _ = preliminaries(n, x)
  o = w0 * w1 * (mu0 - mu1)**2
  return argmax(x, o), o

def Otsu_equivalent(n, x=None):
  """Equivalent to Otsu's method."""
  x, _, _, _, _, _, _, d0, d1 = preliminaries(n, x)
  o = np.sum(n) * np.sum(n * x**2) - np.sum(n * x)**2 - np.sum(n) * (d0 + d1)
  return argmax(x, o), o

def MET(n, x=None):
  """Minimum Error Thresholding."""
  x, w0, w1, _, _, _, _, d0, d1 = preliminaries(n, x)
  ell = (1 + w0 * np.log(clip(d0 / w0)) + w1 * np.log(clip(d1 / w1))
      - 2 * (w0 * np.log(clip(w0))      + w1 * np.log(clip(w1))))
  return argmax(x, -ell), ell  # argmin()

def wprctile(n, x=None, omega=0.5):
  """Weighted percentile, with weighted median as default."""
  assert omega >= 0 and omega <= 1
  x, _, _, p0, p1, _, _, _, _ = preliminaries(n, x)
  h = -omega * np.log(clip(p0)) - (1. - omega) * np.log(clip(p1))
  return argmax(x, -h), h  # argmin()

def im2hist(im, zero_extents=False):
  # Convert an image to grayscale, bin it, and optionally zero out the first and last bins.
  max_val = np.iinfo(im.dtype).max
  x = np.arange(max_val+1)
  e = np.arange(-0.5, max_val+1.5)
  assert len(im.shape) in [2, 3]
  im_bw = np.amax(im[...,:3], -1) if len(im.shape) == 3 else im
  n = np.histogram(im_bw, e)[0]
  if zero_extents:
    n[0] = 0
    n[-1] = 0
  return n, x, im_bw

def binarize(im):    
    n, x, im_bw = im2hist(im)
    MET(n, x=None)


def HED2HEM(ihc_rgb0,wh=0.0,we=1.0,sigma = 3):

    ihc_rgb = staintools.LuminosityStandardizer.standardize(ihc_rgb0)
    
    ihc_hed = rgb2hed(ihc_rgb)
    h0 = ihc_hed[:, :, 0]
    e0 = ihc_hed[:, :, 1]
    d0 = ihc_hed[:, :, 2]
    # Create an RGB image for each of the stains
    null = np.zeros_like(h0)
    ihc_h = hed2rgb(np.stack((h0, null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, e0, null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, d0), axis=-1))
    
    # Rescale hematoxylin and DAB channels
    h = rescale_intensity(h0, out_range=(0, 1),
                          in_range=(np.percentile(h0, 0.0), np.percentile(h0, 99)))
    e = rescale_intensity(e0, out_range=(0, 1),
                          in_range=(np.percentile(e0, 0.0), np.percentile(e0, 99)))
    d = rescale_intensity(d0, out_range=(0, 1),
                          in_range=(np.percentile(d0, 1.0), np.percentile(d0, 99)))
    # recombine to get 'flourescent' image
    zdh = np.dstack((h, e, d))
    # reconstruct
    h1 = gaussian_filter(h, sigma=sigma)
    e1 = gaussian_filter(e, sigma=sigma)
    #d1 = gaussian_filter(d, sigma=sigma)
    
    hrec = (h0+wh*h1*d0)/(1+wh)
    erec = (e0+we*e1*d0)/(1+we)
    
    #hrec = gaussian_filter(hrec, sigma = 1.0)
    #erec = gaussian_filter(erec, sigma = 1.0)
    
    
    rec = hed2rgb(np.stack((hrec, erec, null), axis=-1))
    rec = np.array(255*rec,dtype=np.uint8)
    rec = staintools.LuminosityStandardizer.standardize(rec)
    
    return rec, zdh, (ihc_h,ihc_e,ihc_d)

if __name__ == '__main__':
    #plt.close('all')
    #ihc_rgb = data.immunohistochemistry()
    from cpathreader import CPathReader
    from pathlib import Path


    bdir = r'C:\Users\fayya\OneDrive - University of Warwick\Desktop\mixstain'
    from glob import glob
    f = glob(bdir+'\*.tif')[5]
    #ihc_rgb = imread(f)
    ihc_rgb0 = imread(r'C:\\ML\\wsiclust-backup\\problem.jpg')
    #ihc_rgb0 = imread(r'C:\Users\fayya\Downloads\AIF_8.tif')#('vf.jpg')#
    #ihc_rgb = imread(r'C:\Users\fayya\OneDrive - University of Warwick\Desktop\mixstain\AIF-2(1) (1).tif')
    #ihc_rgb = imread(r'C:\Users\fayya\OneDrive - University of Warwick\Desktop\stpatch.jpg')
    # Separate the stains from the IHC image
    #rec, zdh, (ihc_h,ihc_e,ihc_d) = HED2HEM(ihc_rgb, wh = 0.2, we = 0.2, sigma = 5.0)
    
    #from skimage.exposure import match_histograms
    slide_path = Path(r"C:\Users\fayya\Downloads\Mitosis Ki67 IHC and IHC+HE Trial Asmaa",
    "Case 1 Ki IHC+H&E.mrxs",)
    P = CPathReader(mpp = 0.2431,mask_mpp = 16,patch_size=4096)  
    #%%
    ihc_rgb0,cprinfo = P.WSI2RandomPatch(slide_path)
    ihc_rgb0 = np.array(ihc_rgb0,dtype=np.uint8)

    #%%
    plt.close('all')
    wh = 0.5
    we = 0.1
    sigma = 5.0
    #reference = imread()
    #ihc_rgb = match_histograms(ihc_rgb, reference, multichannel=True)
    # stain_target = r'C:\Users\fayya\OneDrive - University of Warwick\Desktop\mixstain\AIF_8 (2).tif'
    
    # snormalizer = staintools.StainNormalizer(method='vahadane')
    # snormalizer.fit(staintools.read_image(stain_target))
    # ihc_rgb = snormalizer.transform(ihc_rgb)
    
    ihc_rgb = staintools.LuminosityStandardizer.standardize(ihc_rgb0)
    
    ihc_hed = rgb2hed(ihc_rgb)
    h0 = ihc_hed[:, :, 0]
    e0 = ihc_hed[:, :, 1]
    d0 = ihc_hed[:, :, 2]
    # Create an RGB image for each of the stains
    null = np.zeros_like(h0)
    ihc_h = hed2rgb(np.stack((h0, null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, e0, null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, d0), axis=-1))
    
    # Rescale hematoxylin and DAB channels
    h = rescale_intensity(h0, out_range=(0, 1),
                          in_range=(np.percentile(h0, 0.0), np.percentile(h0, 99)))
    e = rescale_intensity(e0, out_range=(0, 1),
                          in_range=(np.percentile(e0, 0.0), np.percentile(e0, 99)))
    d = rescale_intensity(d0, out_range=(0, 1),
                          in_range=(np.percentile(d0, 0.0), np.percentile(d0, 100)))
    # recombine to get 'flourescent' image
    zdh = np.dstack((h, e, d))
    # reconstruct
    h1 = gaussian_filter(h, sigma=sigma)
    e1 = gaussian_filter(e, sigma=sigma)
    #d1 = gaussian_filter(d, sigma=sigma)
    
    hrec = (h0+wh*h1*d0)/(1+wh)
    erec = (e0+we*e1*d0)/(1+we)
    
    #hrec = gaussian_filter(hrec, sigma = 1.0)
    #erec = gaussian_filter(erec, sigma = 1.0)
    
    
    rec = hed2rgb(np.stack((hrec, erec, null), axis=-1))
    rec = np.array(255*rec,dtype=np.uint8)
    rec = staintools.LuminosityStandardizer.standardize(rec)

    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()
    
    ax[0].imshow(ihc_rgb0)
    ax[0].set_title("Original image")
    
    ax[1].imshow(ihc_h)
    ax[1].set_title("Hematoxylin")
    
    ax[2].imshow(ihc_e)
    ax[2].set_title("Eosin")  
    
    ax[3].imshow(ihc_d,cmap='gray')
    ax[3].set_title("DAB")
    
    for a in ax.ravel():
        a.axis('off')
    
    fig.tight_layout()
    
    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()
    
    ax[0].imshow(ihc_rgb0)
    ax[0].set_title("Original image")
    
    ax[1].imshow(h1,cmap='gray')
    ax[1].set_title("Hematoxylin")
    
    ax[2].imshow(e1,cmap='gray')
    ax[2].set_title("Eosin")  
    
    ax[3].imshow(d0,cmap='gray')
    ax[3].set_title("DAB")
    
    for a in ax.ravel():
        a.axis('off')
    
    fig.tight_layout()
    
    fig = plt.figure()
    axis = plt.subplot(1, 1, 1, sharex=ax[0], sharey=ax[0])
    axis.imshow(zdh)
    axis.set_title('Stain-separated image (rescaled)')
    axis.axis('off')
    plt.show()
    
    
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()
    
    ax[0].imshow(ihc_rgb0)
    ax[0].set_title("Original image")
    
    ax[1].imshow(rec)
    ax[1].set_title("Reconstructed")
    #%%
    d = rescale_intensity(d0, out_range=(0, 1),
                      in_range=(np.percentile(d0, 0.0), np.percentile(d0, 99)))
    d8 = np.array(255*d,dtype = np.uint8)
    n, x, im_bw = im2hist(d8)    
    zz,_=MET(n,x)
    plt.figure();plt.imshow(d8>zz,cmap='gray')