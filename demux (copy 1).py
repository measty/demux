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

from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from stainseparator import rgb2hed, hed2rgb
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
# Example IHC image

import staintools
import numpy as np

#%%
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
    def __init__(self):
        return None
    def fit(self,src_c,src_b,tgt_c,tgt_b):
        self.src_p = hist2percentile(src_c,np.linspace(0,1,255),src_b)
        self.tgt_p = hist2percentile(tgt_c,np.linspace(0,1,255),tgt_b)        
        return self
    def transform(self,src):
        return np.interp(src.ravel(),self.src_p,self.tgt_p).reshape(src.shape)
if __name__ == '__main__':
    #plt.close('all')
    #ihc_rgb = data.immunohistochemistry()
    from cpathreader import CPathReader
    from pathlib import Path

    snormalizer = staintools.StainNormalizer(method='vahadane')
    snormalizer.fit(staintools.LuminosityStandardizer.standardize(staintools.read_image(r'./target2.jpg')))
    
    slide_path = Path(r"./Mitosis_Ki67_IHC_and_IHC+HE_Trial_Asmaa/",
    "Case 2 Ki IHC+H&E.mrxs",)
    P = CPathReader(mpp = 0.2431,mask_mpp = 8,patch_size=4096)  
    N,cprinfo = P.WSI2SNPatches(slide_path,out_path='./temp',normalize = False)
    
    
    
    #%% Get stain-wise histograms
    #ihc_rgb0,cprinfo = P.WSI2RandomPatch(slide_path)
    from glob import glob
    from tqdm import tqdm
    plist = glob('./temp/'+slide_path.name+'/*.jpg')
    xy = [Path(f).name.split('.')[0].split('-') for f in plist]    
    
    n_jobs = -1
    bins = np.linspace(-2,2,num=1000)
    def gethist(f):
        ihc_rgb0 = imread(f)        
        ihc_rgb0 = np.array(ihc_rgb0,dtype=np.uint8)
        ihc_rgb = staintools.LuminosityStandardizer.standardize(ihc_rgb0)        
        ihc_hed = rgb2hed(ihc_rgb)
        h0 = ihc_hed[:, :, 0]
        e0 = ihc_hed[:, :, 1]
        d0 = ihc_hed[:, :, 2]
        # null = np.zeros_like(h0)
        # ihc_h = hed2rgb(np.stack((h0, null, null), axis=-1))
        # ihc_e = hed2rgb(np.stack((null, e0, null), axis=-1))
        # ihc_d = hed2rgb(np.stack((null, null, d0), axis=-1))
        
        nh=np.histogram(h0.flatten(),bins)[0]
        ne=np.histogram(e0.flatten(),bins)[0]
        nd=np.histogram(d0.flatten(),bins)[0]
        return (nh,ne,nd)
    Z = Parallel(n_jobs=n_jobs)(delayed(gethist)(f) for f in tqdm(plist, total=len(plist)))
    1/0
    nh,ne,nd = np.zeros(len(bins)-1),np.zeros(len(bins)-1),np.zeros(len(bins)-1)
    for f in tqdm(plist):
        ihc_rgb0 = imread(f)        
        ihc_rgb0 = np.array(ihc_rgb0,dtype=np.uint8)
        ihc_rgb = staintools.LuminosityStandardizer.standardize(ihc_rgb0)        
        ihc_hed = rgb2hed(ihc_rgb)
        h0 = ihc_hed[:, :, 0]
        e0 = ihc_hed[:, :, 1]
        d0 = ihc_hed[:, :, 2]
        # null = np.zeros_like(h0)
        # ihc_h = hed2rgb(np.stack((h0, null, null), axis=-1))
        # ihc_e = hed2rgb(np.stack((null, e0, null), axis=-1))
        # ihc_d = hed2rgb(np.stack((null, null, d0), axis=-1))
        
        nh+=np.histogram(h0.flatten(),bins)[0]
        ne+=np.histogram(e0.flatten(),bins)[0]
        nd+=np.histogram(d0.flatten(),bins)[0]
        
    #%% Do stain transfer
    plt.close('all')
    odir = './processed/'
    from skimage.io import imsave
    import os
    slide_path = Path(slide_path)
    output = Path(odir, slide_path.name)
    if not output.exists():
        if not output.parent.exists():
            output.parent.mkdir()
        output.mkdir()
        
    wh = 0.9
    we = 0.1
    sigma = 5.0
    hmin,hmax = hist2percentile(nh, 0.01,bins),hist2percentile(nh, 0.99,bins)
    emin,emax = hist2percentile(ne, 0.01,bins),hist2percentile(ne, 0.99,bins)
    dmin,dmax = hist2percentile(nd, 0.01,bins),hist2percentile(nd, 0.99,bins)
    D2H = GlobalHistMatcher().fit(nd,bins,nh,bins)
    D2E = GlobalHistMatcher().fit(nd,bins,ne,bins)
    thresh = hist2percentile(nd,0.99,bins)
    #idx = 30
    for f in tqdm(plist):#[idx:idx+1]
        ihc_rgb0 = imread(f)        
        ihc_rgb0 = np.array(ihc_rgb0,dtype=np.uint8)
        ihc_rgb = staintools.LuminosityStandardizer.standardize(ihc_rgb0)        
        ihc_hed = rgb2hed(ihc_rgb)
        h0 = ihc_hed[:, :, 0]
        e0 = ihc_hed[:, :, 1]
        d0 = ihc_hed[:, :, 2]
        null = np.zeros_like(h0)
        h = rescale_intensity(h0, out_range=(0, 1),
                          in_range=(hmin, hmax))
        e = rescale_intensity(e0, out_range=(0, 1),
                              in_range=(emin, emax))
        d = rescale_intensity(d0, out_range=(0, 1),
                              in_range=(dmin, dmax))
        zdh = np.dstack((h, e, d))
        h1 = gaussian_filter(h, sigma=sigma)
        e1 = gaussian_filter(e, sigma=sigma)
        #d1 = gaussian_filter(d, sigma=sigma)
        d2h = D2H.transform(d0)
        d2e = D2E.transform(d0)
        hrec = (h0+wh*h1*d2h)/(1+wh)#(h0+wh*h1*d0)/(1+wh)
        erec = (e0+we*e1*d2e)/(1+we)#(e0+we*e1*d0)/(1+we)
        rec = hed2rgb(np.stack((hrec, erec, null), axis=-1)) #HE Reconstruction
        hd = hed2rgb(np.stack((h0, null, d0), axis=-1)) #HD Reconstruction
        dt = d0>thresh #corresponding D-trhesholded image
        imsave(os.path.join(output,Path(f).name),rec)
        

#%%
    # d = rescale_intensity(d0, out_range=(0, 1),
    #                   in_range=(hist2percentile(nd,0.0,bins), hist2percentile(nd,0.99,bins)))
    fig, axes = plt.subplots(1, 3, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()    
    ax[0].imshow(ihc_rgb0)
    ax[0].set_title("Original image")        
    #ax[1].imshow(snormalizer.transform(rec))
    ax[1].imshow(rec)
    ax[1].set_title("Reconstructed")
    #ax[2].imshow(d0>hist2percentile(nd, 0.99,bins))
    ax[2].imshow(dt,cmap='gray')
    ax[2].set_title("D thresholded")
    
           