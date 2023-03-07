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
import os
vipshome = r'C:\Users\meast\vips-dev-8.12\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

from PIL import Image
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import data
from stainseparator import rgb2hed, hed2rgb, combine_stains, separate_stains, rgb_from_hed, hed_from_rgb
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
# Example IHC image
from skimage.morphology import disk
from skimage.filters import median
from some_staintools import LuminosityStandardizer
import numpy as np
c_rgb_from_hed = np.array([[0.458,0.814,0.356],[0.259,0.866,0.428],[0.269,0.568,0.778]])#before lum adj
#c_rgb_from_hed = np.array([[0.412,0.829,0.377],[0.223,0.884,0.411],[0.253,0.631,0.734]])#after lum adj
# c_rgb_from_hed = np.array([[0.399,0.855,0.33],[0.17,0.921,0.35],[0.269,0.568,0.778]])#
#c_rgb_from_hed = np.array([[0.458,0.814,0.356],[0.17,0.921,0.35],[0.269,0.568,0.778]])#
#c_rgb_from_hed = rgb_from_hed
c_hed_from_rgb = np.linalg.inv(c_rgb_from_hed)
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

def restain_tile(ihc_rgb0, wdh=0.7, wde=0.4, weh=0.8, wed=0.8, sigma=5.0, stains='HE', matchers=None, lev=10):
        #ihc_rgb0 = imread(f)        
        #ihc_rgb0 = np.array(ihc_rgb0,dtype=np.uint8)
        print(f'input args are: wdh={wdh}, wde={wde}, weh={weh}, wed={wed}, stains={stains}')
        ihc_rgb = ihc_rgb0[:,:,0:3]
        #ihc_rgb = staintools.LuminosityStandardizer.standardize(ihc_rgb0)
        if lev>6 and lev<9:
            ihc_rgb = LuminosityStandardizer.standardize(ihc_rgb,98)      
        ihc_hed = separate_stains(ihc_rgb, c_hed_from_rgb)#rgb2hed(ihc_rgb)
        h0 = ihc_hed[:, :, 0]
        e0 = ihc_hed[:, :, 1]
        d0 = ihc_hed[:, :, 2]
        null = np.zeros_like(h0)
        
        h = rescale_intensity(h0, out_range=(0, 1),
                          in_range=matchers['hmm'])
        e = rescale_intensity(e0, out_range=(0, 1),
                              in_range=matchers['emm'])
        d = rescale_intensity(d0, out_range=(0, 1),
                              in_range=matchers['dmm'])
                            
        #h=h0
        #e=e0
        #d=d0
        zdh = np.dstack((h, e, d))
        if sigma>0:
            h1 = gaussian_filter(h, sigma=sigma)
            e1 = gaussian_filter(e, sigma=sigma)
            d1 = gaussian_filter(d, sigma=sigma)
        else:
            h1=h
            e1=e
            d1=d
        if stains=='HE':
            d2h = matchers['D2H'].transform(d0)
            d2e = matchers['D2E'].transform(d0)
            hrec = (h0+wdh*h1*d2h)/(1+wdh)#(h0+wh*h1*d0)/(1+wh)
            erec = (e0+wde*e1*d2e)/(1+wde)#(e0+we*e1*d0)/(1+we)
            # hrec = gaussian_filter(hrec, sigma=1.0)
            # erec = gaussian_filter(erec, sigma=1.0)
            rec = combine_stains(np.stack((hrec, erec, null), axis=-1), c_rgb_from_hed)#hed2rgb(np.stack((hrec, erec, null), axis=-1)) #HE Reconstruction
        elif stains=='HD':    
            #hdh,hdd = h0,d0
            e2h = matchers['E2H'].transform(e0)
            e2d = matchers['E2D'].transform(e0)
            hdh = (h0+weh*h1*e2h)/(1+weh)
            hdd = (d0+wed*d1*e2d)/(1+wed)
            # hdh = median(h0, disk(2))#gaussian_filter(h0, sigma=1)#
            # hdd = median(d0, disk(2))#gaussian_filter(d0, sigma=1)#median(d0, disk(2))#
            rec = hed2rgb(np.stack((hdh, null, hdd), axis=-1))#hed2rgb(np.stack((h0, null, d0), axis=-1)) #HD Reconstruction
            #hd = combine_stains(np.stack((hdh, null, hdd), axis=-1), c_rgb_from_hed)#hed2rgb(np.stack((h0, null, d0), axis=-1)) #HD Reconstruction
            #rec=hed2rgb(np.stack((h0, null, 0.5*d*d0), axis=-1))
        elif stains=='D':
            whd=0.2
            e2d = matchers['E2D'].transform(e0)
            h2d = matchers['H2D'].transform(h0)
            drec=(d0+wed*d1*e2d+whd*d1*h2d)/(1+wed+whd)
            odsum=(e0+d0+h0)/5
            drec=drec+odsum

            rec = hed2rgb(np.stack((0.15*h0, null, drec), axis=-1))
        else:
            print('unknown stain')
        
        #dt = d0>=thresh #corresponding D-trhesholded image
        
        #1/0
        #imsave(os.path.join(out_he,Path(f).name),rec)
        #imsave(os.path.join(out_hd,Path(f).name),hd)
        return (rec*255).astype('uint8')
        #return Image.fromarray((rec*255).astype('uint8'))


def gethist(f):
        bins = np.linspace(-2,2,num=1000)
        ihc_rgb0 = imread(f)        
        ihc_rgb0 = np.array(ihc_rgb0,dtype=np.uint8)
        #ihc_rgb = staintools.LuminosityStandardizer.standardize(ihc_rgb0)
        #ihc_rgb = LuminosityStandardizer.standardize(ihc_rgb0)   #should we do this?
        ihc_rgb=ihc_rgb0     
        ihc_hed = separate_stains(ihc_rgb, c_hed_from_rgb)#rgb2hed(ihc_rgb)
        h0 = ihc_hed[:, :, 0]
        e0 = ihc_hed[:, :, 1]
        d0 = ihc_hed[:, :, 2]

        #plt.subplot(1,2,1)
        #plt.scatter(d0.flatten()[0::20],h0.flatten()[0::20])
        #plt.subplot(1,2,2)
        #plt.scatter(d0.flatten()[0::20],e0.flatten()[0::20])
        #plt.show()

        # null = np.zeros_like(h0)
        # ihc_h = hed2rgb(np.stack((h0, null, null), axis=-1))
        # ihc_e = hed2rgb(np.stack((null, e0, null), axis=-1))
        # ihc_d = hed2rgb(np.stack((null, null, d0), axis=-1))
        
        nh=np.histogram(h0.flatten(),bins)[0]
        ne=np.histogram(e0.flatten(),bins)[0]
        nd=np.histogram(d0.flatten(),bins)[0]
        return (nh,ne,nd)

if __name__ == '__main__':
    #plt.close('all')
    #ihc_rgb = data.immunohistochemistry()
    from cpathreader import CPathReader
    from pathlib import Path

    #snormalizer = staintools.StainNormalizer(method='vahadane')
    #snormalizer.fit(staintools.LuminosityStandardizer.standardize(staintools.read_image(r'./target2.jpg')))
    
    #slide_path = Path(r"E:/PRISMATIC/Mitosis_Ki67_IHC_and_IHC+HE_Trial_Asmaa/", "Case_1_Ki_IHC+H_and_E.mrxs",)
    slide_path=list(Path(r'E:\PRISMATIC\Dubble staining').glob('*.ndpi'))[5]
    #slide_path = Path(r"E:/PRISMATIC/PHH3-HE-Asmaa-Nottingham/", "1M03.mrxs",)
    P = CPathReader(mpp = 0.2431,mask_mpp = 8,patch_size=4096)  
    N,cprinfo = P.WSI2SNPatches(slide_path,out_path=r'E:/PRISMATIC/demux/temp/',normalize = False, n_jobs=8)
    
    
    #%% Get stain-wise histograms
    #ihc_rgb0,cprinfo = P.WSI2RandomPatch(slide_path)
    from glob import glob
    from tqdm import tqdm
    plist = glob(r'E:/PRISMATIC/demux/temp/'+slide_path.name+'/*.jpg')
    xy = [Path(f).name.split('.')[0].split('-') for f in plist]    

    n_jobs = 4# 8 #-1
    bins = np.linspace(-2,2,num=1000)
    
    print('getting histograms')
    Z = Parallel(n_jobs=n_jobs)(delayed(gethist)(f) for f in tqdm(plist, total=len(plist)))
    Zs = np.sum(Z,axis=0)
    nh,ne,nd = Zs[0],Zs[1],Zs[2]

    #for f in plist:
        #gethist(f)
    
    # nh,ne,nd = np.zeros(len(bins)-1),np.zeros(len(bins)-1),np.zeros(len(bins)-1)
    # for f in tqdm(plist):
    #     ihc_rgb0 = imread(f)        
    #     ihc_rgb0 = np.array(ihc_rgb0,dtype=np.uint8)
    #     ihc_rgb = staintools.LuminosityStandardizer.standardize(ihc_rgb0)        
    #     ihc_hed = rgb2hed(ihc_rgb)
    #     h0 = ihc_hed[:, :, 0]
    #     e0 = ihc_hed[:, :, 1]
    #     d0 = ihc_hed[:, :, 2]
    #     # null = np.zeros_like(h0)
    #     # ihc_h = hed2rgb(np.stack((h0, null, null), axis=-1))
    #     # ihc_e = hed2rgb(np.stack((null, e0, null), axis=-1))
    #     # ihc_d = hed2rgb(np.stack((null, null, d0), axis=-1))
        
    #     nh+=np.histogram(h0.flatten(),bins)[0]
    #     ne+=np.histogram(e0.flatten(),bins)[0]
    #     nd+=np.histogram(d0.flatten(),bins)[0]
        
    #%% Do stain transfer
    
    
    plt.close('all')
    odir = r'E:\PRISMATIC\demux\processed'
    from skimage.io import imsave
    import os
    slide_path = Path(slide_path)
    output = Path(odir).joinpath(slide_path.name)
    if not output.exists():
        if not output.parent.exists():
            output.parent.mkdir()
        output.mkdir()
    out_he = output/'HE'
    out_hd = output/'HD'
    
    if not out_he.exists(): out_he.mkdir()
    if not out_hd.exists(): out_hd.mkdir()
    
    np.savetxt(str(output)+'_hist.txt',Zs)
    wh = 0.7
    we = 0.1
    sigma = 5.0
    hmin,hmax = hist2percentile(nh, 0.01,bins),hist2percentile(nh, 0.99,bins)
    emin,emax = hist2percentile(ne, 0.01,bins),hist2percentile(ne, 0.99,bins)
    dmin,dmax = hist2percentile(nd, 0.01,bins),hist2percentile(nd, 0.99,bins)
    D2H = GlobalHistMatcher().fit(nd,bins,nh,bins)
    D2E = GlobalHistMatcher().fit(nd,bins,ne,bins)
    E2D = GlobalHistMatcher().fit(ne,bins,nd,bins)
    E2H = GlobalHistMatcher().fit(ne,bins,nh,bins)
    thresh = hist2percentile(nd,0.95,bins)
    thresh2 = hist2percentile(nd,0.75,bins)
    idx = 33
    for f in tqdm(plist):#plist[idx:idx+1]:#
        ihc_rgb0 = imread(f)        
        ihc_rgb0 = np.array(ihc_rgb0,dtype=np.uint8)
        ihc_rgb = ihc_rgb0
        #ihc_rgb = staintools.LuminosityStandardizer.standardize(ihc_rgb0)
        ihc_rgb = LuminosityStandardizer.standardize(ihc_rgb0)      
        ihc_hed = separate_stains(ihc_rgb, c_hed_from_rgb)#rgb2hed(ihc_rgb)
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
        d1 = gaussian_filter(d, sigma=sigma)
        d2h = D2H.transform(d0)
        d2e = D2E.transform(d0)
        hrec = (h0+wh*h1*d2h)/(1+wh)#(h0+wh*h1*d0)/(1+wh)
        erec = (e0+we*e1*d2e)/(1+we)#(e0+we*e1*d0)/(1+we)
        # hrec = gaussian_filter(hrec, sigma=1.0)
        # erec = gaussian_filter(erec, sigma=1.0)
        rec = combine_stains(np.stack((hrec, erec, null), axis=-1), c_rgb_from_hed)#hed2rgb(np.stack((hrec, erec, null), axis=-1)) #HE Reconstruction
        
        hdh,hdd = h0,d0
        e2h = E2H.transform(e0)
        e2d = E2D.transform(e0)
        weh = 1.0;hdh = (h0+weh*h1*e2h)/(1+weh)
        wed = 1.0;hdd = (d0+wed*d1*e2d)/(1+wed)
        # hdh = median(h0, disk(2))#gaussian_filter(h0, sigma=1)#
        # hdd = median(d0, disk(2))#gaussian_filter(d0, sigma=1)#median(d0, disk(2))#
        hd = hed2rgb(np.stack((hdh, null, hdd), axis=-1))#hed2rgb(np.stack((h0, null, d0), axis=-1)) #HD Reconstruction
        #hd = combine_stains(np.stack((hdh, null, hdd), axis=-1), c_rgb_from_hed)#hed2rgb(np.stack((h0, null, d0), axis=-1)) #HD Reconstruction
        ## this works better hed2rgb(np.stack((h0, null, 0.5*d*d0), axis=-1))
        
        dt = d0>=thresh #corresponding D-trhesholded image
        
        #1/0
        imsave(os.path.join(out_he,Path(f).name),rec)
        imsave(os.path.join(out_hd,Path(f).name),hd)
    #%%
    from mergepatches import *
    MergePatches(out_he,ofname = str(output/('reconstructed_HE_'+output.name)))
    MergePatches(out_hd,ofname = str(output/('reconstructed_HD_'+output.name)))



#%%
    # d = rescale_intensity(d0, out_range=(0, 1),
    #                   in_range=(hist2percentile(nd,0.0,bins), hist2percentile(nd,0.99,bins)))
    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()    
    ax[0].imshow(ihc_rgb0)
    ax[0].set_title("Original image")        
    #ax[1].imshow(snormalizer.transform(rec))
    ax[1].imshow(rec)
    ax[1].set_title("Reconstructed")
    #ax[2].imshow(d0>hist2percentile(nd, 0.99,bins))
    ax[2].imshow(dt,cmap='gray')
    ax[2].set_title("D thresholded")
    ax[3].imshow(hd)
    ax[3].set_title("HD")
    
           