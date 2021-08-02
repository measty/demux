#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 12:56:11 2021

@author: u1876024
"""


import numpy as np
import pyvips
from pathlib import Path
from glob import glob
from skimage.io import imread
import os
from tqdm import tqdm
def numpy2vips(a):
    dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
    }
    height, width, bands = a.shape
    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      dtype_to_format[str(a.dtype)])
    return vi

def MergePatches(idir,ofname = './out.tif',save_path='./_temp'):        
    G = glob(str(Path(idir)/'*.jpg'))
    xy = np.array([list(map(int,Path(f).name.split('.')[0].split('-'))) for f in G])
    psize = imread(G[0]).shape
    csize = np.max(xy,axis=0)-np.min(xy,axis=0)+np.array(psize[:-1])
    canvas_shape = csize[::-1]
    xyr = xy-np.min(xy,axis=0)
    xyr = np.hstack((xyr,xyr+psize[:-1]))    
    out_ch = psize[-1]
    cum_canvas = np.lib.format.open_memmap(
        save_path,
        mode="w+",
        shape=tuple(canvas_shape) + (out_ch,),
        dtype=np.uint8,
    )
    for patch_idx in tqdm(range(len(G))):
        patch = imread(G[patch_idx])
        bound_in_wsi = xyr[patch_idx]
        # position is assumed to be in XY coordinate        
        # convert to XY to YX, and in tl, br
        tl_in_wsi = np.array(bound_in_wsi[:2][::-1])
        br_in_wsi = np.array(bound_in_wsi[2:][::-1])        
        cum_canvas[tl_in_wsi[0] : br_in_wsi[0], tl_in_wsi[1] : br_in_wsi[1]] = patch

    vi = numpy2vips(cum_canvas); 
    vi.tiffsave(ofname, tile=True, compression='jpeg', bigtiff=True, pyramid=True)
    cum_canvas._mmap.close()
    del cum_canvas
    os.remove(save_path)    



if __name__=='__main__':
    idir = './processed/Case 2 Ki IHC+H&E.mrxs/HE'
    MergePatches(idir,ofname='./case2hd.tif')