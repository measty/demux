#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 12:56:11 2021

@author: u1876024
"""
import os
vipshome = r'C:\Users\meast\vips-dev-8.12\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

import numpy as np
import pyvips
from pathlib import Path
from glob import glob
from skimage.io import imread
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

def MergePatches(idir,ofname = './out.tif',save_path='E:/PRISMATIC/demux/_temp',sep=None, ext='jpg', comp='jpeg'):        
    G = glob(str(Path(idir)/f'*.{ext}'))
    if sep:
        xy = np.array([list(map(int,Path(f).stem.split(sep)[1].split('-'))) for f in G])
    else:
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
    vi.tiffsave(ofname, tile=True, compression=comp, bigtiff=True, pyramid=True)
    cum_canvas._mmap.close()
    del cum_canvas
    os.remove(save_path)    



if __name__=='__main__':
    #idir = './processed/Case 2 Ki IHC+H&E.mrxs/HE'
    #MergePatches(idir,ofname='./case2hd.tif')

    #slide_path = Path(r'E:\PRISMATIC\PHH3-HE-Asmaa-Nottingham\\',"1M02.mrxs",)
    slide_path=Path(r'E:\PRISMATIC\Mitosis_Ki67_IHC_and_IHC+HE_Trial_Asmaa\Case_4_Ki_IHC+H_and_E.mrxs')
    odir = 'E:/PRISMATIC/demux/processed/'
    #odir = 'E:/Meso_TCGA/processed/'
    #slide_path = Path(slide_path)
    output = Path(odir, slide_path.name)
    out_he = output/'HE'
    out_hd = output/'HD'
    orig= Path(r'E:/PRISMATIC/demux').joinpath('temp',slide_path.name)
    base_path=Path(r'E:\Meso_TCGA\Slides_tiled\TCGA-SC-A6LN-01Z-00-DX1.379BF588-5A65-4BF8-84CF-5136085D8A47')
    tiles = base_path.joinpath('tiles')
    #masks = base_path.joinpath('masks')
    outs=base_path.joinpath('outputs_g')
    #MergePatches(out_he,ofname = str(output/('reconstructed_HE_'+output.name)))
    #MergePatches(out_hd,ofname = str(output/('reconstructed_HD_'+output.name)))
    #MergePatches(orig,ofname = str(output/('matching_orig_'+output.name)))
    #MergePatches(tiles,ofname=str(tiles.joinpath('recon.tiff')),sep='_')
    #MergePatches(masks,ofname=str(masks.joinpath('recon.tiff')),sep='_',comp='deflate')
    MergePatches(outs,ofname = str(outs.joinpath('recon.tiff')),sep='_',ext='png',comp='deflate')