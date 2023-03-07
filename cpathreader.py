# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 21:18:14 2020
Extract patches and stain normalize them (parallel)
NOTE: In windows, use %run rather than execute 
@author: fayyaz
"""


from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
#import staintools
from PIL import Image
from joblib import Parallel, delayed
import tiatoolbox.tools.stainnorm as snorm
import extractors
class CPRInfo:
    def __init__(self,slide_path,output,cp,ext,N):
        self.slide_path = slide_path 
        self.output_path = output
        self.thumb = ext.thumb
        self.mask = ext.mask
        self.mpp = ext.mpp
        self.mask_mpp = ext.mask_mpp
        self.patch_size = ext.patch_size
        self.stride = ext.stride
        self.stride_px = ext.stride_px
        self.threshold = ext.threshold
        self.shape = ext.shape
        self.stain_method = cp.stain_method
        self.stain_target = cp.stain_target
        self.slide_properties = dict(ext.slide.properties.items())
        self.slide_level_count = ext.slide.level_count
        self.slide_dimensions = ext.slide.dimensions
        self.slide_level_dimensions = ext.slide.level_dimensions
        self.N = N

        
class CPathReader:
    def __init__(self,patch_size=224,stain_target="target.png",stain_method="vahadane",**kwargs):
        """
        Read WSI/folder/raw images, stain normalize patches

        Parameters
        ----------
        patch_size : TYPE, optional
            DESCRIPTION. The default is 224.
        stain_target : TYPE, optional
            DESCRIPTION. The default is "target.png".
        stain_method : TYPE, optional
            DESCRIPTION. The default is "vahadane".
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.patch_size = patch_size
        self.kwargs = kwargs        
        self.stain_method = stain_method
        self.stain_target = stain_target
        self.snormalizer = snorm.get_normaliser(stain_method)  # "macenko" is much faster
        target_image_path='D:\Meso\TMA_stain_target.tiff'
        target_image = cv2.imread(target_image_path)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        self.snormalizer.fit(target_image)
    def WSI2Thumb(self,slide_path,mpp=8):  
        from segmenters import MultiThresholdSegmenter
        from openslide import OpenSlide

        slide = OpenSlide(str(slide_path))
        seg = MultiThresholdSegmenter(thumbnail_mpp=mpp)
        seg.slide = slide
        thumb = seg.thumbnail(force=True)
        return thumb
        
    def WSI2RandomPatch(self,slide_path):
        """
        Return random stain normalized patch (for testing)

        Parameters
        ----------
        slide_path : TYPE
            DESCRIPTION.

        Returns
        -------
        patch : TYPE
            DESCRIPTION.
        cprinfo : TYPE dictionary
            contains useful information fields such as thumbnail.

        """
        ext = extractors.SimplePatchExtractor(
            slide_path, patch_size=self.patch_size, **self.kwargs)
        (x, y), tissue_area, patch = ext.getRandomTissuePatch()
        #patch = self.stainNormalizePatch(patch)
        cprinfo = CPRInfo(slide_path,'',self,ext,0)
        return patch,cprinfo.__dict__
    def stainNormalizePatch(self,patch):
        """
        Stain normalize single image 

        Parameters
        ----------
        patch : TYPE
            DESCRIPTION.

        Returns
        -------
        patch : TYPE
            DESCRIPTION.

        """
        transformed = self.snormalizer.transform(np.array(patch))
        patch = Image.fromarray(transformed)
        return patch                
    def WSI2SNPatches(self,slide_path,out_path='./',normalize = True,n_jobs=-1):
        """
        Get stain normalized patches from a whole slide image an save to folder
        Also saves an information dictionary "infodictwb.pkl"
        And a thumbnail '.thumb.png'

        Parameters
        ----------
        slide_path : TYPE
            DESCRIPTION. path to slide
        out_path : TYPE, optional
            DESCRIPTION. The default is './'. Parent directory. A folder of the 
            same name as the wsi is created in this folder to save patch files
            each named after its coordinates in the WSI
        n_jobs : TYPE, optional
            DESCRIPTION. The default is -1.

        Returns
        -------
        N : TYPE
            DESCRIPTION. Number of patch files saved
        cprinfo : TYPE Dictionary
            DESCRIPTION. contains useful information about the slide and images

        """
        slide_path = Path(slide_path)
        output = Path(out_path, slide_path.name)
        if not output.exists():
            if not output.parent.exists():
                output.parent.mkdir()
            output.mkdir()
        ext = extractors.SimplePatchExtractor(
            slide_path, patch_size=self.patch_size,**self.kwargs)
        
        def writeout(pobj):
            (x, y), tissue_area, patch = pobj
            if patch is not None:
                # Beware: this normaliser will produce a lot of annoying warnings from spams
                if normalize: 
                    patch = self.stainNormalizePatch(patch)
                patch.save(output / f"{x:05}-{y:05}.jpg") 
                return 1
            return 0
                
        Z = Parallel(n_jobs=n_jobs)(delayed(writeout)(pobj) for pobj in tqdm(iter(ext), total=len(ext)))
        N = sum(Z)
        cprinfo = CPRInfo(slide_path,output,self,ext,N)
        import pickle
        pickle.dump( cprinfo.__dict__, open( output / "infodictwb.pkl", "wb" ) )
        cprinfo.thumb.save(output/'.thumb.png')
        return N,cprinfo.__dict__
    
    def Patches2SNPatches(self,plist,odir='./',n_jobs=-1):
        """
        Stain normalize patches and save
        Given a string descriptor of file lists or a file list of image files, it
        stain normalizes each image and saves it to a given output directory
        with the same name

        Parameters
        ----------
        plist : TYPE (string or list)
            DESCRIPTION. example './myfolder/*.jpg' or list of image file names
        odir : TYPE string, optional
            DESCRIPTION. The default is './'. Output directory
        n_jobs : TYPE, optional int (number of parallel jobs)
            DESCRIPTION. The default is -1.

        Returns
        -------
        None.

        """
        if type(plist) is str:
            from glob import glob 
            plist = glob(plist)
        odir = Path(odir)
        def writout(f):            
            patch = cv2.imread(f)
            patch = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            f = Path(f)
            patch = self.stainNormalizePatch(patch)
            patch.save(odir / f.name)
           
        Parallel(n_jobs=n_jobs)(delayed(writout)(f) for f in tqdm(plist))
        
           
if __name__=='__main__':       
        
    slide_path = Path(r"C:\Users\fayya\Downloads\Mitosis Ki67 IHC and IHC+HE Trial Asmaa",
    "Case 1 Ki IHC+H&E.mrxs",)
    P = CPathReader(mpp = 0.2431,mask_mpp = 16,patch_size=4096)
    #%% Testing a random patch
    import matplotlib.pyplot as plt
    patch,cprinfo = P.WSI2RandomPatch(slide_path)
    from demultiplexed_staining import HED2HEM
    #%%
    rec, zdh,  (ihc_h,ihc_e,ihc_d)  = HED2HEM(patch,wh=0.2,we=0.2,sigma = 5.0)
    
    rec = np.array(255*rec,dtype=np.uint8)
    recsn = P.stainNormalizePatch(rec)
    
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()
    
    ax[0].imshow(patch)
    ax[0].set_title("Original image")
    
    ax[1].imshow(recsn)
    ax[1].set_title("Reconstructed")
    
    # ax[2].imshow(recsn)
    # ax[2].set_title("Reconstructed SN")
    
    print(cprinfo)
    #plt.imshow(patch)
    1/0
    #%% WSI 2 patches
    N,cprinfo = P.WSI2SNPatches(slide_path)
    print(cprinfo)
    #%% Stain normalize all patches in a given folder
    #P.Patches2SNPatches(plist=r'.\TCGA-36-1574-01A-01-TS1.0ebf58a0-dd01-40b2-9f37-5970f65cf9bb.svs\*.jpg',odir='./temp',n_jobs=1)
    




    

    
