#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 14:38:17 2020
Extract features using a pretrained model
@author: Fayyaz Minhas
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
import os
import natsort
from pathlib import Path

USE_CUDA = torch.cuda.is_available()
device = {True:'cuda',False:'cpu'}[USE_CUDA] 
device = torch.device(device)
def toNumpy(v):
    if type(v) is not torch.Tensor: return np.asarray(v)
    if USE_CUDA:
        return v.detach().cpu().numpy()
    return v.detach().numpy()
def fname2coords(img_loc):
    """
    Return a tuple of patch coordinates based on the string file name img_loc
    """
    return tuple(int(x) for x in Path(img_loc).name.split('.')[0].split('-'))
    
class WSIPatchData(Dataset):
    """Custom dataset that includes image file paths. 
    """
    def __init__(self, main_dir, transform,fname2coords=fname2coords):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = glob(main_dir)        
        self.total_imgs = natsort.natsorted(all_imgs)
        self.fname2coords = fname2coords
    def __len__(self):
        return len(self.total_imgs)

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, idx):
        #img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")        
        tensor_image = self.transform(image)
        #x = img_loc.split('_')[-1].split('.')[0]        
        #c = tuple(int(s) for s in x.strip("()").split(","))+
        # get coordinates from the image file name 
        c = self.fname2coords( img_loc)
        
        return tensor_image,c,img_loc

class WSINNFE:
    """
    WSI image neural network feature extraction
    """
    def __init__(self,transformations = None, model = None, bsize = 128):
        if transformations is None: 
            transformations = transforms.Compose([
                #transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.transformations = transformations
        self.bsize = bsize
        if model is None:
            model = models.resnet50(pretrained=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.model = model
        self.FE =  torch.nn.Sequential(*list(model.children())[:-1])
        model.to(device)
        
    def getFeatures(self,fdesc,ofile=None,fname2coords=fname2coords):
        ds = WSIPatchData(fdesc, transform = self.transformations,fname2coords=fname2coords)
        dsl = torch.utils.data.DataLoader(ds, batch_size=self.bsize, shuffle=False)
        C,H,Fn = [],[],[]
        
        for x, (cx,cy),fn in tqdm(dsl):
            # Move to device
            x = x.to(device)    
            h = toNumpy(self.FE(x))
            cx,cy = toNumpy(cx),toNumpy(cy)
            C.append(np.vstack((cx,cy)))
            H.append(h)
            Fn.extend(fn)
        C,H = np.hstack(C).T,np.squeeze(np.vstack(H))   
        D = {'coords':C,'features':H,'fnames':Fn}
        if ofile is not None:
            import pickle
            with open(ofile,'wb') as ofh:
                pickle.dump(D,ofh)            
        return D

if __name__=='__main__':
    nnfe = WSINNFE()
    f = r'C:\Users\fayya\OneDrive\Desktop\wsiclust\TCGA-36-1574-01A-01-TS1.0ebf58a0-dd01-40b2-9f37-5970f65cf9bb.svs\*.jpg'
    D = nnfe.getFeatures(f,ofile='temp.feats.pkl')
    

        

    