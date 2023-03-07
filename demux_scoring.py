import os
vipshome = r'C:\Users\meast\openslide\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

from pathlib import Path
from sysconfig import get_path
import numpy as np
from tiatoolbox.wsicore.wsireader import get_wsireader
from skimage.io import imread
from stainseparator import separate_stains
import matplotlib.pyplot as plt
from skimage.morphology import binary_opening, remove_small_objects, remove_small_holes, disk
from skimage.filters import gaussian

rng=np.random.default_rng()

gt_path=Path(r'E:\PRISMATIC\Mitosis_Ki67_IHC_and_IHC+HE_Trial_Asmaa\Case_4_Ki_IHC_only.mrxs')
rec_path=Path(r'E:\PRISMATIC\demux\processed\Case_4_Ki_IHC+H_and_E.mrxs\reconstructed_HD_Case_4_Ki_IHC+H_and_E.mrxs')
rec_patches=Path(r'E:\PRISMATIC\demux\processed\Case_4_Ki_IHC+H_and_E.mrxs\HD')

c_rgb_from_hed = np.array([[0.458,0.814,0.356],[0.259,0.866,0.428],[0.269,0.568,0.778]])
c_hed_from_rgb = np.linalg.inv(c_rgb_from_hed)

plist=list(rec_patches.glob('*.jpg'))
idx=rng.integers(0,len(plist),1)
idx=idx[0]

reader=get_wsireader(gt_path)

im=imread(plist[idx])
tl=[int(x) for x in plist[idx].stem.split('-')]

gt=reader.read_rect(tl,size=(4096,4096))

im_hed = separate_stains(im, c_hed_from_rgb)
im_d=im_hed[:,:,2]
im_d=gaussian(im_d,5)
thresh1=0.5*np.percentile(im_d,[95])
thresh2=np.percentile(im_d[im_d>thresh1].ravel(),[80])
im_dt=im_d>thresh2
kernel=disk(3)
im_dt=binary_opening(im_dt,kernel)
im_dt=remove_small_objects(im_dt,10)

gt_hed = separate_stains(gt, c_hed_from_rgb)
gt_d=gt_hed[:,:,2]
gt_d=gaussian(gt_d,5)
thresh1=0.5*np.percentile(gt_d,[95])
thresh2=np.percentile(gt_d[gt_d>thresh1].ravel(),[80])
gt_dt=gt_d>thresh2
gt_dt=binary_opening(gt_dt,kernel)
gt_dt=remove_small_objects(gt_dt,10)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
ax = axes.ravel()    
ax[0].imshow(gt_dt)
ax[0].set_title("Original image")        
#ax[1].imshow(snormalizer.transform(rec))
ax[1].imshow(im_dt)
ax[1].set_title("Reconstructed")
plt.show()
print('done!')











