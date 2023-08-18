from demux_fixed import VirtualRestainer
from pathlib import Path
import numpy as np
from demux_fixed import hist2percentile, GlobalHistMatcher
import matplotlib.pyplot as plt
from PIL import Image

slide_path = Path(r"/media/u2071810/Data/Multiplexstaining/Fouzia_samples2")
#tile_path = Path(r"/media/u2071810/Data/Multiplexstaining/Asmaa Multiplex Staining/tiles_skimg")
filter = "*X.jpg"
slides = list(slide_path.glob(filter))
stains = ["HED","HE","HD"]
mode = "tile"

for slide in slides[:5]:
    for stain in stains:
        print(f"Restaining {slide}")
        restainer = VirtualRestainer(slide, coupling_coeffs={"wdh":0.5, "wde":0.8, "weh":0.1, "wed":0.1})
        Zs = restainer._get_hist(restainer.patch_extractor.wsi.img)
        #Zs = np.sum(Z,axis=0)
        nh,ne,nd = Zs[0],Zs[1],Zs[2]

        bins = np.linspace(-2,2,num=1000)

        restainer.hrange = (hist2percentile(nh, 0.01,bins),hist2percentile(nh, 0.99,bins))
        restainer.erange = (hist2percentile(ne, 0.01,bins),hist2percentile(ne, 0.99,bins))
        restainer.drange = (hist2percentile(nd, 0.01,bins),hist2percentile(nd, 0.99,bins))
        restainer.D2H = GlobalHistMatcher().fit(nd,bins,nh,bins)
        restainer.D2E = GlobalHistMatcher().fit(nd,bins,ne,bins)
        restainer.E2D = GlobalHistMatcher().fit(ne,bins,nd,bins)
        restainer.E2H = GlobalHistMatcher().fit(ne,bins,nh,bins)
        restainer.H2D = GlobalHistMatcher().fit(nh,bins,nd,bins)
        restained = restainer.restain_tile(restainer.patch_extractor.wsi.img, stains=stain, lum_norm=False)
        # save image
        restained = Image.fromarray(restained)
        restained.save(slide_path / f"{slide.stem}_{stain}_v3.jpg")

print("Done")