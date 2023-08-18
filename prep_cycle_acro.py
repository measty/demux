from demux_fixed import VirtualRestainer
from pathlib import Path
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor, PatchExtractor
from tiatoolbox.wsicore import WSIReader
import numpy as np
from tqdm import tqdm
from PIL import Image

#slide_path = Path(r"/media/u2071810/Data1/Multiplexstaining/Asmaa_Multiplex_Staining")
slide_path = Path(r"/media/u2071810/Data1/Acrobat/acrobat_train_pyramid_1_of_2")
tile_path = Path(r"/media/u2071810/Data1/Acrobat/acro_tiles_cycle")
patch_size = 256
n_tiles = 2000

def from_triple_stained():
    if not tile_path.exists():
        tile_path.mkdir()
    filter = "*PHH3_HE.svs"
    slides = list(slide_path.glob(filter))
    stains = ["HED","HE","HD"]
    mode = "tile"
    info_str = "_info2.pkl"
    valid_slides = []

    for slide in slides:
        info_path = slide.parent / (slide.stem + info_str)
        if info_path.exists():
            valid_slides.append(slide)

    slides = valid_slides

    # split slides into train, val, test
    split_ratios = [0.8, 0.1, 0.1]
    n_slides = len(slides)
    n_train = int(n_slides * split_ratios[0])
    n_val = int(n_slides * split_ratios[1])
    n_test = n_slides - n_train - n_val
    splits = {}
    splits["train"] = slides[:n_train]
    splits["val"] = slides[n_train:n_train+n_val]
    splits["test"] = slides[n_train+n_val:]

    for split in splits:
        for slide in splits[split]:
            print(f"Restaining {slide}")
            restainer = VirtualRestainer(slide, coupling_coeffs={"wdh":0.8, "wde":0.1, "weh":0.1, "wed":0.1})
            info_path = slide.parent / (slide.stem + info_str)
            try:
                restainer.load_info(info_path)
            except:
                print("No info file found")
                continue
            if mode == "wsi":
                if not (slide.parent / (slide.stem + "_restained19_HE.tiff")).exists():
                    restainer.save_restained_WSI(slide.parent / (slide.stem + "_restained19.tiff"), stains=stains, lum_norm=False)
            else:
                restainer.save_sample_tiles(tile_path, stains=stains, lum_norm=False, tile_size=patch_size, n_tiles=n_tiles, save_d=False, sep_folders={"HE": f"B/{split}", "HD": f"A/{split}", "HED": f"C/{split}"})

    print("Done")

def from_restained():
    filter = "*_HE.tiff"
    he_slides = sorted(list(slide_path.glob(filter)))
    he_slides = [slide for slide in he_slides if "PHH3" not in slide.stem]
    filter = "*.tiff"
    hd_slides = sorted(list(slide_path.glob(filter)))
    hd_slides = [slide for slide in hd_slides if "HE" != slide.stem.split('_')[-1]]
    if len(he_slides) != len(hd_slides):
        print("Number of slides do not match")
        #raise ValueError
        he_slides = he_slides * (len(hd_slides) // len(he_slides))

    # split slides into train, val, test
    split_ratios = [1, 0, 0] #[0.8, 0.1, 0.1]
    n_slides = min(len(he_slides), len(hd_slides))
    n_train = int(n_slides * split_ratios[0])
    n_val = int(n_slides * split_ratios[1])
    n_test = n_slides - n_train - n_val
    he_splits = {}
    he_splits["train"] = he_slides[:n_train]
    he_splits["val"] = he_slides[n_train:n_train+n_val]
    he_splits["test"] = he_slides[n_train+n_val:]
    hd_splits = {}
    hd_splits["train"] = hd_slides[:n_train]
    hd_splits["val"] = hd_slides[n_train:n_train+n_val]
    hd_splits["test"] = hd_slides[n_train+n_val:]

    for split in he_splits:
        (tile_path / f"A/{split}").mkdir(exist_ok=True, parents=True)
        (tile_path / f"B/{split}").mkdir(exist_ok=True, parents=True)
        for he_slide, hd_slide in zip(he_splits[split], hd_splits[split]):
            if he_slide.stem.split('_')[0] != hd_slide.stem.split('_')[0]:
                print("Slides do not match")
                #raise ValueError
            patch_extractor_he = SlidingWindowPatchExtractor(
                input_img=he_slide,
                patch_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                input_mask='otsu', #'morphological',
                min_mask_ratio=0.1, # only discard patches with very low tissue content
                within_bound=True,
            )
            patch_extractor_hd = SlidingWindowPatchExtractor(
                input_img=hd_slide,
                patch_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                input_mask='otsu', #'morphological',
                min_mask_ratio=0.1, # only discard patches with very low tissue content
                within_bound=True,
            )
            # masking works badly on HD so we use the same mask as HE
            #patch_extractor_hd.coordinate_list = patch_extractor_he.coordinate_list
            #patch_extractor_hd.locations_df = patch_extractor_he.locations_df
            sample_inds_he = np.random.choice(len(patch_extractor_he), min(n_tiles, len(patch_extractor_he)), replace=False)
            sample_inds_hd = np.random.choice(len(patch_extractor_hd), min(n_tiles, len(patch_extractor_hd)), replace=False)
            for i, (ind_he, ind_hd) in tqdm(enumerate(zip(sample_inds_he, sample_inds_hd))):
                he_tile = patch_extractor_he[int(ind_he)]
                hd_tile = patch_extractor_hd[int(ind_hd)]

                Image.fromarray(he_tile).save(tile_path / f"B/{split}/{he_slide.stem.split('_')[0]}_{i}.png")
                Image.fromarray(hd_tile).save(tile_path / f"A/{split}/{hd_slide.stem.split('_')[0]}_{i}.png")

    print("Done")

if __name__ == "__main__":
    from_restained()