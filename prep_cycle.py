from demux_fixed import VirtualRestainer
from pathlib import Path
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor, PatchExtractor
from tiatoolbox.wsicore import WSIReader
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse

slide_path = Path(r"/media/u2071810/Data1/Multiplexstaining/Asmaa_Multiplex_Staining")
tile_path = Path(r"/media/u2071810/Data1/Multiplexstaining/Asmaa_Multiplex_Staining/tiles_cycle_lores")
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
    filter = "*_HE.svs"
    he_slides = sorted(list(slide_path.glob(filter)))
    he_slides = [slide for slide in he_slides if "PHH3" not in slide.stem]
    filter = "*_PHH3.svs"
    hd_slides = sorted(list(slide_path.glob(filter)))
    hd_slides = [slide for slide in hd_slides if "HE" not in slide.stem]
    if len(he_slides) != len(hd_slides):
        print("Number of slides do not match")
        raise ValueError

    # split slides into train, val, test
    split_ratios = [0.9, 0.0, 0.1]
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
                raise ValueError
            patch_extractor_he = SlidingWindowPatchExtractor(
                input_img=he_slide,
                patch_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                input_mask='extended', #'morphological',
                min_mask_ratio=0.1, # only discard patches with very low tissue content
                within_bound=True,
                resolution=1.0,
                units='mpp'
            )
            patch_extractor_hd = SlidingWindowPatchExtractor(
                input_img=hd_slide,
                patch_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                input_mask=None, #'morphological',
                min_mask_ratio=0.1, # only discard patches with very low tissue content
                within_bound=True,
                resolution=1.0,
                units='mpp'
            )
            # masking works badly on HD so we use the same mask as HE
            patch_extractor_hd.coordinate_list = patch_extractor_he.coordinate_list
            patch_extractor_hd.locations_df = patch_extractor_he.locations_df
            sample_inds = np.random.choice(len(patch_extractor_he), n_tiles, replace=False)
            for i, ind in tqdm(enumerate(sample_inds)):
                he_tile = patch_extractor_he[int(ind)]
                hd_tile = patch_extractor_hd[int(ind)]

                Image.fromarray(he_tile).save(tile_path / f"B/{split}/{he_slide.stem.split('_')[0]}_{i}.png")
                Image.fromarray(hd_tile).save(tile_path / f"A/{split}/{hd_slide.stem.split('_')[0]}_{i}.png")

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #add args for pathA, pathB and savepath
    parser.add_argument("--mode", type=str, default="tile", help="tile or wsi")
    parser.add_argument("--patch_size", type=int, default=256, help="patch size")
    parser.add_argument("--n_tiles", type=int, default=1000, help="number of tiles")
    parser.add_argument("--stains", type=str, default="HED", help="stains to use")
    parser.add_argument("--slide_path", type=str, default="/home/alexanderliao/data/PHH3", help="path to slides")
    parser.add_argument("--tile_path", type=str, default="/home/alexanderliao/data/PHH3/tiles", help="path to save tiles")
    
    from_restained()