"""Code for calculating statistics of cell detections on wsis/tiles"""

import os
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from tiatoolbox.wsicore import WSIReader
import numpy as np
from tqdm import tqdm
import pandas as pd
from tiatoolbox.annotation.storage import SQLiteStore, Annotation
from pathlib import Path
from argparse import ArgumentParser

def get_cell_stats_box(store: SQLiteStore, box: tuple[int, int, int, int], types: list):
    """Queries store for cells in box and returns stats"""
    cells = store.query(geometry=box)
    # each cell has a properties dict, make a dataframe of all the properties
    df = pd.DataFrame([c.properties for c in cells.values()])
    # get the area of each cells shapely geometry (cell.geometry.area)
    areas = [c.geometry.area for c in cells.values()]
    df['area'] = areas
    # get the number of cells
    n_cells = len(df)
    # type column is categorical, get the counts of each type in types. Some may be zero
    type_counts = df['type'].value_counts().reindex(types, fill_value=0)

    # histogram of distribution of areas
    bins = np.linspace(0, 2000, 200) # np.hstack(([0], np.linspace(0, 2000, 200)))
    area_hist, _ = np.histogram(df['area'], bins=bins)

    return n_cells, type_counts, area_hist, bins

def get_random_patch_locations(slide: str | Path, num_patches: int, patch_size: int, stride: int = None):
    """Use patch extractor to get random patch locations"""
    if stride is None:
        stride = patch_size
    extractor = SlidingWindowPatchExtractor(
        input_img=slide,
        patch_size=(patch_size, patch_size),
        stride=(stride, stride),
        input_mask='otsu',
        min_mask_ratio=0.9,
        within_bound=True,
        resolution=0,
        units="level",
    )
    coords = extractor.coordinate_list
    if num_patches >= len(coords):
        return coords
    inds = np.random.choice(len(coords), num_patches, replace=False)
    return coords[inds]

if __name__ == "__main__":
    # arg parsing: wsi, store, save_dir, n_tiles, patch_size, stride
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to store or folder of stores")
    parser.add_argument("--wsi", type=str, help="Path to wsi in case of wsi mode")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save dir")
    parser.add_argument("--n_tiles", type=int, required=True, help="Number of tiles to sample")
    parser.add_argument("--patch_size", type=int, required=True, help="Patch size")
    parser.add_argument("--stride", type=int, help="Stride")

    args = parser.parse_args()
    store_path = Path(args.input)
    save_dir = Path(args.save_dir)
    n_tiles = args.n_tiles
    patch_size = args.patch_size
    stride = args.stride if args.stride is not None else patch_size
    wsi = args.wsi if args.wsi is not None else None
    types = [0, 1, 2, 3, 4]

    if not Path(store_path).is_dir() and wsi:
        # in wsi mode with a single store for the wsi
        mode = 'wsi'
        reader = WSIReader.open(wsi)
        slide_dims = reader.info.slide_dimensions[::-1]
        # get global stats on whole slide
        store = SQLiteStore(store_path)
        global_n_cells, global_type_counts, global_area_hist, global_area_bins = get_cell_stats_box(store, (0, 0, slide_dims[0], slide_dims[1]), types)
        print(f"Global stats: {global_n_cells} cells, {global_type_counts} type counts, {global_area_hist} area hist")
        patches = get_random_patch_locations(wsi, n_tiles, patch_size, stride)
    elif Path(store_path).is_dir():
        # in tile mode with a folder of stores
        mode = 'tile'
        stores = list(Path(store_path).glob('*.db'))
        np.random.shuffle(stores)
        patches = [[0, 0, patch_size, patch_size] for _ in range(min(n_tiles, len(stores)))]

    n_cells, type_counts, area_hist = [], [], []
    # get stats on random patches
    for i, box in tqdm(enumerate(patches)):
        if mode == "wsi":
            n, tc, ah, _ = get_cell_stats_box(store, box, types)
        else:
            store = SQLiteStore(stores[i])
            n, tc, ah, _ = get_cell_stats_box(store, box, types)
        n_cells.append(n)
        type_counts.append(tc)
        area_hist.append(ah)
        

    # save stats in csv
    df = pd.DataFrame(type_counts)
    df['n_cells'] = n_cells
    df['area_hist'] = area_hist
    df.to_csv(save_dir / f"{input.stem}_cell_stats.csv")






