from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.misc import imread, store_from_dat
from tiatoolbox.wsicore.wsireader import WSIReader

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from pathlib import Path
from shutil import rmtree
import pickle
import argparse


if __name__=='__main__':


    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--wsi_folder', type=Path, required=True, help='Path to folder containing WSI files.')
    parser.add_argument('--mask_folder', type=Path, help='Path to folder containing mask files.')
    parser.add_argument('--save_dir', type=Path, required=True, help='Path to folder where output files will be saved.')
    parser.add_argument('--filter_str', type=str, default='*HE.png', help='Filter string for selecting input files.')
    parser.add_argument('--mode', type=str, default='tile', choices=['wsi', 'tile'], help='Processing mode (wsi or tile).')
    parser.add_argument('--names', type=str, nargs='+', help='List of slide names from the filtered list to process.')
                        

    args = parser.parse_args()

    wsi_folder = Path(args.wsi_folder)
    mask_folder = Path(args.mask_folder) if args.mask_folder else None
    save_dir = Path(args.save_dir)
    filter_str = args.filter_str
    mode = args.mode
    names = args.names if args.names else None

    doing_now = save_dir / "current_files.pkl"
    if not save_dir.exists():
        save_dir.mkdir()
    tmp_save_dir = str(save_dir / "tmp")

    done_files = save_dir.glob('*.dat')
    done_files = [f.stem for f in done_files]
    if doing_now.exists():
        try:
            with open(doing_now, 'rb') as f:
                doing_files = pickle.load(f)
        except:
            print('problem with current files pkl')
            doing_files = []
    else:
        doing_files = []
    doing_files = [] #force it to do all the files, remove if another run
    # is in progress
    skip_files = doing_files + done_files
    number_to_do = 10
    slide_list = list(wsi_folder.glob(filter_str))
    if names:
        named_files = []
        for name in names:
            named_files.extend([f for f in slide_list if name in f.stem])
        slide_list = named_files
    to_do = []
    masks = []
    #get the masks
    for slide in slide_list:
        if slide.stem in skip_files:
            #weve already done it
            continue
        slide_name = slide.name
        mask = mask_folder / f"{slide_name}" / f"{slide_name}_mask_use.png" if mask_folder else None
        if mask and mask.exists():
            masks.append(mask)
        else:
            print(f"no mask for file: {slide_name}")
            masks.append(None)
        to_do.append(slide)
        if len(to_do) >= number_to_do:
            break

    with open(doing_now, 'wb') as f:
        pickle.dump(doing_files + [td.stem for td in to_do], f)

    # Instantiate the nucleus instance segmentor
    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        num_loader_workers=6,
        num_postproc_workers=12,
        batch_size=16,
        auto_generate_mask=True,
        verbose=False,
    )
    if mode == 'wsi':
        inst_segmentor.ioconfig.tile_shape = (4000, 4000)
    else:
        unit = 'baseline'
        res = 1.0
        inst_segmentor.ioconfig.input_resolutions = [{'units': unit, 'resolution': res}]
        inst_segmentor.ioconfig.output_resolutions = [{'units': unit, 'resolution': res}, {'units': unit, 'resolution': res}, {'units': unit, 'resolution': res}]
        inst_segmentor.ioconfig.save_resolution = {'units': unit, 'resolution': res}
        inst_segmentor.ioconfig.highest_input_resolution = {'units': unit, 'resolution': res}

    print(f'processing: {to_do}')

    for i, (slide_path, mask_path) in enumerate(zip(to_do, masks)):
        tmp_save_dir = str(Path(tmp_save_dir).with_name(Path(tmp_save_dir).name.split('_')[0] + f'_{i}'))
        #remove tmp save dir if it exists
        if Path(tmp_save_dir).exists():
            rmtree(tmp_save_dir)

        # predict on a slide
        wsi_output = inst_segmentor.predict(
                [slide_path],
                masks=[mask_path],
                save_dir=tmp_save_dir,
                mode=mode,
                on_gpu=True,
                crash_on_exception=True,
            )

        # move the file to the save_dir
        res_path = Path(save_dir) / f'{slide_path.stem}.db'
        try:
            #put the annotations in a store
            if mode == 'wsi':
                mpp = WSIReader.open(slide_path).info.mpp
                SQ = store_from_dat(Path(tmp_save_dir) / "0.dat", scale_factor=0.25/mpp)
            else:
                SQ = store_from_dat(Path(tmp_save_dir) / "0.dat")
            SQ.dump(res_path)
        except:
            print(f'failed to process: {slide_path}')
            #remove tmp save dir if it exists
            if Path(tmp_save_dir).exists():
                rmtree(tmp_save_dir)
            continue

        #remove tmp save dir if it exists
        if Path(tmp_save_dir).exists():
            rmtree(tmp_save_dir)

        print(f'sucessfully processed slide {i}: {slide_path}')
        
    if Path(tmp_save_dir).exists():
            rmtree(tmp_save_dir)
    #if history.exists():
    #    with open(history, 'rb') as f:
    #        done_files = pickle.load(f)

    # check the files
    for slide in to_do:
        out_file = Path(save_dir) / f'{slide.stem}.db'
        if out_file.exists():
            print(f'done {slide}')
            done_files.append(slide)

    #with open(history, 'wb') as f:
    #    pickle.dump(done_files, f)

    if doing_now.exists():
        with open(doing_now, 'rb') as f:
            doing_files = pickle.load(f)

    for file in to_do:
        doing_files.remove(file.stem)

    with open(doing_now, 'wb') as f:
        pickle.dump(doing_files, f)