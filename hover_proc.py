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

wsi_folder = Path(r"/media/u2071810/Data/Multiplexstaining/Asmaa Multiplex Staining/")
mask_folder = None #Path(r"/media/u2071810/Data/ABCTB/histoqc_output_may_2022")
save_dir = Path(r"/home/u2071810/Data/Demux/overlays/")
tmp_save_dir = str(save_dir / "tmp")
doing_now = save_dir / "current_files.pkl"
filter_str = '*restained6_HE.tiff'

if __name__=='__main__':
    if not save_dir.exists():
        save_dir.mkdir()

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
    doing_files = []
    skip_files = doing_files + done_files
    number_to_do = 50
    slide_list = list(wsi_folder.glob(filter_str))
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
        batch_size=48,
        auto_generate_mask=True,
        verbose=False,
    )
    inst_segmentor.ioconfig.tile_shape = (4000, 4000)

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
                mode="wsi",
                on_gpu=True,
                crash_on_exception=True,
            )

        # move the file to the save_dir
        res_path = Path(save_dir) / f'{slide_path.stem}.db'
        try:
            #put the annotations in a store
            mpp = WSIReader.open(slide_path).info.mpp
            SQ = store_from_dat(Path(tmp_save_dir) / "0.dat", scale_factor=0.25/mpp)
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