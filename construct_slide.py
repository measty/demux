from demux_fixed import VirtualRestainer
from pathlib import Path
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from tiatoolbox.wsicore import WSIReader
import numpy as np
from tqdm import tqdm
from tiatoolbox.utils.image import imresize
import pyvips as vips
import os


def construct_slide(
    slide_path,
    mask,
    patch_size=256,
    proc_fn: callable = None,
    resolution=0,
    units="level",
    stride=None,
    save_path=None,
    back_heuristic="none",
    var_thresh=None,
    **kwargs,
):
    filename = Path(slide_path)
    if save_path is None:
        save_path = filename.with_name(filename.stem + "_proc.tiff")

    tmp_path = save_path.parent / (filename.stem + "_tmp_.npy")
    wsi = WSIReader.open(slide_path)
    canvas_shape = wsi.info.slide_dimensions[::-1]
    mpp = wsi.info.mpp
    back_level = 245
    out_ch = 4 if stride is not None else 3

    patch_extractor = SlidingWindowPatchExtractor(
        input_img=slide_path,
        patch_size=(patch_size, patch_size),
        stride=(stride, stride) if stride is not None else (patch_size, patch_size),
        input_mask=mask,  #'morphological',
        min_mask_ratio=0.1,  # only discard patches with very low tissue content
        within_bound=True,
        resolution=resolution,
        units=units,
    )

    locs = patch_extractor.coordinate_list[:, :2]

    cum_canvas = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        shape=tuple(canvas_shape) + (out_ch,),
        dtype=np.uint16 if stride is not None else np.uint8,
    )

    if stride is not None:
        cum_canvas[:] = 0
    else:
        cum_canvas[:] = back_level

    for i, tile in tqdm(enumerate(patch_extractor), total=len(patch_extractor)):
        # if variance of tile vals less than threshold, skip
        if var_thresh and np.var(tile) < var_thresh:
            rec = tile
        else:
            rec = proc_fn(tile)
            # if variance of processed tile vals less than threshold, skip
            if var_thresh and np.var(rec) < var_thresh:
                rec = tile
        # if tile is very dark, replace with background level
        if np.mean(rec) < 55:
            rec = np.ones_like(rec) * back_level
        x, y = locs[i]
        if resolution > 0 and units == "mpp":
            x, y = int(x * resolution / mpp[0]), int(y * resolution / mpp[1])
            out_size = (np.array(rec.shape[:2]) * resolution / mpp).astype(int) + 1
            rec = imresize(rec, output_size=out_size)
        if y + rec.shape[0] > canvas_shape[0] or x + rec.shape[1] > canvas_shape[1]:
            print("patch out of bounds, cropping.")
            rec = rec[: canvas_shape[0] - y, : canvas_shape[1] - x]
            if rec.shape[0] == 0 or rec.shape[1] == 0:
                continue
        if stride is None:
            cum_canvas[y : y + rec.shape[0], x : x + rec.shape[1], :3] = rec
        else:
            # keep track of how many times each pixel has been written to
            cum_canvas[y : y + rec.shape[0], x : x + rec.shape[1], 3] += 1
            # add the new tile to the canvas
            cum_canvas[y : y + rec.shape[0], x : x + rec.shape[1], :3] += rec
    if stride is not None:
        # set pixels that havent been written to background level
        cum_canvas[cum_canvas[:, :, 3] == 0, :3] = back_level
        # set pixel counts of background pixels to 1 to avoid divide by zero
        cum_canvas[cum_canvas[:, :, 3] == 0, 3] = 1
        # divide by the number of times each pixel was written to, patchwise to avoid memory issues
        for i in tqdm(range(0, cum_canvas.shape[0], patch_size)):
            for j in range(0, cum_canvas.shape[1], patch_size):
                cum_canvas[
                    i : min(i + patch_size, cum_canvas.shape[0]),
                    j : min(j + patch_size, cum_canvas.shape[1]),
                    :3,
                ] = (
                    cum_canvas[
                        i : min(i + patch_size, cum_canvas.shape[0]),
                        j : min(j + patch_size, cum_canvas.shape[1]),
                        :3,
                    ]
                    / cum_canvas[
                        i : min(i + patch_size, cum_canvas.shape[0]),
                        j : min(j + patch_size, cum_canvas.shape[1]),
                        3:4,
                    ]
                ).astype(
                    np.uint8
                )
        cum_canvas = cum_canvas[:, :, :3]

    # make a vips image and save it as a pyramidal tiff
    # height, width, bands = cum_canvas.shape
    # linear = cum_canvas.reshape(width * height * bands)
    vips_img = vips.Image.new_from_memory(
        cum_canvas[:, :, :3].astype(np.uint8).tobytes(),
        canvas_shape[1],
        canvas_shape[0],
        3,
        "uchar",
    )
    # set resolution metadata - tiffsave expects res in pixels per mm regardless of resunit
    # save_path = save_path / (filename.stem + '_proc.tiff')
    vips_img.tiffsave(
        save_path,
        tile=True,
        pyramid=True,
        compression="jpeg",
        Q=85,
        bigtiff=True,
        xres=1000 / mpp[0],
        yres=1000 / mpp[1],
        resunit="cm",
        tile_width=512,
        tile_height=512,
    )
    print(f"saved slide {filename.stem} to {save_path}")
    # close memmap and clean up
    cum_canvas._mmap.close()
    del cum_canvas
    os.remove(tmp_path)


if __name__ == "__main__":
    slide_path = Path(
        r"/media/u2071810/Data/Multiplexstaining/Asmaa_Multiplex_Staining"
    )
    save_path = Path(r"/home/u2071810/Data/Demux/demux_restains")
    filter = "*PHH3_HE.svs"
    slides = list(slide_path.glob(filter))
    stains = ["HE", "HD"]
    mode = "wsi"
    slides_keep = []
    for slide in slides:
        info_path = slide.parent / (slide.stem + "_info.pkl")
        if info_path.exists():
            slides_keep.append(slide)
    slides = slides_keep

    for slide in slides[:2]:
        print(f"Restaining {slide}")
        restainer = VirtualRestainer(
            slide, coupling_coeffs={"wdh": 0.8, "wde": 0.1, "weh": 0.1, "wed": 0.1}
        )
        info_path = slide.parent / (slide.stem + "_info.pkl")
        try:
            restainer.load_info(info_path)
        except:
            print("No info file found")
            continue
        if mode == "wsi":
            if not (save_path / (slide.stem + "_restained_HE.tiff")).exists():
                construct_slide(
                    slide,
                    "otsu",
                    patch_size=4096,
                    proc_fn=restainer,
                    resolution=0,
                    stride=2048,
                    save_path=save_path / (slide.stem + "_restained_HE.tiff"),
                )

    # if "*" in slide_path.name:
    #     slide_filter = slide_path.name
    #     slide_path = slide_path.parent
    # else:
    #     slide_filter = "*"
    # save_path.mkdir(exist_ok=True)
    # if slide_path.is_dir():
    #     slides = list(slide_path.glob(slide_filter))
    # else:
    #     slides = [slide_path]
    # slides.sort()

    # print(f"found {len(slides)} slides")
    # print(f"processing {min(len(slides), opt.num_test)} slides")
    # for slide in slides[:min(len(slides), opt.num_test)]:
    #     # try to get mask
    #     if isinstance(mask_opt, Path):
    #         mask = mask_opt / (slide.stem + '_mask.png')
    #     else:
    #         mask = mask_opt
    #     print(f"starting slide {slide}")
    #     construct_slide(slide, mask, model=ModelWrapper(model), resolution=resolution, units=units, stride=stride, save_path=save_path, back_heuristic=back_heuristic, var_thresh=var_thresh)
