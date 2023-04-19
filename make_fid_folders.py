from filtered_patch_ds import WSIFolderPatchDataset
from pathlib import Path
#from pympler.asizeof import asizeof

if __name__ == "__main__":

    slide_path = Path(r"E:\PRISMATIC\Asmaa_Multiplex_Staining")
    patch_path = Path(r"E:\PRISMATIC\fid_patches\restained5")

    def filter_fn_he(slide_path: Path):
        return slide_path.name.endswith("_HE.svs") and "PHH3" not in slide_path.name

    def filter_fn_hed(slide_path: Path):
        return slide_path.name.endswith("PHH3_HE.svs")

    def filter_fn_hd(slide_path: Path):
        return slide_path.name.endswith("_PHH3.svs") and "HE" not in slide_path.name

    def filter_fn_restained(slide_path: Path):
        return slide_path.name.endswith("_restained5_HE.tiff")

    def filter_fn_restained_hd(slide_path: Path):
        return slide_path.name.endswith("_restained3_HD.tiff")


    wsi_dataset = WSIFolderPatchDataset(
        [slide_path],
        patch_input_shape=(512, 512),
        stride_shape=(512, 512),
        slide_filter=filter_fn_restained,
        resolution=0,
        units="level",
        min_mask_ratio=0.75,
    )

    # make some random patches from real HE slides
    wsi_dataset.save_random_patches(patch_path, 500, include_slide_name=True)
    print(f"saved patches to {patch_path}")