from demux_fixed import VirtualRestainer
from pathlib import Path

slide_path = Path(r"/media/u2071810/Data1/Multiplexstaining/Asmaa_Multiplex_Staining")
tile_path = Path(r"/media/u2071810/Data1/Multiplexstaining/Asmaa_Multiplex_Staining/tiles_pix2pix2")
if not tile_path.exists():
    tile_path.mkdir()
filter = "*PHH3_HE.svs"
slides = list(slide_path.glob(filter))
stains = ["HE","HD"]
mode = "wsi"
slides_keep = []
for slide in slides:
    info_path = slide.parent / (slide.stem + "_info_hybrid.pkl")
    if info_path.exists():
        slides_keep.append(slide)
slides = slides_keep

for slide in slides[:10]:
    print(f"Restaining {slide}")
    restainer = VirtualRestainer(slide, coupling_coeffs={"wdh":0.8, "wde":0.1, "weh":0.1, "wed":0.1})
    info_path = slide.parent / (slide.stem + "_info_hybrid.pkl")
    try:
        restainer.load_info(info_path)
    except:
        print("No info file found")
        continue
    if mode == "wsi":
        if not (slide.parent / (slide.stem + "_restained_hy_HE.tiff")).exists():
            restainer.save_restained_WSI(slide.parent / (slide.stem + "_restained_hy.tiff"), stains=stains, lum_norm=False)
    else:
        restainer.save_sample_tiles(tile_path, stains=stains, lum_norm=False, tile_size=512, n_tiles=100, save_d=False, sep_folders=True)

print("Done")
