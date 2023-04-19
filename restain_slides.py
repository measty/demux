from demux_fixed import VirtualRestainer
from pathlib import Path

slide_path = Path(r"/media/u2071810/Data/Multiplexstaining/Asmaa Multiplex Staining")
tile_path = Path(r"/media/u2071810/Data/Multiplexstaining/Asmaa Multiplex Staining/tiles_hybrid")
if not tile_path.exists():
    tile_path.mkdir()
filter = "*PHH3_HE.svs"
slides = list(slide_path.glob(filter))
stains = ["HE","HD"]
mode = "tile"

for slide in slides[:5]:
    print(f"Restaining {slide}")
    restainer = VirtualRestainer(slide, coupling_coeffs={"wdh":0.8, "wde":0.1, "weh":0.1, "wed":0.1})
    info_path = slide.parent / (slide.stem + "_info_hybrid.pkl")
    try:
        restainer.load_info(info_path)
    except:
        print("No info file found")
        continue
    if mode == "wsi":
        if not (slide.parent / (slide.stem + "_restained18_HE.tiff")).exists():
            restainer.save_restained_WSI(slide.parent / (slide.stem + "_restained18.tiff"), stains=stains, lum_norm=False)
    else:
        restainer.save_sample_tiles(tile_path, stains=stains, lum_norm=False, tile_size=2000, n_tiles=20)

print("Done")
