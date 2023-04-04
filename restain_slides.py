from demux_fixed import VirtualRestainer
from pathlib import Path

slide_path = Path(r"/media/u2071810/Data/Multiplexstaining/Asmaa Multiplex Staining")
filter = "*PHH3_HE.svs"
slides = list(slide_path.glob(filter))
stains = ["HE","HD"]

for slide in slides[:5]:
    print(f"Restaining {slide}")
    restainer = VirtualRestainer(slide, coupling_coeffs={"wdh":0.8, "wde":0.1, "weh":0.1, "wed":0.1})
    info_path = slide.parent / (slide.stem + "_info2.pkl")
    try:
        restainer.load_info(info_path)
    except:
        print("No info file found")
        continue
    if not (slide.parent / (slide.stem + "_restained5_HE.tiff")).exists():
        restainer.save_restained_WSI(slide.parent / (slide.stem + "_restained5.tiff"), stains=stains, lum_norm=False)

print("Done")
