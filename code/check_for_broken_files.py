import os
import tifffile as tiff
from PIL import Image


bad_files = []

root = "./data/preprocessed_png_256"
check_for_tiff = False

for fname in os.listdir(root):
    if not fname.lower().endswith(".png"):
        continue
    path = os.path.join(root, fname)
    try:
        if check_for_tiff:
            with tiff.TiffFile(path) as tif:
                _ = tif.asarray()
        else:
            im = Image.open(path).convert("RGB")
    except Exception as e:
        print("BAD:", path, e)
        bad_files.append(path)

print("Total bad files:", len(bad_files))
