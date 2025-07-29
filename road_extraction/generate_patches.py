import os
import rasterio
import numpy as np
from rasterio.windows import Window
from tqdm import tqdm

img_path = "CBERS4_PAN.tif"
mask_path = "mascara_rodovias.tif"
output_dir = "dataset"
patch_size = 256

img_out_dir = os.path.join(output_dir, "images")
mask_out_dir = os.path.join(output_dir, "masks")
os.makedirs(img_out_dir, exist_ok=True)
os.makedirs(mask_out_dir, exist_ok=True)

with rasterio.open(img_path) as img_src, rasterio.open(mask_path) as mask_src:
    width = img_src.width
    height = img_src.height
    count = 0

    for i in tqdm(range(0, height, patch_size)):
        for j in range(0, width, patch_size):
            if i + patch_size > height or j + patch_size > width:
                continue
            window = Window(j, i, patch_size, patch_size)
            img_patch = img_src.read(1, window=window)
            mask_patch = mask_src.read(1, window=window)

            if np.sum(mask_patch) == 0:
                continue

            count += 1
            img_out = os.path.join(img_out_dir, f"patch_{count:04d}.tif")
            mask_out = os.path.join(mask_out_dir, f"patch_{count:04d}.tif")

            profile = img_src.profile
            profile.update({"height": patch_size, "width": patch_size, "transform": rasterio.windows.transform(window, img_src.transform)})

            with rasterio.open(img_out, "w", **profile) as dst_img:
                dst_img.write(img_patch, 1)
            with rasterio.open(mask_out, "w", **profile) as dst_mask:
                dst_mask.write(mask_patch, 1)

print(f"✅ {count} patches salvos.")
