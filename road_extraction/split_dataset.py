import os
import shutil
import random

orig_images = "dataset/images"
orig_masks = "dataset/masks"
output_base = "dataset_split"
train_ratio = 0.8
random.seed(42)

all_patches = sorted(os.listdir(orig_images))
random.shuffle(all_patches)

n_total = len(all_patches)
n_train = int(n_total * train_ratio)

train_patches = all_patches[:n_train]
val_patches = all_patches[n_train:]

def copy_files(patches, dst_img_dir, dst_mask_dir):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)
    for patch in patches:
        shutil.copy2(os.path.join(orig_images, patch), os.path.join(dst_img_dir, patch))
        shutil.copy2(os.path.join(orig_masks, patch), os.path.join(dst_mask_dir, patch))

copy_files(train_patches, f"{output_base}/train/images", f"{output_base}/train/masks")
copy_files(val_patches, f"{output_base}/val/images", f"{output_base}/val/masks")

print(f"✅ Total: {n_total} | Treino: {len(train_patches)} | Validação: {len(val_patches)}")
