import os
import numpy as np
import rasterio
from PIL import Image
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

class RoadIntersectionDataset(Dataset):
    '''
    bands_mode:
      - "rgb"    → 3 canais (R,G,B). Se houver 4 bandas, descarta NIR.
      - "rgbnir" → 4 canais (R,G,B,NIR). Se houver só 3, adiciona NIR vazio.
    '''
    def __init__(self, images_dir, masks_dir=None, transform=None, is_training=False, bands_mode="rgb"):
        assert bands_mode in ("rgb", "rgbnir")
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.is_training = is_training
        self.bands_mode = bands_mode

        self.images = sorted([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
        self.masks = sorted([f for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f))]) if masks_dir else None
        if self.masks is not None:
            assert len(self.images) == len(self.masks), "Número de imagens e máscaras não bate!"

        print("🔍 Calculando estatísticas do dataset...")
        self.mean, self.std = self._calculate_dataset_stats()
        print(f"📊 Média por canal: {self.mean}")
        print(f"📊 Desvio padrão por canal: {self.std}")

        self.transform = transform if transform is not None else self._get_default_transforms(is_training)

    def _read_image_hwc_allbands(self, path):
        with rasterio.open(path) as src:
            arr = src.read()
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        return arr.transpose(1, 2, 0)

    def _select_bands(self, image_hwc):
        H, W, C = image_hwc.shape
        if self.bands_mode == "rgb":
            if C >= 3:
                return image_hwc[:, :, :3]
            if C == 2:
                return np.concatenate([image_hwc, image_hwc[:, :, :1]], axis=2)
            return np.repeat(image_hwc, 3, axis=2)
        else:  # "rgbnir"
            if C >= 4:
                return image_hwc[:, :, :4]
            if C == 3:
                nir_zero = np.zeros((H, W, 1), dtype=image_hwc.dtype)
                return np.concatenate([image_hwc, nir_zero], axis=2)
            if C == 2:
                tmp3 = np.concatenate([image_hwc, image_hwc[:, :, :1]], axis=2)
                nir_zero = np.zeros((H, W, 1), dtype=image_hwc.dtype)
                return np.concatenate([tmp3, nir_zero], axis=2)
            tmp3 = np.repeat(image_hwc, 3, axis=2)
            nir_zero = np.zeros((H, W, 1), dtype=image_hwc.dtype)
            return np.concatenate([tmp3, nir_zero], axis=2)

    def _read_image_selected(self, path):
        hwc = self._read_image_hwc_allbands(path)
        return self._select_bands(hwc)

    def _calculate_dataset_stats(self):
        '''
        Estatísticas por canal após a seleção de bandas (3 ou 4).
        '''
        sums = None
        sums2 = None
        count = 0
        for img_name in tqdm(self.images, desc="Calculando estatísticas"):
            img_path = os.path.join(self.images_dir, img_name)
            try:
                image = self._read_image_selected(img_path)
            except Exception:
                continue
            H, W, C = image.shape
            flat = image.reshape(-1, C).astype(np.float64)
            if sums is None:
                sums = flat.sum(axis=0)
                sums2 = (flat ** 2).sum(axis=0)
            else:
                sums += flat.sum(axis=0)
                sums2 += (flat ** 2).sum(axis=0)
            count += flat.shape[0]
        mean = (sums / count).tolist()
        var = (sums2 / count) - (np.array(mean) ** 2)
        std = np.sqrt(np.clip(var, 0, None)).tolist()
        return mean, std

    def _get_default_transforms(self, is_training=False):
        base = [
            A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0),
            ToTensorV2()
        ]
        if is_training:
            aug = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            ]
            return A.Compose(aug + base)
        return A.Compose(base)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = self._read_image_selected(img_path)

        if self.masks is not None:
            mask_path = os.path.join(self.masks_dir, self.masks[idx])
            mask = np.array(Image.open(mask_path).convert("L"))
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            mask = (mask > 0.5).float().unsqueeze(0)
            return image, mask
        else:
            transformed = A.Compose([
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0),
                ToTensorV2()
            ])(image=image)
            image = transformed['image']
            return image, self.images[idx]


def get_training_transforms(mean, std):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ])

def get_validation_transforms(mean, std):
    return A.Compose([
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ])

if __name__ == "__main__":
    '''
    Exemplos de uso:
      - RGB (3 canais): instancie sua UNet com in_channels=3.
      - RGB+NIR (4 canais): instancie sua UNet com in_channels=4.  (A UNet suporta in_channels variável.)  # :contentReference[oaicite:0]{index=0}
    '''
    # RGB (sem NIR)
    train_rgb = RoadIntersectionDataset(
        images_dir="dataset/train/images",
        masks_dir="dataset/train/masks",
        is_training=True,
        bands_mode="rgb"
    )
    val_rgb = RoadIntersectionDataset(
        images_dir="dataset/val/images",
        masks_dir="dataset/val/masks",
        is_training=False,
        bands_mode="rgb"
    )
    print(f"[RGB] Train={len(train_rgb)}  Val={len(val_rgb)}  mean={train_rgb.mean}  std={train_rgb.std}")

    # RGB+NIR
    train_rgbnir = RoadIntersectionDataset(
        images_dir="dataset/train/images",
        masks_dir="dataset/train/masks",
        is_training=True,
        bands_mode="rgbnir"
    )
    val_rgbnir = RoadIntersectionDataset(
        images_dir="dataset/val/images",
        masks_dir="dataset/val/masks",
        is_training=False,
        bands_mode="rgbnir"
    )
    print(f"[RGBNIR] Train={len(train_rgbnir)}  Val={len(val_rgbnir)}  mean={train_rgbnir.mean}  std={train_rgbnir.std}")
