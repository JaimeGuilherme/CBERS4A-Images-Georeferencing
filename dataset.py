import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class RoadIntersectionDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))

        if masks_dir is not None:
            self.masks = sorted(os.listdir(masks_dir))
            assert len(self.images) == len(self.masks), "Número de imagens e máscaras não bate!"
        else:
            self.masks = None

        self.default_image_transform = T.ToTensor()
        self.default_mask_transform = T.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        if self.masks is not None:
            mask_path = os.path.join(self.masks_dir, self.masks[idx])
            mask = Image.open(mask_path).convert("L")

            if self.transform:
                image, mask = self.transform(image, mask)
            else:
                image = self.default_image_transform(image)
                mask = self.default_mask_transform(mask)
                mask = (mask > 0.5).float()

            return image, mask
        else:
            if self.transform:
                image = self.transform(image)
            else:
                image = self.default_image_transform(image)

            return image, self.images[idx]
