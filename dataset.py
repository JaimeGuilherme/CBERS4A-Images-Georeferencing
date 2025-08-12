import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class RoadIntersectionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.tif')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.tif')])
        assert len(self.images) == len(self.masks), 'Quantidade de imagens e máscaras difere'
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        img_name = self.images[idx]; mask_name = self.masks[idx]
        assert os.path.splitext(img_name)[0] == os.path.splitext(mask_name)[0], f"Nome de imagem e máscara diferentes: {img_name}, {mask_name}"
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()
        if self.transform:
            image = self.transform(image)
        return image, mask
