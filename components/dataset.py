import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

class RoadIntersectionDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, transform=None, is_training=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.is_training = is_training
        self.images = sorted(os.listdir(images_dir))

        if masks_dir is not None:
            self.masks = sorted(os.listdir(masks_dir))
            assert len(self.images) == len(self.masks), "Número de imagens e máscaras não bate!"
        else:
            self.masks = None

        # Calcular estatísticas do dataset (média e desvio padrão por canal)
        print("🔍 Calculando estatísticas do dataset...")
        self.mean, self.std = self._calculate_dataset_stats()
        print(f"📊 Média por canal: {self.mean}")
        print(f"📊 Desvio padrão por canal: {self.std}")

        # Definir transformações baseadas se é treinamento ou não
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms(is_training)

    def _calculate_dataset_stats(self):
        """Calcular média e desvio padrão por canal do dataset"""
        pixel_values_r = []
        pixel_values_g = []
        pixel_values_b = []
        
        # Iterar por todas as imagens para calcular estatísticas
        for img_name in tqdm(self.images, desc="Calculando estatísticas"):
            img_path = os.path.join(self.images_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            image = np.array(image) #/ 255.0  # Normalizar para [0, 1]
            
            # Coletar valores por canal
            pixel_values_r.extend(image[:, :, 0].flatten())
            pixel_values_g.extend(image[:, :, 1].flatten())
            pixel_values_b.extend(image[:, :, 2].flatten())
        
        # Calcular média e desvio padrão por canal
        mean_r = np.mean(pixel_values_r)
        mean_g = np.mean(pixel_values_g)
        mean_b = np.mean(pixel_values_b)
        
        std_r = np.std(pixel_values_r)
        std_g = np.std(pixel_values_g)
        std_b = np.std(pixel_values_b)
        
        return [mean_r, mean_g, mean_b], [std_r, std_g, std_b]

    def _get_default_transforms(self, is_training=False):
        """Criar pipeline de transformações usando Albumentations"""
        
        if is_training:
            # Transformações para treinamento (com augmentations)
            transform_list = [
                # Transformações geométricas
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                
                # Transformações de cor/brilho
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                
                # Normalização com estatísticas calculadas do dataset
                A.Normalize(
                    mean=self.mean,
                    std=self.std,
                    max_pixel_value=255.0
                ),
                
                # Converter para tensor
                ToTensorV2()
            ]
        else:
            # Transformações para validação/teste (sem augmentations)
            transform_list = [
                # Apenas normalização e conversão para tensor
                A.Normalize(
                    mean=self.mean,
                    std=self.std,
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ]

        return A.Compose(transform_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Carregar imagem
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)  # Albumentations trabalha com numpy arrays

        if self.masks is not None:
            # Carregar máscara
            mask_path = os.path.join(self.masks_dir, self.masks[idx])
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask)  # Converter para numpy array
            
            # Aplicar transformações em imagem e máscara simultaneamente
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
            # Converter máscara para formato binário (0 ou 1)
            mask = (mask > 0.5).float().unsqueeze(0)  # Adicionar dimensão de canal
            
            return image, mask
        else:
            # Apenas imagem (para inferência)
            # Criar transform sem máscara usando as estatísticas calculadas
            image_only_transform = A.Compose([
                A.Normalize(
                    mean=self.mean,
                    std=self.std,
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ])
            
            transformed = image_only_transform(image=image)
            image = transformed['image']
            
            return image, self.images[idx]


# Função auxiliar para criar transforms customizados
def get_training_transforms(mean, std):
    """
    Criar transformações de treinamento customizadas
    
    Args:
        mean: lista com médias por canal [R, G, B]
        std: lista com desvios padrão por canal [R, G, B]
    """
    transform_list = [
        # Augmentations geométricas
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Augmentations de cor
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.6
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.4
        ),
        
        # Normalização com estatísticas do dataset
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255.0
        ),
        
        # Converter para tensor
        ToTensorV2()
    ]
    
    return A.Compose(transform_list)


def get_validation_transforms(mean, std):
    """
    Criar transformações de validação (sem augmentations)
    
    Args:
        mean: lista com médias por canal [R, G, B]
        std: lista com desvios padrão por canal [R, G, B]
    """
    transform_list = [
        # Apenas normalização
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255.0
        ),
        
        # Converter para tensor
        ToTensorV2()
    ]
    
    return A.Compose(transform_list)


# Exemplo de uso:
if __name__ == "__main__":
    # Dataset para treinamento com augmentations padrão
    train_dataset = RoadIntersectionDataset(
        images_dir="dataset/train/images",
        masks_dir="dataset/train/masks",
        is_training=True
    )
    
    # Dataset para validação sem augmentations
    val_dataset = RoadIntersectionDataset(
        images_dir="dataset/val/images", 
        masks_dir="dataset/val/masks",
        is_training=False
    )
    
    # Dataset com transforms customizados (usando estatísticas do dataset de treino)
    custom_train_transforms = get_training_transforms(
        mean=train_dataset.mean, 
        std=train_dataset.std
    )
    custom_train_dataset = RoadIntersectionDataset(
        images_dir="dataset/train/images",
        masks_dir="dataset/train/masks", 
        transform=custom_train_transforms
    )
    
    print(f"Train dataset: {len(train_dataset)} amostras")
    print(f"Val dataset: {len(val_dataset)} amostras")
    print(f"Média calculada: {train_dataset.mean}")
    print(f"Desvio padrão calculado: {train_dataset.std}")
    
    # Testar uma amostra
    image, mask = train_dataset[0]
    print(f"Forma da imagem: {image.shape}")
    print(f"Forma da máscara: {mask.shape}")
    print(f"Valores únicos na máscara: {torch.unique(mask)}")
    print(f"Range da imagem normalizada: [{image.min():.3f}, {image.max():.3f}]")
