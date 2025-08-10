import os
import shutil
import random
import numpy as np
import rasterio
import fiona
from rasterio.windows import Window
from PIL import Image, ImageDraw

def criar_mascara(points, img_width, img_height, buffer_pixels=3):
    mask = Image.new('L', (img_width, img_height), 0)
    draw = ImageDraw.Draw(mask)
    for pt in points:
        x, y = pt
        bbox = [x - buffer_pixels, y - buffer_pixels, x + buffer_pixels, y + buffer_pixels]
        draw.ellipse(bbox, fill=255)
    return np.array(mask) > 0

def carregar_pontos(arquivo_pontos, transform, img_width, img_height):
    points_px = []
    with fiona.open(arquivo_pontos, 'r') as src:
        for feat in src:
            geom = feat.get('geometry')
            if geom is None or geom.get('type') != 'Point':
                continue
            lon, lat = geom['coordinates']
            px, py = ~transform * (lon, lat)
            px, py = int(px), int(py)
            if 0 <= px < img_width and 0 <= py < img_height:
                points_px.append((px, py))
    return points_px

def salvar_tif(path, array, count, height, width, transform, crs):
    dtype = array.dtype
    with rasterio.open(path, 'w', driver='GTiff', height=height, width=width, count=count, dtype=dtype, transform=transform, crs=crs) as dst:
        if count == 1:
            dst.write(array, 1)
        else:
            for i in range(count):
                dst.write(array[i], i+1)

def gerar_patches_com_buffer(imagem_path, pontos_path, saida_path, patch_size=256, buffer_pixels=3):
    os.makedirs(os.path.join(saida_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(saida_path, 'masks'), exist_ok=True)
    with rasterio.open(imagem_path) as src:
        transform = src.transform
        crs = src.crs
        img_width = src.width
        img_height = src.height
        points_px = carregar_pontos(pontos_path, transform, img_width, img_height)
        patch_id = 0
        for top in range(0, img_height, patch_size):
            for left in range(0, img_width, patch_size):
                width = min(patch_size, img_width - left)
                height = min(patch_size, img_height - top)
                window = Window(left, top, width, height)
                patch = src.read(window=window)
                patch_img = patch.astype(np.float32)
                patch_img -= patch_img.min()
                if patch_img.max() > 0:
                    patch_img /= patch_img.max()
                patch_img *= 255
                patch_img = patch_img.astype('uint8')
                if patch_img.shape[0] < 3:
                    patch_img = np.repeat(patch_img, 3, axis=0)
                elif patch_img.shape[0] > 3:
                    patch_img = patch_img[:3]
                patch_points = []
                for (px, py) in points_px:
                    if left <= px < left + width and top <= py < top + height:
                        patch_points.append((px - left, py - top))
                if len(patch_points) == 0:
                    continue
                mask = criar_mascara(patch_points, width, height, buffer_pixels)
                mask_uint8 = (mask * 255).astype('uint8').reshape(1, height, width)
                patch_transform = src.window_transform(window)
                img_filename = f"patch_{patch_id:04d}.tif"
                salvar_tif(os.path.join(saida_path, 'images', img_filename), patch_img, count=3, height=height, width=width, transform=patch_transform, crs=crs)
                salvar_tif(os.path.join(saida_path, 'masks', img_filename), mask_uint8, count=1, height=height, width=width, transform=patch_transform, crs=crs)
                patch_id += 1
    print(f"Patches gerados: {patch_id}")
    separar_patches(saida_path, 'dataset', 0.6, 0.3, 0.1)

def separar_patches(patches_dir, output_dir, train_pct=0.6, val_pct=0.3, test_pct=0.1, seed=42):
    images_path = os.path.join(patches_dir, 'images')
    masks_path = os.path.join(patches_dir, 'masks')
    imagens = sorted(os.listdir(images_path))
    mascaras = sorted(os.listdir(masks_path))
    assert len(imagens) == len(mascaras), 'Número de imagens e máscaras difere'
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train','val','test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)
    indices = list(range(len(imagens)))
    random.seed(seed)
    random.shuffle(indices)
    total = len(indices)
    train_end = int(total * train_pct)
    val_end = train_end + int(total * val_pct)
    for i, idx in enumerate(indices):
        if i < train_end:
            split='train'
        elif i < val_end:
            split='val'
        else:
            split='test'
        shutil.copy(os.path.join(images_path, imagens[idx]), os.path.join(output_dir, split, 'images', imagens[idx]))
        shutil.copy(os.path.join(masks_path, mascaras[idx]), os.path.join(output_dir, split, 'masks', mascaras[idx]))
    print('Separação concluída.')

if __name__ == '__main__':
    input_dir = 'input'
    output_dir = 'dataset'
    patch_size = 256
    buffer_pixels = 8

    imagem_tif = None
    pontos_path = None

    for file in os.listdir(input_dir):
        if file.lower().endswith('.tif'):
            imagem_tif = os.path.join(input_dir, file)
        elif file.lower().endswith(('.geojson', '.shp', '.gpkg')):
            pontos_path = os.path.join(input_dir, file)

    if imagem_tif is None or pontos_path is None:
        print("Erro: Não encontrou arquivo .tif e/ou arquivo de pontos (.geojson, .shp ou .gpkg) na pasta input.")
    else:
        print(f"Usando imagem: {imagem_tif}")
        print(f"Usando arquivo de pontos: {pontos_path}")
        gerar_patches_com_buffer(imagem_tif, pontos_path, output_dir, patch_size=patch_size, buffer_pixels=buffer_pixels)
