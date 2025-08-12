import os
import shutil
import random
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import math
import fiona
from tqdm import tqdm

<<<<<<< Updated upstream
def criar_mascara(points, img_width, img_height, buffer_pixels=8):
=======
def criar_mascara(points, img_width, img_height, buffer_pixels=6):
>>>>>>> Stashed changes
    from PIL import Image, ImageDraw
    mask = Image.new('L', (img_width, img_height), 0)
    draw = ImageDraw.Draw(mask)

    for pt in points:
        x, y = pt
        bbox = [x - buffer_pixels, y - buffer_pixels, x + buffer_pixels, y + buffer_pixels]
        draw.ellipse(bbox, fill=255)

    return np.array(mask) > 0

<<<<<<< Updated upstream
def carregar_pontos_bufferados(arquivo_pontos, transform, img_width, img_height, buffer_pixels=8, img_crs_epsg=32724):
=======
def carregar_pontos_bufferados(arquivo_pontos, transform, img_width, img_height, buffer_pixels=6, img_crs_epsg=32724):
>>>>>>> Stashed changes
    points_px = []
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{img_crs_epsg}", always_xy=True)

    with fiona.open(arquivo_pontos, 'r') as src:
        for feat in src:
            geom = feat['geometry']
            if geom['type'] != 'Point':
                continue
            lon, lat = geom['coordinates']
            x_m, y_m = transformer.transform(lon, lat)
            px, py = ~transform * (x_m, y_m)
            px, py = int(px), int(py)
            if 0 <= px < img_width and 0 <= py < img_height:
                points_px.append((px, py))
    return points_px

def salvar_tif(path, array, count, height, width, transform, crs):
    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=array.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(array)

def separar_patches(patches_dir, output_dir, train_pct=0.6, val_pct=0.3, test_pct=0.1, seed=42):
    random.seed(seed)
    images_path = os.path.join(patches_dir, 'images')
    masks_path = os.path.join(patches_dir, 'masks')
    imagens = sorted(os.listdir(images_path))
    mascaras = sorted(os.listdir(masks_path))
    assert len(imagens) == len(mascaras), "Número de imagens e máscaras não bate"

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)

    total = len(imagens)
    train_end = int(total * train_pct)
    val_end = train_end + int(total * val_pct)
    indices = list(range(total))
    random.shuffle(indices)

    for i, idx in enumerate(indices):
        if i < train_end:
            split = 'train'
        elif i < val_end:
            split = 'val'
        else:
            split = 'test'

        img_name = imagens[idx]
        mask_name = mascaras[idx]
        shutil.copy(os.path.join(images_path, img_name), os.path.join(output_dir, split, 'images', img_name))
        shutil.copy(os.path.join(masks_path, mask_name), os.path.join(output_dir, split, 'masks', mask_name))

    print(f"Distribuição concluída:\n Treino: {train_end}\n Validação: {val_end - train_end}\n Teste: {total - val_end}")

<<<<<<< Updated upstream
def gerar_patches_com_buffer(imagem_path, pontos_path, saida_path, patch_size=256, buffer_pixels=8, limite_patches=None):
=======
def gerar_patches_com_buffer(imagem_path, pontos_path, saida_path, patch_size=256, buffer_pixels=6, limite_patches=None):
>>>>>>> Stashed changes
    os.makedirs(os.path.join(saida_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(saida_path, 'masks'), exist_ok=True)

    with rasterio.open(imagem_path) as src:
        transform = src.transform
        crs = src.crs
        img_width = src.width
        img_height = src.height
        points_px = carregar_pontos_bufferados(pontos_path, transform, img_width, img_height, buffer_pixels)

        print(f"Total pontos dentro da imagem: {len(points_px)}")

        patch_id = 0
<<<<<<< Updated upstream
        total_iter = (img_height // patch_size + 1) * (img_width // patch_size + 1)

        for top in tqdm(range(0, img_height, patch_size), desc="Gerando patches", total=img_height // patch_size + 1):
            for left in range(0, img_width, patch_size):
                width = min(patch_size, img_width - left)
                height = min(patch_size, img_height - top)

                patch_points = [(px - left, py - top) for (px, py) in points_px if left <= px < left + width and top <= py < top + height]
=======

        rows = math.ceil(img_height / patch_size)

        for row_idx, top in enumerate(tqdm(range(0, img_height, patch_size), desc="Gerando patches", total=rows)):
            for col_idx, left in enumerate(range(0, img_width, patch_size)):
                width = min(patch_size, img_width - left)
                height = min(patch_size, img_height - top)

                patch_points = [
                    (px - left, py - top)
                    for (px, py) in points_px
                    if left <= px < left + width and top <= py < top + height
                ]
>>>>>>> Stashed changes
                if not patch_points:
                    continue  # pula patch sem pontos

                window = Window(left, top, width, height)
                patch = src.read(window=window)

                patch_img = patch.astype(np.float32)
                patch_img -= patch_img.min()
                if patch_img.max() > 0:
                    patch_img /= patch_img.max()
                patch_img *= 255
                patch_img = patch_img.astype(np.uint8)

                if patch_img.shape[0] < 3:
                    patch_img = np.repeat(patch_img, 3, axis=0)
                elif patch_img.shape[0] > 3:
                    patch_img = patch_img[:3]

                mask = criar_mascara(patch_points, width, height, buffer_pixels)
                mask_uint8 = (mask * 255).astype(np.uint8).reshape(1, height, width)

                patch_transform = src.window_transform(window)

                img_filename = f"patch_{patch_id:04d}.tif"
                salvar_tif(os.path.join(saida_path, 'images', img_filename), patch_img, 3, height, width, patch_transform, crs)
                salvar_tif(os.path.join(saida_path, 'masks', img_filename), mask_uint8, 1, height, width, patch_transform, crs)

                patch_id += 1
                if limite_patches and patch_id >= limite_patches:
                    print(f"⚡ Limite de {limite_patches} patches atingido.")
                    break
            if limite_patches and patch_id >= limite_patches:
                break

        print(f"Patches gerados: {patch_id}")

    print("\nSeparando dataset em treino, validação e teste...")
    separar_patches(saida_path, "dataset", 0.6, 0.3, 0.1, 42)
    print("✅ Dataset pronto.")

if __name__ == "__main__":
    caminho_imagem = "input/imagem_georreferenciada.tif"
    caminho_pontos = "input/intersecoes_osm.gpkg"
    caminho_saida = "dataset_patches"
<<<<<<< Updated upstream
    gerar_patches_com_buffer(caminho_imagem, caminho_pontos, caminho_saida, limite_patches=10)
=======
    gerar_patches_com_buffer(caminho_imagem, caminho_pontos, caminho_saida, limite_patches=100)
>>>>>>> Stashed changes
