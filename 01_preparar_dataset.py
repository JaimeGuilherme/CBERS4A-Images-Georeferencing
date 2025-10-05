# 01_preparar_dataset.py

import os
import shutil
import random
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
import pandas as pd
from pyproj import Transformer
import math
from tqdm import tqdm


def carregar_pontos_gpkg(pasta):
    print(f"\n📌 Processando Pontos gpkg...")
    gdfs = []
    for arquivo in os.listdir(pasta):
        if arquivo.lower().endswith(".gpkg"):
            caminho = os.path.join(pasta, arquivo)
            try:
                gdf = gpd.read_file(caminho)
                gdfs.append(gdf)
            except Exception as e:
                print(f"Erro ao carregar {caminho}: {e}")
    if not gdfs:
        raise FileNotFoundError("Nenhum .gpkg encontrado na pasta!")
    gdf_unificado = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    return gdf_unificado


def criar_mascara(points, img_width, img_height, buffer_pixels=5):
    from PIL import Image, ImageDraw
    mask = Image.new('L', (img_width, img_height), 0)
    draw = ImageDraw.Draw(mask)
    for pt in points:
        x, y = pt
        bbox = [x - buffer_pixels, y - buffer_pixels, x + buffer_pixels, y + buffer_pixels]
        draw.ellipse(bbox, fill=255)
    return np.array(mask) > 0


def carregar_pontos(gdf_pontos, transform, img_width, img_height, img_crs):
    if gdf_pontos.crs is None:
        raise ValueError("O GeoDataFrame de pontos não tem CRS definido. Defina gdf_pontos.set_crs(EPSG:xxxx).")
    transformer = Transformer.from_crs(gdf_pontos.crs, img_crs, always_xy=True)
    points_px = []
    for _, row in gdf_pontos.iterrows():
        geom = row.geometry
        if geom is None or geom.geom_type != "Point":
            continue
        x_src, y_src = geom.x, geom.y
        x, y = transformer.transform(x_src, y_src)
        px, py = ~transform * (x, y)
        px, py = int(px), int(py)
        if 0 <= px < img_width and 0 <= py < img_height:
            points_px.append((px, py))
    return points_px


def salvar_tif(path, array, height, width, transform, crs):
    count = array.shape[0]
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


def selecionar_bandas(patch, prefer='all'):
    C, H, W = patch.shape

    def take_n(n):
        if C >= n:
            return patch[:n, :, :]
        out = np.zeros((n, H, W), dtype=patch.dtype)
        out[:C, :, :] = patch
        return out

    if isinstance(prefer, (list, tuple)):
        idx = [i - 1 for i in prefer if 1 <= i <= C]
        if not idx:
            idx = list(range(min(3, C)))
        return patch[idx, :, :]

    if isinstance(prefer, str):
        key = prefer.strip().lower()
        if key in ('all', '*'):
            return patch
        if key in ('rgb',):
            return take_n(3)
        if key in ('rgbnir', 'rgba', 'nirrgb'):
            return take_n(4)

    return take_n(3)


def gerar_patches_multiplas_imagens(
    caminho_imagens,
    pontos,
    saida_path,
    patch_size=256,
    buffer_pixels=5,
    limite_patches=None,
    bandas='all',
    normalizar=True
):
    os.makedirs(os.path.join(saida_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(saida_path, 'masks'), exist_ok=True)
    patch_id = 0
    interrompido_por_limite = False
    imagens_tif = sorted([f for f in os.listdir(caminho_imagens) if f.lower().endswith(".tif")])
    for nome_img in imagens_tif:
        imagem_path = os.path.join(caminho_imagens, nome_img)
        print(f"\n📌 Processando {nome_img}...")
        with rasterio.open(imagem_path) as src:
            transform = src.transform
            crs = src.crs
            img_width = src.width
            img_height = src.height
            n_bandas = src.count
            bounds = src.bounds
            print(f"   Bandas: {n_bandas} | CRS: {crs} | Bounds: {bounds}")
            try:
                gdf_img_crs = pontos.to_crs(crs)
                in_bbox = gdf_img_crs.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
                print(f"   Pontos no bbox (aprox.): {len(in_bbox)}")
            except Exception as e:
                print(f"   Aviso: não consegui checar bbox -> {e}")
            points_px = carregar_pontos(pontos, transform, img_width, img_height, crs)
            print(f"   Pontos dentro desta imagem (em pixels): {len(points_px)}")
            if len(points_px) == 0:
                print("   ⚠ Nenhum ponto dentro da imagem. Verifique CRS dos pontos e extensão da cena.")
                continue
            rows = math.ceil(img_height / patch_size)
            for row_idx, top in enumerate(tqdm(range(0, img_height, patch_size), desc=f"Patches {nome_img}", total=rows)):
                for col_idx, left in enumerate(range(0, img_width, patch_size)):
                    width = min(patch_size, img_width - left)
                    height = min(patch_size, img_height - top)
                    patch_points = [
                        (px - left, py - top)
                        for (px, py) in points_px
                        if left <= px < left + width and top <= py < top + height
                    ]
                    if not patch_points:
                        continue
                    window = Window(left, top, width, height)
                    patch = src.read(window=window)
                    patch_img = selecionar_bandas(patch, prefer=bandas)
                    if normalizar:
                        patch_img = patch_img.astype(np.float32)
                        for b in range(patch_img.shape[0]):
                            band = patch_img[b]
                            mn, mx = band.min(), band.max()
                            if mx > mn:
                                band = (band - mn) / (mx - mn) * 255.0
                            else:
                                band = np.zeros_like(band)
                            patch_img[b] = band
                        patch_img = patch_img.astype(np.uint8)
                    mask = criar_mascara(patch_points, width, height, buffer_pixels)
                    mask_uint8 = (mask * 255).astype(np.uint8).reshape(1, height, width)
                    patch_transform = src.window_transform(window)
                    img_filename = f"{os.path.splitext(nome_img)[0]}_patch_{patch_id:05d}.tif"
                    salvar_tif(os.path.join(saida_path, 'images', img_filename), patch_img, height, width, patch_transform, crs)
                    salvar_tif(os.path.join(saida_path, 'masks', img_filename), mask_uint8, height, width, patch_transform, crs)
                    patch_id += 1
                    if limite_patches and patch_id >= limite_patches:
                        print(f"⚡ Limite de {limite_patches} patches atingido.")
                        interrompido_por_limite = True
                        break
                if interrompido_por_limite:
                    break
        if interrompido_por_limite:
            break
    print(f"\n✅ Total de patches gerados: {patch_id}")
    print("\nSeparando dataset em treino, validação e teste...")
    separar_patches(saida_path, "dataset_separated", 0.8, 0.1, 0.1, 42)
    print("✅ Dataset pronto.")


if __name__ == "__main__":
    caminho_imagens = "input/imagens_tif"
    caminho_pontos = "input/pontos_gpkg"
    caminho_saida = "dataset_patches"
    pontos = carregar_pontos_gpkg(caminho_pontos)
    gerar_patches_multiplas_imagens(
        caminho_imagens,
        pontos,
        caminho_saida,
        patch_size=256,
        buffer_pixels=5,
        limite_patches=None,
        bandas='all',
        normalizar=True
    )