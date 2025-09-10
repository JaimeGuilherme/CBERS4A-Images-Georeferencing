import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pyproj import Transformer
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point
from scipy.ndimage import label, center_of_mass
from sklearn.neighbors import NearestNeighbors
import rasterio
from rasterio.windows import Window

from components.dataset import RoadIntersectionDataset
from components.unet import UNet
from components.utils import load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def carregar_pontos_bufferados(gdf_pontos, transform, img_width, img_height, buffer_pixels=5, img_crs_epsg=32724):
    points_px = []
    transformer = Transformer.from_crs(gdf_pontos.crs, f"EPSG:{img_crs_epsg}", always_xy=True)

    for _, row in gdf_pontos.iterrows():
        if row.geometry is None or row.geometry.geom_type != "Point":
            continue

        lon, lat = row.geometry.x, row.geometry.y
        x_m, y_m = transformer.transform(lon, lat)
        px, py = ~transform * (x_m, y_m)
        px, py = int(px), int(py)

        if 0 <= px < img_width and 0 <= py < img_height:
            points_px.append((px, py))

    return points_px

def gerar_patches(imagem_path, saida_path, points_px, patch_size=256, limite_patches=None):
    os.makedirs(saida_path, exist_ok=True)
    patches = []
    patch_id = 0
    interrompido_por_limite = False

    with rasterio.open(imagem_path) as src:
        img_width, img_height = src.width, src.height
        base_meta = src.meta.copy()

        total_rows = (img_height + patch_size - 1) // patch_size

        for top in tqdm(range(0, img_height, patch_size), desc="Gerando patches", total=total_rows):
            for left in range(0, img_width, patch_size):
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

                patch_filename = f"patch_{patch_id:05d}.tif"
                patch_path = os.path.join(saida_path, patch_filename)

                meta = base_meta.copy()
                meta.update({
                    "height": height,
                    "width": width,
                    "transform": src.window_transform(window),
                    "count": 3,
                    "dtype": patch_img.dtype
                })

                with rasterio.open(patch_path, "w", **meta) as dst:
                    dst.write(patch_img)

                patches.append(patch_filename)
                patch_id += 1

                if limite_patches and patch_id >= limite_patches:
                    print(f"⚡ Limite de {limite_patches} patches atingido.")
                    interrompido_por_limite = True
                    break
            if interrompido_por_limite:
                break

    return patches

def inferir_pontos(patches_dir, modelo_path, batch_size=1):
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    checkpoint = load_checkpoint(modelo_path, model)
    threshold = checkpoint.get("best_threshold", 0.5)

    dataset = RoadIntersectionDataset(patches_dir, masks_dir=None, transform=None, is_training=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    output, nomes = [], []
    model.eval()
    with torch.no_grad():
        for imgs, nomes_patches in tqdm(dataloader, desc="Inferindo patches"):
            imgs = imgs.to(DEVICE)
            saida = torch.sigmoid(model(imgs))
            preds = (saida > threshold).float()
            output.append(preds.cpu().numpy())
            nomes.extend(nomes_patches)

    dados = []
    for pred, nome_patch in tqdm(zip(output, nomes), total=len(nomes), desc="Extraindo pontos"):
        mask = pred.squeeze().astype(np.uint8)
        labeled, num_features = label(mask)
        if num_features == 0:
            continue
        centros = center_of_mass(mask, labeled, range(1, num_features + 1))
        patch_path = os.path.join(patches_dir, nome_patch)
        with rasterio.open(patch_path) as src:
            patch_transform = src.transform
            crs = src.crs
        for centro in centros:
            y, x = centro
            lon, lat = rasterio.transform.xy(patch_transform, int(round(y)), int(round(x)))
            dados.append({"geometry": Point(lon, lat)})
    gdf = gpd.GeoDataFrame(dados, crs=crs if dados else None)
    return gdf

def carregar_pontos_gpkg(pasta):
    gdfs = []
    for arquivo in tqdm(os.listdir(pasta), desc="Carregando arquivos GPKG"):
        if arquivo.lower().endswith(".gpkg"):
            caminho = os.path.join(pasta, arquivo)
            gdf = gpd.read_file(caminho)
            gdfs.append(gdf)
    if not gdfs:
        raise FileNotFoundError("Nenhum .gpkg encontrado!")
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

def associar_pontos(detectados, pasta_osm, output_path, max_distance=20):
    osm = carregar_pontos_gpkg(pasta_osm)
    if detectados.crs != osm.crs:
        detectados = detectados.to_crs(osm.crs)

    coords_detect = np.array([(geom.x, geom.y) for geom in detectados.geometry])
    coords_osm = np.array([(geom.x, geom.y) for geom in osm.geometry])

    nbrs = NearestNeighbors(n_neighbors=1).fit(coords_osm)
    distances, indices = nbrs.kneighbors(coords_detect)

    pares = []
    for i, (dist, idx) in tqdm(enumerate(zip(distances.flatten(), indices.flatten())),
                               total=len(distances),
                               desc="Associando pontos"):
        if dist <= max_distance:
            pares.append({
                "id_detectado": int(detectados.index[i]),
                "id_osm": int(osm.index[idx]),
                "distancia": float(dist),
                "geometry_detectado": detectados.geometry.iloc[i],
                "geometry_osm": osm.geometry.iloc[idx]
            })

    gdf_pares = gpd.GeoDataFrame(pares, geometry="geometry_detectado", crs=osm.crs)
    gdf_pares.to_file(output_path, driver="GeoJSON")
    return gdf_pares

if __name__ == "__main__":
    pasta_imagens = "main_input/imagens_tif"
    pasta_patches = "temp_patches"
    modelo_path = "checkpoints/best_model.pth"
    pasta_osm = "main_input/pontos_gpkg"
    saida_pares = "main_output/pares_homologos.geojson"

    nome_imagem = "imagem1.tif"
    imagem = os.path.join(pasta_imagens, nome_imagem)
    if not os.path.exists(imagem):
        raise FileNotFoundError(f"Arquivo {nome_imagem} não encontrado em {pasta_imagens}")

    nome_gpkg = "intersecoes_osm.gpkg"
    caminho = os.path.join(pasta_osm, nome_gpkg)
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo {nome_gpkg} não encontrado em {pasta_osm}")

    print(f"\n📌 Carregando {nome_gpkg}...")
    pontos = gpd.read_file(caminho)

    os.makedirs("main_output", exist_ok=True)
    os.makedirs(pasta_patches, exist_ok=True)

    nome_img = os.path.splitext(nome_imagem)[0]
    pasta_patches_img = os.path.join(pasta_patches, nome_img)
    os.makedirs(pasta_patches_img, exist_ok=True)

    print(f"📌 Quebrando imagem {nome_img} em patches...")
    with rasterio.open(imagem) as src:
        transform = src.transform
        img_width = src.width
        img_height = src.height

    points_px = carregar_pontos_bufferados(pontos, transform, img_width, img_height, buffer_pixels=5)
    gerar_patches(imagem, pasta_patches_img, points_px)

    print(f"📌 Rodando inferência em {nome_img}...")
    gdf_detectados = inferir_pontos(pasta_patches_img, modelo_path)
    gdf_detectados["imagem_origem"] = nome_img

    gdf_detectados_total = gpd.GeoDataFrame(gdf_detectados, crs=gdf_detectados.crs)

    print("📌 Associando pontos detectados com OSM...")
    gdf_pares = associar_pontos(gdf_detectados_total, pasta_osm, saida_pares)

    print("✅ Pipeline concluído. Pares salvos em", saida_pares)
