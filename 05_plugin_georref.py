import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point
from scipy.ndimage import label, center_of_mass
from sklearn.neighbors import NearestNeighbors
import rasterio
from rasterio.windows import Window

from dataset import RoadIntersectionDataset
from unet import UNet
from utils import load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gerar_patches(imagem_path, saida_path, patch_size=256):
    os.makedirs(saida_path, exist_ok=True)
    patches = []
    with rasterio.open(imagem_path) as src:
        img_width, img_height = src.width, src.height
        for top in range(0, img_height, patch_size):
            for left in range(0, img_width, patch_size):
                width = min(patch_size, img_width - left)
                height = min(patch_size, img_height - top)
                window = Window(left, top, width, height)
                patch = src.read(window=window)

                if patch.shape[0] < 3:
                    patch = np.repeat(patch, 3, axis=0)
                elif patch.shape[0] > 3:
                    patch = patch[:3]

                patch_filename = f"patch_{top}_{left}.tif"
                patch_path = os.path.join(saida_path, patch_filename)

                meta = src.meta.copy()
                meta.update({
                    "height": height,
                    "width": width,
                    "transform": src.window_transform(window),
                    "count": 3
                })
                with rasterio.open(patch_path, "w", **meta) as dst:
                    dst.write(patch)

                patches.append(patch_filename)
    return patches

def inferir_pontos(patches_dir, modelo_path, batch_size=1):
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    checkpoint = load_checkpoint(modelo_path, model)
    threshold = checkpoint.get("best_threshold", 0.5)

    dataset = RoadIntersectionDataset(patches_dir, masks_dir=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    resultados, nomes = [], []
    model.eval()
    with torch.no_grad():
        for imgs, nomes_patches in tqdm(dataloader, desc="Inferindo"):
            imgs = imgs.to(DEVICE)
            saida = torch.sigmoid(model(imgs))
            preds = (saida > threshold).float()
            resultados.append(preds.cpu().numpy())
            nomes.extend(nomes_patches)

    dados = []
    for pred, nome_patch in zip(resultados, nomes):
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
    for arquivo in os.listdir(pasta):
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
    for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
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
    imagem = "input/imagem.tif"
    pasta_patches = "temp_patches"
    modelo_path = "checkpoints/best_model.pth"
    pasta_osm = "input/pontos_gpkg"
    saida_pares = "results/pares_homologos.geojson"

    os.makedirs("results", exist_ok=True)

    print("📌 Quebrando imagem em patches...")
    gerar_patches(imagem, pasta_patches)

    print("📌 Rodando inferência...")
    gdf_detectados = inferir_pontos(pasta_patches, modelo_path)

    print("📌 Associando pontos...")
    gdf_pares = associar_pontos(gdf_detectados, pasta_osm, saida_pares)

    print("✅ Pipeline concluído. Pares salvos em", saida_pares)
