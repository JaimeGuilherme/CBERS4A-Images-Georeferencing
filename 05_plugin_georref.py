import os
import glob
import torch
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from unet import UNet
from utils import load_model
from scipy.ndimage import label

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parâmetros
PATCH_SIZE = 256
OVERLAP = 0.3  # 30% sobreposição
THRESHOLD = 0.5
MAX_ASSOC_DISTANCE = 20  # distância máxima para parear pontos (em unidades da CRS)
CHECKPOINTS_DIR = "checkpoints"

def transform_image_patch(patch_np):
    """
    Recebe patch numpy no formato (3, H, W), uint8 [0-255].
    Converte para tensor float normalizado [0,1] shape (1,3,H,W).
    """

    # Caso patch tenha formato (3, H, W), ok. Se (H, W, 3), transpõe.
    if patch_np.ndim == 3 and patch_np.shape[0] != 3:
        patch_np = patch_np.transpose(2, 0, 1)

    # Converte para float e normaliza [0,1]
    patch_np = patch_np.astype(np.float32) / 255.0

    patch_tensor = torch.from_numpy(patch_np)
    patch_tensor = patch_tensor.unsqueeze(0)  # batch dimension

    return patch_tensor

def carregar_modelos(checkpoints_dir):
    paths = glob.glob(os.path.join(checkpoints_dir, "*.pth"))
    modelos = []
    for path in paths:
        model = UNet(in_channels=3, out_channels=1).to(DEVICE)
        load_model(model, path)
        model.eval()
        modelos.append(model)
        print(f"Modelo carregado: {path}")
    return modelos

def inferir_patch(models, patch_tensor):
    # Recebe tensor 1x3x256x256, retorna saída média das probabilidades
    with torch.no_grad():
        preds = []
        for model in models:
            out = torch.sigmoid(model(patch_tensor.to(DEVICE)))
            preds.append(out.cpu().numpy())
        mean_pred = np.mean(preds, axis=0)
    return mean_pred  # formato (1,1,256,256)

def dividir_em_patches(imagem_path):
    with rasterio.open(imagem_path) as src:
        img_width = src.width
        img_height = src.height
        passo = int(PATCH_SIZE * (1 - OVERLAP))

        patches = []
        coords = []

        for y in range(0, img_height - PATCH_SIZE + 1, passo):
            for x in range(0, img_width - PATCH_SIZE + 1, passo):
                window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                patch = src.read([1,2,3], window=window)  # canais 1,2,3 para RGB
                patches.append(patch)
                coords.append((x, y))
        return patches, coords, src.transform, src.crs

def detectar_pontos(pred_map, threshold=THRESHOLD):
    # pred_map: numpy 2D (256x256) float probabilidades
    binary_map = pred_map > threshold
    labeled, n_features = label(binary_map)
    pontos = []
    for i in range(1, n_features + 1):
        ys, xs = np.where(labeled == i)
        x_centro = int(np.mean(xs))
        y_centro = int(np.mean(ys))
        pontos.append((x_centro, y_centro))
    return pontos

def converter_para_coords(pontos_patch, patch_origin, transform):
    # pontos_patch: lista (x,y) em pixels do patch
    # patch_origin: (x_off, y_off) no raster original
    coords = []
    for (x, y) in pontos_patch:
        px = patch_origin[0] + x
        py = patch_origin[1] + y
        x_geo, y_geo = rasterio.transform.xy(transform, py, px)
        coords.append(Point(x_geo, y_geo))
    return coords

def associar_por_vizinho(pontos_detectados, pontos_osm, max_dist=MAX_ASSOC_DISTANCE):
    coords_detect = np.array([(p.x, p.y) for p in pontos_detectados])
    coords_osm = np.array([(p.x, p.y) for p in pontos_osm.geometry])

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(coords_osm)
    distances, indices = nbrs.kneighbors(coords_detect)

    pares = []
    for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if dist <= max_dist:
            pares.append({
                'geometry_detectado': pontos_detectados[i],
                'geometry_osm': pontos_osm.geometry.iloc[idx],
                'distancia': dist
            })

    gdf_pares = gpd.GeoDataFrame(pares, geometry='geometry_detectado', crs=pontos_osm.crs)
    return gdf_pares

def main(imagem_path, pontos_osm_path, output_pares_path):
    print("Carregando modelos treinados...")
    modelos = carregar_modelos(CHECKPOINTS_DIR)

    print("Dividindo imagem em patches...")
    patches, coords, transform, crs = dividir_em_patches(imagem_path)

    todos_pontos_detectados = []

    print("Inferindo patches e detectando pontos...")
    for patch, origin in tqdm(zip(patches, coords), total=len(patches)):
        # Normalizar e converter patch para tensor
        patch_tensor = transform_image_patch(patch)  # implementada no seu dataset.py
        pred_prob = inferir_patch(modelos, patch_tensor)[0,0]  # shape (256,256)

        pontos_patch = detectar_pontos(pred_prob, THRESHOLD)
        pontos_geo = converter_para_coords(pontos_patch, origin, transform)
        todos_pontos_detectados.extend(pontos_geo)

    print(f"Total pontos detectados: {len(todos_pontos_detectados)}")

    pontos_osm = gpd.read_file(pontos_osm_path)

    print("Associando pontos detectados com OSM...")
    gdf_pares = associar_por_vizinho(todos_pontos_detectados, pontos_osm, MAX_ASSOC_DISTANCE)

    print(f"Pares homologos encontrados: {len(gdf_pares)}")
    gdf_pares.to_file(output_pares_path, driver="GeoJSON")
    print(f"Pares salvos em: {output_pares_path}")

if __name__ == "__main__":
    # Exemplos de caminho (altere para seus caminhos)
    imagem_nova = "input/imagem_nova.tif"
    pontos_osm = "input/intersecoes_osm_nova.gpkg"
    saida_pares = "output/pares_homologos.geojson"

    main(imagem_nova, pontos_osm, saida_pares)
