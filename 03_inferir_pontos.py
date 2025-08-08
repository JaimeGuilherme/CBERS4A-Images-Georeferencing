import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
from scipy.ndimage import label, center_of_mass
from dataset import RoadIntersectionDataset
from unet import UNet
from utils import load_checkpoint
import rasterio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inferir_modelos_multiplos(modelos, dataloader, device, threshold=0.5):
    for model in modelos:
        model.eval()

    resultados = []

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Inferindo", leave=False)
        for imagens, _ in loop:
            imagens = imagens.to(device)

            soma_saidas = None
            for model in modelos:
                saida = torch.sigmoid(model(imagens))
                soma_saidas = saida if soma_saidas is None else soma_saidas + saida

            media_saidas = soma_saidas / len(modelos)
            preds = (media_saidas > threshold).float()

            resultados.append(preds.cpu().numpy())

    return resultados

def extrair_pontos_com_patches(preds_binarias, nomes_patches, patch_offsets, transform, crs=None):
    dados = []
    for pred, patch_name, (x_offset, y_offset) in zip(preds_binarias, nomes_patches, patch_offsets):
        mask = pred.squeeze().astype(np.uint8)  # garante 2D

        labeled, num_features = label(mask)
        centros = center_of_mass(mask, labeled, range(1, num_features + 1))

        for centro in centros:
            if len(centro) != 2:
                continue

            y, x = centro
            if np.isnan(x) or np.isnan(y):
                continue

            # Coordenadas absolutas na imagem
            abs_x = x_offset + x
            abs_y = y_offset + y

            # Coordenadas geográficas (lon, lat)
            lon, lat = rasterio.transform.xy(transform, abs_y, abs_x)

            ponto = Point(lon, lat)
            dados.append({'geometry': ponto, 'patch': patch_name})

    print(f"[DEBUG] Total pontos detectados: {len(dados)}")
    if dados:
        print(f"[DEBUG] Exemplo ponto: {dados[0]}")

    gdf = gpd.GeoDataFrame(dados)
    gdf = gdf.set_geometry('geometry')
    if crs:
        gdf.set_crs(crs, inplace=True)

    return gdf

if __name__ == "__main__":
    caminho_modelos = sorted([os.path.join("checkpoints", f) for f in os.listdir("checkpoints") if f.endswith(".pth")])
    caminho_test_img = "dataset/test/images"
    caminho_test_mask = "dataset/test/masks"
    batch_size = 1
    threshold = 0.5
    caminho_saida_geojson = "resultados/pontos_detectados.geojson"

    # Carregar transform e crs da primeira imagem para usar nas coordenadas geográficas
    primeira_imagem_path = os.path.join(caminho_test_img, sorted(os.listdir(caminho_test_img))[0])
    with rasterio.open(primeira_imagem_path) as src:
        transform = src.transform
        crs = src.crs

    test_dataset = RoadIntersectionDataset(caminho_test_img, caminho_test_mask)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Calcular offsets dos patches pela posição do arquivo (extraindo índice do nome do patch)
    nomes_patches = sorted([f for f in os.listdir(caminho_test_img) if f.endswith(".tif")])
    patch_offsets = []
    for nome in nomes_patches:
        # Exemplo nome patch_0292.tif
        idx_str = nome.split("_")[1].split(".")[0]
        idx = int(idx_str)
        step = 256
        x_offset = (idx % 50) * step  # ajusta 50 conforme sua organização
        y_offset = (idx // 50) * step
        patch_offsets.append((x_offset, y_offset))

    modelos = []
    print("📦 Carregando modelos:")
    for caminho_modelo in caminho_modelos:
        print(f" - {os.path.basename(caminho_modelo)}")
        model = UNet(in_channels=3, out_channels=1).to(DEVICE)
        load_checkpoint(caminho_modelo, model)
        modelos.append(model)

    print("\n🚀 Inferindo média dos modelos...")
    resultados_binarios = inferir_modelos_multiplos(modelos, test_loader, DEVICE, threshold=threshold)

    gdf_pontos = extrair_pontos_com_patches(resultados_binarios, nomes_patches, patch_offsets, transform, crs)

    os.makedirs(os.path.dirname(caminho_saida_geojson), exist_ok=True)
    gdf_pontos.to_file(caminho_saida_geojson, driver='GeoJSON')

    print(f"\n✅ Pontos detectados salvos em: {caminho_saida_geojson}")
    print(f"Total de pontos detectados: {len(gdf_pontos)}")
