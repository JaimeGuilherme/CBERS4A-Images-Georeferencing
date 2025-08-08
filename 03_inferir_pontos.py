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
from utils import load_model

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

def extrair_pontos_com_patches(preds_binarias, nomes_patches):
    dados = []
    for pred, patch_name in zip(preds_binarias, nomes_patches):
        mask = pred[0].astype(np.uint8)  # remove canal
        labeled, num_features = label(mask)
        centros = center_of_mass(mask, labeled, range(1, num_features + 1))

        for y, x in centros:
            if np.isnan(x) or np.isnan(y):
                continue
            ponto = Point(int(round(x)), int(round(y)))
            dados.append({'geometry': ponto, 'patch': patch_name})

    return gpd.GeoDataFrame(dados)

if __name__ == "__main__":
    caminho_modelos = sorted([os.path.join("checkpoints", f) for f in os.listdir("checkpoints") if f.endswith(".pth")])
    caminho_test_img = "dataset/test/images"
    caminho_test_mask = "dataset/test/masks"
    batch_size = 1
    threshold = 0.5
    caminho_saida_geojson = "resultados/pontos_detectados.geojson"

    test_dataset = RoadIntersectionDataset(caminho_test_img, caminho_test_mask, overlap=0.3)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    modelos = []
    print("📦 Carregando modelos:")
    for caminho_modelo in caminho_modelos:
        print(f" - {os.path.basename(caminho_modelo)}")
        model = UNet(in_channels=3, out_channels=1).to(DEVICE)
        load_model(model, caminho_modelo)
        modelos.append(model)

    print("\n🚀 Inferindo média dos modelos...")
    resultados_binarios = inferir_modelos_multiplos(modelos, test_loader, DEVICE, threshold=threshold)

    nomes_patches = sorted([f for f in os.listdir(caminho_test_img) if f.endswith(".tif")])
    gdf_pontos = extrair_pontos_com_patches(resultados_binarios, nomes_patches)

    os.makedirs(os.path.dirname(caminho_saida_geojson), exist_ok=True)
    gdf_pontos.to_file(caminho_saida_geojson, driver='GeoJSON')

    print(f"\n✅ Pontos detectados salvos em: {caminho_saida_geojson}")
    print(f"Total de pontos detectados: {len(gdf_pontos)}")
