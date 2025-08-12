import os, torch
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inferir_modelo(model, dataloader, device, threshold):
    model.eval()
    resultados = []
    with torch.no_grad():
        loop = tqdm(dataloader, desc='Inferindo', leave=False)
        for imagens, _ in loop:
            imagens = imagens.to(device)
            saida = torch.sigmoid(model(imagens))
            preds = (saida > threshold).float()
            resultados.append(preds.cpu().numpy())
    return resultados

def extrair_pontos_com_patches(preds_binarias, nomes_patches):
    dados = []
    for pred, patch_name in zip(preds_binarias, nomes_patches):
        mask = pred.squeeze().astype(np.uint8)
        labeled, num_features = label(mask)
        if num_features == 0:
            continue
        centros = center_of_mass(mask, labeled, range(1, num_features + 1))
        patch_path = os.path.join('dataset/test/images', patch_name)
        with rasterio.open(patch_path) as src:
            patch_transform = src.transform
            crs = src.crs
        for centro in centros:
            y, x = centro
            lon, lat = rasterio.transform.xy(patch_transform, int(round(y)), int(round(x)))
            dados.append({'geometry': Point(lon, lat), 'patch': patch_name, 'x_pixel': int(round(x)), 'y_pixel': int(round(y))})
    gdf = gpd.GeoDataFrame(dados)
    if len(dados) > 0:
        gdf.set_crs(crs, inplace=True)
    return gdf

if __name__ == '__main__':
    # Escolher o melhor modelo
    caminho_modelo = 'checkpoints/best_model.pth'
    caminho_test_img = 'dataset/test/images'
    caminho_test_mask = 'dataset/test/masks'
    batch_size = 1
    caminho_saida_geojson = 'resultados/pontos_detectados.geojson'
    pasta_saida_mascaras = 'resultados/mascaras_patches'

    os.makedirs(os.path.dirname(caminho_saida_geojson), exist_ok=True)
    os.makedirs(pasta_saida_mascaras, exist_ok=True)

    # Carregar modelo e threshold salvo
    print('📦 Carregando melhor modelo:', caminho_modelo)
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    checkpoint = load_checkpoint(caminho_modelo, model)
    threshold = checkpoint.get('best_threshold', 0.5)
    print(f'🔍 Usando threshold salvo: {threshold}')

    # Dataset
    test_dataset = RoadIntersectionDataset(caminho_test_img, caminho_test_mask)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    nomes_patches = sorted([f for f in os.listdir(caminho_test_img) if f.endswith('.tif')])

    # Inferência
    resultados_binarios = inferir_modelo(model, test_loader, DEVICE, threshold)

    # Salvar máscaras
    for pred, nome_patch in zip(resultados_binarios, nomes_patches):
        mask_patch = (pred[0, 0] > 0).astype(np.uint8) * 255
        patch_path = os.path.join(caminho_test_img, nome_patch)
        with rasterio.open(patch_path) as src:
            meta = src.meta.copy()
        meta.update({'count': 1, 'dtype': mask_patch.dtype, 'driver': 'GTiff'})
        saida_patch = os.path.join(pasta_saida_mascaras, f'mask_{nome_patch}')
        with rasterio.open(saida_patch, 'w', **meta) as dst:
            dst.write(mask_patch, 1)

    # Extrair pontos
    gdf_pontos = extrair_pontos_com_patches(resultados_binarios, nomes_patches)

    if not gdf_pontos.empty:
        gdf_pontos.to_file(caminho_saida_geojson, driver='GeoJSON')
        print('\n✅ Pontos detectados salvos em:', caminho_saida_geojson)
    else:
        print('\n⚠️ Nenhum ponto detectado para salvar.')

    print('✅ Máscaras salvas em:', pasta_saida_mascaras)
    print('Total de pontos detectados:', len(gdf_pontos))
