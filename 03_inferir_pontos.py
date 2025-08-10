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

def inferir_modelos_multiplos(modelos, dataloader, device, threshold=0.5):
    for model in modelos:
        model.eval()
    resultados = []
    with torch.no_grad():
        loop = tqdm(dataloader, desc='Inferindo', leave=False)
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
        mask = pred.squeeze().astype(np.uint8)
        labeled, num_features = label(mask)
        if num_features == 0:
            continue
        centros = center_of_mass(mask, labeled, range(1, num_features + 1))
        patch_path = os.path.join('dataset/test/images', patch_name)
        with rasterio.open(patch_path) as src:
            patch_transform = src.transform; crs = src.crs
        for centro in centros:
            y, x = centro
            abs_x = int(round(x)); abs_y = int(round(y))
            lon, lat = rasterio.transform.xy(patch_transform, abs_y, abs_x)
            dados.append({'geometry': Point(lon, lat), 'patch': patch_name, 'x_pixel': int(round(x)), 'y_pixel': int(round(y))})
    gdf = gpd.GeoDataFrame(dados)
    if len(dados)>0:
        gdf.set_crs(crs, inplace=True)
    return gdf

if __name__ == '__main__':
    caminho_modelos = sorted([os.path.join('checkpoints', f) for f in os.listdir('checkpoints') if f.endswith('.pth')])
    caminho_test_img = 'dataset/test/images'; caminho_test_mask = 'dataset/test/masks'
    batch_size = 1; threshold = 0.5
    caminho_saida_geojson = 'resultados/pontos_detectados.geojson'; pasta_saida_mascaras = 'resultados/mascaras_patches'
    os.makedirs(os.path.dirname(caminho_saida_geojson), exist_ok=True); os.makedirs(pasta_saida_mascaras, exist_ok=True)
    primeira = sorted([f for f in os.listdir(caminho_test_img) if f.endswith('.tif')])[0]
    with rasterio.open(os.path.join(caminho_test_img, primeira)) as src:
        crs = src.crs
    test_dataset = RoadIntersectionDataset(caminho_test_img, caminho_test_mask)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    nomes_patches = sorted([f for f in os.listdir(caminho_test_img) if f.endswith('.tif')])
    modelos = []
    print('📦 Carregando modelos:')
    for caminho_modelo in caminho_modelos:
        print(' -', os.path.basename(caminho_modelo))
        model = UNet(in_channels=3, out_channels=1).to(DEVICE)
        load_checkpoint(caminho_modelo, model)
        modelos.append(model)
    print('\n🚀 Inferindo média dos modelos...')
    resultados_binarios = inferir_modelos_multiplos(modelos, test_loader, DEVICE, threshold=threshold)
    for pred, nome_patch in zip(resultados_binarios, nomes_patches):
        mask_patch = pred.squeeze().astype(np.uint8)
        patch_path = os.path.join(caminho_test_img, nome_patch)
        with rasterio.open(patch_path) as src:
            meta = src.meta.copy()
        meta.update({'count':1, 'dtype':mask_patch.dtype, 'driver':'GTiff'})
        saida_patch = os.path.join(pasta_saida_mascaras, f'mask_{nome_patch}')
        with rasterio.open(saida_patch, 'w', **meta) as dst:
            dst.write(mask_patch, 1)
    gdf_pontos = extrair_pontos_com_patches(resultados_binarios, nomes_patches)
    gdf_pontos.to_file(caminho_saida_geojson, driver='GeoJSON')
    print('\n✅ Pontos detectados salvos em:', caminho_saida_geojson)
    print('✅ Máscaras salvas em:', pasta_saida_mascaras)
    print('Total de pontos detectados:', len(gdf_pontos))
