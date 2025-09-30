import os, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
from scipy.ndimage import label, center_of_mass
import rasterio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp

from components.dataset import RoadIntersectionDataset
from components.unet import UNet
from components.utils import load_checkpoint_raw

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BANDS_MODE = "rgbnir"   # "rgbnir" (padrão, 4 canais) ou "rgb" (3 canais)
ARCH = "smp_unet"       # "custom" (sua UNet - padrão) ou "smp_unet" (se tiver)

# === CONFIGURAÇÕES DE PARALELIZAÇÃO ===
NUM_WORKERS_DATALOADER = 4  # Para carregar dados em paralelo
NUM_WORKERS_SAVE = 8        # Para salvar máscaras em paralelo (I/O)
NUM_WORKERS_EXTRACT = mp.cpu_count()  # Para extrair pontos (CPU-bound)

def build_model(arch: str, in_ch: int):
    if arch == "custom":
        return UNet(in_channels=in_ch, out_channels=1).to(DEVICE)
    elif arch == "smp_unet":
        try:
            import segmentation_models_pytorch as smp
        except Exception as e:
            raise RuntimeError("Para usar ARCH='smp_unet', instale segmentation_models_pytorch.") from e
        return smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=in_ch, classes=1).to(DEVICE)
    else:
        raise ValueError(f"ARCH inválido: {arch}")

def adapt_first_conv_if_needed(model, checkpoint_state):
    '''
    Ajusta a 1ª conv se o checkpoint foi treinado com 3 canais e o modelo atual tem 4 (ou vice-versa).
    Mesma lógica usada no treino.
    '''
    state = checkpoint_state.get('model_state_dict', checkpoint_state)
    model_state = model.state_dict()
    possible_keys = [k for k in model_state.keys() if k.endswith("weight") and model_state[k].dim() == 4]
    if not possible_keys:
        model.load_state_dict(state, strict=False); return
    first_conv_key = possible_keys[0]
    if first_conv_key not in state:
        model.load_state_dict(state, strict=False); return
    w_ckpt = state[first_conv_key]
    w_model = model_state[first_conv_key]
    in_ckpt, in_model = w_ckpt.shape[1], w_model.shape[1]
    if in_ckpt == in_model:
        model.load_state_dict(state, strict=False); return
    if in_ckpt == 3 and in_model == 4:
        w_new = w_model.clone()
        w_new[:, :3, :, :] = w_ckpt
        w_new[:, 3:4, :, :] = w_ckpt.mean(dim=1, keepdim=True)
        state[first_conv_key] = w_new
    elif in_ckpt == 4 and in_model == 3:
        state[first_conv_key] = w_ckpt[:, :3, :, :].contiguous()
    model.load_state_dict(state, strict=False)

def inferir_modelo(model, dataloader, device, threshold):
    model.eval()
    output, nomes = [], []
    with torch.no_grad():
        for imgs, nomes_patches in tqdm(dataloader, desc='Inferindo', leave=False):
            imgs = imgs.to(device)
            saida = torch.sigmoid(model(imgs))  # (B, 1, H, W)
            preds = (saida > threshold).float().cpu().numpy()
            for i in range(preds.shape[0]):
                mask = preds[i].squeeze()  # garante (H, W)
                output.append(mask)
            nomes.extend(nomes_patches)
    return output, nomes

def salvar_mascara_worker(args):
    """Worker para salvar uma única máscara (paralelizável)"""
    pred, nome_patch, caminho_test_img, pasta_saida_mascaras = args
    
    mask_patch = pred.squeeze().astype(np.uint8) * 255
    if mask_patch.ndim != 2:
        raise ValueError(f"Máscara com dimensões inesperadas: {mask_patch.shape}")

    patch_path = os.path.join(caminho_test_img, nome_patch)
    with rasterio.open(patch_path) as src:
        meta = src.meta.copy()
    
    meta.update({'count': 1, 'dtype': mask_patch.dtype, 'driver': 'GTiff'})
    saida_patch = os.path.join(pasta_saida_mascaras, f'mask_{nome_patch}')
    
    with rasterio.open(saida_patch, 'w', **meta) as dst:
        dst.write(mask_patch, 1)
    
    return nome_patch

def salvar_mascaras_paralelo(output_bin, nomes_patches, caminho_test_img, pasta_saida_mascaras, num_workers):
    """Salva máscaras em paralelo usando ThreadPoolExecutor (I/O bound)"""
    args_list = [(pred, nome, caminho_test_img, pasta_saida_mascaras) 
                 for pred, nome in zip(output_bin, nomes_patches)]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(salvar_mascara_worker, args_list),
                  total=len(args_list),
                  desc="Salvando máscaras"))

def extrair_pontos_worker(args):
    """Worker para extrair pontos de uma única máscara (paralelizável)"""
    pred, nome_patch, patch_path = args
    
    mask = pred.squeeze().astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[0]  # garante 2D

    labeled, num_features = label(mask)
    if num_features == 0:
        return []

    centros = center_of_mass(mask, labeled, range(1, num_features + 1))
    
    # Abre o raster para pegar transform e CRS
    with rasterio.open(patch_path) as src:
        patch_transform = src.transform
        crs = src.crs
    
    dados_patch = []
    for centro in centros:
        if len(centro) == 2:
            y, x = centro
        elif len(centro) == 3:
            _, y, x = centro
        else:
            continue

        lon, lat = rasterio.transform.xy(patch_transform, int(round(y)), int(round(x)))
        dados_patch.append({
            'geometry': Point(lon, lat),
            'patch': nome_patch,
            'x_pixel': int(round(x)),
            'y_pixel': int(round(y)),
            'crs': crs
        })
    
    return dados_patch

def extrair_pontos_paralelo(preds_binarias, nomes_patches, pasta_imagens, num_workers):
    """Extrai pontos em paralelo usando ProcessPoolExecutor (CPU bound)"""
    patch_paths = [os.path.join(pasta_imagens, nome) for nome in nomes_patches]
    args_list = list(zip(preds_binarias, nomes_patches, patch_paths))
    
    todos_dados = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(extrair_pontos_worker, args_list),
                           total=len(args_list),
                           desc="Extraindo pontos"))
        
        for dados in results:
            todos_dados.extend(dados)
    
    if not todos_dados:
        return gpd.GeoDataFrame()
    
    # Extrai CRS do primeiro ponto (todos devem ter o mesmo)
    crs = todos_dados[0].pop('crs')
    
    gdf = gpd.GeoDataFrame(todos_dados)
    gdf.set_crs(crs, inplace=True)
    
    return gdf

if __name__ == '__main__':
    caminho_modelo = 'checkpoints/best_model.pth'
    caminho_test_img = 'dataset_separated/test/images'
    batch_size = 64
    caminho_saida_geojson = 'output/pontos_detectados.geojson'
    pasta_saida_mascaras = 'output/mascaras_patches'

    os.makedirs(pasta_saida_mascaras, exist_ok=True)
    os.makedirs(os.path.dirname(caminho_saida_geojson), exist_ok=True)

    in_ch = 3 if BANDS_MODE == "rgb" else 4
    print('Carregando modelo:', caminho_modelo)
    print(f'Usando {NUM_WORKERS_DATALOADER} workers para DataLoader')
    print(f'Usando {NUM_WORKERS_SAVE} workers para salvar máscaras')
    print(f'Usando {NUM_WORKERS_EXTRACT} workers para extrair pontos')
    
    model = build_model(ARCH, in_ch)
    checkpoint = load_checkpoint_raw(caminho_modelo, map_location=DEVICE)
    adapt_first_conv_if_needed(model, checkpoint)

    threshold = checkpoint.get('best_threshold', 0.5)
    print(f'Usando threshold: {threshold}')

    # DataLoader com num_workers para carregar dados em paralelo
    test_dataset = RoadIntersectionDataset(
        caminho_test_img, masks_dir=None, transform=None, is_training=False, bands_mode=BANDS_MODE
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=NUM_WORKERS_DATALOADER,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print('\n=== INFERÊNCIA ===')
    output_bin, nomes_patches = inferir_modelo(model, test_loader, DEVICE, threshold)

    print('\n=== SALVANDO MÁSCARAS EM PARALELO ===')
    salvar_mascaras_paralelo(output_bin, nomes_patches, caminho_test_img, 
                             pasta_saida_mascaras, NUM_WORKERS_SAVE)

    print('\n=== EXTRAINDO PONTOS EM PARALELO ===')
    gdf_pontos = extrair_pontos_paralelo(output_bin, nomes_patches, 
                                         caminho_test_img, NUM_WORKERS_EXTRACT)

    if not gdf_pontos.empty:
        gdf_pontos.to_file(caminho_saida_geojson, driver='GeoJSON')
        print(f'\nPontos detectados salvos em: {caminho_saida_geojson}')
    else:
        print('\nNenhum ponto detectado para salvar.')

    print(f'Máscaras salvas em: {pasta_saida_mascaras}')
    print(f'Total de pontos detectados: {len(gdf_pontos)}')