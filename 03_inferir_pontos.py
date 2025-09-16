# 03_inferir_pontos.py
import os, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
from scipy.ndimage import label, center_of_mass
import rasterio

from components.dataset import RoadIntersectionDataset
from components.unet import UNet
from components.utils import load_checkpoint

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BANDS_MODE = "rgbnir"   # "rgbnir" (padrão, 4 canais) ou "rgb" (3 canais)
ARCH = "custom"         # "custom" (sua UNet - padrão) ou "smp_unet" (se tiver)

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
            saida = torch.sigmoid(model(imgs))
            preds = (saida > threshold).float()
            output.append(preds.cpu().numpy())
            nomes.extend(nomes_patches)
    return output, nomes

def extrair_pontos(preds_binarias, nomes_patches, pasta_imagens):
    dados = []
    for pred, nome_patch in tqdm(zip(preds_binarias, nomes_patches),
                                 desc="Extraindo pontos",
                                 total=len(nomes_patches)):
        mask = pred.squeeze().astype(np.uint8)
        labeled, num_features = label(mask)
        if num_features == 0:
            continue
        centros = center_of_mass(mask, labeled, range(1, num_features + 1))
        patch_path = os.path.join(pasta_imagens, nome_patch)
        with rasterio.open(patch_path) as src:
            patch_transform = src.transform
            crs = src.crs
        for y, x in centros:
            lon, lat = rasterio.transform.xy(patch_transform, int(round(y)), int(round(x)))
            dados.append({
                'geometry': Point(lon, lat),
                'patch': nome_patch,
                'x_pixel': int(round(x)),
                'y_pixel': int(round(y))
            })
    gdf = gpd.GeoDataFrame(dados)
    if len(dados) > 0:
        gdf.set_crs(crs, inplace=True)
    return gdf

if __name__ == '__main__':
    caminho_modelo = 'checkpoints/best_model.pth'
    caminho_test_img = 'dataset_separated/test/images'
    batch_size = 1
    caminho_saida_geojson = 'output/pontos_detectados.geojson'
    pasta_saida_mascaras = 'output/mascaras_patches'

    os.makedirs(pasta_saida_mascaras, exist_ok=True)
    os.makedirs(os.path.dirname(caminho_saida_geojson), exist_ok=True)

    in_ch = 3 if BANDS_MODE == "rgb" else 4
    print('Carregando modelo:', caminho_modelo)
    model = build_model(ARCH, in_ch)
    checkpoint = load_checkpoint(caminho_modelo, model)
    adapt_first_conv_if_needed(model, checkpoint)

    threshold = checkpoint.get('best_threshold', 0.5)
    print(f'Usando threshold: {threshold}')

    test_dataset = RoadIntersectionDataset(
        caminho_test_img, masks_dir=None, transform=None, is_training=False, bands_mode=BANDS_MODE
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    output_bin, nomes_patches = inferir_modelo(model, test_loader, DEVICE, threshold)

    for pred, nome_patch in tqdm(zip(output_bin, nomes_patches),
                                 total=len(nomes_patches),
                                 desc="Salvando máscaras"):
        mask_patch = (pred[0, 0] > 0).astype(np.uint8) * 255
        patch_path = os.path.join(caminho_test_img, nome_patch)
        with rasterio.open(patch_path) as src:
            meta = src.meta.copy()
        meta.update({'count': 1, 'dtype': mask_patch.dtype, 'driver': 'GTiff'})
        saida_patch = os.path.join(pasta_saida_mascaras, f'mask_{nome_patch}')
        with rasterio.open(saida_patch, 'w', **meta) as dst:
            dst.write(mask_patch, 1)

    gdf_pontos = extrair_pontos(output_bin, nomes_patches, caminho_test_img)

    if not gdf_pontos.empty:
        gdf_pontos.to_file(caminho_saida_geojson, driver='GeoJSON')
        print(f'Pontos detectados salvos em: {caminho_saida_geojson}')
    else:
        print('Nenhum ponto detectado para salvar.')

    print(f'Máscaras salvas em: {pasta_saida_mascaras}')
    print(f'Total de pontos detectados: {len(gdf_pontos)}')
