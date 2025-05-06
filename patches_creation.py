
import os
import numpy as np
import rasterio
import random
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

# CONFIGURAÇÕES
tif_original_path = "campinas_alinhado.tif"
mask_path = "estrada_campinas_rasterizado_alinhado.tif"
output_dir = "dataset_sincronizado_random"
patch_size = 224
max_offset_m = 50  # deslocamento máximo em metros (positivo ou negativo)

# Função: Gerar deslocamento aleatório (em metros)
def gerar_translacao_aleatoria(max_offset):
    dx = random.uniform(-max_offset, max_offset)
    dy = random.uniform(-max_offset, max_offset)
    return dx, dy

# Função: Deslocar conteúdo da imagem
def deslocar_conteudo(imagem, dx_m, dy_m, res_x, res_y):
    dx_px = int(dx_m / res_x)
    dy_px = int(dy_m / res_y)
    deslocado = np.zeros_like(imagem)
    for b in range(imagem.shape[0]):
        deslocado[b] = np.roll(imagem[b], shift=(dy_px, dx_px), axis=(0, 1))
    return deslocado

# Função principal
def gerar_patches_com_translacao_aleatoria():
    orig_out = os.path.join(output_dir, "original")
    desloc_out = os.path.join(output_dir, "deslocado")
    mask_out = os.path.join(output_dir, "mask")
    os.makedirs(orig_out, exist_ok=True)
    os.makedirs(desloc_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    with rasterio.open(tif_original_path) as orig_src,          rasterio.open(mask_path) as mask_src:

        profile = orig_src.profile.copy()
        nodata = orig_src.nodata
        res_x, res_y = orig_src.res
        transform_orig = orig_src.transform

        full_image = orig_src.read()
        width, height = orig_src.width, orig_src.height

        count = 0

        for row in range(0, height - patch_size + 1, patch_size):
            for col in range(0, width - patch_size + 1, patch_size):
                window = Window(col, row, patch_size, patch_size)
                window_bounds = rasterio.windows.bounds(window, transform_orig)

                try:
                    row_mask, col_mask = mask_src.index(window_bounds[0], window_bounds[3])
                    window_mask = Window(col_mask, row_mask, patch_size, patch_size)

                    if row_mask + patch_size > mask_src.height or col_mask + patch_size > mask_src.width:
                        continue

                    img_patch = full_image[:, row:row + patch_size, col:col + patch_size]
                    mask_patch = mask_src.read(1, window=window_mask)

                    if nodata is not None:
                        valid_pixels = (img_patch[0] != nodata).all()
                    else:
                        valid_pixels = not np.isnan(img_patch[0]).any()

                    if valid_pixels and mask_patch.sum() > 0:
                        # Gerar deslocamento aleatório
                        dx, dy = gerar_translacao_aleatoria(max_offset_m)
                        deslocado = deslocar_conteudo(full_image, dx, dy, res_x, res_y)
                        desloc_patch = deslocado[:, row:row + patch_size, col:col + patch_size]

                        patch_name = f"patch_{count}.tif"
                        desloc_name = f"patch_{count}_dx{dx:.2f}_dy{dy:.2f}.tif"

                        patch_profile = profile.copy()
                        patch_profile.update({
                            'height': patch_size,
                            'width': patch_size,
                            'count': 1,
                            'dtype': 'uint8',
                            'compress': 'none'
                        })

                        # Salva imagem original
                        patch_profile.update(transform=rasterio.windows.transform(window, transform_orig))
                        with rasterio.open(os.path.join(orig_out, patch_name), 'w', **patch_profile) as dst:
                            dst.write(img_patch[0], 1)

                        # Salva imagem deslocada (mesmo transform, mas dados transladados)
                        with rasterio.open(os.path.join(desloc_out, desloc_name), 'w', **patch_profile) as dst:
                            dst.write(desloc_patch[0], 1)

                        # Salva máscara
                        patch_profile.update(transform=rasterio.windows.transform(window_mask, mask_src.transform))
                        with rasterio.open(os.path.join(mask_out, patch_name), 'w', **patch_profile) as dst:
                            dst.write(mask_patch, 1)

                        count += 1

                except Exception:
                    continue

    print(f"[✓] {count} patches gerados com deslocamento aleatório.")

# Executa
if __name__ == "__main__":
    gerar_patches_com_translacao_aleatoria()
