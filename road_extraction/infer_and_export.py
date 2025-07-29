import torch
import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from rasterio.windows import Window
from rasterio.features import shapes
from train_unet import UNet
from tqdm import tqdm

def export_pred_to_vector(pred_bin, reference_raster, output_vector):
    with rasterio.open(reference_raster) as src:
        transform = src.transform
        crs = src.crs
    results = (
        {"geometry": shape(geom), "properties": {"value": value}}
        for geom, value in shapes(pred_bin.astype(np.uint8), mask=pred_bin.astype(bool), transform=transform)
        if value == 1
    )
    gdf = gpd.GeoDataFrame.from_features(results)
    gdf.set_crs(crs, inplace=True)
    gdf.to_file(output_vector, driver="GPKG")
    print("✅ Vetor salvo como:", output_vector)

# === Caminhos ===
raster_path = "CBERS4_PAN.tif"
modelo_path = "unet_rodovias.pth"
saida_pred_tif = "pred_binaria.tif"
saida_vetor = "rodovias_preditas.gpkg"
patch_size = 256

# === Carregar modelo treinado ===
model = UNet()
model.load_state_dict(torch.load(modelo_path, map_location="cpu"))
model.eval()

# === Abrir imagem original ===
with rasterio.open(raster_path) as src:
    profile = src.profile
    width = src.width
    height = src.height
    transform = src.transform
    crs = src.crs

    # Inicializar array de predição
    pred_full = np.zeros((height, width), dtype=np.float32)

    # Total de blocos horizontais e verticais
    n_rows = height // patch_size
    n_cols = width // patch_size

    # Percorrer janelas com barra de progresso
    for i in tqdm(range(n_rows), desc="Inferência por blocos"):
        for j in range(n_cols):
            y = i * patch_size
            x = j * patch_size
            if y + patch_size > height or x + patch_size > width:
                continue

            window = Window(x, y, patch_size, patch_size)
            patch = src.read(1, window=window).astype(np.float32) / 255.0
            patch_tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                pred_patch = model(patch_tensor).squeeze().numpy()

            pred_full[y:y+patch_size, x:x+patch_size] = pred_patch

# === Binarizar
pred_bin = pred_full > 0.5

# === Salvar predição binária
profile.update(dtype=rasterio.uint8, count=1)
with rasterio.open(saida_pred_tif, "w", **profile) as dst:
    dst.write(pred_bin.astype(np.uint8), 1)

print("🖼️ Predição binária salva em:", saida_pred_tif)

# === Exportar como vetor GPKG
export_pred_to_vector(pred_bin, raster_path, saida_vetor)
