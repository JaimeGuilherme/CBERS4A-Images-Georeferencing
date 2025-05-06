import rasterio
from rasterio.warp import reproject, Resampling
import rasterio.features
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from tqdm import tqdm

# Path das imagens
# gpkg_path = 'vetor_recortado.gpkg'
# tif_original = 'imagem/VCP/campinas.tif'
# tif_alinhado = 'campinas_alinhado.tif'
# gpkg_rasterizado = 'estrada_campinas_rasterizado_alinhado.tif'

# Campo de categorização do GPKG (opcional)
campo_categoria = 'CLASSE'

# Etapa 1: Carregar o GPKG e obter informações
gdf = gpd.read_file(gpkg_path)
gdf = gdf[gdf.is_valid]
gpkg_crs = gdf.crs
gpkg_bounds = gdf.total_bounds

# Etapa 2: Abrir o raster original e alinhar com a malha do GPKG
with rasterio.open(tif_original) as src:
    src_crs = src.crs
    src_transform = src.transform
    src_res = (src.transform.a, -src.transform.e)
    src_dtype = src.dtypes[0]
    src_count = src.count

    # Reprojetar GPKG se necessário
    if gpkg_crs != src_crs:
        gdf = gdf.to_crs(src_crs)
        gpkg_bounds = gdf.total_bounds

    # Calcular origem alinhada
    origin_x = gpkg_bounds[0] - (gpkg_bounds[0] % src_res[0])
    origin_y = gpkg_bounds[3] - (gpkg_bounds[3] % src_res[1])

    # Calcular nova largura e altura
    width = int((gpkg_bounds[2] - origin_x) / src_res[0])
    height = int((origin_y - gpkg_bounds[1]) / src_res[1])

    new_transform = rasterio.transform.from_origin(origin_x, origin_y, *src_res)

    aligned_profile = src.profile.copy()
    aligned_profile.update({
        'height': height,
        'width': width,
        'transform': new_transform
    })

    # Criar array alinhado
    aligned_data = np.zeros((src_count, height, width), dtype=src_dtype)

    for i in range(1, src_count + 1):
        reproject(
            source=rasterio.band(src, i),
            destination=aligned_data[i - 1],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=new_transform,
            dst_crs=src_crs,
            resampling=Resampling.nearest
        )

# Salvar TIFF realinhado
with rasterio.open(tif_alinhado, 'w', **aligned_profile) as dst:
    dst.write(aligned_data)

print(f"TIF alinhado salvo: {tif_alinhado}")

# Etapa 3: Rasterizar o GPKG com base no mesmo grid

# Preparar shapes para rasterização
if campo_categoria and campo_categoria in gdf.columns:
    categorias_unicas = gdf[campo_categoria].dropna().unique()
    mapa_valores = {valor: i + 1 for i, valor in enumerate(sorted(categorias_unicas))}

    shapes = (
        (geom, mapa_valores[attr])
        for geom, attr in tqdm(zip(gdf.geometry, gdf[campo_categoria]), total=len(gdf))
        if attr in mapa_valores
    )
else:
    shapes = ((geom, 1) for geom in tqdm(gdf.geometry))

# Criar raster com os mesmos parâmetros do TIFF realinhado
raster_profile = aligned_profile.copy()
raster_profile.update({
    'count': 1,
    'dtype': rasterio.uint8
})

with rasterio.open(gpkg_rasterizado, 'w', **raster_profile) as dst:
    rasterizado = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=new_transform,
        fill=0,
        all_touched=True
    )
    dst.write(rasterizado, 1)

print(f"Raster do GPKG alinhado ao TIFF salvo em: {gpkg_rasterizado}")
