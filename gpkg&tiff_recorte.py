import geopandas as gpd
import rasterio
from shapely.geometry import box

# Path das imagens de entrada
# tif_path = "imagem/VCP/campinas.tif"
# gpkg_path = "gpkg/roads_sudeste.gpkg"

# Path imagem de saida
# saida_recorte = "vetor_recortado.gpkg"

# Abre o raster e obtém o bounding box
with rasterio.open(tif_path) as src:
    bbox = box(*src.bounds)
    raster_crs = src.crs

# Abre o vetor
vetor = gpd.read_file(gpkg_path)

# Reprojeta o vetor, se necessário
if vetor.crs != raster_crs:
    vetor = vetor.to_crs(raster_crs)

# Cria GeoDataFrame da área do raster
bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs=raster_crs)

# Recorta o vetor
vetor_recortado = gpd.overlay(vetor, bbox_gdf, how='intersection')

# Salva o resultado
vetor_recortado.to_file(saida_recorte, layer="recorte", driver="GPKG")