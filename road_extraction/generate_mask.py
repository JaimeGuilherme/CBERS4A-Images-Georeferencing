import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import mapping
import numpy as np

imagem_tif = "CBERS4_PAN.tif"
vetor_rodovias = "rodovias_osm.gpkg"
saida_mascara = "mascara_rodovias.tif"

with rasterio.open(imagem_tif) as src:
    perfil = src.profile
    transform = src.transform
    shape = (src.height, src.width)
    crs = src.crs

rodovias = gpd.read_file(vetor_rodovias)
if rodovias.crs != crs:
    rodovias = rodovias.to_crs(crs)

geometrias = [(geom, 1) for geom in rodovias.geometry if geom is not None and geom.is_valid]

mascara = rasterize(
    geometrias,
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype="uint8"
)

perfil.update({"count": 1, "dtype": "uint8", "nodata": 0})

with rasterio.open(saida_mascara, "w", **perfil) as dst:
    dst.write(mascara, 1)

print("✅ Máscara salva:", saida_mascara)
