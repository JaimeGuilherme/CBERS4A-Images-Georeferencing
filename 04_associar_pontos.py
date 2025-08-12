import os, geopandas as gpd
from sklearn.neighbors import NearestNeighbors
import numpy as np

def associar_pontos(pontos_detectados_path, pontos_osm_path, output_path, max_distance=20):
    detectados = gpd.read_file(pontos_detectados_path)
    osm = gpd.read_file(pontos_osm_path)
    if detectados.crs != osm.crs:
        detectados = detectados.to_crs(osm.crs)
    coords_detect = np.array([(geom.x, geom.y) for geom in detectados.geometry])
    coords_osm = np.array([(geom.x, geom.y) for geom in osm.geometry])
    if len(coords_detect)==0 or len(coords_osm)==0:
        print('Nenhum ponto para associar.'); return
    nbrs = NearestNeighbors(n_neighbors=1).fit(coords_osm)
    distances, indices = nbrs.kneighbors(coords_detect)
    pares = []
    for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if dist <= max_distance:
            pares.append({'id_detectado': int(detectados.index[i]), 'id_osm': int(osm.index[idx]), 'distancia': float(dist), 'geometry_detectado': detectados.geometry.iloc[i], 'geometry_osm': osm.geometry.iloc[idx]})
    gdf_pares = gpd.GeoDataFrame(pares, geometry='geometry_detectado', crs=osm.crs)
    gdf_pares.to_file(output_path, driver='GeoJSON')
    print('Pares salvos em', output_path, 'total:', len(pares))

if __name__ == '__main__':
    os.makedirs('resultados', exist_ok=True)
    associar_pontos('resultados/pontos_detectados.geojson', 'input/intersecoes_osm.gpkg', 'resultados/pares_homologos.geojson')
