import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
import os

def associar_pontos(detectados_path, originais_path, output_path):
    # Carrega pontos detectados e pontos originais
    gdf_detectados = gpd.read_file(detectados_path)
    gdf_originais = gpd.read_file(originais_path)

    # Ajusta CRS caso sejam diferentes
    if gdf_detectados.crs != gdf_originais.crs:
        gdf_detectados = gdf_detectados.to_crs(gdf_originais.crs)

    coords_detectados = [[p.x, p.y] for p in gdf_detectados.geometry]
    coords_originais = [[p.x, p.y] for p in gdf_originais.geometry]

    # Cria modelo de vizinho mais próximo
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(coords_originais)

    # Para cada ponto detectado, acha o índice do ponto original mais próximo
    distancias, indices = nn.kneighbors(coords_detectados)

    # Monta lista de pares homólogos
    pares = []
    for i, (dist, idx) in enumerate(zip(distancias[:, 0], indices[:, 0])):
        pares.append({
            'geometry': gdf_detectados.geometry.iloc[i],
            'id_detectado': i,
            'id_original': idx,
            'distancia': dist
        })

    # Cria GeoDataFrame com os pares associados
    gdf_pares = gpd.GeoDataFrame(pares, crs=gdf_detectados.crs)

    # Salva resultado para uso no QGIS
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf_pares.to_file(output_path, driver="GeoJSON")
    print(f"[✔] {len(gdf_pares)} pares homólogos salvos em: {output_path}")

if __name__ == "__main__":
    detectados_path = "resultados/pontos_detectados.geojson"
    originais_path = "input/intersecoes_osm.gpkg"
    output_path = "resultados/pontos_homologos.geojson"

    associar_pontos(detectados_path, originais_path, output_path)
