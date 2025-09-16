# 04_associar_pontos.py
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point

from components.metrics import association_metrics

def carregar_pontos_gpkg(pasta):
    print(f"\n📌 Processando Pontos GPKG na pasta {pasta}...")
    gdfs = []
    for arquivo in os.listdir(pasta):
        if arquivo.lower().endswith(".gpkg"):
            caminho = os.path.join(pasta, arquivo)
            try:
                gdf = gpd.read_file(caminho)
                gdfs.append(gdf)
                print(f"   ✅ Carregado: {arquivo} ({len(gdf)} pontos)")
            except Exception as e:
                print(f"   ❌ Erro ao carregar {arquivo}: {e}")
    if not gdfs:
        raise FileNotFoundError("Nenhum .gpkg encontrado na pasta!")
    gdf_unificado = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    print(f"   🔹 Total de pontos unificados: {len(gdf_unificado)}")
    return gdf_unificado

def _coords_xy(gdf):
    return np.column_stack((gdf.geometry.x.values, gdf.geometry.y.values))

def _pairwise_dist(a, b):
    """
    Distância euclidiana entre dois conjuntos de pontos 2D.
    a: (Na, 2), b: (Nb, 2) -> (Na, Nb)
    """
    a2 = np.sum(a*a, axis=1)[:, None]
    b2 = np.sum(b*b, axis=1)[None, :]
    ab = a @ b.T
    d2 = np.clip(a2 + b2 - 2*ab, 0.0, None)
    return np.sqrt(d2)

def associar_pontos(pontos_detectados_path, pasta_osm_path, output_path, max_distance=20.0):
    '''
    Faz matching 1–para–1 entre detectados e OSM usando Hungarian (globalmente ótimo).
    Mantém apenas pares com distância <= max_distance.
    '''
    detectados = gpd.read_file(pontos_detectados_path)
    osm = carregar_pontos_gpkg(pasta_osm_path)

    if detectados.crs != osm.crs:
        detectados = detectados.to_crs(osm.crs)

    detectados = detectados[detectados.geometry.notna() & detectados.geometry.geom_type.isin(["Point"])].copy()
    osm = osm[osm.geometry.notna() & osm.geometry.geom_type.isin(["Point"])].copy()

    if len(detectados) == 0 or len(osm) == 0:
        print('Nenhum ponto para associar.')
        return

    XYd = _coords_xy(detectados)
    XYo = _coords_xy(osm)
    D = _pairwise_dist(XYd, XYo)

    row_ind, col_ind = linear_sum_assignment(D)

    chosen_dist = D[row_ind, col_ind]
    ok = chosen_dist <= max_distance

    pares = []
    for i_ok, j_ok, dist in zip(row_ind[ok], col_ind[ok], chosen_dist[ok]):
        geom_d = detectados.geometry.iloc[i_ok]
        geom_o = osm.geometry.iloc[j_ok]
        pares.append({
            'id_detectado': int(detectados.index[i_ok]),
            'id_osm': int(osm.index[j_ok]),
            'distancia': float(dist),
            'det_x': float(geom_d.x),
            'det_y': float(geom_d.y),
            'osm_x': float(geom_o.x),
            'osm_y': float(geom_o.y),
            'geometry': geom_d
        })

    gdf_pares = gpd.GeoDataFrame(pares, geometry='geometry', crs=detectados.crs)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf_pares.to_file(output_path, driver='GeoJSON')
    print('✅ Pares salvos em', output_path, 'total:', len(gdf_pares))

    m = association_metrics(
        distances=np.array([p['distancia'] for p in pares]) if len(pares) else np.array([]),
        n_detectados=len(detectados),
        n_osm=len(osm),
        max_distance=max_distance
    )
    print("\n📊 Métricas de associação")
    for k, v in m.items():
        print(f" - {k}: {v}")

if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    associar_pontos(
        pontos_detectados_path='output/pontos_detectados.geojson',
        pasta_osm_path='input/pontos_gpkg',
        output_path='output/pares_homologos.geojson',
        max_distance=20.0
    )
