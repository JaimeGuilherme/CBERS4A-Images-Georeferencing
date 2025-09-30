# 04_associar_pontos.py
import os
import math
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point
import rasterio

from components.metrics import association_metrics

# ------------------------ utilitários ------------------------

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

def _infer_image_name_from_patch(patch_name: str) -> str:
    """
    Deduz o nome da imagem original a partir do nome do patch.
    Ex.: 'IMG_001_patch_00012.tif' -> 'IMG_001.tif'
    """
    base = patch_name
    if "_patch_" in patch_name:
        base = patch_name.split("_patch_")[0] + ".tif"
    return base

def _find_raster_for_patch(pasta_imagens_orig: str, patch_name: str) -> str | None:
    """
    Tenta localizar o TIFF original correspondente a um patch.
    Retorna caminho completo ou None.
    """
    base_name = _infer_image_name_from_patch(patch_name)
    cand = os.path.join(pasta_imagens_orig, base_name)
    if os.path.exists(cand):
        return cand
    prefix = base_name.replace(".tif", "")
    for f in os.listdir(pasta_imagens_orig):
        if f.lower().endswith(".tif") and f.startswith(prefix):
            return os.path.join(pasta_imagens_orig, f)
    return None

def _map_to_pixel_xy(gdf_points: gpd.GeoDataFrame, raster_path: str) -> np.ndarray:
    """
    Converte coordenadas de mapa (x,y) -> coordenadas de pixel (col=x, row=y) do raster.
    Retorna array shape (N, 2) com (pixelX, pixelY). Se falhar, retorna NaN.
    """
    pxpy = np.full((len(gdf_points), 2), np.nan, dtype=float)
    if raster_path is None or len(gdf_points) == 0:
        return pxpy
    try:
        with rasterio.open(raster_path) as src:
            inv = ~src.transform
            xs = gdf_points.geometry.x.values
            ys = gdf_points.geometry.y.values
            cols, rows = inv * (xs, ys)
            pxpy[:, 0] = np.round(cols).astype(float)
            pxpy[:, 1] = np.round(rows).astype(float)
    except Exception as e:
        print(f"   ⚠️ Falha ao calcular pixel XY para {raster_path}: {e}")
    return pxpy

def _pick_metric_crs(gdf: gpd.GeoDataFrame):
    """
    Retorna um CRS projetado (UTM) apropriado ao centróide do gdf.
    """
    if gdf.crs is None or gdf.crs.is_geographic:
        cen = gdf.geometry.unary_union.centroid
        lon, lat = float(cen.x), float(cen.y)
        zone = int(math.floor((lon + 180.0) / 6.0) + 1)
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        return f"EPSG:{epsg}"
    return gdf.crs

# ------------------------ matching tileado ------------------------

def _tile_key(x: float, y: float, tile_size: float) -> tuple[int, int]:
    return (int(math.floor(x / tile_size)), int(math.floor(y / tile_size)))

def _build_osm_tile_index(osm_xy: np.ndarray, tile_size: float):
    """
    Cria índice: tile -> lista de índices OSM naquele tile.
    """
    tile_index = {}
    for j, (x, y) in enumerate(osm_xy):
        tk = _tile_key(x, y, tile_size)
        tile_index.setdefault(tk, []).append(j)
    return tile_index

def _neighbors(tk: tuple[int, int]):
    tx, ty = tk
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            yield (tx + dx, ty + dy)

def _hungarian_per_tile(det_xy: np.ndarray, osm_xy: np.ndarray, assign_radius: float):
    """
    Executa Hungarian por tiles. Retorna listas:
    - det_idx_sel, osm_idx_sel, dist_sel
    """
    tile_size = assign_radius * 2.0
    tile_index = _build_osm_tile_index(osm_xy, tile_size)

    # agrupa candidatos por tile do detectado
    edges_by_tile = {}  # tk -> list of (i_det, j_osm, dist)
    for i, (xd, yd) in enumerate(det_xy):
        tk = _tile_key(xd, yd, tile_size)
        cand_js = []
        for nb in _neighbors(tk):
            cand_js.extend(tile_index.get(nb, []))
        if not cand_js:
            continue
        # filtra por raio
        c_xy = osm_xy[np.array(cand_js)]
        dx = c_xy[:, 0] - xd
        dy = c_xy[:, 1] - yd
        d = np.hypot(dx, dy)
        ok = d <= assign_radius
        if not np.any(ok):
            continue
        for j_local, dist in zip(np.array(cand_js)[ok], d[ok]):
            edges_by_tile.setdefault(tk, []).append((i, int(j_local), float(dist)))

    det_idx_sel, osm_idx_sel, dist_sel = [], [], []

    # roda Hungarian por tile
    INF = 1e9
    for tk, edges in edges_by_tile.items():
        dets = sorted({i for (i, _, _) in edges})
        osms = sorted({j for (_, j, _) in edges})
        map_det = {i: k for k, i in enumerate(dets)}
        map_osm = {j: k for k, j in enumerate(osms)}
        D = np.full((len(dets), len(osms)), INF, dtype=np.float32)
        for (i, j, dist) in edges:
            D[map_det[i], map_osm[j]] = min(D[map_det[i], map_osm[j]], dist)

        row_ind, col_ind = linear_sum_assignment(D)
        for r, c in zip(row_ind, col_ind):
            d = float(D[r, c])
            if d <= assign_radius and d < INF:
                det_idx_sel.append(dets[r])
                osm_idx_sel.append(osms[c])
                dist_sel.append(d)

    return det_idx_sel, osm_idx_sel, dist_sel

# ------------------------ pipeline principal ------------------------

def associar_pontos(
    pontos_detectados_path: str,
    pasta_osm_path: str,
    output_path: str,
    pasta_imagens_orig: str,
    max_distance: float = 20.0
):
    """
    Faz matching 1–para–1 (Hungarian) entre detectados (correto) e OSM (original/errado) em tiles.
    Salva:
      - GeoJSON com pares (geometry = detectado)
      - CSV 'pares_homologos.csv' (resumo)
      - CSV 'scp_homologos.csv' (src=original, dst=inferido)
      - CSV 'gcp_qgis.csv' (mapX,mapY,pixelX,pixelY,enable,id,image) para Georreferenciador do QGIS
      - CSV 'analitico_pares.csv' (para gráficos/estatísticas)
      - CSVs de não pareados
      - CSV 'metricas_associacao.csv'
    """
    # --- leitura
    detectados = gpd.read_file(pontos_detectados_path)  # inferidos
    osm = carregar_pontos_gpkg(pasta_osm_path)          # originais (errados)

    # alinhar CRS (projeta para métrico se necessário)
    if detectados.crs != osm.crs:
        detectados = detectados.set_crs(detectados.crs, allow_override=True)
        osm = osm.set_crs(osm.crs, allow_override=True)
        # projeta ambos para o CRS do detectado (se existirem e forem iguais) apenas para alinhar
        # depois escolhemos um métrico:
        if detectados.crs is not None and osm.crs is not None:
            osm = osm.to_crs(detectados.crs)

    # escolher CRS métrico
    metric_crs = _pick_metric_crs(detectados if len(detectados) else osm)
    if (detectados.crs is None) or detectados.crs.is_geographic or (str(detectados.crs) != str(metric_crs)):
        print(f"ℹ️ Reprojetando para CRS métrico: {metric_crs}")
        detectados = detectados.to_crs(metric_crs)
    if (osm.crs is None) or osm.crs.is_geographic or (str(osm.crs) != str(metric_crs)):
        osm = osm.to_crs(metric_crs)

    # filtros mínimos
    detectados = detectados[detectados.geometry.notna() & detectados.geometry.geom_type.isin(["Point"])].copy()
    osm = osm[osm.geometry.notna() & osm.geometry.geom_type.isin(["Point"])].copy()

    if len(detectados) == 0 or len(osm) == 0:
        print('Nenhum ponto para associar.')
        return

    # --- matching tileado
    XYd = _coords_xy(detectados).astype(np.float32)
    XYo = _coords_xy(osm).astype(np.float32)

    det_idx_sel, osm_idx_sel, dist_sel = _hungarian_per_tile(XYd, XYo, assign_radius=float(max_distance))

    # --- construir pares finais
    pares = []
    for i_det, j_osm, dist in zip(det_idx_sel, osm_idx_sel, dist_sel):
        gd = detectados.iloc[i_det]
        go = osm.iloc[j_osm]
        image_name = None
        if "patch" in detectados.columns and isinstance(gd.get("patch", None), str):
            image_name = _infer_image_name_from_patch(gd["patch"])
        pares.append({
            'id_detectado': int(detectados.index[i_det]),
            'id_osm': int(osm.index[j_osm]),
            'distancia': float(dist),
            'det_x': float(gd.geometry.x),
            'det_y': float(gd.geometry.y),
            'osm_x': float(go.geometry.x),
            'osm_y': float(go.geometry.y),
            'image': image_name,
            'geometry': gd.geometry
        })

    # --- salvar GeoJSON dos pares (geom = detectado)
    gdf_pares = gpd.GeoDataFrame(pares, geometry='geometry', crs=detectados.crs)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf_pares.to_file(output_path, driver='GeoJSON')
    print('✅ Pares salvos em', output_path, 'total:', len(gdf_pares))

    # --- CSV básico de pares
    pares_csv_path = os.path.join(os.path.dirname(output_path), 'pares_homologos.csv')
    cols_csv = ['id_detectado', 'id_osm', 'distancia', 'det_x', 'det_y', 'osm_x', 'osm_y', 'image']
    pd.DataFrame([{k: p.get(k, None) for k in cols_csv} for p in pares]).to_csv(pares_csv_path, index=False)
    print(f'📝 CSV de pares salvo em {pares_csv_path}')

    # --- pixelX/pixelY por imagem (inferida do patch)
    # preenche det_px/det_py/osm_px/osm_py
    for img in sorted({p['image'] for p in pares if p['image']}):
        raster_path = _find_raster_for_patch(pasta_imagens_orig, img)
        if raster_path is None:
            print(f"   ⚠️ Raster não encontrado para '{img}' em {pasta_imagens_orig}. Pixels ficarão NA.")
        idxs = [i for i, p in enumerate(pares) if p['image'] == img]
        if not idxs:
            continue

        gdf_det_tmp = gpd.GeoDataFrame(
            geometry=[Point(pares[i]['det_x'], pares[i]['det_y']) for i in idxs], crs=detectados.crs
        )
        gdf_osm_tmp = gpd.GeoDataFrame(
            geometry=[Point(pares[i]['osm_x'], pares[i]['osm_y']) for i in idxs], crs=detectados.crs
        )
        det_pxpy = _map_to_pixel_xy(gdf_det_tmp, raster_path)
        osm_pxpy = _map_to_pixel_xy(gdf_osm_tmp, raster_path)

        for k, i in enumerate(idxs):
            pares[i]['det_px'] = float(det_pxpy[k, 0]) if np.isfinite(det_pxpy[k, 0]) else None
            pares[i]['det_py'] = float(det_pxpy[k, 1]) if np.isfinite(det_pxpy[k, 1]) else None
            pares[i]['osm_px'] = float(osm_pxpy[k, 0]) if np.isfinite(osm_pxpy[k, 0]) else None
            pares[i]['osm_py'] = float(osm_pxpy[k, 1]) if np.isfinite(osm_pxpy[k, 1]) else None

    # --- CSV GCP (QGIS Georreferenciador): usa o ponto inferido (correto)
    gcp_rows = []
    for idx, p in enumerate(pares, start=1):
        gcp_rows.append({
            'id': idx,
            'mapX': p['det_x'],
            'mapY': p['det_y'],
            'pixelX': p.get('det_px', None),
            'pixelY': p.get('det_py', None),
            'enable': 1,
            'image': p.get('image', None)
        })
    gcp_csv_path = os.path.join(os.path.dirname(output_path), 'gcp_qgis.csv')
    pd.DataFrame(gcp_rows, columns=['id','mapX','mapY','pixelX','pixelY','enable','image']).to_csv(gcp_csv_path, index=False)
    print(f'📌 CSV GCP (QGIS) salvo em {gcp_csv_path}')

    # --- CSV “SCP/Homólogos” (src=original/errado, dst=inferido/correto)
    scp_rows = []
    for p in pares:
        scp_rows.append({
            'srcX': p['osm_x'], 'srcY': p['osm_y'],
            'dstX': p['det_x'], 'dstY': p['det_y'],
            'image': p.get('image', None)
        })
    scp_csv_path = os.path.join(os.path.dirname(output_path), 'scp_homologos.csv')
    pd.DataFrame(scp_rows, columns=['srcX','srcY','dstX','dstY','image']).to_csv(scp_csv_path, index=False)
    print(f'🔗 CSV Homólogos (SCP) salvo em {scp_csv_path}')

    # --- CSV analítico (gráficos/estatísticas)
    analitico_rows = []
    for idx, p in enumerate(pares, start=1):
        analitico_rows.append({
            'id': idx, 'image': p.get('image', None),
            'orig_x': p['osm_x'], 'orig_y': p['osm_y'],
            'orig_px': p.get('osm_px', None), 'orig_py': p.get('osm_py', None),
            'inf_x': p['det_x'],  'inf_y': p['det_y'],
            'inf_px': p.get('det_px', None), 'inf_py': p.get('det_py', None),
            'dist_m': p['distancia']
        })
    analitico_csv_path = os.path.join(os.path.dirname(output_path), 'analitico_pares.csv')
    pd.DataFrame(analitico_rows).to_csv(analitico_csv_path, index=False)
    print(f'📈 CSV analítico salvo em {analitico_csv_path}')

    # --- não pareados
    det_pareados = set(det_idx_sel)
    osm_pareados = set(osm_idx_sel)

    det_nao_pareados = detectados.iloc[[i for i in range(len(detectados)) if i not in det_pareados]].copy()
    osm_nao_pareados = osm.iloc[[j for j in range(len(osm)) if j not in osm_pareados]].copy()

    if len(det_nao_pareados) > 0:
        det_np_csv = os.path.join(os.path.dirname(output_path), 'detectados_nao_pareados.csv')
        pd.DataFrame({
            'id_detectado': det_nao_pareados.index.astype(int),
            'x': det_nao_pareados.geometry.x.astype(float),
            'y': det_nao_pareados.geometry.y.astype(float)
        }).to_csv(det_np_csv, index=False)
        print(f'🟡 Detectados não pareados: {len(det_nao_pareados)} (CSV: {det_np_csv})')

    if len(osm_nao_pareados) > 0:
        osm_np_csv = os.path.join(os.path.dirname(output_path), 'osm_nao_pareados.csv')
        pd.DataFrame({
            'id_osm': osm_nao_pareados.index.astype(int),
            'x': osm_nao_pareados.geometry.x.astype(float),
            'y': osm_nao_pareados.geometry.y.astype(float)
        }).to_csv(osm_np_csv, index=False)
        print(f'🟠 OSM não pareados: {len(osm_nao_pareados)} (CSV: {osm_np_csv})')

    # --- métricas de associação
    m = association_metrics(
        distances=np.array(dist_sel, dtype=float) if len(dist_sel) else np.array([]),
        n_detectados=len(detectados),
        n_osm=len(osm),
        max_distance=max_distance
    )
    print("\n📊 Métricas de associação")
    for k, v in m.items():
        print(f" - {k}: {v}")

    metricas_csv_path = os.path.join(os.path.dirname(output_path), 'metricas_associacao.csv')
    pd.DataFrame([m]).to_csv(metricas_csv_path, index=False)
    print(f'📄 CSV de métricas salvo em {metricas_csv_path}')

# ------------------------ main ------------------------

if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    associar_pontos(
        pontos_detectados_path='output/pontos_detectados.geojson',
        pasta_osm_path='input/pontos_gpkg',
        output_path='output/pares_homologos.geojson',
        pasta_imagens_orig='input/imagens_tif',
        max_distance=20.0
    )
