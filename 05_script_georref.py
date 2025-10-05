# 05_script_georref.py
# -----------------------------------------------------------------------------
# Execução: python 05_script_georref.py
# Estrutura esperada:
#   ./main_input/
#       imagem.tif (ou .tiff)
#       cruzamentos.gpkg
#       modelo.pth
# Saídas:
#   ./main_output/pontos_inferidos.gpkg
#   ./main_output/georeferencer.points
#   ./main_output/pares_homologos.geojson (opcional, auditoria)
# -----------------------------------------------------------------------------

import os
import math
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
from scipy.ndimage import label, center_of_mass
from scipy.optimize import linear_sum_assignment
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import DataLoader

# ====== imports flexíveis para o seu projeto ======
def _try_import():
    # tenta com o pacote "components"
    try:
        from components.dataset import RoadIntersectionDataset
        from components.unet import UNet
        from components.utils import load_checkpoint as _load_ckpt, load_checkpoint_raw as _load_ckpt_raw
        return RoadIntersectionDataset, UNet, _load_ckpt, _load_ckpt_raw
    except Exception:
        pass
    # tenta módulos no diretório raiz (dataset.py, unet.py, utils.py)
    try:
        from dataset import RoadIntersectionDataset
        from unet import UNet
        from utils import load_checkpoint as _load_ckpt, load_checkpoint_raw as _load_ckpt_raw
        return RoadIntersectionDataset, UNet, _load_ckpt, _load_ckpt_raw
    except Exception as e:
        raise ImportError(
            "Não foi possível importar RoadIntersectionDataset/UNet/utils. "
            "Garanta que estão em 'components/' ou ao lado deste script."
        ) from e

RoadIntersectionDataset, UNet, load_checkpoint, load_checkpoint_raw = _try_import()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
#   Descoberta de arquivos
# =========================
def achar_entrada(pasta_main_input="main_input"):
    if not os.path.isdir(pasta_main_input):
        raise FileNotFoundError(f"Pasta '{pasta_main_input}' não encontrada.")
    # geotiff
    tif_list = sorted(glob.glob(os.path.join(pasta_main_input, "*.tif")) + glob.glob(os.path.join(pasta_main_input, "*.tiff")))
    if not tif_list:
        raise FileNotFoundError("Nenhuma imagem GeoTIFF (.tif/.tiff) encontrada em ./main_input/")
    imagem = tif_list[0]

    # gpkg
    gpkg_list = sorted(glob.glob(os.path.join(pasta_main_input, "*.gpkg")))
    if not gpkg_list:
        raise FileNotFoundError("Nenhum GPKG encontrado em ./main_input/")
    gpkg = gpkg_list[0]

    # pth
    pth_list = sorted(glob.glob(os.path.join(pasta_main_input, "*.pth")))
    if not pth_list:
        raise FileNotFoundError("Nenhum checkpoint .pth encontrado em ./main_input/")
    modelo = pth_list[0]

    return imagem, gpkg, modelo


# =========================
#   Utilidades
# =========================
def carregar_pontos_gpkg(path_ou_pasta: str) -> gpd.GeoDataFrame:
    if os.path.isdir(path_ou_pasta):
        gdfs = []
        for f in os.listdir(path_ou_pasta):
            if f.lower().endswith(".gpkg"):
                gdfs.append(gpd.read_file(os.path.join(path_ou_pasta, f)))
        if not gdfs:
            raise FileNotFoundError("Nenhum .gpkg encontrado na pasta!")
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    else:
        if not path_ou_pasta.lower().endswith(".gpkg"):
            raise ValueError("Forneça um .gpkg válido ou uma pasta com .gpkg")
        return gpd.read_file(path_ou_pasta)

def pontos_em_pixels(gdf_pontos: gpd.GeoDataFrame, img_transform, img_width, img_height, img_crs) -> list[tuple[int,int]]:
    if gdf_pontos.crs is None:
        raise ValueError("O GPKG de entrada não tem CRS definido.")
    transformer = Transformer.from_crs(gdf_pontos.crs, img_crs, always_xy=True)
    pts = []
    for _, row in gdf_pontos.iterrows():
        geom = row.geometry
        if geom is None or geom.geom_type != "Point":
            continue
        x_src, y_src = geom.x, geom.y
        x, y = transformer.transform(x_src, y_src)
        px, py = ~img_transform * (x, y)
        px, py = int(px), int(py)
        if 0 <= px < img_width and 0 <= py < img_height:
            pts.append((px, py))
    return pts

def gerar_patches_interesse(imagem_path: str, saida_patches: str,
                            points_px: list[tuple[int,int]], patch_size=256):
    os.makedirs(saida_patches, exist_ok=True)
    patches = []
    pid = 0
    with rasterio.open(imagem_path) as src:
        W, H = src.width, src.height
        meta_base = src.meta.copy()
        total_rows = (H + patch_size - 1)//patch_size
        for top in tqdm(range(0, H, patch_size), desc="Gerando patches", total=total_rows):
            for left in range(0, W, patch_size):
                w = min(patch_size, W - left)
                h = min(patch_size, H - top)
                # patch válido apenas se contiver algum ponto de referência
                if not any(left <= px < left + w and top <= py < top + h for (px, py) in points_px):
                    continue
                window = Window(left, top, w, h)
                patch = src.read(window=window)  # (C, h, w)
                # Normalização simples banda a banda para [0,255]
                patch = patch.astype(np.float32)
                for b in range(patch.shape[0]):
                    mn, mx = float(patch[b].min()), float(patch[b].max())
                    if mx > mn:
                        patch[b] = (patch[b] - mn) / (mx - mn) * 255.0
                    else:
                        patch[b] = 0.0
                patch = patch.astype(np.uint8)
                # Ajuste de canais (3 ou 4)
                C = patch.shape[0]
                if C >= 4:
                    patch = patch[:4]
                elif C == 2:
                    patch = np.concatenate([patch, patch[:1]], axis=0)  # vira 3
                elif C == 1:
                    patch = np.repeat(patch, 3, axis=0)

                meta = meta_base.copy()
                meta.update({
                    "height": h, "width": w,
                    "transform": src.window_transform(window),
                    "count": patch.shape[0],
                    "dtype": patch.dtype,
                    "driver": "GTiff"
                })
                fname = f"patch_{pid:05d}.tif"
                fpath = os.path.join(saida_patches, fname)
                with rasterio.open(fpath, "w", **meta) as dst:
                    dst.write(patch)
                patches.append((fname, left, top))  # guardamos offsets
                pid += 1
    return patches

def build_model(in_ch: int):
    # UNet simples do seu projeto
    model = UNet(in_channels=in_ch, out_channels=1).to(DEVICE)
    return model

def adapt_first_conv_if_needed(model, checkpoint_state):
    state = checkpoint_state.get('model_state_dict', checkpoint_state)
    model_state = model.state_dict()
    conv_keys = [k for k in model_state.keys() if k.endswith("weight") and model_state[k].dim() == 4]
    if not conv_keys:
        model.load_state_dict(state, strict=False); return
    first_conv_key = conv_keys[0]
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

def inferir_pontos_de_patches(patches_dir: str, modelo_path: str, bands_mode: str = "rgbnir", batch_size: int = 16):
    in_ch = 3 if bands_mode == "rgb" else 4
    model = build_model(in_ch)
    ckpt = load_checkpoint_raw(modelo_path, map_location=DEVICE) if 'load_checkpoint_raw' in globals() else load_checkpoint(modelo_path, model)
    # threshold salvo no checkpoint (fallback 0.5)
    threshold = ckpt.get("best_threshold", 0.5) if isinstance(ckpt, dict) else 0.5
    # caso load_checkpoint() já carregue os pesos, tentamos adaptar se necessário
    if isinstance(ckpt, dict):
        adapt_first_conv_if_needed(model, ckpt)

    ds = RoadIntersectionDataset(patches_dir, masks_dir=None, transform=None, is_training=False,
                                 bands_mode=("rgb" if in_ch == 3 else "rgbnir"))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    pontos = []
    crs_img = None
    model.eval()
    with torch.no_grad():
        for imgs, nomes in tqdm(dl, desc="Inferindo patches"):
            imgs = imgs.to(DEVICE)
            saida = torch.sigmoid(model(imgs))
            preds = (saida > threshold).float().cpu().numpy()  # (B,1,H,W)
            for bi in range(preds.shape[0]):
                mask = preds[bi].squeeze().astype(np.uint8)
                labeled, num_features = label(mask)
                if num_features == 0:
                    continue
                centros = center_of_mass(mask, labeled, range(1, num_features + 1))
                patch_name = nomes[bi]
                patch_path = os.path.join(patches_dir, patch_name)
                with rasterio.open(patch_path) as src:
                    patch_transform = src.transform
                    crs_img = src.crs
                for c in centros:
                    if len(c) == 2:
                        y, x = c
                    elif len(c) == 3:
                        _, y, x = c
                    else:
                        continue
                    x_geo, y_geo = rasterio.transform.xy(patch_transform, int(round(y)), int(round(x)))
                    pontos.append({
                        "geometry": Point(x_geo, y_geo),
                        "patch": patch_name,
                        "x_local": int(round(x)),
                        "y_local": int(round(y))
                    })
    gdf = gpd.GeoDataFrame(pontos, crs=crs_img) if len(pontos) else gpd.GeoDataFrame()
    return gdf

# -------- associação Húngara por tiles --------
def _coords_xy(gdf: gpd.GeoDataFrame) -> np.ndarray:
    return np.column_stack((gdf.geometry.x.values, gdf.geometry.y.values))

def _tile_key(x: float, y: float, tile_size: float) -> tuple[int, int]:
    return (int(math.floor(x / tile_size)), int(math.floor(y / tile_size)))

def _build_tile_index(xy: np.ndarray, tile_size: float):
    idx = {}
    for i, (x, y) in enumerate(xy):
        tk = _tile_key(x, y, tile_size)
        idx.setdefault(tk, []).append(i)
    return idx

def _neighbors(tk: tuple[int,int]):
    tx, ty = tk
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            yield (tx+dx, ty+dy)

def hungarian_match_per_tile(det_xy: np.ndarray, ref_xy: np.ndarray, assign_radius: float):
    tile_size = assign_radius * 2.0
    ref_index = _build_tile_index(ref_xy, tile_size)
    edges_by_tile = {}

    for i, (xd, yd) in enumerate(det_xy):
        tk = _tile_key(xd, yd, tile_size)
        cand = []
        for nb in _neighbors(tk):
            cand.extend(ref_index.get(nb, []))
        if not cand:
            continue
        c_xy = ref_xy[np.array(cand)]
        d = np.hypot(c_xy[:,0]-xd, c_xy[:,1]-yd)
        ok = d <= assign_radius
        if not np.any(ok):
            continue
        for j_local, dist in zip(np.array(cand)[ok], d[ok]):
            edges_by_tile.setdefault(tk, []).append((i, int(j_local), float(dist)))

    det_idxs, ref_idxs, dists = [], [], []
    INF = 1e9
    for tk, edges in edges_by_tile.items():
        dets = sorted({i for (i, _, _) in edges})
        refs = sorted({j for (_, j, _) in edges})
        map_det = {i:k for k,i in enumerate(dets)}
        map_ref = {j:k for k,j in enumerate(refs)}
        D = np.full((len(dets), len(refs)), INF, dtype=np.float32)
        for (i, j, dist) in edges:
            D[map_det[i], map_ref[j]] = min(D[map_det[i], map_ref[j]], dist)
        row_ind, col_ind = linear_sum_assignment(D)
        for r, c in zip(row_ind, col_ind):
            d = float(D[r, c])
            if d <= assign_radius and d < INF:
                det_idxs.append(dets[r])
                ref_idxs.append(refs[c])
                dists.append(d)
    return det_idxs, ref_idxs, dists

def escrever_points_qgis(pares, main_output_dir, nome="georeferencer.points"):
    os.makedirs(main_output_dir, exist_ok=True)
    fpath = os.path.join(main_output_dir, nome)
    with open(fpath, "w", encoding="utf-8") as f:
        for row in pares:
            # mapX,mapY,pixelX,pixelY,enable
            f.write(f"{row['mapX']},{row['mapY']},{row['pixelX']},{row['pixelY']},1\n")
    return fpath

def pick_metric_crs(gdf_base: gpd.GeoDataFrame):
    if gdf_base.crs is None or getattr(gdf_base.crs, "is_geographic", False):
        cen = gdf_base.geometry.unary_union.centroid
        lon, lat = float(cen.x), float(cen.y)
        zone = int(math.floor((lon + 180.0)/6.0) + 1)
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        return f"EPSG:{epsg}"
    return gdf_base.crs


# =========================
#   MAIN
# =========================
def main():
    # 0) entradas
    imagem_path, gpkg_path, modelo_path = achar_entrada("main_input")
    os.makedirs("main_output", exist_ok=True)
    patches_dir = os.path.join("main_output", "patches")
    os.makedirs(patches_dir, exist_ok=True)

    print(f"🛰️ Imagem: {imagem_path}")
    print(f"🧭 Pontos (GPKG): {gpkg_path}")
    print(f"🧠 Modelo: {modelo_path}")

    # 1) ler imagem + pontos referência
    pontos_ref = carregar_pontos_gpkg(gpkg_path)
    with rasterio.open(imagem_path) as src:
        transform = src.transform
        crs_img = src.crs
        W, H = src.width, src.height

    # 2) pontos ref -> pixels; gerar patches apenas onde há interesse
    px_pts = pontos_em_pixels(pontos_ref, transform, W, H, crs_img)
    if not px_pts:
        raise RuntimeError("Nenhum ponto do GPKG cai dentro da imagem fornecida.")
    _ = gerar_patches_interesse(imagem_path, patches_dir, px_pts, patch_size=256)

    # 3) inferência
    gdf_inferidos = inferir_pontos_de_patches(patches_dir, modelo_path, bands_mode="rgbnir", batch_size=16)
    inferidos_gpkg = os.path.join("main_output", "pontos_inferidos.gpkg")
    if len(gdf_inferidos):
        gdf_inferidos.to_file(inferidos_gpkg, driver="GPKG")
    print(f"✅ Pontos inferidos: {len(gdf_inferidos)} | salvo em: {inferidos_gpkg}")

    if len(gdf_inferidos) == 0:
        print("⚠️ Nenhum ponto inferido. Encerrando.")
        return

    # 4) associação Húngara e .points
    # reprojeta ambos para CRS métrico comum (para distâncias em metros)
    metric_crs = pick_metric_crs(gdf_inferidos)
    gdf_det = gdf_inferidos.to_crs(metric_crs)
    gdf_ref = pontos_ref.to_crs(metric_crs) if pontos_ref.crs != metric_crs else pontos_ref.copy()

    det_xy = _coords_xy(gdf_det).astype(np.float32)
    ref_xy = _coords_xy(gdf_ref).astype(np.float32)

    det_idx, ref_idx, dists = hungarian_match_per_tile(det_xy, ref_xy, assign_radius=20.0)

    # Para o QGIS: mapX,mapY são coordenadas do GPKG (em CRS da imagem!);
    # pixelX,pixelY são coordenadas de pixel da IMAGEM COMPLETA.
    # Portanto:
    #   - projetamos pontos de referência para CRS da imagem (mapX,mapY).
    #   - calculamos pixelX,pixelY dos DETECTADOS usando ~transform da IMAGEM (global).
    ref_qgis = gdf_ref.to_crs(crs_img)  # map coords no CRS da imagem
    pares_points = []
    with rasterio.open(imagem_path) as src:
        inv = ~src.transform
        for i_det, j_ref, _d in zip(det_idx, ref_idx, dists):
            det_geom_img_crs = gdf_inferidos.iloc[i_det].geometry  # já está no CRS da imagem
            px, py = inv * (det_geom_img_crs.x, det_geom_img_crs.y)
            pares_points.append({
                "mapX": float(ref_qgis.iloc[j_ref].geometry.x),
                "mapY": float(ref_qgis.iloc[j_ref].geometry.y),
                "pixelX": float(px),
                "pixelY": float(py),
            })

    points_path = escrever_points_qgis(pares_points, "main_output", "georeferencer.points")
    print(f"📍 Arquivo .points: {points_path} | pares: {len(pares_points)}")

    # arquivo de auditoria (opcional): pares_homologos.geojson com pontos detectados
    try:
        pares_geo = []
        for i_det, j_ref, d in zip(det_idx, ref_idx, dists):
            pares_geo.append({
                "geometry": gdf_inferidos.iloc[i_det].geometry,
                "id_det": int(i_det),
                "id_ref": int(j_ref),
                "dist_m": float(d)
            })
        if pares_geo:
            gpd.GeoDataFrame(pares_geo, geometry="geometry", crs=gdf_inferidos.crs).to_file(
                os.path.join("main_output", "pares_homologos.geojson"), driver="GeoJSON"
            )
    except Exception:
        pass

if __name__ == "__main__":
    main()
