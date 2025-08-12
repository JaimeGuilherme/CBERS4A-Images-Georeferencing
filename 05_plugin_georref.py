import os, glob, torch, numpy as np, geopandas as gpd, rasterio
from rasterio.windows import Window
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from unet import UNet
from utils import load_checkpoint
from scipy.ndimage import label

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 256; OVERLAP = 0.3; THRESHOLD = 0.5; MAX_ASSOC_DISTANCE = 20; CHECKPOINTS_DIR = 'checkpoints'

def transform_image_patch(patch_np):
    if patch_np.ndim == 3 and patch_np.shape[0] != 3:
        patch_np = patch_np.transpose(2,0,1)
    patch_np = patch_np.astype(np.float32) / 255.0
    patch_tensor = torch.from_numpy(patch_np)
    patch_tensor = patch_tensor.unsqueeze(0)
    return patch_tensor

def carregar_modelos(checkpoints_dir):
    paths = sorted(glob.glob(os.path.join(checkpoints_dir, '*.pth')))
    modelos = []
    for path in paths:
        model = UNet(in_channels=3, out_channels=1).to(DEVICE)
        load_checkpoint(path, model)
        model.eval(); modelos.append(model)
    return modelos

def dividir_em_patches_do_raster(imagem_path, patch_size=PATCH_SIZE, overlap=OVERLAP):
    patches, coords, transforms = [], [], []
    with rasterio.open(imagem_path) as src:
        img_width = src.width; img_height = src.height
        step = int(patch_size * (1-overlap))
        for y in range(0, img_height - patch_size + 1, step):
            for x in range(0, img_width - patch_size + 1, step):
                window = Window(x, y, patch_size, patch_size)
                patch = src.read([1,2,3], window=window)
                patches.append(patch); coords.append((x,y)); transforms.append(src.window_transform(window))
        crs = src.crs
    return patches, coords, transforms, crs

def detectar_pontos(pred_map, threshold=THRESHOLD):
    binary_map = pred_map > threshold
    labeled, n_features = label(binary_map)
    pontos = []
    for i in range(1, n_features+1):
        ys, xs = np.where(labeled==i)
        x_centro = int(np.mean(xs)); y_centro = int(np.mean(ys))
        pontos.append((x_centro, y_centro))
    return pontos

def main(imagem_path, pontos_osm_path, output_pares_path):
    modelos = carregar_modelos(CHECKPOINTS_DIR)
    patches, coords, transforms, crs = dividir_em_patches_do_raster(imagem_path)
    todos_pontos_detectados = []
    for patch, origin, tr in tqdm(zip(patches, coords, transforms), total=len(patches)):
        patch_tensor = transform_image_patch(patch)
        with torch.no_grad():
            preds = []
            for m in modelos:
                out = torch.sigmoid(m(patch_tensor.to(DEVICE)))
                preds.append(out.cpu().numpy())
            mean_pred = np.mean(preds, axis=0)[0,0]
        pontos_patch = detectar_pontos(mean_pred)
        for xpix, ypix in pontos_patch:
            lon, lat = rasterio.transform.xy(tr, ypix, xpix)
            todos_pontos_detectados.append(Point(lon, lat))
    gdf_detect = gpd.GeoDataFrame(geometry=todos_pontos_detectados, crs=crs)
    gdf_osm = gpd.read_file(pontos_osm_path)
    if gdf_detect.crs != gdf_osm.crs:
        gdf_detect = gdf_detect.to_crs(gdf_osm.crs)
    coords_detect = np.array([(p.x, p.y) for p in gdf_detect.geometry])
    coords_osm = np.array([(p.x, p.y) for p in gdf_osm.geometry])
    nbrs = NearestNeighbors(n_neighbors=1).fit(coords_osm)
    distances, indices = nbrs.kneighbors(coords_detect)
    pares = []
    for i,(d,idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if d <= MAX_ASSOC_DISTANCE:
            pares.append({'id_detectado': int(i),'id_osm': int(gdf_osm.index[idx]),'distancia': float(d),'geometry_detectado': gdf_detect.geometry.iloc[i],'geometry_osm': gdf_osm.geometry.iloc[idx]})
    gdf_pares = gpd.GeoDataFrame(pares, geometry='geometry_detectado', crs=gdf_osm.crs)
    os.makedirs(os.path.dirname(output_pares_path), exist_ok=True)
    gdf_pares.to_file(output_pares_path, driver='GeoJSON')
    print('Pares salvos em', output_pares_path, 'total:', len(pares))

if __name__ == '__main__':
    imagem_nova = 'input/imagem_nova.tif'; pontos_osm = 'input/intersecoes_osm_nova.gpkg'; saida = 'output/pares_homologos.geojson'
    main(imagem_nova, pontos_osm, saida)
