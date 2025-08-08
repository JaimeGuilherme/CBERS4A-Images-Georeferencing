import os
import torch
import numpy as np
import rasterio
import geopandas as gpd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from skimage import measure
from shapely.geometry import Point

# Define patch size
PATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset for sliding window inference
class SlidingWindowDataset(Dataset):
    def __init__(self, image_array, step=PATCH_SIZE):
        self.image = image_array
        self.step = step
        self.patches = []
        self.coords = []
        h, w, _ = image_array.shape
        for y in range(0, h - PATCH_SIZE + 1, step):
            for x in range(0, w - PATCH_SIZE + 1, step):
                patch = image_array[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                self.patches.append(patch)
                self.coords.append((x, y))

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        x, y = self.coords[idx]
        return self.transform(patch), (x, y)

def predict_intersections(image_path, model_path, output_geojson):
    # Load image
    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3])  # RGB
        image = np.moveaxis(image, 0, -1)
        transform = src.transform
        crs = src.crs

    # Normalize image
    image = image.astype(np.uint8)

    # Load model
    model = torch.load(model_path, map_location=DEVICE)
    model.eval()

    # Dataset and loader
    dataset = SlidingWindowDataset(image)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Collect predicted intersection points
    detected_points = []

    for inputs, (x_offset, y_offset) in loader:
        inputs = inputs.to(DEVICE)
        with torch.no_grad():
            output = model(inputs)
            mask_pred = (output > 0.5).float().cpu().numpy()[0, 0]

        # Detect connected components (blobs)
        labeled = measure.label(mask_pred)
        props = measure.regionprops(labeled)

        for prop in props:
            # Get centroid in patch
            cy, cx = prop.centroid
            # Convert to image coordinates
            abs_x = x_offset.item() + int(cx)
            abs_y = y_offset.item() + int(cy)

            # Convert to geographic coordinates
            lon, lat = rasterio.transform.xy(transform, abs_y, abs_x)
            detected_points.append(Point(lon, lat))

    # Save as GeoJSON
    gdf = gpd.GeoDataFrame(geometry=detected_points, crs=crs)
    gdf.to_file(output_geojson, driver="GeoJSON")
    print(f"[✔] {len(detected_points)} pontos detectados salvos em: {output_geojson}")

if __name__ == "__main__":
    # Exemplo de uso
    image_path = "05_plugin_georref/imagem_nova.tif"
    model_path = "02_train_model/model_final.pth"
    output_geojson = "05_plugin_georref/pontos_detectados.geojson"

    predict_intersections(image_path, model_path, output_geojson)
