# Road Extraction — U-Net Based Pipeline

This directory implements a complete pipeline for extracting road networks from CBERS-4A satellite imagery using a U-Net convolutional neural network. The extracted roads are then exported as vector data for future georeferencing applications.

## 🔁 Overview of Pipeline

1. **Generate binary road masks** from OSM vector data.
2. **Create image/mask patches** (256x256).
3. **Split dataset** into training and validation sets.
4. **Train U-Net** with patches.
5. **Run inference on full satellite image** and export to `.gpkg` vector file.

---

## 📜 Script Descriptions

### 1. `generate_mask.py`

- **Purpose:** Rasterizes the OSM vector road data to create a binary mask aligned with the CBERS image.

- **Inputs:**
  - `input_vector_path`: Path to the OSM road vector file (GeoPackage or Shapefile).
  - `reference_raster_path`: Path to the georeferenced CBERS raster image.

- **Outputs:**
  - `output_mask_path`: Path to the output binary raster mask (GeoTIFF).

- **Dependencies:** `geopandas`, `rasterio`, `shapely`

---

### 2. `generate_patches.py`

- **Purpose:** Cuts both the CBERS image and the corresponding binary mask into aligned patches of size 256x256.

- **Inputs:**
  - `image_path`: Path to the full CBERS image.
  - `mask_path`: Path to the full binary mask image.

- **Outputs:**
  - `patches/images/`: Directory with image patches.
  - `patches/masks/`: Directory with mask patches.

- **Dependencies:** `numpy`, `rasterio`, `tqdm`

---

### 3. `split_dataset.py`

- **Purpose:** Splits the patches into training and validation sets (default 80/20).

- **Inputs:**
  - `patches/images/`
  - `patches/masks/`

- **Outputs:**
  - `dataset/train/images/`, `dataset/train/masks/`
  - `dataset/val/images/`, `dataset/val/masks/`

- **Dependencies:** `os`, `shutil`, `random`

---

### 4. `train_unet.py`

- **Purpose:** Trains a U-Net on the training set for road segmentation.

- **Inputs:**
  - `dataset/train/images/`, `dataset/train/masks/`
  - `dataset/val/images/`, `dataset/val/masks/`

- **Outputs:**
  - `model.pth`: Trained model file.

- **Dependencies:** `torch`, `torchvision`, `numpy`, `PIL`, `tqdm`

---

### 5. `infer_and_export.py`

- **Purpose:** Applies the trained U-Net model to the full CBERS image using sliding-window inference and exports the binary result as a georeferenced vector layer.

- **Inputs:**
  - `reference_raster_path`: Original CBERS image.
  - `model_path`: Trained `.pth` file.

- **Outputs:**
  - `roads_predicted.gpkg`: GeoPackage file with vectorized road polygons.

- **Dependencies:** `torch`, `rasterio`, `geopandas`, `numpy`, `shapely`, `tqdm`

---

## ✅ Requirements

Install all required dependencies with:

```bash
pip install -r ../requirements.txt
