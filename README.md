# Road Extraction Project with Deep Learning

This repository contains the development of two approaches for georeferencing CBERS-4A satellite imagery using neural networks and OpenStreetMap (OSM) data. The approaches are organized in separate directories.

## Repository Structure

```
.
├── Road Extraction/           # Second approach (full road extraction)
│   ├── generate_mask.py
│   ├── generate_patches.py
│   ├── split_dataset.py
│   ├── train_unet.py
│   ├── infer_and_export.py
│   ├── unet_model.pth
│   └── README.md  
├── Road Intersections/       # First approach (road intersections)
│   └── ...
└── README.md                 # This File
```

## 🛣️ Second Approach: `Road Extraction/`

This approach uses a U-Net convolutional neural network to segment **entire road networks** from CBERS-4A satellite imagery. The predicted result is then converted into vector format and will be used later to perform automatic georeferencing of unaligned imagery.

### 🔧 Requirements

- Python ≥ 3.9
- Packages:
  - torch
  - numpy
  - rasterio
  - geopandas
  - scikit-learn
  - tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

### 📌 Execution Order

Run the following scripts in order:

0. **Enter the Road Extraction folder:**
```bash
cd road_extraction
```

1. **Generate raster mask from OSM road vectors:**
```bash
python generate_mask.py
```

2. **Generate 256x256 image/mask patches:**
```bash
python generate_patches.py
```

3. **Split patches into training/validation sets (80/20):**
```bash
python split_dataset.py
```

4. **Train U-Net model using patches:**
```bash
python train_unet.py
```

5. **Run inference in blocks and export vector result:**
```bash
python infer_and_export.py
```

### 📤 Outputs

- .gpkg vector file containing predicted roads.
- Can later be used to extract homologous points and assist with automatic image georeferencing.

## 📦 First Approach: `Road Intersections/`

This approach focuses on detecting road intersections using deep learning techniques. Its scripts and documentation are organized in a parallel folder.

## 📚 Credits

This project is part of an undergraduate research work and uses public data from [INPE](http://www.inpe.br/) and [OpenStreetMap](https://www.openstreetmap.org/).

## 📄 Licença

This project is licensed under the [MIT License](LICENSE).