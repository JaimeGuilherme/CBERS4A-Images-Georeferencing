# Detecção de Interseções de Rodovias com U-Net

Este projeto realiza a detecção de interseções de rodovias a partir de imagens de satélite, utilizando uma arquitetura de rede neural U-Net para segmentação e posterior extração de pontos vetorizados.

---

## **Organização dos Arquivos**

```
teste/
│
├── 01_preparar_dataset.py        # Script para preparar o dataset
├── 02_treinar_unet.py            # Script de treinamento da U-Net
├── 03_inferir_pontos.py          # Script para inferência e geração de pontos
├── 04_associar_pontos.py         # Script para associar pontos detectados com OSM
├── 05_plugin_georref.py          # Plugin QGIS para georreferenciamento
├── unet.py                       # Arquitetura da U-Net
├── dataset.py                    # Classe de carregamento do dataset
├── utils.py                      # Funções utilitárias
├── losses.py                     # Funções de perda (BCE, Focal Loss, etc.)
├── metrics.py                    # Funções para métricas (IoU, precisão, recall)
├── requirements.txt              # Dependências do projeto
├── README.md                     # Este arquivo
│
├── checkpoints/                  # Pesos salvos durante o treinamento
│
├── dataset/
│   ├── train/
│   │   ├── images/               # Imagens de treino
│   │   ├── masks/                # Máscaras de treino
│   ├── val/
│   │   ├── images/               # Imagens de validação
│   │   ├── masks/                # Máscaras de validação
│   ├── test/
│       ├── images/               # Imagens de teste
│       ├── masks/                # Máscaras de teste
│
├── dataset_patches/
│   ├── images/                   # Pedaços (patches) das imagens
│   ├── masks/                    # Pedaços (patches) das máscaras
│
├── input/                        # Pasta para arquivos de entrada
├── output/                       # Pasta para arquivos de saída
│
└── resultados/
    ├── mascaras_patches/         # Máscaras resultantes por patch
```

---

## **Fluxo de Uso**

### 1️⃣ Preparar o dataset
```bash
python 01_preparar_dataset.py
```
**Entradas:**
- Imagens originais de satélite (GeoTIFF ou similar)
- Máscaras originais das interseções (mesmo alinhamento espacial)

**Saídas:**
- Patches de imagens e máscaras nas pastas:
  ```
  dataset/train/images/
  dataset/train/masks/
  dataset/val/images/
  dataset/val/masks/
  dataset/test/images/
  dataset/test/masks/
  ```

---

### 2️⃣ Treinar a U-Net
```bash
python 02_treinar_unet.py
```
**Entradas:**
- Patches de imagens e máscaras da pasta `dataset/train/` e `dataset/val/`

**Saídas:**
- Modelos treinados salvos em `checkpoints/`:
  - `checkpoint_epoch_X.pth`
  - `best_model.pth`
- Métricas (IoU, precisão, recall) exibidas no console
- (Opcional) logs no TensorBoard

---

### 3️⃣ Inferir pontos
```bash
python 03_inferir_pontos.py
```
**Entradas:**
- Modelos salvos na pasta `checkpoints/`
- Patches de teste em `dataset/test/images` e `dataset/test/masks`

**Saídas:**
- `resultados/pontos_detectados.geojson` (pontos detectados georreferenciados)
- `resultados/mascaras_patches/` (máscaras binárias georreferenciadas por patch)

---

### 4️⃣ Associar pontos com OSM
```bash
python 04_associar_pontos.py
```
**Entradas:**
- `resultados/pontos_detectados.geojson`
- Arquivo vetorial do OpenStreetMap com interseções (GeoJSON, Shapefile, etc.)

**Saídas:**
- GeoJSON com pontos detectados e campo indicando a associação com ponto OSM

---

### 5️⃣ Georreferenciar no QGIS
- Utilizar `05_plugin_georref.py` como plugin no QGIS para:
  - Carregar camadas de pontos detectados
  - Visualizar sobreposição com vetores OSM
  - Ajustar georreferenciamento manualmente se necessário

---

## **Requisitos**

Instalar as dependências:
```bash
pip install -r requirements.txt
```

Principais bibliotecas:
- `torch`
- `tqdm`
- `numpy`
- `rasterio`
- `geopandas`
- `shapely`
- `scipy`

---

## **Saídas Principais**
- `resultados/pontos_detectados.geojson` → Pontos vetorizados das interseções
- `resultados/mascaras_patches/` → Máscaras segmentadas por patch

---

## **Autor**
Projeto desenvolvido para automação da detecção e vetorização de interseções rodoviárias a partir de imagens de satélite.