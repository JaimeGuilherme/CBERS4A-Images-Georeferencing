# Projeto: Detecção e Associação de Interseções de Rodovias para Georreferenciamento

Este projeto automatiza a detecção de pontos de interseção de rodovias em imagens de satélite usando Deep Learning (U-Net) e gera pares de pontos homólogos com vetores OSM para uso no georreferenciador do QGIS.

---

## Requisitos

- Python 3.8 ou superior
- Instale as dependências com:

```bash
pip install -r requirements.txt
```

### Conteúdo do requirements.txt:

```
torch>=1.13.0
torchvision>=0.14.0
numpy
tqdm
rasterio
geopandas
scikit-learn
shapely
scipy
Pillow
```

---

## Organização dos Arquivos

```
├── 01_preparar_dataset.py              # Gera patches de imagem e máscaras bufferizadas dos pontos
├── 02_treinar_unet.py                 # Treina a U-Net com os patches gerados
├── 03_inferir_pontos.py               # Inferência da U-Net para detectar pontos em patches de teste
├── 04_associar_pontos.py              # Associação dos pontos detectados com pontos OSM (pares homólogos)
├── 05_plugin_georref.py               # Processo final de georreferenciamento usando imagem nova e pares
├── dataset.py                         # Dataset personalizado para carregar patches (usado no treinamento/inferência)
├── requirements.txt                   # Lista de dependências Python do projeto
├── README.md                          # Documentação, instruções e organização do projeto
├── input/
│   ├── imagem_georreferenciada.tif    # Imagem correta e georreferenciada para gerar patches
│   ├── imagem_nova.tif                # Imagem nova (não georreferenciada) para aplicar georreferenciamento
│   ├── intersecoes_osm.gpkg           # Vetor OSM com pontos de interseção para a imagem georreferenciada
│   ├── intersecoes_osm_nova.gpkg      # Vetor OSM para a nova imagem (usado no passo final)
├── dataset_patches/
│   ├── images/                        # Patches de imagens gerados pelo script 01
│   ├── masks/                         # Máscaras dos patches (buffer dos pontos)
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── checkpoints/
│   └── *.pth                          # Modelos treinados U-Net salvos aqui
├── output/
│   ├── pontos_detectados.geojson     # Pontos detectados pela IA após inferência
│   └── pares_homologos.geojson       # Pares de pontos homólogos para georreferenciamento QGIS
```

---

## Passo a passo para executar

### 1. Preparar Dataset (gera patches e máscaras bufferizadas)

```bash
python 01_preparar_dataset.py
```

**Entrada:**

- `input/imagem_georreferenciada.tif`
- `input/intersecoes_osm.gpkg`

**Saída:**

- Patches e máscaras em `dataset_patches/`
- Dataset separado em treino, validação e teste em `dataset_patches/train/`, `val/` e `test/`

---

### 2. Treinar U-Net (treina o modelo e salva checkpoints)

```bash
python 02_treinar_unet.py
```

**Entrada:**

- Patches de treino e validação em `dataset_patches/train/` e `dataset_patches/val/`

**Saída:**

- Modelos treinados salvos em `checkpoints/`

---

### 3. Inferir Pontos no Dataset de Teste

```bash
python 03_inferir_pontos.py
```

**Entrada:**

- Patches de teste em `dataset_patches/test/images/`
- Modelos treinados em `checkpoints/`

**Saída:**

- Pontos detectados salvos em `output/pontos_detectados.geojson`

---

### 4. Associar Pontos Detectados com OSM (gera pares homólogos)

```bash
python 04_associar_pontos.py
```

**Entrada:**

- Pontos detectados em `output/pontos_detectados.geojson`
- Pontos OSM em `input/intersecoes_osm.gpkg`

**Saída:**

- Pares homólogos salvos em `output/pares_homologos.geojson`

---

### 5. Produto Final: Georreferenciamento para Imagem Nova

```bash
python 05_plugin_georref.py
```

**Entrada:**

- `input/imagem_nova.tif`
- `input/intersecoes_osm_nova.gpkg`
- Modelos treinados em `checkpoints/`

**Saída:**

- Pares homólogos para georreferenciamento em `output/pares_homologos.geojson`

---

## Notas importantes

- Certifique-se que todas as imagens e vetores estejam na mesma projeção CRS.
- Buffers nas máscaras são criados apenas no passo de preparação dos patches.
- A sobreposição (overlap) dos patches é configurada no script final (05) para melhorar a detecção.
- A distância máxima para associação dos pontos é configurável no script 05 (default 20 unidades CRS).
