# TFM-FIgLib

Repositorio asociado al bloque **FIgLib/HPWREN** del Trabajo Fin de Máster *Sistema multimodal para la detección temprana de incendios forestales mediante inteligencia artificial*.

Este repositorio recoge el material necesario para revisar y reproducir la parte de clasificación visual y detección temporal desarrollada sobre FIgLib/HPWREN. El dataset completo no se redistribuye por tamaño y derechos de uso; se incluye un pequeño ejemplo de estructura de datos y las instrucciones necesarias para organizar el resto.

## Estructura

```text
TFM-FIgLib/
├── README.md
├── requirements.txt
├── notebooks/
│   └── comparacion_figlib_224_vs_384.ipynb
├── src/
│   ├── visual_model.py
│   ├── temporal_gru.py
│   └── preprocessing.py
├── scripts/
│   ├── predict_visual.py
│   └── predict_temporal.py
├── training/
│   ├── train_visual_reference.py
│   └── train_temporal_reference.py
├── models/
│   ├── best_temporal_gru_full384_38.pt
│   └── README.md
├── results/
│   └── key_metrics.json
├── sample_data/
│   ├── example_probabilities.csv
│   └── README.md
└── docs/
    └── reproducibility.md
```

## Datos y particionado

FIgLib/HPWREN está organizado en secuencias asociadas a eventos. La separación entre entrenamiento, validación y test se realizó **a nivel de evento**, evitando que imágenes temporalmente próximas del mismo incendio apareciesen en subconjuntos distintos.

Para el clasificador visual se utilizó un etiquetado conservador:

- negativo: `t <= -1200 s`;
- positivo: `t >= +900 s`;
- región intermedia: excluida del entrenamiento visual supervisado.

Para el modelo temporal, la etiqueta positiva se definió desde `t >= 0 s`, manteniendo `t <= -1200 s` como región negativa y excluyendo la franja intermedia de la pérdida supervisada.

## Modelo visual Full384

Arquitectura: **ResNet18** con entrada 384x384.

Preprocesado de evaluación:

```text
Resize(440) -> CenterCrop(384) -> ToTensor() -> ImageNet normalization
```

Resultados principales sobre test:

| Métrica | Full384 |
|---|---:|
| Balanced Accuracy | 0.7989 |
| Precision | 0.9019 |
| Recall | 0.6906 |
| AUC | 0.8386 |
| FPR | 0.0927 |

El checkpoint visual final (`best_resnet18_full_511_384_controlled.pt`) ocupa aproximadamente 44.8 MB y se mantiene en Google Drive para no duplicar un binario grande en GitHub.

**Checkpoint Full384:** https://drive.google.com/file/d/1_8itlXVmfy9Htq8BqSTipIAhSl4rKMRI/view

## Modelo temporal GRU

La GRU recibe por observación tres variables:

```text
p_t, delta_p_t, delta_t
```

Se utilizan ventanas causales de ocho observaciones. La configuración del checkpoint final es:

- `input_size = 3`;
- `hidden_size = 32`;
- una capa GRU;
- cabeza `32 -> 16 -> 1`;
- ReLU + Dropout(0.2);
- umbral operativo seleccionado en validation: `0.85`.

Resultados principales:

| Métrica | Rolling mean 5 | GRU |
|---|---:|---:|
| Detección post-t=0 | 0.88 | 0.96 |
| Antes de 300 s | 0.36 | 0.56 |
| Antes de 600 s | 0.58 | 0.72 |
| Antes de 900 s | 0.72 | 0.90 |
| Antes de 1800 s | 0.88 | 0.94 |
| Clear-pre alert | 0.18 | 0.28 |
| Mediana hasta primera alerta | 361 s | 210.5 s |
| Eventos no detectados | 6 | 2 |

El checkpoint temporal se incluye directamente en `models/best_temporal_gru_full384_38.pt`.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Uso básico

### Inferencia visual

```bash
python scripts/predict_visual.py \
  --image ruta/a/imagen.jpg \
  --checkpoint models/best_resnet18_full_511_384_controlled.pt
```

El checkpoint visual debe estar disponible localmente en la ruta indicada antes de ejecutar la inferencia.

### Inferencia temporal

```bash
python scripts/predict_temporal.py \
  --csv sample_data/example_probabilities.csv \
  --checkpoint models/best_temporal_gru_full384_38.pt
```

El CSV de ejemplo contiene `p_smoke` y `feat_dt` ya normalizado. Si se parte únicamente de timestamps, debe utilizarse el mismo factor de normalización temporal del experimento original; el checkpoint no serializa ese valor.

## Código de entrenamiento

La carpeta `training/` contiene implementaciones de referencia construidas a partir de la configuración experimental documentada en la memoria. Sirven para reproducir la arquitectura y los hiperparámetros principales. Los experimentos históricos se realizaron originalmente en notebooks de Google Colab y no todos los notebooks intermedios se conservan como artefactos autocontenidos.

## Reproducibilidad

Los experimentos completos se ejecutaron sobre Google Colab y Google Drive. Este repositorio conserva el código, el checkpoint temporal, la referencia al checkpoint visual, un notebook de comparación de resolución, métricas exportadas y un pequeño ejemplo de formato. No se redistribuyen las decenas de miles de imágenes del dataset ni resultados gráficos pesados.

Los detalles metodológicos y las limitaciones de reproducción se encuentran en `docs/reproducibility.md` y en el Anexo D de la memoria.

## Referencias

- Dewangan et al. (2022), *FIgLib & SmokeyNet: Dataset and Deep Learning Model for Real-Time Wildland Fire Smoke Detection*, Remote Sensing, 14(4), 1007. https://doi.org/10.3390/rs14041007
- Cho et al. (2014), *Learning Phrase Representations using RNN Encoder--Decoder for Statistical Machine Translation*. https://doi.org/10.3115/v1/D14-1179

## Autor

Antón Soto — Máster en Big Data, Data Science e Inteligencia Artificial, Universidad Complutense de Madrid.
