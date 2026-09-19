# Modelos

## GRU temporal

El checkpoint pequeño `best_temporal_gru_full384_38.pt` se incluye directamente en el repositorio.

Metadatos almacenados en el checkpoint:

- época: 49;
- validation AUC: 0.8628440584761514;
- `seq_len`: 8;
- características: `feat_p`, `feat_dp`, `feat_dt`.

## Clasificador visual Full384

El checkpoint visual final `best_resnet18_full_511_384_controlled.pt` ocupa aproximadamente 44.8 MB. Para evitar duplicarlo como binario grande en GitHub, se mantiene en Google Drive:

https://drive.google.com/file/d/1_8itlXVmfy9Htq8BqSTipIAhSl4rKMRI/view

La ruta esperada para la inferencia visual es:

```text
models/best_resnet18_full_511_384_controlled.pt
```

Esta ubicación es la utilizada por los ejemplos del repositorio.
