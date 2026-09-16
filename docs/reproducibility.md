# Reproducibilidad del bloque FIgLib/HPWREN

## 1. Alcance

Este repositorio documenta la parte del TFM dedicada a clasificación visual de humo y detección temporal temprana sobre FIgLib/HPWREN. Los resultados completos y el desarrollo experimental se describen en el Anexo D de la memoria.

## 2. Dataset

El dataset completo no se incluye en GitHub. Debe obtenerse desde la fuente original de FIgLib/HPWREN y organizarse por secuencias/eventos. En los experimentos del TFM se trabajó con 519 secuencias; tras depuración se utilizó un conjunto de aproximadamente 41 187 imágenes físicas, de las cuales 23 480 entraron en el entrenamiento supervisado visual bajo las reglas temporales descritas en la memoria.

La partición se realiza a nivel de evento, no a nivel de frame. El objetivo es impedir que imágenes casi idénticas de una misma secuencia aparezcan simultáneamente en entrenamiento y evaluación.

## 3. Etiquetado visual

El instante `t=0` es una referencia del evento y no debe interpretarse como el instante exacto en el que el humo pasa a ser visible.

Para el clasificador visual:

- `t <= -1200 s`: negativo;
- `t >= +900 s`: positivo;
- `-1200 < t < +900`: excluido del entrenamiento visual supervisado.

La franja intermedia sí puede utilizarse posteriormente para evaluar comportamiento temporal.

## 4. Full224 y Full384

La comparación controlada conserva el mismo split por evento y una metodología de entrenamiento equivalente.

Preprocesado de evaluación Full224:

```text
Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet)
```

Preprocesado de evaluación Full384:

```text
Resize(440) -> CenterCrop(384) -> ToTensor() -> Normalize(ImageNet)
```

Configuración principal de Full384 documentada en el experimento:

- backbone: ResNet18;
- optimizador: AdamW;
- learning rate: `3e-4`;
- weight decay: `1e-4`;
- scheduler: cosine;
- batch físico: 16;
- acumulación de gradiente: 4;
- batch efectivo: 64;
- AMP activado.

Resultados Full384 en test a umbral 0.5:

```text
Accuracy          0.78756
Balanced Accuracy 0.79894
Precision         0.90195
Recall            0.69057
F1                0.78223
AUC               0.83860
TN                1527
FP                156
FN                643
TP                1435
FPR               0.09269
```

## 5. Bootstrap pareado por evento

Para comparar Full224 y Full384 se utilizó bootstrap pareado a nivel de evento con 10 000 réplicas. Cada réplica remuestrea con reemplazo los eventos de test y calcula las métricas de ambos modelos sobre exactamente la misma muestra de eventos. Esto preserva el emparejamiento y evita tratar miles de frames correlacionados como observaciones independientes.

Mejoras robustas observadas para Full384 frente a Full224:

- Balanced Accuracy: `+0.053906`, IC 95 % `[+0.011823, +0.094373]`;
- Precision: `+0.07995`, IC 95 % `[+0.029951, +0.132518]`;
- FPR: `-0.086156`, IC 95 % `[-0.149079, -0.029908]`.

## 6. Multiescala

La estrategia multiescala combinó una representación global con regiones locales. Mejoró el recall de `0.6906` a `0.7594`, pero aumentó la FPR de `0.0927` a `0.1771`. Por ese compromiso, Full384 se mantuvo como baseline visual principal.

## 7. Regla temporal heurística

Sobre las probabilidades Full384 se evaluaron reglas causales. La referencia seleccionada fue una media móvil de cinco observaciones. En el conjunto temporal de prueba:

- detección post-`t=0`: 0.88;
- detección antes de 900 s: 0.72;
- clear-pre alert: 0.18;
- mediana hasta primera alerta: 361 s;
- eventos no detectados: 6.

## 8. GRU temporal

La GRU utiliza ventanas causales de longitud 8. Para cada observación se construyen tres características:

```text
feat_p  = p_t
feat_dp = p_t - p_(t-1)
feat_dt = intervalo temporal normalizado entre observaciones
```

El checkpoint almacenado conserva `seq_len=8` y las tres características anteriores. Arquitectura:

- GRU: input 3, hidden 32, una capa;
- cabeza: `32 -> 16 -> 1`;
- ReLU;
- dropout 0.2.

Entrenamiento documentado:

- BCEWithLogitsLoss con `pos_weight`;
- AdamW, `lr=1e-3`, `weight_decay=1e-4`;
- máximo 50 épocas;
- early stopping con paciencia 8;
- mejor checkpoint: época 49;
- AUC de validation del checkpoint: 0.862844;
- umbral operativo seleccionado en validation: 0.85.

Resultados en test:

```text
Post t=0            0.96
Antes de 300 s      0.56
Antes de 600 s      0.72
Antes de 900 s      0.90
Antes de 1800 s     0.94
Clear-pre alert     0.28
Mediana             210.5 s
Media               394.1 s
Eventos no detect.  2
```

**Nota sobre `feat_dt`:** el checkpoint original guarda el nombre de la característica pero no serializa el factor exacto utilizado para normalizar el intervalo temporal. Para una reproducción bit a bit del experimento temporal debe utilizarse el mismo preprocesado del notebook de entrenamiento original. El código incluido aquí permite cargar el checkpoint y ejecutar la arquitectura, pero no inventa un factor de normalización que no esté almacenado en el propio modelo.

## 9. Modelos

- Full384: se conserva en Drive por tamaño (~44.8 MB).
- GRU: se incluye en GitHub porque su checkpoint es pequeño (~20 KB).

## 10. Limitaciones de reproducibilidad

FIgLib es un benchmark centrado en eventos, no una grabación 24/7 de vigilancia continua. Por tanto, una FPR por frame no equivale directamente a falsas alarmas por unidad de tiempo en un despliegue real. Además, cualquier evaluación sobre cámaras, paisajes o climatologías distintas debe tratarse como validación fuera de dominio.
