# Sample de datos

El dataset completo FIgLib/HPWREN no se incluye en el repositorio. Esta carpeta contiene únicamente ejemplos pequeños para comprobar el formato de entrada de los scripts.

`example_probabilities.csv` representa una secuencia ficticia de probabilidades visuales. Sus valores **no pertenecen al benchmark ni deben utilizarse para calcular métricas**.

Columnas:

- `timestamp`: instante de la observación, en segundos dentro del ejemplo;
- `p_smoke`: probabilidad visual de humo producida por Full384;
- `feat_dt`: versión ya normalizada del intervalo temporal utilizada como tercera característica de la GRU.

Para reproducir los resultados del TFM deben utilizarse las secuencias reales FIgLib/HPWREN y el mismo preprocesado temporal descrito en el Anexo D.
