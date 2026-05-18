# Audio Classification — Fine-tuning de modelos de audio con HuggingFace Transformers

Repositorio con el código de fine-tuning y evaluación de modelos de clasificación de audio basados en arquitecturas Transformer. El objetivo es comparar el rendimiento de tres arquitecturas preentrenadas — **DistilHuBERT**, **HuBERT** y **Wav2Vec2** — sobre cinco datasets acústicos de distinta naturaleza.

---

## Descripción general

El proyecto consta de dos scripts principales:

### `train.py` — Entrenamiento y fine-tuning

Realiza el fine-tuning de un modelo preentrenado de audio sobre un dataset de clasificación. Sus funcionalidades principales son:

- Carga y preprocesa el dataset seleccionado (local o descargado desde Kaggle).
- Aplica aumentación de datos en train: ruido gaussiano y variación de ganancia aleatoria.
- Normaliza y trunca las señales de audio a una duración máxima configurable por dataset.
- Realiza el fine-tuning del modelo con `Trainer` de HuggingFace, con parada temprana (`EarlyStoppingCallback`) basada en accuracy.
- Registra curvas de aprendizaje (accuracy y loss) por época, tanto en formato JSON como en PNG.
- Guarda el modelo y el feature extractor en disco al finalizar el entrenamiento.
- Evalúa el modelo sobre el split de test con un `pipeline` de HuggingFace y genera métricas completas: accuracy global, accuracy por clase y matriz de confusión.

### `diagnosis.py` — Diagnóstico de un modelo ya entrenado

Carga un modelo guardado en disco y ejecuta una evaluación completa sobre el split de test del dataset indicado. Genera un informe en texto plano con accuracy global, accuracy por clase y matriz de confusión, guardado en la carpeta `./resultados/`.

### `results.py` + `table.py` — Análisis de resultados agregados

`results.py` almacena los resultados de accuracy de los cinco experimentos por modelo y dataset. `table.py` los consume, calcula media ± desviación típica y exporta una tabla resumen a `results_table.csv`.

---

## Modelos

Se evalúan tres arquitecturas preentrenadas de audio:

| Modelo | ID en HuggingFace Hub |
|---|---|
| DistilHuBERT | `ntu-spml/distilhubert` |
| HuBERT Base | `facebook/hubert-base-ls960` |
| Wav2Vec2 Base | `facebook/wav2vec2-base-960h` |

Los modelos fine-tuneados resultantes están disponibles públicamente en HuggingFace en el perfil **[nataliaaolmo](https://huggingface.co/nataliaaolmo)**.

---

## Datasets

El proyecto soporta cinco datasets. Cuatro de ellos se descargan automáticamente desde Kaggle mediante `kagglehub`; el dataset local debe aportarse manualmente.

| Nombre clave | Descripción | Fuente |
|---|---|---|
| `local` | Dataset de géneros musicales en MP3 | **No incluido en este repositorio** — debe colocarse manualmente en `./dataset/` organizado en subcarpetas por clase |
| `urbansound8k` | 8732 clips de sonidos urbanos en 10 clases | [chrisfilo/urbansound8k](https://www.kaggle.com/datasets/chrisfilo/urbansound8k) |
| `gender_voice` | Clips de voz etiquetados por género | [murtadhanajim/gender-recognition-by-voiceoriginal](https://www.kaggle.com/datasets/murtadhanajim/gender-recognition-by-voiceoriginal) |
| `instruments` | Sonidos de instrumentos musicales | [abdulvahap/music-instrunment-sounds-for-classification](https://www.kaggle.com/datasets/abdulvahap/music-instrunment-sounds-for-classification) |
| `cats_dogs` | Clasificación de audio de gatos y perros | [stealthtechnologies/cats-vs-dogs-audio-classification](https://www.kaggle.com/datasets/stealthtechnologies/cats-vs-dogs-audio-classification) |

> **Nota sobre el dataset local:** este dataset no está subido al repositorio por razones de tamaño y licencia. Para usarlo, crea la carpeta `./dataset/` con una subcarpeta por cada clase, conteniendo los archivos `.mp3` correspondientes.

> **Nota sobre Kaggle:** para que `kagglehub` pueda descargar datasets, necesitas tener configuradas tus credenciales de Kaggle (`~/.kaggle/kaggle.json`). Consulta la [documentación oficial](https://www.kaggle.com/docs/api).

---

## Estructura del repositorio

```
.
├── train.py           # Script principal de fine-tuning y evaluación
├── diagnosis.py       # Diagnóstico de un modelo ya entrenado
├── results.py         # Diccionario con los resultados de los experimentos
├── table.py           # Genera la tabla resumen con media ± std
├── dataset/           # Dataset local (NO incluido — aportarlo manualmente)
└── resultados/        # Carpeta de salida para informes de diagnosis.py
```

Los checkpoints y artefactos generados durante el entrenamiento se guardan en carpetas con el formato `{model_short}-{dataset}-finetuned{N}/`.

---

## Requisitos

Python **3.10** o superior. Se recomienda usar un entorno virtual.

### Instalación

```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
# venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### Dependencias principales

| Paquete | Versión utilizada | Función |
|---|---|---|
| `torch` | 2.6.0+cu124 | Motor de deep learning (CUDA 12.4) |
| `transformers` | 4.57.6 | Modelos, Trainer, pipelines de HuggingFace |
| `datasets` | 2.21.0 | Carga y preprocesamiento de datasets de audio |
| `evaluate` | 0.4.6 | Métricas de evaluación |
| `accelerate` | 1.10.1 | Backend de entrenamiento distribuido/GPU |
| `kagglehub` | 0.3.13 | Descarga de datasets desde Kaggle |
| `numpy` | 1.23.5 | Operaciones numéricas y manipulación de arrays |
| `pandas` | 1.5.1 | Lectura de metadatos CSV (UrbanSound8K) |
| `matplotlib` | 3.6.1 | Generación de gráficas de curvas de aprendizaje |
| `safetensors` | 0.7.0 | Serialización eficiente de pesos del modelo |
| `tokenizers` | 0.22.2 | Tokenización rápida (dependencia de transformers) |

> **GPU:** el código está configurado para entrenamiento en GPU con CUDA 12.4. Para ejecutarlo en CPU, ajusta `fp16=False` en `TrainingArguments` (ya está así por defecto en el script).

---

## Uso

### 1. Configurar el experimento

Al inicio de `train.py` se definen tres variables de configuración:

```python
MODEL_ID      = "ntu-spml/distilhubert"   # o "facebook/hubert-base-ls960" / "facebook/wav2vec2-base-960h"
DATASET_NAME  = "gender_voice"             # uno de: local, urbansound8k, gender_voice, instruments, cats_dogs
FINETUNE_NUM  = 1                          # número de experimento, para distinguir runs
```

### 2. Entrenar

```bash
python train.py
```

El script descargará automáticamente el dataset desde Kaggle (si no está en caché), preprocesará los audios, entrenará el modelo durante hasta 10 épocas con parada temprana y guardará:

- El modelo fine-tuneado en `./{EXPERIMENT_ID}/`
- La curva de aprendizaje en `./{EXPERIMENT_ID}/learning_curve-{EXPERIMENT_ID}.json` y `.png`

### 3. Evaluar un modelo ya entrenado

Edita las variables `MODEL_ID`, `DATASET` y `FINETUNE_N` en `diagnosis.py` para que apunten al experimento que quieres evaluar, y ejecuta:

```bash
python diagnosis.py
```

El informe se guarda en `./resultados/{EXPERIMENT_ID}.txt`.

### 4. Generar la tabla de resultados agregados

```bash
python table.py
```

Genera `results_table.csv` con la media ± desviación típica de accuracy de los cinco experimentos por modelo y dataset.

---

## Parámetros de entrenamiento

| Parámetro | Valor |
|---|---|
| Learning rate | `1e-5` |
| Batch size (train/eval) | `8` |
| Épocas máximas | `10` |
| Warmup ratio | `0.1` |
| Max grad norm | `0.5` |
| Label smoothing | `0.1` |
| Early stopping patience | `3 épocas` |
| Early stopping threshold | `0.01` |
| Métrica de selección | `accuracy` |

---

## Resultados

La tabla siguiente muestra la accuracy media ± desviación típica sobre 5 experimentos por combinación modelo-dataset:

|  | local | urbansound8k | gender_voice | instruments | cats_dogs |
|---|---|---|---|---|---|
| **DistilHuBERT** | 0.9840 ± 0.0120 | 0.9259 ± 0.0030 | 1.0000 ± 0.0000 | 0.9841 ± 0.0040 | 0.9962 ± 0.0046 |
| **HuBERT** | 0.9780 ± 0.0150 | 0.9151 ± 0.0086 | 0.9999 ± 0.0002 | 0.9467 ± 0.0195 | 1.0000 ± 0.0000 |
| **Wav2Vec2** | 0.8840 ± 0.0850 | 0.3700 ± 0.0700 | 0.9621 ± 0.0560 | 0.7626 ± 0.0860 | 0.5715 ± 0.0820 |

Los modelos entrenados están disponibles en [huggingface.co/nataliaaolmo](https://huggingface.co/nataliaaolmo).
