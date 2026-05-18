from datasets import Dataset, Audio, ClassLabel, DatasetDict
from collections import Counter, defaultdict
import os
import glob
import numpy as np
import time
import evaluate
import json
import torch
import pandas as pd
import kagglehub
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification
from transformers import EarlyStoppingCallback
from transformers import TrainerCallback
from transformers import Trainer
from transformers import pipeline

# ── Modelo ────────────────────────────────────────────────────────────────────
MODEL_ID = "ntu-spml/distilhubert"
#MODEL_ID = "facebook/hubert-base-ls960"
#MODEL_ID = "facebook/wav2vec2-base-960h"

# ── Dataset ───────────────────────────────────────────────────────────────────
# "local"
# "urbansound8k"
# "gender_voice"
# "instruments"
# "cats_dogs"
DATASET_NAME = "gender_voice"

# ── Número de experimento ─────────────────────────────────────────────────────
FINETUNE_NUM = 1

IS_WAV2VEC2 = "wav2vec2" in MODEL_ID.lower()

# Prefijo único que identifica este experimento — se usa en todos los archivos
model_short  = MODEL_ID.split("/")[-1]
EXPERIMENT_ID = f"{model_short}-{DATASET_NAME}-finetuned{FINETUNE_NUM}"

# ══════════════════════════════════════════════════════════════════════════════
#  AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════
def _build_label_maps(labels: list[str]):
    unique = sorted(set(labels))
    label2id = {l: i for i, l in enumerate(unique)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def _make_dataset(audio_paths, str_labels, label2id, id2label):
    int_labels = [label2id[l] for l in str_labels]
    ds = Dataset.from_dict({"audio": audio_paths, "label": int_labels}) 
    ds = ds.cast_column("audio", Audio()) 
    ds = ds.cast_column(
        "label",
        ClassLabel(
            num_classes=len(id2label),
            names=[id2label[i] for i in range(len(id2label))]
        )
    )
    return ds


def _kaggle_download(dataset_handle: str, dest_dir: str) -> str:
    cache_root = os.path.expanduser("~/.cache/kagglehub/datasets")
    slug = dataset_handle.replace("/", os.sep)
    potential_cache = os.path.join(cache_root, slug)
    if os.path.isdir(potential_cache):
        versions = [d for d in os.listdir(potential_cache)
                    if os.path.isdir(os.path.join(potential_cache, d))]
        if versions:
            cached_path = os.path.join(potential_cache, sorted(versions)[-1])
            print(f"Dataset ya en cache: {cached_path}")
            return cached_path
    print(f"Descargando {dataset_handle} desde Kaggle...")
    path = kagglehub.dataset_download(dataset_handle)
    print(f"Descargado en: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_local(data_dir: str = "./dataset"):
    audio_paths, str_labels = [], []
    for genre in sorted(os.listdir(data_dir)):
        genre_path = os.path.join(data_dir, genre)
        for f in glob.glob(os.path.join(genre_path, "*.mp3")):
            audio_paths.append(f)
            str_labels.append(genre)
    label2id, id2label = _build_label_maps(str_labels)
    ds = _make_dataset(audio_paths, str_labels, label2id, id2label)
    return ds.train_test_split(shuffle=True, test_size=0.1,
                               stratify_by_column="label"), label2id, id2label


def load_urbansound8k():
    base = _kaggle_download("chrisfilo/urbansound8k", "./datasets/urbansound8k")
    csv_candidates = glob.glob(os.path.join(base, "**", "UrbanSound8K.csv"), recursive=True) 
    if not csv_candidates:
        raise FileNotFoundError(f"No se encontro UrbanSound8K.csv en {base}")
    meta = pd.read_csv(csv_candidates[0]) 
    audio_root = os.path.dirname(csv_candidates[0]) 
    audio_paths, str_labels = [], [] 
    missing = 0
    for _, row in meta.iterrows(): 
        path = os.path.join(audio_root, f"fold{row['fold']}", row['slice_file_name']) 
        if os.path.isfile(path):
            audio_paths.append(path)
            str_labels.append(row["class"])
        else:
            missing += 1
    print(f"Archivos encontrados: {len(audio_paths)}  |  No encontrados: {missing}")
    if not audio_paths:
        raise RuntimeError(f"No se encontro ningun archivo de audio en {base}")
    label2id, id2label = _build_label_maps(str_labels)
    ds = _make_dataset(audio_paths, str_labels, label2id, id2label)
    return ds.train_test_split(shuffle=True, test_size=0.1,
                               stratify_by_column="label"), label2id, id2label


def load_gender_voice():
    base = _kaggle_download(
        "murtadhanajim/gender-recognition-by-voiceoriginal",
        "./datasets/gender_voice"
    )
    audio_paths, str_labels = [], []
    for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
        for f in glob.glob(os.path.join(base, "**", ext), recursive=True):
            label = os.path.basename(os.path.dirname(f)).lower()
            audio_paths.append(f)
            str_labels.append(label)
    if not audio_paths:
        raise RuntimeError(f"No se encontraron archivos de audio en {base}.")
    label2id, id2label = _build_label_maps(str_labels)
    ds = _make_dataset(audio_paths, str_labels, label2id, id2label)
    return ds.train_test_split(shuffle=True, test_size=0.1,
                               stratify_by_column="label"), label2id, id2label


def load_instruments():
    base = _kaggle_download(
        "abdulvahap/music-instrunment-sounds-for-classification",
        "./datasets/instruments"
    )
    subdirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if len(subdirs) == 1:
        base = os.path.join(base, subdirs[0])
        inner = os.path.join(base, "music_dataset")
        if os.path.isdir(inner):
            base = inner
    print(f"Usando carpeta base: {base}")
    audio_paths, str_labels = [], []
    
    for instrument in os.listdir(base):
        inst_path = os.path.join(base, instrument)
        if not os.path.isdir(inst_path) or instrument.startswith("."):
            continue
        label = instrument.lower().replace(" ", "_")
        for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
            for f in glob.glob(os.path.join(inst_path, "**", ext), recursive=True):
                audio_paths.append(f)
                str_labels.append(label)
    if not audio_paths:
        raise RuntimeError(f"No se encontraron audios en {base}")
    print(f"Total audios encontrados: {len(audio_paths)}")
    label2id, id2label = _build_label_maps(str_labels)
    ds = _make_dataset(audio_paths, str_labels, label2id, id2label)
    return ds.train_test_split(shuffle=True, test_size=0.1,
                               stratify_by_column="label"), label2id, id2label


def load_cats_dogs():
    base = _kaggle_download(
        "stealthtechnologies/cats-vs-dogs-audio-classification",
        "./datasets/cats_dogs"
    )
    audio_paths, str_labels = [], []
    for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
        for f in glob.glob(os.path.join(base, "**", ext), recursive=True):
            label = os.path.basename(os.path.dirname(f)).lower()
            if "cat" in label:   label = "cat"
            elif "dog" in label: label = "dog"
            audio_paths.append(f)
            str_labels.append(label)
    if not audio_paths:
        raise RuntimeError(f"No se encontraron archivos de audio en {base}.")
    label2id, id2label = _build_label_maps(str_labels)
    ds = _make_dataset(audio_paths, str_labels, label2id, id2label)
    return ds.train_test_split(shuffle=True, test_size=0.1,
                               stratify_by_column="label"), label2id, id2label


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE LOS DATOS Y DATOS RELEVANTES
# ══════════════════════════════════════════════════════════════════════════════
LOADERS = {
    "local":        load_local,
    "urbansound8k": load_urbansound8k,
    "gender_voice": load_gender_voice,
    "instruments":  load_instruments,
    "cats_dogs":    load_cats_dogs,
}

if DATASET_NAME not in LOADERS:
    raise ValueError(f"DATASET_NAME='{DATASET_NAME}' no reconocido. Opciones: {list(LOADERS.keys())}")

print(f"\n{'='*60}")
print(f"  Modelo:      {MODEL_ID}")
print(f"  Dataset:     {DATASET_NAME}")
print(f"  Experimento: finetuned{FINETUNE_NUM}  ({EXPERIMENT_ID})")
print(f"{'='*60}\n")

dataset, label2id, id2label = LOADERS[DATASET_NAME]()
num_labels = len(id2label)

print(f"\nEtiquetas: {id2label}")
print(f"Train: {len(dataset['train'])} muestras  |  Test: {len(dataset['test'])} muestras")

train_counts = Counter(dataset["train"]["label"])
test_counts  = Counter(dataset["test"]["label"])
print("\n--- Distribucion de clases ---")
print(f"  {'clase':<15}  {'train':>8}  {'test':>6}")
print(f"  {'-'*33}")
for id_ in sorted(id2label.keys()):
    print(f"  {id2label[id_]:<15}  {train_counts[id_]:>8}  {test_counts[id_]:>6}")

print("\n--- Duracion media por clase (primeras 200 muestras de train) ---")
dur_por_clase = defaultdict(list)
for sample in dataset["train"].select(range(min(200, len(dataset["train"])))):
    arr = sample["audio"]["array"]
    sr  = sample["audio"]["sampling_rate"]
    dur_por_clase[id2label[sample["label"]]].append(len(arr) / sr)
for clase, durs in sorted(dur_por_clase.items()):
    print(f"  {clase:<15} -> media {np.mean(durs):.2f}s  min {np.min(durs):.2f}s  max {np.max(durs):.2f}s")

# ──────────────────────────────────────────────────────────────────────────────
#  FEATURE EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────────
feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_ID,
    do_normalize=True,
    return_attention_mask=True,
)
sampling_rate = feature_extractor.sampling_rate
dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

# ──────────────────────────────────────────────────────────────────────────────
#  PREPROCESAMIENTO AUDIO Y REGULARIZACIÓN DE DATOS
# ──────────────────────────────────────────────────────────────────────────────
MAX_DURATION_MAP = {
    "instruments":  3.0,
    "cats_dogs":    5.0,
    "gender_voice": 5.0,
    "urbansound8k": 4.0,
    "local":        30.0,
}
max_duration = MAX_DURATION_MAP.get(DATASET_NAME, 30.0)
max_len      = int(sampling_rate * max_duration)
rng = np.random.default_rng(42)

def regulate_audio(array: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        noise_level = rng.uniform(0.001, 0.005)
        array = array + noise_level * rng.standard_normal(len(array)).astype(np.float32)
    if rng.random() < 0.5:
        gain_db = rng.uniform(-3.0, 3.0)
        array = array * (10 ** (gain_db / 20.0))
    return array.astype(np.float32)

def preprocess(examples, regulation=False):
    audio_arrays = []
    for x in examples["audio"]:
        arr = x["array"].astype(np.float32)

        if regulation:
            arr = regulate_audio(arr, rng)
        peak = np.max(np.abs(arr))
        if peak > 0:
            arr = arr / (peak + 1e-6)

        arr = np.clip(arr, -1.0, 1.0)
        audio_arrays.append(arr)

    return feature_extractor(
        audio_arrays,
        sampling_rate=sampling_rate,
        max_length=max_len,
        truncation=True,
        return_attention_mask=True,
    )

preprocess_train = lambda x: preprocess(x, regulation=True)
preprocess_eval  = lambda x: preprocess(x, regulation=False)

#Curva de aprendizaje

print("Preprocesando train completo para evaluacion (sin regularización)...")
train_eval_encoded = dataset["train"].map(
    preprocess_eval,
    remove_columns=["audio"],
    batched=True,
    batch_size=100,
)

print("Preprocesando train (con regularización)...")
train_encoded = dataset["train"].map(
    preprocess_train,
    remove_columns=["audio"],
    batched=True,
    batch_size=32,
)

print("Preprocesando test (sin regularización)...")
test_encoded = dataset["test"].map(
    preprocess_eval,
    remove_columns=["audio"],
    batched=True,
    batch_size=100,
)

dataset_encoded = DatasetDict({
    "train":      train_encoded,
    "test":       test_encoded,
    "train_eval": train_eval_encoded,
})

# ──────────────────────────────────────────────────────────────────────────────
#  MODELO
# ──────────────────────────────────────────────────────────────────────────────
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_ID,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

# ──────────────────────────────────────────────────────────────────────────────
#  CALLBACK PARA LA CURVA DE APRENDIZAJE
# ──────────────────────────────────────────────────────────────────────────────

class LearningCurveCallback(TrainerCallback):
    def __init__(self, train_eval_dataset, experiment_id):
        self.train_eval_dataset = train_eval_dataset
        self.experiment_id      = experiment_id
        self.history            = []
        self.trainer            = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or not state.epoch:
            return

        epoch = int(state.epoch)

        acc_test  = metrics.get("eval_accuracy")
        loss_test = metrics.get("eval_loss")

        train_pred = self.trainer.predict(
            self.train_eval_dataset,
            metric_key_prefix="train_curve"
        )
        acc_train  = train_pred.metrics.get("train_curve_accuracy")
        loss_train = train_pred.metrics.get("train_curve_loss")

        self.history.append({
            "epoch":          epoch,
            "accuracy_train": acc_train,
            "accuracy_test":  acc_test,
            "loss_train":     loss_train,
            "loss_test":      loss_test,
        })

        print(f"\n[Curva época {epoch}] "
              f"acc_train={acc_train:.4f}  acc_test={acc_test:.4f}  "
              f"loss_train={loss_train:.4f}  loss_test={loss_test:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        os.makedirs(args.output_dir, exist_ok=True)

        json_path = os.path.join(
            args.output_dir,
            f"learning_curve-{self.experiment_id}.json"
        )
        png_path = os.path.join(
            args.output_dir,
            f"learning_curve-{self.experiment_id}.png"
        )

        with open(json_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nCurva guardada en: {json_path}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs     = [r["epoch"]          for r in self.history]
            acc_train  = [r["accuracy_train"] for r in self.history]
            acc_test   = [r["accuracy_test"]  for r in self.history]
            loss_train = [r["loss_train"]     for r in self.history]
            loss_test  = [r["loss_test"]      for r in self.history]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(
                f"Curvas de aprendizaje\n"
                f"{MODEL_ID}  ·  {DATASET_NAME}  ·  finetuned{FINETUNE_NUM}",
                fontsize=11
            )

            ax = axes[0]
            ax.plot(epochs, acc_train, "o-", label="Train", color="steelblue")
            ax.plot(epochs, acc_test,  "s-", label="Test",  color="tomato")
            ax.set_xlabel("Época")
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy por época")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(epochs)
            ax.set_ylim(0, 1.05)

            ax = axes[1]
            if any(v is not None for v in loss_train):
                ax.plot(epochs, loss_train, "o-", label="Train", color="steelblue")
            if any(v is not None for v in loss_test):
                ax.plot(epochs, loss_test,  "s-", label="Test",  color="tomato")
            ax.set_xlabel("Época")
            ax.set_ylabel("Loss")
            ax.set_title("Loss por época")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(epochs)

            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Grafica guardada en: {png_path}")

        except ImportError:
            print("[AVISO] matplotlib no instalado: pip install matplotlib")


# ──────────────────────────────────────────────────────────────────────────────
#  TRAINING ARGUMENTS
# ──────────────────────────────────────────────────────────────────────────────
from transformers import TrainingArguments

run_name   = EXPERIMENT_ID
output_dir = f"./{EXPERIMENT_ID}"

training_args = TrainingArguments(
    output_dir=output_dir,
    run_name=run_name,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    warmup_ratio=0.1, 
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
    fp16=False,
    max_grad_norm=0.5,
    label_smoothing_factor=0.1, 
    push_to_hub=False,
)

# ──────────────────────────────────────────────────────────────────────────────
#  METRICAS
# ──────────────────────────────────────────────────────────────────────────────
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# ──────────────────────────────────────────────────────────────────────────────
#  EARLY STOPPING
# ──────────────────────────────────────────────────────────────────────────────

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.01,
)

# ──────────────────────────────────────────────────────────────────────────────
#  ENTRENAMIENTO
# ──────────────────────────────────────────────────────────────────────────────

learning_curve_cb = LearningCurveCallback(
    train_eval_dataset=dataset_encoded["train_eval"],
    experiment_id=EXPERIMENT_ID,
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=dataset_encoded["train"],
    eval_dataset=dataset_encoded["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping, learning_curve_cb],
)

learning_curve_cb.set_trainer(trainer)

print(f"\nIniciando entrenamiento: {EXPERIMENT_ID}")
print(f"  Checkpoints en: {output_dir}\n")
start = time.time()
trainer.train()

elapsed = time.time() - start
print(f"\nEntrenamiento completado en {int(elapsed // 60)}m {elapsed % 60:.2f}s")

final_model_path = output_dir
trainer.save_model(final_model_path)
feature_extractor.save_pretrained(final_model_path)
print(f"Modelo guardado en: {final_model_path}")

# ──────────────────────────────────────────────────────────────────────────────
#  VALIDACION COMPLETA SOBRE EL SPLIT DE TEST
# ──────────────────────────────────────────────────────────────────────────────

classifier = pipeline(
    "audio-classification",
    model=final_model_path,
    feature_extractor=final_model_path,
)

print(f"\nEjecutando validacion sobre {len(dataset['test'])} muestras de test...\n")

correct           = 0
total             = 0
per_class_correct = defaultdict(int)
per_class_total   = defaultdict(int)
confusion         = defaultdict(lambda: defaultdict(int))

start_val = time.time()

for i in range(len(dataset["test"])):
    sample         = dataset["test"][i]
    label_real     = id2label[sample["label"]]
    result         = classifier(
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
    )
    label_predicho = result[0]["label"]

    per_class_total[label_real] += 1
    confusion[label_real][label_predicho] += 1
    if label_predicho == label_real:
        correct += 1
        per_class_correct[label_real] += 1
    total += 1

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{total}]  accuracy parcial: {correct/total:.4f}")

elapsed_val = time.time() - start_val

print(f"\n{'='*60}")
print(f"  RESULTADOS FINALES  —  {EXPERIMENT_ID}")
print(f"{'='*60}")
print(f"  Accuracy global:  {correct}/{total} = {correct/total:.4f} ({correct/total*100:.2f}%)")
print(f"  Tiempo validacion: {int(elapsed_val // 60)}m {elapsed_val % 60:.2f}s")
print(f"\n  Accuracy por clase:")
for clase in sorted(per_class_total.keys()):
    c = per_class_correct[clase]
    t = per_class_total[clase]
    print(f"    {clase:<25} {c:>3}/{t:<3}  ({c/t*100:.1f}%)")

all_classes = sorted(per_class_total.keys())
col_w = 10
if num_labels <= 10:
    print(f"\n  Matriz de confusion (filas=real, columnas=predicho):")
    print(f"  {'real / pred':<16}" + "".join(f"{c:>{col_w}}" for c in all_classes))
    print("  " + "-" * (16 + col_w * len(all_classes)))
    for real in all_classes:
        print(
            f"  {real:<16}" +
            "".join(f"{confusion[real][pred]:>{col_w}}" for pred in all_classes)
        )
else:
    print(f"\n  Clases con peor accuracy (bottom 5):")
    class_acc = {c: per_class_correct[c] / per_class_total[c] for c in per_class_total}
    for clase, acc in sorted(class_acc.items(), key=lambda x: x[1])[:5]:
        c = per_class_correct[clase]
        t = per_class_total[clase]
        top_wrong = sorted(
            [(p, n) for p, n in confusion[clase].items() if p != clase],
            key=lambda x: -x[1]
        )[:3]
        wrong_str = ", ".join(f"{p}({n})" for p, n in top_wrong)
        print(f"    {clase:<25} {c:>4}/{t:<4} ({acc*100:.1f}%)  confunde con: {wrong_str}")

print(f"\nArchivos de este experimento en {output_dir}/:")
print(f"  learning_curve-{EXPERIMENT_ID}.json")
print(f"  learning_curve-{EXPERIMENT_ID}.png")