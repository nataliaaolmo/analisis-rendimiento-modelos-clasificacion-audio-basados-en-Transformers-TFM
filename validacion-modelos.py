import os
import time
import glob
import numpy as np
import pandas as pd

from collections import defaultdict
from datasets import Dataset, Audio, ClassLabel
import kagglehub

from transformers import (
    pipeline,
    AutoFeatureExtractor,
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

MODEL_ID   = "ntu-spml/distilhubert"
DATASET    = "gender_voice"
FINETUNE_N = 1

model_short   = MODEL_ID.split("/")[-1]
EXPERIMENT_ID = f"{model_short}-{DATASET}-finetuned{FINETUNE_N}"

MODEL_PATH    = f"./{EXPERIMENT_ID}"

# carpeta centralizada para guardar TODOS los diagnósticos
RESULTS_DIR = "./resultados"
os.makedirs(RESULTS_DIR, exist_ok=True)

# nombre base de archivos
RESULT_BASENAME = f"{EXPERIMENT_ID}"

TXT_RESULT_PATH = os.path.join(
    RESULTS_DIR,
    f"{RESULT_BASENAME}.txt"
)

# ══════════════════════════════════════════════════════════════════════════════
#  AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def _build_label_maps(labels):
    unique = sorted(set(labels))
    label2id = {l: i for i, l in enumerate(unique)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def _make_dataset(audio_paths, str_labels, label2id, id2label):
    int_labels = [label2id[l] for l in str_labels]

    ds = Dataset.from_dict({
        "audio": audio_paths,
        "label": int_labels
    })

    ds = ds.cast_column("audio", Audio())

    ds = ds.cast_column(
        "label",
        ClassLabel(
            num_classes=len(id2label),
            names=[id2label[i] for i in range(len(id2label))]
        )
    )

    return ds


def _kaggle_download(dataset_handle, dest_dir=None):
    cache_root = os.path.expanduser("~/.cache/kagglehub/datasets")
    slug = dataset_handle.replace("/", os.sep)

    potential_cache = os.path.join(cache_root, slug)

    if os.path.isdir(potential_cache):
        versions = [
            d for d in os.listdir(potential_cache)
            if os.path.isdir(os.path.join(potential_cache, d))
        ]

        if versions:
            return os.path.join(
                potential_cache,
                sorted(versions)[-1]
            )

    return kagglehub.dataset_download(dataset_handle)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADER
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

    base = _kaggle_download("chrisfilo/urbansound8k")

    csv_candidates = glob.glob(
        os.path.join(base, "**", "UrbanSound8K.csv"),
        recursive=True
    )

    if not csv_candidates:
        raise FileNotFoundError(
            f"No se encontró UrbanSound8K.csv en {base}"
        )

    meta = pd.read_csv(csv_candidates[0])

    audio_root = os.path.dirname(csv_candidates[0])

    audio_paths = []
    str_labels  = []

    missing = 0

    for _, row in meta.iterrows():

        path = os.path.join(
            audio_root,
            f"fold{row['fold']}",
            row['slice_file_name']
        )

        if os.path.isfile(path):
            audio_paths.append(path)
            str_labels.append(row["class"])
        else:
            missing += 1

    print(f"Archivos encontrados: {len(audio_paths)}")
    print(f"Archivos faltantes:   {missing}")

    label2id, id2label = _build_label_maps(str_labels)

    ds = _make_dataset(
        audio_paths,
        str_labels,
        label2id,
        id2label
    )

    # IMPORTANTE:
    # mismo split SIEMPRE
    return ds.train_test_split(
        shuffle=True,
        test_size=0.1,
        stratify_by_column="label",
        seed=42,
    ), label2id, id2label

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
    
    #aquí no existe csv, Las clases se obtienen directamente de los nombres de las carpetas
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

LOADERS = {
    "local":        load_local,
    "urbansound8k": load_urbansound8k,
    "gender_voice": load_gender_voice,
    "instruments":  load_instruments,
    "cats_dogs":    load_cats_dogs,
}

# ══════════════════════════════════════════════════════════════════════════════
#  DIAGNÓSTICO
# ══════════════════════════════════════════════════════════════════════════════

def run_diagnosis(model_path, dataset_name):

    print(f"\n{'='*70}")
    print(f" DIAGNÓSTICO")
    print(f"{'='*70}")

    # ──────────────────────────────────────────────────────────────────────────
    # modelo
    # ──────────────────────────────────────────────────────────────────────────

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"No existe el directorio del modelo: {model_path}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # cargar feature extractor
    # ──────────────────────────────────────────────────────────────────────────

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

    sampling_rate = feature_extractor.sampling_rate

    # ──────────────────────────────────────────────────────────────────────────
    # dataset
    # ──────────────────────────────────────────────────────────────────────────

    dataset, label2id, id2label = LOADERS[dataset_name]()

    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=sampling_rate)
    )

    test_set = dataset["test"]

    print(f"\nTest samples: {len(test_set)}")
    print(f"Sampling rate: {sampling_rate}")

    # ──────────────────────────────────────────────────────────────────────────
    # pipeline
    # ──────────────────────────────────────────────────────────────────────────

    classifier = pipeline(
        "audio-classification",
        model=model_path,
        feature_extractor=model_path,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # métricas
    # ──────────────────────────────────────────────────────────────────────────

    correct = 0
    total   = 0

    pred_counts = defaultdict(int)

    per_class_correct = defaultdict(int)
    per_class_total   = defaultdict(int)

    confusion = defaultdict(lambda: defaultdict(int))

    examples = []

    start = time.time()

    # ──────────────────────────────────────────────────────────────────────────
    # inferencia
    # ──────────────────────────────────────────────────────────────────────────

    for i in range(len(test_set)):

        sample = test_set[i]

        label_real = id2label[sample["label"]]

        result = classifier(
            sample["audio"]["array"],
            sampling_rate=sample["audio"]["sampling_rate"],
            top_k=len(id2label),
        )

        label_pred = result[0]["label"]

        pred_counts[label_pred] += 1

        per_class_total[label_real] += 1

        confusion[label_real][label_pred] += 1

        if label_pred == label_real:
            correct += 1
            per_class_correct[label_real] += 1

        total += 1

        examples.append({
            "real": label_real,
            "pred": label_pred,
            "top_score": float(result[0]["score"]),
            "scores": {
                r["label"]: float(r["score"])
                for r in result
            }
        })

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(test_set)}]")

    elapsed = time.time() - start

    accuracy = correct / total

    # ──────────────────────────────────────────────────────────────────────────
    # construir resultado final
    # ──────────────────────────────────────────────────────────────────────────

    result_data = {
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "dataset": DATASET,
        "finetune_version": FINETUNE_N,

        "metrics": {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "elapsed_seconds": elapsed,
        },

        "per_class_accuracy": {
            cls: {
                "correct": per_class_correct[cls],
                "total": per_class_total[cls],
                "accuracy":
                    per_class_correct[cls] / per_class_total[cls]
            }
            for cls in sorted(per_class_total.keys())
        },

        "prediction_distribution": dict(pred_counts),

        "confusion_matrix": {
            real: dict(preds)
            for real, preds in confusion.items()
        },

        "examples": examples,
    }
    
    # ──────────────────────────────────────────────────────────────────────────
    # guardar TXT legible
    # ──────────────────────────────────────────────────────────────────────────

    with open(TXT_RESULT_PATH, "w") as f:

        f.write(f"EXPERIMENTO: {EXPERIMENT_ID}\n")
        f.write("="*70 + "\n\n")

        f.write(f"Modelo:   {MODEL_ID}\n")
        f.write(f"Dataset:  {DATASET}\n")
        f.write(f"Version:  finetuned{FINETUNE_N}\n\n")

        f.write(f"Accuracy global: "
                f"{correct}/{total} = {accuracy:.4f}\n")

        f.write(f"Tiempo: {elapsed:.2f}s\n\n")

        f.write("Accuracy por clase\n")
        f.write("-"*70 + "\n")

        for cls in sorted(per_class_total.keys()):

            c = per_class_correct[cls]
            t = per_class_total[cls]

            f.write(
                f"{cls:<20} "
                f"{c:>4}/{t:<4} "
                f"({c/t*100:.2f}%)\n"
            )

        f.write("\n")

        # matriz de confusión
        all_classes = sorted(per_class_total.keys())

        col_w = 12

        f.write("MATRIZ DE CONFUSIÓN\n")
        f.write("-"*70 + "\n")

        f.write(
            f"{'real/pred':<20}" +
            "".join(f"{c:>{col_w}}" for c in all_classes) +
            "\n"
        )

        for real in all_classes:

            row = (
                f"{real:<20}" +
                "".join(
                    f"{confusion[real][pred]:>{col_w}}"
                    for pred in all_classes
                )
            )

            f.write(row + "\n")

    # ──────────────────────────────────────────────────────────────────────────
    # resumen consola
    # ──────────────────────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"Accuracy final: {accuracy:.4f}")
    print(f"Resultados guardados:")
    print(f"  TXT  -> {TXT_RESULT_PATH}")
    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    run_diagnosis(
        MODEL_PATH,
        DATASET,
    )