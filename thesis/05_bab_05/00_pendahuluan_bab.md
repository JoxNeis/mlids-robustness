# **BAB 5 IMPLEMENTASI**

Bab ini menyajikan implementasi dari rancangan yang telah diuraikan pada Bab 4. Implementasi diwujudkan dalam bentuk kode program berbahasa Python yang dijalankan pada *Jupyter Notebook* (*mlids-scikit.ipynb*). Setiap tahap pada *pipeline*, mulai dari konversi dataset ke format Parquet hingga analisis hasil pengujian ketahanan, ditulis sebagai fungsi yang dapat dijalankan ulang secara berurutan. Pemecahan pekerjaan menjadi fungsi-fungsi kecil, yang masing-masing menangani satu file, satu kelompok baris, satu skenario, atau satu model, memudahkan pemrosesan paralel, pemeriksaan hasil antara pada setiap tahap, dan pengulangan tahap tertentu tanpa mengulang tahap lainnya. Setiap tahap juga menampilkan status dan bilah kemajuan (*progress bar*) sehingga pelaksanaan tahap yang membutuhkan waktu lama dapat dipantau.

Bab ini disusun sejajar dengan Bab 4. Setiap subbab pada bab ini memuat kode program dari rancangan pada subbab dengan nomor yang sama pada Bab 4, misalnya Subbab 5.7 memuat implementasi dari rancangan pada Subbab 4.7. Di dalam setiap subbab, kode program disusun mengikuti urutan langkah pada algoritma di Bab 4, dan setiap kode program didahului penjelasan mengenai langkah yang diimplementasikannya. Kode program disajikan sebagaimana pada *notebook*, termasuk nama variabel dan fungsi yang berbahasa Inggris. Kode yang berasal dari beberapa sel *notebook* digabung dalam satu kode program dengan sel-selnya dipisahkan oleh satu baris kosong. Konstanta pengaturan, seperti lokasi folder, ukuran kelompok baris, dan ruang pencarian hiperparameter, dikumpulkan pada awal setiap tahap dan digunakan sebagai nilai bawaan (*default*) parameter fungsi, sehingga pengaturan dapat diubah tanpa mengubah logika fungsi. Fungsi bantu yang digunakan oleh banyak tahap disajikan pada bagian awal bab ini, sedangkan fungsi bantu yang digunakan oleh satu tahap disajikan pada subbab tahap tersebut.

Library yang digunakan dirangkum pada Tabel 5.1. Algoritma *K Nearest-Neighbor*, *Support Vector Machine*, *Random Forest*, dan *Logistic Regression* menggunakan implementasi CPU dari library *scikit-learn*, sedangkan *XGBoost* menggunakan library *xgboost*.

Tabel 5.1 Library yang Digunakan pada Implementasi

| Kelompok | Library | Kegunaan pada implementasi |
| :--- | :--- | :--- |
| Pengolahan data | *pandas* dan *NumPy* | Struktur data tabular, pembacaan CSV per *chunk*, dan operasi numerik |
| Pengolahan data | *Polars* | Pembacaan file Parquet secara *lazy*, pembacaan per kelompok baris, dan agregasi dengan mesin *streaming* |
| Pengolahan data | *PyArrow* | Penulisan file Parquet per kelompok baris serta pembacaan skema dan metadata file |
| Komputasi paralel | *joblib* | Pemrosesan file secara paralel dengan *backend* *loky* serta penyimpanan dan pemuatan objek |
| Praproses | *scikit-learn* | *train_test_split*, *LabelEncoder*, *FunctionTransformer*, *StandardScaler*, *PCA*, *StratifiedShuffleSplit*, dan *GridSearchCV* |
| Praproses | *imbalanced-learn* | *Pipeline* dan SMOTE |
| Model | *scikit-learn* | *KNeighborsClassifier*, *SVC*, *LinearSVC*, *RandomForestClassifier*, dan *LogisticRegression* |
| Model | *xgboost* | *XGBClassifier* |
| Evaluasi | *scikit-learn* (*sklearn.metrics*) | Akurasi, akurasi seimbang, presisi, *recall*, F1-*score*, MCC, *Cohen's kappa*, dan *confusion matrix* |
| Visualisasi | *Matplotlib* | Diagram batang, grafik garis, dan peta panas (*heatmap*) |
| Pendukung | *hashlib*, *re*, *json*, dan *shutil* | *Hash* BLAKE2b, ekstraksi tanggal dari nama file, file JSON, dan penyalinan file |

Kode Program 5.1 memuat seluruh perintah impor yang dikelompokkan menurut kegunaannya, yaitu pustaka standar Python, sistem dan file, pengolahan data, visualisasi, praproses, model, dan evaluasi. Sel pertama menetapkan variabel *USE_CUDA*, yang hanya menentukan perangkat komputasi *XGBoost* (Subbab 5.13). Apabila *USE_CUDA* bernilai *True* dan direktori *include* CUDA tersedia pada lingkungan Python yang digunakan, variabel lingkungan *CUDA_PATH* diatur agar library berbasis GPU dapat menemukan file CUDA, dan ukuran *cache* kompilasi CUDA diperbesar menjadi 4 GB. Kelas *Pipeline* diimpor dari *imbalanced-learn*, bukan dari *scikit-learn*, karena hanya *Pipeline* dari *imbalanced-learn* yang dapat memuat tahap SMOTE (Subbab 5.7).

```python
USE_CUDA = True

import time
import warnings
from typing import Any, Callable, Iterator, Literal, cast

import os
import sys
import gc
import shutil
import hashlib
import joblib
import json

import re
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

if USE_CUDA:
    cuda_root = os.path.join(sys.prefix, "targets", "x86_64-linux")
    if os.path.isdir(os.path.join(cuda_root, "include")):
        os.environ.setdefault("CUDA_PATH", cuda_root)
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 1024**3))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.preprocessing import LabelEncoder, FunctionTransformer

from imblearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, ParameterGrid

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import make_scorer, get_scorer, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, confusion_matrix
```

Kode Program 5.1 Impor library dan pengaturan perangkat

Kode Program 5.2 memuat fungsi bantu penampil status yang digunakan oleh seluruh tahap. Fungsi *print_status* menimpa baris yang sama pada keluaran *notebook* sehingga status terbaru selalu terlihat tanpa memenuhi layar, sedangkan *end_status* menutup baris tersebut dengan pesan akhir. Fungsi *format_duration* mengubah detik menjadi format jam, menit, dan detik. Fungsi *print_progress* menampilkan bilah kemajuan beserta persentase, waktu yang telah berlalu, dan perkiraan sisa waktu. Kelas *FitProgress* mencatat jumlah pelatihan yang telah selesai dan yang gagal pada pencarian hiperparameter (Subbab 5.8).

```python
def print_status(message: str) -> None:
    print(f"\r{message:<80}", end="", flush=True)


def end_status(message: str = "") -> None:
    print(f"\r{message:<80}", flush=True)

def format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(int(seconds), 60)
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}"

def print_progress(
    current: int,
    total: int,
    prefix: str = "",
    width: int = 30,
    started: float | None = None,
    suffix: str = "",
) -> None:
    fraction = current / total if total else 0
    filled = int(width * fraction)
    bar = "█" * filled + "-" * (width - filled)
    message = f"{prefix} |{bar}| {current}/{total} ({fraction:.0%})"
    if started is not None:
        elapsed = time.time() - started
        message += f" | {format_duration(elapsed)} elapsed"
        if 0 < current < total:
            message += f", ~{format_duration(elapsed / current * (total - current))} left"
    if suffix:
        message += f" | {suffix}"
    print_status(message)

class FitProgress:
    def __init__(self, total: int, prefix: str) -> None:
        self.total = total
        self.prefix = prefix
        self.done = 0
        self.failed = 0
        self.started = time.time()
        self.show()

    def show(self) -> None:
        print_progress(self.done, self.total, self.prefix, started=self.started, suffix=f"{self.failed} failed")

    def update(self, failed: bool = False) -> None:
        self.done += 1
        if failed:
            self.failed += 1
        self.show()
```

Kode Program 5.2 Fungsi penampil status dan bilah kemajuan

Kode Program 5.3 memuat fungsi bantu pengelolaan file. Konstanta *ROW_INDEX_COLUMN* menetapkan nama kolom nomor baris yang digunakan pada pembagian dataset (Subbab 5.5). Fungsi *get_file_paths_in_folder* mengembalikan lokasi seluruh file dengan ekstensi tertentu yang telah diurutkan secara alfabetis, sehingga urutan pemrosesan bersifat deterministik. Fungsi *load* dan *dump* memuat dan menyimpan objek Python, misalnya *LabelEncoder* dan *pipeline* yang telah dilatih, menggunakan *joblib*.

```python
ROW_INDEX_COLUMN = "row_index"

def get_file_paths_in_folder(folder_path: str, file_extension: str) -> list[str]:
    file_paths = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(f".{file_extension}"):
            file_paths.append(os.path.join(folder_path, file_name))
    return sorted(file_paths)

def load(object_path: str) -> Any:
    return joblib.load(object_path)

def dump(obj: Any, dump_path: str) -> None:
    joblib.dump(obj, dump_path)
```

Kode Program 5.3 Fungsi bantu pengelolaan file

Kode Program 5.4 menetapkan lokasi folder dan file yang digunakan pada seluruh tahap. Folder *cache* menyimpan daftar kelas dan pengode label. Folder *scikit-learn* menyimpan hasil yang bergantung pada model, yaitu hasil pencarian hiperparameter, model terlatih, prediksi, dan evaluasi, dengan subfolder *wout-smote* dan *with-smote* untuk kedua pengaturan SMOTE. Folder *data-pipeline* menyimpan data hasil setiap tahap praproses dengan awalan nomor sesuai urutan tahapnya, yaitu *01-raw-parquet* (Subbab 5.1), *02-cleaned-parquet* (Subbab 5.3), *03-deduplicated-parquet* (Subbab 5.4), *04-splitted-parquet* (Subbab 5.5), dan *06-noisy-parquet* (Subbab 5.14).

```python
# cached path
PATH_FOLDER_CACHE = "cache"
PATH_FOLDER_RESULTS = "scikit-learn"  # the models and results of this notebook, apart from the cuML ones
PATH_FOLDER_TRAINED_MODEL = os.path.join(PATH_FOLDER_RESULTS, "trained-models")
PATH_FOLDER_GRID_SEARCH = os.path.join(PATH_FOLDER_RESULTS, "grid-search")
PATH_FOLDER_EVALUATION = os.path.join(PATH_FOLDER_RESULTS, "evaluation")
PATH_FOLDER_NOISE_EVALUATION = "noise-evaluation"
PATH_FOLDER_PREDICTIONS = os.path.join(PATH_FOLDER_RESULTS, "predictions")

# class label's
PATH_JSON_ORIGINAL_CLASSES_LIST = os.path.join(PATH_FOLDER_CACHE, "original-classes.json")
PATH_JSON_MINIMIZED_CLASSES_LIST = os.path.join(PATH_FOLDER_CACHE, "minimized-classes.json")
PATH_JSON_CURRENTLY_USED_CLASSES_LIST = PATH_JSON_ORIGINAL_CLASSES_LIST

# transformers
PATH_LABEL_ENCODER_TRANSFORMER = os.path.join(PATH_FOLDER_CACHE, "label-encoder-transformer.pkl")
PATH_STANDARD_SCALER_TRANSFORMER = os.path.join(PATH_FOLDER_CACHE, "standard-scaler-transformer.pkl")
PATH_PCA_TRANSFORMER = os.path.join(PATH_FOLDER_CACHE, "pca.npz")

# data pipeline path
PATH_FOLDER_ORIGINAL_DATASET = "cse-cic-ids2018"
PATH_FOLDER_DATA_PIPELINE = "data-pipeline"
PATH_FOLDER_TRAINED_MODEL_WOUT_SMOTE = os.path.join(PATH_FOLDER_TRAINED_MODEL, "wout-smote")
PATH_FOLDER_TRAINED_MODEL_WITH_SMOTE = os.path.join(PATH_FOLDER_TRAINED_MODEL, "with-smote")
PATH_FOLDER_GRID_SEARCH_WOUT_SMOTE = os.path.join(PATH_FOLDER_GRID_SEARCH, "wout-smote")
PATH_FOLDER_GRID_SEARCH_WITH_SMOTE = os.path.join(PATH_FOLDER_GRID_SEARCH, "with-smote")
PATH_FOLDER_EVALUATION_MATRICES = os.path.join(PATH_FOLDER_EVALUATION, "matrices")
PATH_FOLDER_PARQUET_DATASET = os.path.join(PATH_FOLDER_DATA_PIPELINE, "01-raw-parquet")
PATH_FOLDER_CLEANED_DATASET = os.path.join(PATH_FOLDER_DATA_PIPELINE, "02-cleaned-parquet")
PATH_FOLDER_DEDUPLICATED_DATASET = os.path.join(PATH_FOLDER_DATA_PIPELINE, "03-deduplicated-parquet")
PATH_FOLDER_SPLITTED_DATASET = os.path.join(PATH_FOLDER_DATA_PIPELINE, "04-splitted-parquet")
PATH_FOLDER_SPLITTED_DATASET_TRAIN = os.path.join(PATH_FOLDER_SPLITTED_DATASET, "train")
PATH_FOLDER_SPLITTED_DATASET_TEST = os.path.join(PATH_FOLDER_SPLITTED_DATASET, "test")
PATH_FOLDER_NOISY_DATASET = os.path.join(PATH_FOLDER_DATA_PIPELINE, "06-noisy-parquet")
PATH_FOLDER_NOISY_DATASET_TEST = os.path.join(PATH_FOLDER_NOISY_DATASET, "test")
```

Kode Program 5.4 Lokasi folder dan file

Kode Program 5.5 menetapkan nama file yang digunakan pada beberapa tahap, yaitu file fitur, label, dan label terkode pada setiap himpunan data, file pemeriksaan gangguan, file prediksi, serta file tabel dan gambar hasil evaluasi.

```python
FEATURES_FILE_NAME = "features.parquet"
LABELS_FILE_NAME = "labels.parquet"
ENCODED_LABELS_FILE_NAME = "encoded-labels.parquet"

# noisy datasets
NOISE_CHECK_FILE_NAME = "noise-check.csv"

# predictions
PREDICTIONS_FILE_NAME = "predictions.parquet"

# evaluation results
OUTCOME_COUNTS_FILE_NAME = "outcome-counts.parquet"
DISPLACEMENT_BINS_FILE_NAME = "displacement-bins.csv"
METRICS_FILE_NAME = "metrics.csv"
DEGRADATION_FILE_NAME = "degradation.csv"
ROBUSTNESS_RANKING_FILE_NAME = "robustness-ranking.csv"
MIXTURE_CHECK_FILE_NAME = "mixture-check.csv"
SMOTE_EFFECT_FILE_NAME = "smote-effect.csv"
CLASS_METRICS_FILE_NAME = "class-metrics.csv"
ERROR_COMPOSITION_FILE_NAME = "error-composition.csv"
PREDICTION_SUMMARIES_FILE_NAME = "prediction-summaries.csv"
KEY_FINDINGS_FILE_NAME = "key-findings.csv"
MODEL_COMPARISON_PLOT_FILE_NAME = "model-comparison.png"
ERROR_COMPOSITION_PLOT_FILE_NAME = "error-composition.png"
FLIP_RATE_PLOT_FILE_NAME = "flip-rate-against-displacement.png"
CONFUSION_MATRIX_PLOT_FILE_NAME = "confusion-matrix.png"
PREDICTION_TRANSITIONS_PLOT_FILE_NAME = "prediction-transitions.png"
```

Kode Program 5.5 Nama file data, prediksi, dan hasil evaluasi

Kode Program 5.6 menetapkan konstanta pengolahan data, yaitu tipe data fitur *float32* (*NUMERIC_DATA_TYPE*), nama kolom label dan kolom kelas, nama kolom jumlah baris, ukuran kelompok baris sebanyak 500.000 baris (*BATCH_SIZE*), dan *random state* 42 yang digunakan pada seluruh proses acak.

```python
NUMERIC_DATA_TYPE = "float32"
LABEL_COLUMN = "label"
CLASS_COLUMN = "class"
ROW_COUNT_COLUMN = "rows"
BATCH_SIZE = 500_000
RANDOM_STATE = 42
```

Kode Program 5.6 Konstanta pengolahan data

Kode Program 5.7 memuat fungsi pembacaan data per kelompok baris yang digunakan pada tahap yang membaca seluruh data latih atau data uji. Fungsi *count_rows* menghitung jumlah baris sebuah *LazyFrame* tanpa memuat isinya. Fungsi *read_in_batches* membaca *LazyFrame* per kelompok baris menggunakan *slice*, yang diteruskan oleh *Polars* hingga ke pembaca Parquet sehingga hanya baris yang dibutuhkan yang dibaca, dan mengembalikan posisi awal setiap kelompok bersama datanya sambil menampilkan bilah kemajuan. Fungsi *load_features* memuat seluruh fitur ke satu matriks *float32* yang dialokasikan sebelumnya, dengan mengisi matriks tersebut kelompok demi kelompok, sehingga puncak penggunaan memori hanya sebesar matriks akhir ditambah satu kelompok (Subbab 4.8.1).

```python
def count_rows(dataset: pl.LazyFrame) -> int:
    return dataset.select(pl.len()).collect().item()

def read_in_batches(
    dataset: pl.LazyFrame,
    prefix: str = "Reading batches",
    batch_size: int = BATCH_SIZE,
) -> Iterator[tuple[int, pl.DataFrame]]:
    row_offsets = range(0, count_rows(dataset), batch_size)
    started = time.time()
    print_progress(0, len(row_offsets), prefix, started=started)
    for batch_index, row_offset in enumerate(row_offsets, start=1):
        yield row_offset, dataset.slice(row_offset, batch_size).collect()
        print_progress(batch_index, len(row_offsets), prefix, started=started)

def load_features(
    features: pl.LazyFrame,
    prefix: str = "Loading features",
    batch_size: int = BATCH_SIZE,
    numeric_data_type: str = NUMERIC_DATA_TYPE,
) -> pd.DataFrame:
    columns = features.collect_schema().names()
    values = np.empty((count_rows(features), len(columns)), dtype=numeric_data_type)
    for row_offset, batch in read_in_batches(features, prefix, batch_size):
        values[row_offset:row_offset + len(batch)] = batch.to_numpy()
    return pd.DataFrame(values, columns=columns)
```

Kode Program 5.7 Pembacaan data per kelompok baris

Kode Program 5.8 memuat fungsi pemrosesan paralel. Fungsi *create_tasks_list* membentuk daftar tugas *joblib.delayed* dari sebuah fungsi, satu tugas untuk setiap argumen pada daftar, dengan argumen tambahan yang sama untuk seluruh tugas. Fungsi *run_tasks_in_parallel* menjalankan daftar tugas tersebut menggunakan *backend* *loky* yang berbasis proses, dengan jumlah *thread* internal setiap proses dibatasi menjadi satu untuk mencegah *oversubscription*, dan menampilkan bilah kemajuan setiap kali satu tugas selesai. Hasil tugas dikembalikan sesuai urutan daftar tugas.

```python
def create_tasks_list(function: Callable, per_task_arguments: list, *shared_arguments: Any) -> list:
    tasks = []
    for per_task_argument in per_task_arguments:
        task = joblib.delayed(function)(per_task_argument, *shared_arguments)
        tasks.append(task)
    return tasks

def run_tasks_in_parallel(tasks: list, n_jobs: int = -1, prefix: str = "Running tasks"):
    started = time.time()
    print_progress(0, len(tasks), prefix, started=started)
    results = []
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        for result in joblib.Parallel(n_jobs=n_jobs, return_as="generator")(tasks):
            results.append(result)
            print_progress(len(results), len(tasks), prefix, started=started)
    return results
```

Kode Program 5.8 Fungsi pemrosesan paralel
