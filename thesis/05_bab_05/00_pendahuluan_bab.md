# **BAB 5 IMPLEMENTASI**

Bab ini menyajikan implementasi dari rancangan yang telah diuraikan pada Bab 4. Implementasi diwujudkan dalam bentuk kode program berbahasa Python yang dijalankan pada *Jupyter Notebook* (*mlids.ipynb*), dengan setiap tahap pada *pipeline*, mulai dari konversi dataset ke format Parquet hingga analisis hasil pengujian ketahanan, ditulis sebagai fungsi yang dapat dijalankan ulang secara berurutan. Pemecahan pekerjaan menjadi fungsi-fungsi kecil, yang masing-masing menangani satu berkas, satu potongan data, atau satu model, memudahkan pemrosesan paralel, pemeriksaan hasil antara pada setiap tahap, dan pengulangan tahap tertentu tanpa mengulang tahap lainnya.

Bab ini disusun sejajar dengan Bab 4. Setiap subbab pada bab ini memuat kode program dari rancangan pada subbab dengan nomor yang sama pada Bab 4, misalnya Subbab 5.7 memuat implementasi dari rancangan pada Subbab 4.7. Di dalam setiap subbab, kode program disusun mengikuti urutan langkah pada algoritma di Bab 4, dan setiap kode program didahului penjelasan mengenai langkah yang diimplementasikannya. Kode program disajikan sebagaimana pada *notebook*, termasuk nama variabel dan fungsi yang berbahasa Inggris. Kode yang berasal dari beberapa sel *notebook* digabung dalam satu kode program dengan sel-selnya dipisahkan oleh satu baris kosong, sehingga penjelasan yang menyebut sel pertama, sel kedua, atau sel terakhir merujuk pada urutan sel tersebut. Konstanta pengaturan, seperti lokasi folder, ukuran *chunk*, dan ruang pencarian hiperparameter, dikumpulkan pada awal setiap tahap sehingga dapat diubah tanpa mengubah logika fungsi. Fungsi bantu yang digunakan oleh beberapa tahap disajikan pada subbab tempat fungsi tersebut pertama kali digunakan, kemudian digunakan kembali pada subbab berikutnya tanpa ditulis ulang.

Pustaka yang digunakan dirangkum pada Tabel 5.1. Seluruh perintah impor beserta konfigurasi penggunaan GPU disajikan pada Kode Program 5.1. Variabel *use_cuda* menentukan apakah implementasi GPU dari pustaka *cuML* digunakan untuk *K Nearest-Neighbor*, *Support Vector Machine*, dan *Random Forest*; apabila bernilai *False*, implementasi *scikit-learn* dengan antarmuka yang sama digunakan pada CPU. Variabel yang sama juga menentukan perangkat komputasi *XGBoost* serta apakah *TensorFlow* dapat menggunakan GPU.

Tabel 5.1 Pustaka yang Digunakan pada Implementasi

| Kelompok | Pustaka | Kegunaan pada implementasi |
| :--- | :--- | :--- |
| Pengolahan data | *pandas* dan *NumPy* | Struktur data tabular, pembacaan CSV per *chunk*, pembacaan dan penulisan Parquet, serta operasi numerik |
| Pengolahan data | *Polars* | Pembacaan berkas Parquet secara *lazy*, agregasi dengan mesin *streaming*, dan pemuatan himpunan data untuk pelatihan |
| Pengolahan data | *PyArrow* | Mesin pembacaan dan penulisan Parquet dengan kompresi Snappy |
| Komputasi paralel | *joblib* | Pemrosesan berkas secara paralel dengan *backend* *loky* dan penyimpanan model |
| Praproses | *scikit-learn* | *train_test_split*, *LabelEncoder*, *StandardScaler*, dan *IncrementalPCA* |
| Praproses | *imbalanced-learn* | SMOTE |
| Model | *cuML* (RAPIDS) | *K Nearest-Neighbor*, *Support Vector Machine*, dan *Random Forest* pada GPU, dengan *scikit-learn* sebagai pengganti pada CPU |
| Model | *TensorFlow* dan *Keras* | *Logistic Regression* (*softmax regression*) |
| Model | *XGBoost* | *XGBoost* |
| Evaluasi | *scikit-learn* (*sklearn.metrics*) | Akurasi, presisi, *recall*, F1-*score*, dan *confusion matrix* |
| Visualisasi | *Matplotlib* | Kurva, diagram, dan peta panas (*heatmap*) |
| Pendukung | *CuPy* | Larik GPU dan pembebasan memori GPU |

Kode Program 5.1 memuat perintah impor yang dikelompokkan menurut kegunaannya, yaitu sistem, pustaka standar, serialisasi, komputasi paralel, pengolahan data, praproses, visualisasi, model, dan evaluasi. Pada bagian awal, *use_cuda* ditetapkan *True* dan, apabila direktori *include* CUDA tersedia pada lingkungan Python yang digunakan, variabel lingkungan *CUDA_PATH* diatur agar pustaka berbasis GPU dapat menemukannya. Impor kelas model berada di dalam percabangan *use_cuda*, dan konfigurasi *TensorFlow* menyesuaikan penggunaan memori GPU atau menyembunyikan GPU.

```python
import os
import sys

use_cuda = True

if use_cuda:
    cuda_root = os.path.join(sys.prefix, "targets", "x86_64-linux")
    if os.path.isdir(os.path.join(cuda_root, "include")):
        os.environ.setdefault("CUDA_PATH", cuda_root)

import ast
import gc
import hashlib
import itertools
import math
import time
from typing import Any, cast
from contextlib import contextmanager

import json

import joblib
from joblib import Parallel, delayed, parallel_config

import numpy as np
import pandas as pd
import polars as pl

from imblearn.over_sampling import SMOTE
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import matplotlib
import matplotlib.pyplot as plt

if use_cuda:
    from cuml.neighbors import KNeighborsClassifier, NearestNeighbors
else:
    from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

if use_cuda:
    from cuml.svm import SVC
else:
    from sklearn.svm import SVC

if use_cuda:
    from cuml.ensemble import RandomForestClassifier
else:
    from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow import keras

if use_cuda:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    tf.config.set_visible_devices([], "GPU")

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
```

Kode Program 5.1 Impor pustaka dan konfigurasi penggunaan GPU
