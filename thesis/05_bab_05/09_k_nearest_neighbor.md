## **5.9 Implementasi *K Nearest-Neighbor* (KNN)**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.9, yaitu pembentukan *pipeline* KNN, pencarian hiperparameter, dan pembentukan model akhir. Kode program disusun mengikuti Langkah 1 hingga 4 pada algoritma di Subbab 4.9 dan menggunakan fungsi pencarian pada Subbab 5.8.

Fungsi *make_knn_pipeline* pada Kode Program 5.63 mengimplementasikan Langkah 1. Fungsi ini menyusun *pipeline* praproses dengan *KNeighborsClassifier* sebagai tahap terakhir bernama *knn*. Model menggunakan pengaturan bawaan *scikit-learn*, yaitu pembobotan *uniform* dan pemilihan metode pencarian tetangga secara otomatis, dengan delapan proses paralel (*n_jobs=MODEL_N_JOBS*).

```python
def make_knn_pipeline(use_smote: bool = True, n_jobs: int = MODEL_N_JOBS) -> Pipeline:
    return make_model_pipeline("knn", KNeighborsClassifier(n_jobs=n_jobs), use_smote)
```

Kode Program 5.63 Pembentukan pipeline KNN

Kode Program 5.64 mengimplementasikan Langkah 2 hingga 4. Sel pertama menetapkan grid hiperparameter *KNN_PARAM_GRID*, yaitu tujuh nilai jumlah tetangga (*knn__n_neighbors*) dan tiga ukuran jarak (*knn__metric*). Sel kedua menjalankan pencarian dengan *search_model* pada data latih sesuai pengaturan *USE_SMOTE*, kemudian menampilkan kombinasi terbaik dan sepuluh kombinasi teratas. Sel terakhir membentuk model akhir dari pembagian terbaik dengan *save_best_fold_model* dan menyimpannya sebagai *knn.pkl*.

```python
KNN_PARAM_GRID = {
    "knn__n_neighbors": [1, 3, 5, 7, 9, 11, 13],
    "knn__metric": ["manhattan", "euclidean", "cosine"],
}

knn_results = search_model(
    "knn",
    make_knn_pipeline,
    KNN_PARAM_GRID,
    train_features,
    train_labels,
    USE_SMOTE,
)
display(knn_results.loc[0, "params"])
display(summarize_grid_search(knn_results).head(10))

save_best_fold_model(
    "knn",
    make_knn_pipeline,
    knn_results,
    train_features,
    train_labels,
    USE_SMOTE,
)
```

Kode Program 5.64 Pencarian hiperparameter dan pembentukan model akhir KNN
