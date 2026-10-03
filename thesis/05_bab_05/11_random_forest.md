## **5.11 Implementasi *Random Forest* (RF)**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.11, yaitu pembentukan *pipeline* *Random Forest*, pencarian hyperparamater, dan pembentukan model akhir. Kode program disusun mengikuti Langkah 1 hingga 4 pada algoritma di Subbab 4.11.

Fungsi *make_rf_pipeline* pada Kode Program 5.67 mengimplementasikan Langkah 1. Fungsi ini menyusun *pipeline* praproses dengan *RandomForestClassifier* sebagai tahap terakhir bernama *rf*, dengan *random state* 42 dan delapan proses paralel. Pengaturan lainnya menggunakan nilai bawaan *scikit-learn*, yaitu kriteria pemisahan *gini* dan pengambilan sampel *bootstrap* untuk setiap pohon.

```python
def make_rf_pipeline(use_smote: bool = True, n_jobs: int = MODEL_N_JOBS) -> Pipeline:
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=n_jobs)
    return make_model_pipeline("rf", model, use_smote)
```

Kode Program 5.67 Pembentukan pipeline Random Forest

Kode Program 5.68 mengimplementasikan Langkah 2 hingga 4. Sel pertama menetapkan grid *RF_PARAM_GRID*, yaitu dua nilai jumlah pohon (*rf__n_estimators*), tiga nilai kedalaman maksimum (*rf__max_depth*), dan dua nilai jumlah fitur per percabangan (*rf__max_features*). Sel kedua dan ketiga menjalankan pencarian, menampilkan hasilnya, dan membentuk model akhir yang disimpan sebagai *rf.pkl*.

```python
RF_PARAM_GRID = {
    "rf__n_estimators": [100, 300],
    "rf__max_depth": [12, 16, 24],
    "rf__max_features": ["sqrt", "log2"],
}

rf_results = search_model(
    "rf",
    make_rf_pipeline,
    RF_PARAM_GRID,
    train_features,
    train_labels,
    USE_SMOTE,
)
display(rf_results.loc[0, "params"])
display(summarize_grid_search(rf_results).head(10))

save_best_fold_model(
    "rf",
    make_rf_pipeline,
    rf_results,
    train_features,
    train_labels,
    USE_SMOTE,
)
```

Kode Program 5.68 Pencarian hyperparamater dan pembentukan model akhir Random Forest
