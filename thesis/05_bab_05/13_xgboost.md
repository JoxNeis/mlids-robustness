## **5.13 Implementasi *XGBoost***

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.13, yaitu pembentukan *pipeline* *XGBoost*, pencarian hiperparameter, dan pembentukan model akhir. Kode program disusun mengikuti Langkah 1 hingga 4 pada algoritma di Subbab 4.13.

Fungsi *make_xgb_pipeline* pada Kode Program 5.71 mengimplementasikan Langkah 1. Fungsi ini menyusun *pipeline* praproses dengan *XGBClassifier* sebagai tahap terakhir bernama *xgb*. Perangkat komputasi ditetapkan *cuda* apabila *USE_CUDA* bernilai *True* dan *cpu* apabila tidak, sedangkan pengaturan tetap lainnya adalah metode *hist* dengan 128 *bin*, *subsample* 0,8, delapan proses paralel, dan *random state* 42. Fungsi objektif *multi:softprob* dipilih secara otomatis oleh *XGBClassifier* karena label memiliki lebih dari dua kelas.

```python
def make_xgb_pipeline(use_smote: bool = True, n_jobs: int = MODEL_N_JOBS) -> Pipeline:
    model = XGBClassifier(
        device="cuda" if USE_CUDA else "cpu",
        tree_method="hist",
        max_bin=128,
        subsample=0.8,
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
    )
    return make_model_pipeline("xgb", model, use_smote)
```

Kode Program 5.71 Pembentukan pipeline XGBoost

Kode Program 5.72 mengimplementasikan Langkah 2 hingga 4. Sel pertama menetapkan grid *XGB_PARAM_GRID*, yaitu dua nilai jumlah pohon (*xgb__n_estimators*), tiga nilai kedalaman maksimum (*xgb__max_depth*), dua nilai laju pembelajaran (*xgb__learning_rate*), dan dua nilai proporsi fitur per pohon (*xgb__colsample_bytree*). Komentar pada sel tersebut mencatat alasan jumlah pohon dicari sebagai hiperparameter, yaitu karena *GridSearchCV* tidak menyediakan data validasi untuk penghentian dini. Sel kedua dan ketiga menjalankan pencarian, menampilkan hasilnya, dan membentuk model akhir yang disimpan sebagai *xgb.pkl*.

```python
# GridSearchCV gives no validation set for early stopping, so the tree count is searched instead
XGB_PARAM_GRID = {
    "xgb__n_estimators": [200, 400],
    "xgb__max_depth": [6, 8, 10],
    "xgb__learning_rate": [0.1, 0.3],
    "xgb__colsample_bytree": [0.6, 1.0],
}

xgb_results = search_model(
    "xgb",
    make_xgb_pipeline,
    XGB_PARAM_GRID,
    train_features,
    train_labels,
    USE_SMOTE,
)
display(xgb_results.loc[0, "params"])
display(summarize_grid_search(xgb_results).head(10))

save_best_fold_model(
    "xgb",
    make_xgb_pipeline,
    xgb_results,
    train_features,
    train_labels,
    USE_SMOTE,
)
```

Kode Program 5.72 Pencarian hiperparameter dan pembentukan model akhir XGBoost
