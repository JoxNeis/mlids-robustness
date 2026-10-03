## **5.12 Implementasi *Logistic Regression* (*Softmax Regression*)**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.12, yaitu pembentukan *pipeline* *Logistic Regression*, pencarian hiperparameter, dan pembentukan model akhir. Kode program disusun mengikuti Langkah 1 hingga 4 pada algoritma di Subbab 4.12.

Fungsi *make_logreg_pipeline* pada Kode Program 5.69 mengimplementasikan Langkah 1. Fungsi ini menyusun *pipeline* praproses dengan *LogisticRegression* sebagai tahap terakhir bernama *logreg*, dengan optimasi *saga* karena hanya *saga* yang mendukung penalti L1 maupun L2 pada model multinomial, serta *random state* 42. Parameter *n_jobs* diteruskan agar seragam dengan algoritma lainnya, meskipun pelatihan multinomial tidak menggunakannya. Pengaturan lainnya menggunakan nilai bawaan *scikit-learn*, yaitu jumlah iterasi maksimum 100 dan toleransi konvergensi 10⁻⁴.

```python
def make_logreg_pipeline(use_smote: bool = True, n_jobs: int = MODEL_N_JOBS) -> Pipeline:
    model = LogisticRegression(solver="saga", random_state=RANDOM_STATE, n_jobs=n_jobs)  # saga takes both l1 and l2
    return make_model_pipeline("logreg", model, use_smote)
```

Kode Program 5.69 Pembentukan pipeline Logistic Regression

Kode Program 5.70 mengimplementasikan Langkah 2 hingga 4. Sel pertama menetapkan grid *LOGREG_PARAM_GRID*, yaitu empat nilai *C* (*logreg__C*) dan dua jenis penalti (*logreg__penalty*). Sel kedua dan ketiga menjalankan pencarian, menampilkan hasilnya, dan membentuk model akhir yang disimpan sebagai *logreg.pkl*.

```python
LOGREG_PARAM_GRID = {
    "logreg__C": [0.01, 0.1, 1.0, 10.0],
    "logreg__penalty": ["l1", "l2"],
}

logreg_results = search_model(
    "logreg",
    make_logreg_pipeline,
    LOGREG_PARAM_GRID,
    train_features,
    train_labels,
    USE_SMOTE,
)
display(logreg_results.loc[0, "params"])
display(summarize_grid_search(logreg_results).head(10))

save_best_fold_model(
    "logreg",
    make_logreg_pipeline,
    logreg_results,
    train_features,
    train_labels,
    USE_SMOTE,
)
```

Kode Program 5.70 Pencarian hiperparameter dan pembentukan model akhir Logistic Regression
