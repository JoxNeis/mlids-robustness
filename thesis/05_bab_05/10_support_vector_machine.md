## **5.10 Implementasi *Support Vector Machine* (SVM)**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.10, yaitu pembentukan *pipeline* SVM, pencarian hyperparamater pada SVM berkernel *rbf* dan SVM linear, serta pembentukan model akhir. Kode program disusun mengikuti Langkah 1 hingga 4 pada algoritma di Subbab 4.10.

Fungsi *make_svm_pipeline* pada Kode Program 5.65 mengimplementasikan Langkah 1. Fungsi ini menyusun *pipeline* praproses dengan *SVC* sebagai tahap terakhir bernama *svm*. *SVC* menggunakan pengaturan bawaan *scikit-learn*, yaitu kernel *rbf*, toleransi konvergensi 10⁻³, iterasi yang tidak dibatasi, dan tanpa estimasi peluang. Fungsi ini tidak menerima parameter jumlah proses paralel karena *SVC* tidak mendukung pelatihan paralel.

```python
def make_svm_pipeline(use_smote: bool = True) -> Pipeline:
    return make_model_pipeline("svm", SVC(), use_smote)
```

Kode Program 5.65 Pembentukan pipeline SVM

Kode Program 5.66 mengimplementasikan Langkah 2 hingga 4. Grid *SVM_PARAM_GRID* berupa daftar dua kamus. Kamus pertama mencari kernel *rbf* dengan tiga nilai *C* dan dua nilai *gamma* pada tahap *svm*. Kamus kedua mengganti seluruh tahap *svm* dengan objek *LinearSVC* berfungsi *loss hinge* dan *random state* 42, kemudian mencari tiga nilai *C* pada objek pengganti tersebut. Dengan cara ini, kedua jenis SVM dinilai dalam satu pencarian. Sel kedua dan ketiga menjalankan pencarian, menampilkan hasilnya, dan membentuk model akhir yang disimpan sebagai *svm.pkl*.

```python
SVM_PARAM_GRID = [
    {"svm__kernel": ["rbf"], "svm__C": [0.1, 1.0, 10.0], "svm__gamma": ["scale", 0.1]},
    {"svm": [LinearSVC(loss="hinge", random_state=RANDOM_STATE)], "svm__C": [0.1, 1.0, 10.0]},
]

svm_results = search_model(
    "svm",
    make_svm_pipeline,
    SVM_PARAM_GRID,
    train_features,
    train_labels,
    USE_SMOTE,
)
display(svm_results.loc[0, "params"])
display(summarize_grid_search(svm_results).head(10))

save_best_fold_model(
    "svm",
    make_svm_pipeline,
    svm_results,
    train_features,
    train_labels,
    USE_SMOTE,
)
```

Kode Program 5.66 Pencarian hyperparamater dan pembentukan model akhir SVM
