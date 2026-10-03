## **5.7 Implementasi *Pipeline* Praproses**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.7, yaitu penyusunan *pipeline* yang terdiri atas konversi tipe data, standardisasi, PCA, SMOTE yang bersifat opsional, dan algoritma klasifikasi. Seluruh tahap disusun menggunakan kelas yang telah tersedia pada *scikit-learn* dan *imbalanced-learn*, sehingga statistik setiap tahap dipelajari pada saat *pipeline* dilatih dan diterapkan pada saat *pipeline* digunakan untuk prediksi.

Kode Program 5.52 menetapkan pengaturan *pipeline*. Konstanta *PCA_VARIANCE_THRESHOLD* menetapkan ambang varians kumulatif PCA sebesar 0,95 sesuai Persamaan 4.4, dan *MODEL_N_JOBS* menetapkan jumlah proses paralel algoritma klasifikasi sebanyak delapan. Fungsi *_to_float32* mengonversi seluruh fitur menjadi *float32* (Subbab 4.7.1). Fungsi ini didefinisikan sebagai fungsi bernama, bukan fungsi anonim, agar *pipeline* yang memuatnya dapat disimpan dan dimuat kembali dengan *joblib*.

```python
PCA_VARIANCE_THRESHOLD = 0.95
MODEL_N_JOBS = 8

def _to_float32(x:pd.DataFrame):
    return x.astype(np.float32)
```

Kode Program 5.52 Pengaturan pipeline dan fungsi konversi tipe data

Kelas *ProgressPipeline* pada Kode Program 5.53 merupakan turunan dari kelas *Pipeline* pada *imbalanced-learn* yang menambahkan pelaporan kemajuan. Fungsi *fit* menerima objek *FitProgress* (pendahuluan Bab 5) sebagai parameter tambahan, menjalankan pelatihan *pipeline* seperti biasa, kemudian memperbarui jumlah pelatihan yang selesai atau yang gagal. Galat pada pelatihan tetap diteruskan setelah dicatat, sehingga pencarian hyperparamater dapat mencatat kombinasi tersebut sebagai kombinasi yang gagal (Subbab 5.8).

```python
class ProgressPipeline(Pipeline):
    def fit(self, X, y=None, progress: FitProgress | None = None, **params):
        try:
            super().fit(X, y, **params)
        except Exception:
            if progress is not None:
                progress.update(failed=True)
            raise
        if progress is not None:
            progress.update()
        return self
```

Kode Program 5.53 Pipeline dengan pelaporan kemajuan pelatihan

Kode Program 5.54 menyusun tahap-tahap *pipeline*. Fungsi *make_preprocessing_steps* mengembalikan tiga tahap praproses, yaitu *cast* berupa *FunctionTransformer* yang menjalankan *_to_float32*, *scaler* berupa *StandardScaler* yang menerapkan Persamaan 4.3, dan *pca* berupa *PCA* dengan jumlah komponen 0,95, yang berarti *scikit-learn* memilih jumlah komponen terkecil yang varians kumulatifnya melebihi 95% sesuai Persamaan 4.4, dengan *random state* 42. Fungsi *make_model_pipeline* menyusun tahap praproses tersebut, menambahkan tahap *smote* berupa *SMOTE* dengan *random state* 42 apabila pengaturan dengan SMOTE digunakan, kemudian menambahkan algoritma klasifikasi sebagai tahap terakhir. Tahap SMOTE ditempatkan setelah PCA sesuai Subbab 4.7.4, dan karena *SMOTE* merupakan *sampler*, *Pipeline* dari *imbalanced-learn* hanya menjalankannya pada saat pelatihan.

```python
def make_preprocessing_steps(pca_variance_threshold: float = PCA_VARIANCE_THRESHOLD) -> list[tuple[str, Any]]:
    return [
        ("cast", FunctionTransformer(_to_float32)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_variance_threshold, random_state=RANDOM_STATE)),
    ]

def make_model_pipeline(model_name: str, model: Any, use_smote: bool = True) -> Pipeline:
    steps = make_preprocessing_steps()
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append((model_name, model))
    return ProgressPipeline(steps)
```

Kode Program 5.54 Penyusunan tahap praproses dan pipeline model

Setiap algoritma klasifikasi memiliki fungsi pembentuk *pipeline* tersendiri yang memanggil *make_model_pipeline* dengan nama tahap dan objek modelnya, yaitu *make_knn_pipeline*, *make_svm_pipeline*, *make_rf_pipeline*, *make_logreg_pipeline*, dan *make_xgb_pipeline*, yang disajikan pada Subbab 5.9 hingga 5.13. Nama tahap tersebut, misalnya *knn* atau *svm*, digunakan sebagai awalan nama hyperparamater pada grid, misalnya *knn__n_neighbors*.
