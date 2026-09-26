## **5.11 Implementasi *Support Vector Machine* (SVM)**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.11, yaitu pemilihan hiperparameter, pelatihan, dan prediksi model *Support Vector Machine*. Kode program disusun mengikuti Langkah 1 hingga 8 pada algoritma di Subbab 4.11 dan menggunakan fungsi bantu pada Subbab 5.9.

Kode Program 5.89 mendefinisikan subfolder model dan ruang pencarian, yaitu kernel *rbf* dan *linear*, nilai *C* sebesar 0,1, 1,0, dan 10,0, serta nilai *gamma* *scale* dan 0,1. Konstanta lainnya adalah ukuran subsampel pencarian sebanyak 50.000 baris latih dan 100.000 baris validasi, ukuran subsampel pelatihan akhir sebanyak 300.000 baris, ukuran *kernel cache* 2.048 MB, toleransi konvergensi 10⁻³, dan tanpa batas iterasi (*SVM_MAX_ITER = -1*). Fungsi *get_svm_name* membentuk nama berkas model dari hiperparameternya.

```python
SVM_SUBFOLDER = "SVM"

list_of_kernels = ["rbf", "linear"]
list_of_c = [0.1, 1.0, 10.0]
list_of_gamma = ["scale", 0.1]

SVM_SEARCH_TRAIN_SAMPLES = 50_000
SVM_SEARCH_DEV_SAMPLES = 100_000
SVM_TRAIN_SAMPLES = 300_000

SVM_CACHE_SIZE = 2048
SVM_TOLERANCE = 1e-3
SVM_MAX_ITER = -1

def get_svm_name(kernel: str, c: float, gamma) -> str:
    return f"svm-k-{kernel}-c-{c}-g-{gamma}.pkl"
```

Kode Program 5.89 Pengaturan model SVM

Fungsi *build_svm_grid* pada Kode Program 5.90 mengimplementasikan Langkah 1. Kernel *rbf* dipasangkan dengan setiap nilai *gamma*, sedangkan kernel *linear* hanya dipasangkan dengan nilai *scale* karena tidak memiliki parameter *gamma*. Fungsi *build_svm* membentuk objek *SVC* dengan kernel, *C*, *gamma*, ukuran *kernel cache*, toleransi, dan batas iterasi yang telah ditetapkan (Langkah 3).

```python
def build_svm_grid(kernels: list[str], list_c: list[float], list_gamma: list) -> list[tuple]:
    combinations = []
    for kernel, c in itertools.product(kernels, list_c):
        if kernel == "linear":
            combinations.append((kernel, c, "scale"))
            continue
        combinations.extend((kernel, c, gamma) for gamma in list_gamma)
    return combinations

def build_svm(
    kernel: str,
    c: float,
    gamma,
    cache_size: int = SVM_CACHE_SIZE,
    tolerance: float = SVM_TOLERANCE,
    max_iter: int = SVM_MAX_ITER,
) -> SVC:
    return SVC(
        C=float(c),
        kernel=kernel,
        gamma=gamma,
        cache_size=cache_size,
        tol=tolerance,
        max_iter=max_iter,
    )
```

Kode Program 5.90 Penyusunan grid dan pembentukan model SVM

Kode Program 5.91 mengimplementasikan Langkah 3 dan 4. Fungsi *support_vector_count* mengembalikan jumlah *support vector* model dari atribut yang tersedia pada implementasi GPU maupun CPU. Fungsi *fit_and_score_svm* melatih SVM pada subsampel latih dengan pencatatan waktu, memprediksi subsampel validasi, dan mengembalikan metrik bersama waktu pelatihan dan jumlah *support vector*, kemudian membebaskan memori.

```python
def support_vector_count(model) -> float:
    try:
        n_support = getattr(model, "n_support_", None)
        if n_support is not None:
            return float(np.sum(to_numpy_2d(n_support)))

        support = getattr(model, "support_", None)
        if support is not None:
            return float(np.size(to_numpy_2d(support)))
    except Exception:
        pass
    return float("nan")

def fit_and_score_svm(kernel, c, gamma, train_x, train_y, dev_x, dev_y) -> dict:
    model = build_svm(kernel, c, gamma)

    fit_started = time.time()
    with step(f"fitting on {train_x.shape[0]:,} rows"):
        model.fit(train_x, train_y)
    fit_seconds = time.time() - fit_started

    with step(f"predicting {dev_x.shape[0]:,} dev rows"):
        pred = to_numpy(model.predict(dev_x))

    scores = evaluate(pred=pred, true=dev_y).iloc[0].to_dict()
    measurements = {"fit_seconds": fit_seconds, "n_support": support_vector_count(model), **scores}

    del model
    free_gpu_memory()
    return measurements
```

Kode Program 5.91 Penghitungan support vector dan penilaian satu kombinasi SVM

Fungsi *search_svm* pada Kode Program 5.92 menyusun grid (Langkah 1), mengambil subsampel berstrata (Langkah 2), dan menjalankan *run_search* dengan *fit_and_score_svm*, sehingga kombinasi yang gagal dicatat dan dilewati (Langkah 3 dan 4). Sel kedua menjalankan pencarian dan menyimpan hasilnya ke *hyperparameter-search/svm-search.csv*. Sel ketiga memilih kombinasi teratas dan menetapkannya pada variabel *SVM_KERNEL*, *SVM_C*, dan *SVM_GAMMA* (Langkah 5).

```python
def search_svm(
    train_x,
    train_y,
    dev_x,
    dev_y,
    combinations: list[tuple] | None = None,
    train_samples: int | None = SVM_SEARCH_TRAIN_SAMPLES,
    dev_samples: int | None = SVM_SEARCH_DEV_SAMPLES,
    dev_min_per_class: int = SEARCH_DEV_MIN_PER_CLASS,
    random_state: int = RANDOM_STATE,
    score_column: str = SCORE_COLUMN,
) -> pd.DataFrame:
    if combinations is None:
        combinations = build_svm_grid(list_of_kernels, list_of_c, list_of_gamma)

    print(f"SVM search over {len(combinations)} combinations")
    free_gpu_memory()

    print("Preparing search subsamples...")
    search_data = (
        *stratified_subsample(train_x, train_y, train_samples, random_state),
        *stratified_subsample(dev_x, dev_y, dev_samples, random_state, min_per_class=dev_min_per_class),
    )

    return run_search(
        combinations, ("kernel", "C", "gamma"), fit_and_score_svm, search_data, score_column
    )

svm_search_results = search_svm(train_x, train_y, dev_x, dev_y)
save_search_results(svm_search_results, "svm-search.csv", PATH_FOLDER_SEARCH_RESULT)
svm_search_results

best_svm = get_best_hyperparameter(svm_search_results, SCORE_COLUMN)
SVM_KERNEL = str(best_svm["kernel"])
SVM_C = float(best_svm["C"])
SVM_GAMMA = best_svm["gamma"]
```

Kode Program 5.92 Pencarian hiperparameter SVM dan pemilihan konfigurasi terbaik

Fungsi *train_svm* pada Kode Program 5.93 mengimplementasikan Langkah 6 dan 7. Fungsi ini mengambil subsampel berstrata sebanyak 300.000 baris dari data latih, melatih *SVC* dengan konfigurasi terpilih, mencetak jumlah *support vector*, dan menyimpan modelnya dengan nama yang memuat hiperparameternya. Sel kedua menjalankannya.

```python
def train_svm(
    train_x,
    train_y,
    kernel: str,
    c: float,
    gamma,
    train_samples: int | None = SVM_TRAIN_SAMPLES,
    random_state: int = RANDOM_STATE,
    subfolder: str = SVM_SUBFOLDER,
    folder_name: str = PATH_FOLDER_MODEL,
) -> str:
    print(f"Training SVM: kernel={kernel} C={c} gamma={gamma}")

    print("Preparing training subsample...")
    features, labels = stratified_subsample(train_x, train_y, train_samples, random_state)
    n_rows, n_features = features.shape

    model = build_svm(kernel, c, gamma)
    with step(f"fitting on {n_rows:,} rows x {n_features} features"):
        model.fit(features, labels)
    print(f"  {support_vector_count(model):,.0f} support vectors")

    with step("saving model"):
        file_path = dump_trained_model(
            model, get_svm_name(kernel, c, gamma), subfolder, folder_name
        )

    print(f"Saved model to {file_path}.")
    return file_path

train_svm(train_x, train_y, SVM_KERNEL, SVM_C, SVM_GAMMA)
```

Kode Program 5.93 Pelatihan model SVM akhir

Fungsi *svm_predict* pada Kode Program 5.94 mengimplementasikan Langkah 8. Fungsi ini memuat model, mencetak jumlah *support vector*-nya, memprediksi data per potongan dengan *predict_in_chunks* ke folder *prediction-result/SVM*, kemudian melepas model dan membebaskan memori GPU. Sel-sel berikutnya menetapkan nama model terpilih dan memprediksi data validasi dan data uji.

```python
def svm_predict(
    model_name: str,
    x,
    split_name: str,
    subfolder: str = SVM_SUBFOLDER,
    chunk_size: int = PREDICT_CHUNK_SIZE,
) -> np.ndarray:
    with step(f"loading {model_name}"):
        model = load_trained_model(model_name, subfolder, PATH_FOLDER_MODEL)
    print(f"  {support_vector_count(model):,.0f} support vectors")

    predictions = predict_in_chunks(
        lambda rows: to_numpy(model.predict(to_features(rows))),
        x,
        get_split_folder(os.path.join(PATH_FOLDER_PREDICTION, subfolder), split_name),
        "svm",
        chunk_size,
    )

    del model
    free_gpu_memory()
    return predictions

svm_model_name = get_svm_name(SVM_KERNEL, SVM_C, SVM_GAMMA)
svm_predict(svm_model_name, dev_x, "dev")

svm_predict(svm_model_name, test_x, "test")
```

Kode Program 5.94 Prediksi dengan model SVM

Kode Program 5.95 menghitung metrik pada data validasi dan data uji dari prediksi yang telah tersimpan, kemudian menyusun laporan klasifikasi per kelas dan *confusion matrix* data uji.

```python
svm_dev_folder = get_split_folder(os.path.join(PATH_FOLDER_PREDICTION, SVM_SUBFOLDER), "dev")
svm_test_folder = get_split_folder(os.path.join(PATH_FOLDER_PREDICTION, SVM_SUBFOLDER), "test")

pd.concat(
    [
        get_evaluation_results(svm_dev_folder, dev_y).assign(split="dev"),
        get_evaluation_results(svm_test_folder, test_y).assign(split="test"),
    ],
    ignore_index=True,
)

svm_test_pred = load_prediction_chunks(svm_test_folder)
get_classification_report(pred=svm_test_pred, true=to_labels(test_y), class_names=CLASS_NAMES)

get_confusion_matrix(pred=svm_test_pred, true=to_labels(test_y), class_names=CLASS_NAMES)
```

Kode Program 5.95 Laporan hasil evaluasi model SVM
