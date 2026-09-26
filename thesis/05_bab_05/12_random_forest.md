## **5.12 Implementasi *Random Forest* (RF)**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.12, yaitu pemilihan hiperparameter, pelatihan, dan prediksi model *Random Forest*. Kode program disusun mengikuti Langkah 1 hingga 8 pada algoritma di Subbab 4.12 dan menggunakan fungsi bantu pada Subbab 5.9.

Kode Program 5.96 mendefinisikan subfolder model dan ruang pencarian, yaitu jumlah pohon 100 dan 300, kedalaman maksimum 12, 16, dan 24, serta *max_features* *sqrt* dan *log2*. Konstanta lainnya adalah ukuran subsampel pencarian sebanyak 1.000.000 baris latih dan 200.000 baris validasi, ukuran data pelatihan akhir (*RF_TRAIN_SAMPLES* bernilai *None*, yaitu seluruh data latih), kriteria pemisahan *gini*, jumlah *bin* 128, dan jumlah *CUDA stream* 4. Fungsi *get_rf_name* membentuk nama berkas model dari hiperparameternya.

```python
RF_SUBFOLDER = "RF"

list_of_n_estimators = [100, 300]
list_of_max_depth = [12, 16, 24]
list_of_max_features = ["sqrt", "log2"]

RF_SEARCH_TRAIN_SAMPLES = 1_000_000
RF_SEARCH_DEV_SAMPLES = 200_000
RF_TRAIN_SAMPLES = None

RF_SPLIT_CRITERION = "gini"
RF_N_BINS = 128
RF_N_STREAMS = 4

def get_rf_name(n_estimators: int, max_depth: int, max_features) -> str:
    return f"rf-n-{n_estimators}-d-{max_depth}-f-{max_features}.pkl"
```

Kode Program 5.96 Pengaturan model Random Forest

Fungsi *build_rf* pada Kode Program 5.97 membentuk *RandomForestClassifier* dari *cuML* dengan parameter *split_criterion*, *n_bins*, *n_streams*, dan *random_state* apabila GPU digunakan, atau dari *scikit-learn* dengan parameter *criterion*, *random_state*, dan *n_jobs=-1* apabila tidak (Langkah 3). Fungsi *fit_and_score_rf* melatih hutan pada subsampel latih, memprediksi subsampel validasi, dan mengembalikan metrik bersama waktu pelatihan dan waktu prediksi (Langkah 3 dan 4).

```python
def build_rf(
    n_estimators: int,
    max_depth: int,
    max_features,
    split_criterion: str = RF_SPLIT_CRITERION,
    n_bins: int = RF_N_BINS,
    n_streams: int = RF_N_STREAMS,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    if use_cuda:
        return RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            max_features=max_features,
            split_criterion=split_criterion,
            n_bins=n_bins,
            n_streams=n_streams,
            random_state=random_state,
        )

    return RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        max_features=max_features,
        criterion=split_criterion,
        random_state=random_state,
        n_jobs=-1,
    )

def fit_and_score_rf(
    n_estimators, max_depth, max_features, train_x, train_y, dev_x, dev_y
) -> dict:
    model = build_rf(n_estimators, max_depth, max_features)

    fit_started = time.time()
    with step(f"fitting {n_estimators} trees on {train_x.shape[0]:,} rows"):
        model.fit(train_x, train_y)
    fit_seconds = time.time() - fit_started

    predict_started = time.time()
    with step(f"predicting {dev_x.shape[0]:,} dev rows"):
        pred = to_numpy(model.predict(dev_x)).astype(np.int32)
    predict_seconds = time.time() - predict_started

    scores = evaluate(pred=pred, true=dev_y).iloc[0].to_dict()
    measurements = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        **scores,
    }

    del model
    free_gpu_memory()
    return measurements
```

Kode Program 5.97 Pembentukan dan penilaian satu hutan

Fungsi *search_rf* pada Kode Program 5.98 menyusun grid dengan *build_grid* (Langkah 1), mengambil subsampel berstrata (Langkah 2), dan menjalankan *run_search* dengan *fit_and_score_rf* (Langkah 3 dan 4). Sel kedua menjalankan pencarian dan menyimpan hasilnya ke *hyperparameter-search/rf-search.csv*. Sel ketiga memilih kombinasi teratas dan menetapkannya pada variabel *RF_N_ESTIMATORS*, *RF_MAX_DEPTH*, dan *RF_MAX_FEATURES* (Langkah 5).

```python
def search_rf(
    train_x,
    train_y,
    dev_x,
    dev_y,
    combinations: list[tuple] | None = None,
    train_samples: int | None = RF_SEARCH_TRAIN_SAMPLES,
    dev_samples: int | None = RF_SEARCH_DEV_SAMPLES,
    dev_min_per_class: int = SEARCH_DEV_MIN_PER_CLASS,
    random_state: int = RANDOM_STATE,
    score_column: str = SCORE_COLUMN,
) -> pd.DataFrame:
    if combinations is None:
        combinations = build_grid(list_of_n_estimators, list_of_max_depth, list_of_max_features)

    print(f"Random Forest search over {len(combinations)} combinations")
    free_gpu_memory()

    print("Preparing search subsamples...")
    search_data = (
        *stratified_subsample(train_x, train_y, train_samples, random_state),
        *stratified_subsample(dev_x, dev_y, dev_samples, random_state, min_per_class=dev_min_per_class),
    )

    return run_search(
        combinations,
        ("n_estimators", "max_depth", "max_features"),
        fit_and_score_rf,
        search_data,
        score_column,
    )

rf_search_results = search_rf(train_x, train_y, dev_x, dev_y)
save_search_results(rf_search_results, "rf-search.csv", PATH_FOLDER_SEARCH_RESULT)
rf_search_results

best_rf = get_best_hyperparameter(rf_search_results, SCORE_COLUMN)
RF_N_ESTIMATORS = int(best_rf["n_estimators"])
RF_MAX_DEPTH = int(best_rf["max_depth"])
RF_MAX_FEATURES = best_rf["max_features"]
```

Kode Program 5.98 Pencarian hiperparameter Random Forest dan pemilihan konfigurasi terbaik

Kode Program 5.99 mengimplementasikan Langkah 6 dan 7. Nilai hiperparameter terpilih ditetapkan kembali secara eksplisit pada *RF_N_ESTIMATORS*, *RF_MAX_DEPTH*, dan *RF_MAX_FEATURES* sesuai hasil pencarian, sehingga pelatihan dapat dijalankan tanpa mengulang pencarian. Fungsi *train_rf* mengambil data latih, seluruhnya karena *RF_TRAIN_SAMPLES* bernilai *None*, melatih hutan dengan konfigurasi terpilih, dan menyimpan modelnya dengan nama yang memuat hiperparameternya. Sel terakhir menjalankannya.

```python
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 24
RF_MAX_FEATURES = "log2"

def train_rf(
    train_x,
    train_y,
    n_estimators: int,
    max_depth: int,
    max_features,
    train_samples: int | None = RF_TRAIN_SAMPLES,
    random_state: int = RANDOM_STATE,
    subfolder: str = RF_SUBFOLDER,
    folder_name: str = PATH_FOLDER_MODEL,
) -> str:
    print(
        f"Training Random Forest: n_estimators={n_estimators} "
        f"max_depth={max_depth} max_features={max_features}"
    )

    print("Preparing training data...")
    features, labels = stratified_subsample(train_x, train_y, train_samples, random_state)
    n_rows, n_features = features.shape

    model = build_rf(n_estimators, max_depth, max_features, random_state=random_state)
    with step(f"fitting {n_estimators} trees on {n_rows:,} rows x {n_features} features"):
        model.fit(features, labels)

    with step("saving model"):
        file_path = dump_trained_model(
            model, get_rf_name(n_estimators, max_depth, max_features), subfolder, folder_name
        )

    print(f"Saved model to {file_path}.")
    return file_path

train_rf(train_x, train_y, RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MAX_FEATURES)
```

Kode Program 5.99 Pelatihan model Random Forest akhir

Fungsi *rf_predict* pada Kode Program 5.100 mengimplementasikan Langkah 8. Fungsi ini memuat model, memprediksi data per potongan dengan *predict_in_chunks* ke folder *prediction-result/RF*, kemudian melepas model dan membebaskan memori GPU. Sel-sel berikutnya menetapkan nama model terpilih dan memprediksi data validasi dan data uji.

```python
def rf_predict(
    model_name: str,
    x,
    split_name: str,
    subfolder: str = RF_SUBFOLDER,
    chunk_size: int = PREDICT_CHUNK_SIZE,
) -> np.ndarray:
    with step(f"loading {model_name}"):
        model = load_trained_model(model_name, subfolder, PATH_FOLDER_MODEL)

    predictions = predict_in_chunks(
        lambda rows: to_numpy(model.predict(to_features(rows))),
        x,
        get_split_folder(os.path.join(PATH_FOLDER_PREDICTION, subfolder), split_name),
        "rf",
        chunk_size,
    )

    del model
    free_gpu_memory()
    return predictions

rf_model_name = get_rf_name(RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MAX_FEATURES)
rf_predict(rf_model_name, dev_x, "dev")

rf_predict(rf_model_name, test_x, "test")
```

Kode Program 5.100 Prediksi dengan model Random Forest

Kode Program 5.101 menghitung metrik pada data validasi dan data uji dari prediksi yang telah tersimpan, kemudian menyusun laporan klasifikasi per kelas dan *confusion matrix* data uji.

```python
rf_dev_folder = get_split_folder(os.path.join(PATH_FOLDER_PREDICTION, RF_SUBFOLDER), "dev")
rf_test_folder = get_split_folder(os.path.join(PATH_FOLDER_PREDICTION, RF_SUBFOLDER), "test")

pd.concat(
    [
        get_evaluation_results(rf_dev_folder, dev_y).assign(split="dev"),
        get_evaluation_results(rf_test_folder, test_y).assign(split="test"),
    ],
    ignore_index=True,
)

rf_test_pred = load_prediction_chunks(rf_test_folder)
get_classification_report(pred=rf_test_pred, true=to_labels(test_y), class_names=CLASS_NAMES)

get_confusion_matrix(pred=rf_test_pred, true=to_labels(test_y), class_names=CLASS_NAMES)
```

Kode Program 5.101 Laporan hasil evaluasi model Random Forest
