## **5.14 Implementasi *XGBoost***

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.14, yaitu pemilihan hiperparameter, pelatihan, dan prediksi model *XGBoost*. Kode program disusun mengikuti Langkah 1 hingga 8 pada algoritma di Subbab 4.14 dan menggunakan fungsi bantu pada Subbab 5.9.

Kode Program 5.108 mendefinisikan subfolder model dan ruang pencarian, yaitu kedalaman maksimum 6, 8, dan 10, laju pembelajaran 0,1 dan 0,3, serta *colsample_bytree* 0,6 dan 1,0. Konstanta lainnya adalah ukuran subsampel pencarian sebanyak 1.000.000 baris latih dan 200.000 baris validasi, ukuran data pelatihan akhir (seluruh data latih), dan pengaturan tetap *XGBoost*, yaitu perangkat komputasi *cuda* atau *cpu* menurut *use_cuda*, metode pohon *hist*, jumlah putaran maksimum 400, kesabaran penghentian dini 20 putaran, 128 *bin*, *subsample* 0,8, dan *min_child_weight* 1,0. Fungsi *get_xgb_name* membentuk nama berkas model dari hiperparameternya.

```python
XGB_SUBFOLDER = "XGB"

list_of_xgb_max_depth = [6, 8, 10]
list_of_xgb_learning_rates = [0.1, 0.3]
list_of_xgb_colsample_bytree = [0.6, 1.0]

XGB_SEARCH_TRAIN_SAMPLES = 1_000_000
XGB_SEARCH_DEV_SAMPLES = 200_000
XGB_TRAIN_SAMPLES = None

XGB_DEVICE = "cuda" if use_cuda else "cpu"
XGB_TREE_METHOD = "hist"
XGB_MAX_ROUNDS = 400
XGB_EARLY_STOPPING_ROUNDS = 20
XGB_MAX_BIN = 128
XGB_SUBSAMPLE = 0.8
XGB_MIN_CHILD_WEIGHT = 1.0

def get_xgb_name(max_depth: int, learning_rate: float, colsample_bytree: float) -> str:
    return f"xgb-d-{max_depth}-lr-{learning_rate}-c-{colsample_bytree}.json"
```

Kode Program 5.108 Pengaturan model XGBoost

Fungsi *build_xgb* pada Kode Program 5.109 membentuk *XGBClassifier* dengan objektif *multi:softprob*, metrik evaluasi *mlogloss*, dan pengaturan tetap yang didefinisikan pada Kode Program 5.108. Fungsi *to_xgb_features* mengonversi fitur menjadi larik *CuPy* apabila GPU digunakan. Fungsi *fit_and_score_xgb* mengimplementasikan Langkah 3 dan 4. Fungsi ini melatih *booster* dengan *eval_set* berupa subsampel validasi dan *early_stopping_rounds*, menghitung jumlah pohon yang digunakan dari iterasi terbaik, memprediksi subsampel validasi, dan mengembalikan metrik bersama jumlah pohon serta waktu pelatihan dan prediksi.

```python
def build_xgb(
    max_depth: int,
    learning_rate: float,
    colsample_bytree: float,
    n_estimators: int = XGB_MAX_ROUNDS,
    early_stopping_rounds: int | None = None,
    subsample: float = XGB_SUBSAMPLE,
    min_child_weight: float = XGB_MIN_CHILD_WEIGHT,
    max_bin: int = XGB_MAX_BIN,
    tree_method: str = XGB_TREE_METHOD,
    device: str = XGB_DEVICE,
    random_state: int = RANDOM_STATE,
) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        colsample_bytree=float(colsample_bytree),
        subsample=float(subsample),
        min_child_weight=float(min_child_weight),
        max_bin=int(max_bin),
        tree_method=tree_method,
        device=device,
        objective="multi:softprob",
        eval_metric="mlogloss",
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
    )

def to_xgb_features(x):
    features = to_features(x)
    if not use_cuda:
        return features

    import cupy as cp

    return cp.asarray(features)

def fit_and_score_xgb(
    max_depth,
    learning_rate,
    colsample_bytree,
    train_x,
    train_y,
    dev_x,
    dev_y,
    max_rounds: int = XGB_MAX_ROUNDS,
    early_stopping_rounds: int = XGB_EARLY_STOPPING_ROUNDS,
) -> dict:
    model = build_xgb(
        max_depth,
        learning_rate,
        colsample_bytree,
        n_estimators=max_rounds,
        early_stopping_rounds=early_stopping_rounds,
    )

    fit_started = time.time()
    with step(f"boosting up to {max_rounds} rounds on {train_x.shape[0]:,} rows"):
        model.fit(train_x, train_y, eval_set=[(dev_x, dev_y)], verbose=False)
    fit_seconds = time.time() - fit_started

    best_iteration = getattr(model, "best_iteration", None)
    n_trees_used = max_rounds if best_iteration is None else int(best_iteration) + 1

    predict_started = time.time()
    with step(f"predicting {dev_x.shape[0]:,} dev rows"):
        pred = to_numpy(model.predict(to_xgb_features(dev_x))).astype(np.int32)
    predict_seconds = time.time() - predict_started

    scores = evaluate(pred=pred, true=dev_y).iloc[0].to_dict()
    measurements = {
        "n_trees_used": n_trees_used,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        **scores,
    }

    del model
    free_gpu_memory()
    return measurements
```

Kode Program 5.109 Pembentukan booster dan penilaian satu kombinasi

Fungsi *search_xgb* pada Kode Program 5.110 menyusun grid dengan *build_grid* (Langkah 1), mengambil subsampel berstrata (Langkah 2), dan menjalankan *run_search* dengan *fit_and_score_xgb*. Sel kedua menjalankan pencarian dan menyimpan hasilnya ke *hyperparameter-search/xgb-search.csv*. Sel ketiga memilih kombinasi teratas dan menetapkannya pada variabel *XGB_MAX_DEPTH*, *XGB_LEARNING_RATE*, *XGB_COLSAMPLE_BYTREE*, dan *XGB_N_ESTIMATORS*, yaitu jumlah pohon yang digunakan pada kombinasi tersebut (Langkah 5).

```python
def search_xgb(
    train_x,
    train_y,
    dev_x,
    dev_y,
    combinations: list[tuple] | None = None,
    train_samples: int | None = XGB_SEARCH_TRAIN_SAMPLES,
    dev_samples: int | None = XGB_SEARCH_DEV_SAMPLES,
    dev_min_per_class: int = SEARCH_DEV_MIN_PER_CLASS,
    random_state: int = RANDOM_STATE,
    score_column: str = SCORE_COLUMN,
) -> pd.DataFrame:
    if combinations is None:
        combinations = build_grid(
            list_of_xgb_max_depth, list_of_xgb_learning_rates, list_of_xgb_colsample_bytree
        )

    print(f"XGBoost search over {len(combinations)} combinations")
    free_gpu_memory()

    print("Preparing search subsamples...")
    search_data = (
        *stratified_subsample(train_x, train_y, train_samples, random_state),
        *stratified_subsample(dev_x, dev_y, dev_samples, random_state, min_per_class=dev_min_per_class),
    )

    return run_search(
        combinations,
        ("max_depth", "learning_rate", "colsample_bytree"),
        fit_and_score_xgb,
        search_data,
        score_column,
    )

xgb_search_results = search_xgb(train_x, train_y, dev_x, dev_y)
save_search_results(xgb_search_results, "xgb-search.csv", PATH_FOLDER_SEARCH_RESULT)
xgb_search_results

best_xgb = get_best_hyperparameter(xgb_search_results, SCORE_COLUMN)
XGB_MAX_DEPTH = int(best_xgb["max_depth"])
XGB_LEARNING_RATE = float(best_xgb["learning_rate"])
XGB_COLSAMPLE_BYTREE = float(best_xgb["colsample_bytree"])
XGB_N_ESTIMATORS = int(best_xgb["n_trees_used"])
```

Kode Program 5.110 Pencarian hiperparameter XGBoost dan pemilihan konfigurasi terbaik

Fungsi *dump_xgb_model* pada Kode Program 5.111 menyimpan model dalam format JSON asli *XGBoost* melalui berkas sementara. Fungsi *train_xgb* mengimplementasikan Langkah 6 dan 7. Fungsi ini membentuk *XGBClassifier* dengan konfigurasi terpilih dan jumlah pohon hasil pencarian, melatihnya pada seluruh data latih tanpa data validasi dan tanpa penghentian dini, dan menyimpan modelnya. Sel terakhir menjalankannya.

```python
def dump_xgb_model(model, name: str, subfolder: str, folder_name: str) -> str:
    target_folder = os.path.join(folder_name, subfolder)
    os.makedirs(target_folder, exist_ok=True)
    file_path = os.path.join(target_folder, name)
    temporary_path = get_temporary_model_path(file_path)
    model.save_model(temporary_path)
    os.replace(temporary_path, file_path)
    return file_path

def train_xgb(
    train_x,
    train_y,
    max_depth: int,
    learning_rate: float,
    colsample_bytree: float,
    n_estimators: int,
    train_samples: int | None = XGB_TRAIN_SAMPLES,
    random_state: int = RANDOM_STATE,
    subfolder: str = XGB_SUBFOLDER,
    folder_name: str = PATH_FOLDER_MODEL,
) -> str:
    print(
        f"Training XGBoost: max_depth={max_depth} learning_rate={learning_rate} "
        f"colsample_bytree={colsample_bytree} n_estimators={n_estimators}"
    )

    print("Preparing training data...")
    features, labels = stratified_subsample(train_x, train_y, train_samples, random_state)
    n_rows, n_features = features.shape

    model = build_xgb(
        max_depth, learning_rate, colsample_bytree, n_estimators=n_estimators, random_state=random_state
    )
    with step(f"boosting {n_estimators} rounds on {n_rows:,} rows x {n_features} features ({XGB_DEVICE})"):
        model.fit(features, labels, verbose=False)

    with step("saving model"):
        file_path = dump_xgb_model(
            model, get_xgb_name(max_depth, learning_rate, colsample_bytree), subfolder, folder_name
        )

    print(f"Saved model to {file_path}.")
    return file_path

train_xgb(
    train_x, train_y, XGB_MAX_DEPTH, XGB_LEARNING_RATE, XGB_COLSAMPLE_BYTREE, XGB_N_ESTIMATORS
)
```

Kode Program 5.111 Pelatihan model XGBoost akhir

Fungsi *load_xgb_model* pada Kode Program 5.112 memuat model dari berkas JSON dan menetapkan ulang perangkat komputasi serta metode pohon sesuai lingkungan. Fungsi *xgb_predict* mengimplementasikan Langkah 8. Fungsi ini memuat model, memprediksi data per potongan dengan fitur berupa larik *CuPy* apabila GPU digunakan, menyimpan hasilnya ke folder *prediction-result/XGB*, kemudian melepas model dan membebaskan memori GPU. Sel-sel berikutnya menetapkan nama model terpilih dan memprediksi data validasi dan data uji.

```python
def load_xgb_model(model_name: str, subfolder: str, folder_name: str) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(os.path.join(folder_name, subfolder, model_name))
    model.set_params(device=XGB_DEVICE, tree_method=XGB_TREE_METHOD)
    return model

def xgb_predict(
    model_name: str,
    x,
    split_name: str,
    subfolder: str = XGB_SUBFOLDER,
    chunk_size: int = PREDICT_CHUNK_SIZE,
) -> np.ndarray:
    with step(f"loading {model_name}"):
        model = load_xgb_model(model_name, subfolder, PATH_FOLDER_MODEL)

    predictions = predict_in_chunks(
        lambda rows: to_numpy(model.predict(to_xgb_features(rows))),
        x,
        get_split_folder(os.path.join(PATH_FOLDER_PREDICTION, subfolder), split_name),
        "xgb",
        chunk_size,
    )

    del model
    free_gpu_memory()
    return predictions

xgb_model_name = get_xgb_name(XGB_MAX_DEPTH, XGB_LEARNING_RATE, XGB_COLSAMPLE_BYTREE)
xgb_predict(xgb_model_name, dev_x, "dev")

xgb_predict(xgb_model_name, test_x, "test")
```

Kode Program 5.112 Prediksi dengan model XGBoost

Kode Program 5.113 menghitung metrik pada data validasi dan data uji dari prediksi yang telah tersimpan, kemudian menyusun laporan klasifikasi per kelas dan *confusion matrix* data uji.

```python
xgb_dev_folder = get_split_folder(os.path.join(PATH_FOLDER_PREDICTION, XGB_SUBFOLDER), "dev")
xgb_test_folder = get_split_folder(os.path.join(PATH_FOLDER_PREDICTION, XGB_SUBFOLDER), "test")

pd.concat(
    [
        get_evaluation_results(xgb_dev_folder, dev_y).assign(split="dev"),
        get_evaluation_results(xgb_test_folder, test_y).assign(split="test"),
    ],
    ignore_index=True,
)

xgb_test_pred = load_prediction_chunks(xgb_test_folder)
get_classification_report(pred=xgb_test_pred, true=to_labels(test_y), class_names=CLASS_NAMES)

get_confusion_matrix(pred=xgb_test_pred, true=to_labels(test_y), class_names=CLASS_NAMES)
```

Kode Program 5.113 Laporan hasil evaluasi model XGBoost
