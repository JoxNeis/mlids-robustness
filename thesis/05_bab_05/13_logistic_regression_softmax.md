## **5.13 Implementasi *Logistic Regression* (*Softmax Regression*)**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.13, yaitu pemilihan hiperparameter, pelatihan, dan prediksi model *Logistic Regression* multikelas. Kode program disusun mengikuti Langkah 1 hingga 9 pada algoritma di Subbab 4.13 dan menggunakan fungsi bantu pada Subbab 5.9.

Kode Program 5.102 mendefinisikan subfolder model dan ruang pencarian, yaitu laju pembelajaran 0,01 dan 0,001, ukuran *batch* 2.048 dan 8.192, serta koefisien L2 sebesar 0, 0,00001, dan 0,0001. Konstanta lainnya adalah ukuran subsampel pencarian sebanyak 1.000.000 baris latih dan 200.000 baris validasi, ukuran data pelatihan akhir (seluruh data latih), jumlah kelas (*LOGREG_N_CLASSES*), jumlah *epoch* maksimum 30, kesabaran penghentian dini 3 *epoch*, dan ukuran *batch* prediksi 8.192. Fungsi *get_logreg_name* membentuk nama berkas model dari hiperparameternya.

```python
LOGREG_SUBFOLDER = "LOGREG"

list_of_learning_rates = [1e-2, 1e-3]
list_of_batch_sizes = [2048, 8192]
list_of_l2 = [0.0, 1e-5, 1e-4]

LOGREG_SEARCH_TRAIN_SAMPLES = 1_000_000
LOGREG_SEARCH_DEV_SAMPLES = 200_000
LOGREG_TRAIN_SAMPLES = None

LOGREG_N_CLASSES = len(CLASS_NAMES)
LOGREG_MAX_EPOCHS = 30
LOGREG_PATIENCE = 3
LOGREG_PREDICT_BATCH = 8192

def get_logreg_name(learning_rate: float, batch_size: int, l2: float) -> str:
    return f"logreg-lr-{learning_rate}-bs-{batch_size}-l2-{l2}.keras"
```

Kode Program 5.102 Pengaturan model Logistic Regression

Fungsi *build_logreg* pada Kode Program 5.103 mengimplementasikan Langkah 3. Fungsi ini menetapkan *random seed*, membentuk model *Sequential* dengan satu lapisan *Dense* yang berunit sebanyak kelas dan beraktivasi *softmax*, menambahkan regularisasi L2 pada bobot apabila koefisiennya lebih besar dari nol, dan mengompilasinya dengan optimasi *Adam* dan *loss* *sparse_categorical_crossentropy*. Fungsi *fit_and_score_logreg* mengimplementasikan Langkah 4 dan 5. Fungsi ini melatih model dengan *EarlyStopping* pada *val_loss*, dengan kesabaran 3 *epoch* dan bobot terbaik dipulihkan, memprediksi subsampel validasi dengan mengambil kelas berpeluang terbesar (*argmax*), dan mengembalikan metrik bersama jumlah *epoch* yang dijalankan serta waktu pelatihan dan prediksi.

```python
def build_logreg(
    n_features: int,
    learning_rate: float,
    l2: float,
    n_classes: int = LOGREG_N_CLASSES,
    random_state: int = RANDOM_STATE,
) -> keras.Model:
    keras.utils.set_random_seed(random_state)

    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            keras.layers.Dense(
                n_classes,
                activation="softmax",
                kernel_regularizer=keras.regularizers.L2(l2) if l2 else None,
                name="softmax",
            ),
        ],
        name="logistic_regression",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def fit_and_score_logreg(
    learning_rate,
    batch_size,
    l2,
    train_x,
    train_y,
    dev_x,
    dev_y,
    max_epochs: int = LOGREG_MAX_EPOCHS,
    patience: int = LOGREG_PATIENCE,
) -> dict:
    model = build_logreg(train_x.shape[1], learning_rate, l2)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )

    fit_started = time.time()
    with step(f"fitting {train_x.shape[0]:,} rows for up to {max_epochs} epochs"):
        history = model.fit(
            train_x,
            train_y,
            validation_data=(dev_x, dev_y),
            epochs=max_epochs,
            batch_size=int(batch_size),
            callbacks=[early_stopping],
            shuffle=True,
            verbose=0,
        )
    fit_seconds = time.time() - fit_started

    predict_started = time.time()
    with step(f"predicting {dev_x.shape[0]:,} dev rows"):
        probabilities = model.predict(dev_x, batch_size=LOGREG_PREDICT_BATCH, verbose=0)
        pred = probabilities.argmax(axis=1).astype(np.int32)
    predict_seconds = time.time() - predict_started

    scores = evaluate(pred=pred, true=dev_y).iloc[0].to_dict()
    measurements = {
        "epochs_run": len(history.history["loss"]),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        **scores,
    }

    del model
    keras.backend.clear_session()
    free_gpu_memory()
    return measurements
```

Kode Program 5.103 Pembentukan model softmax dan penilaian satu kombinasi

Fungsi *search_logreg* pada Kode Program 5.104 menyusun grid dengan *build_grid* (Langkah 1), mengambil subsampel berstrata (Langkah 2), dan menjalankan *run_search* dengan *fit_and_score_logreg*. Sel kedua menjalankan pencarian dan menyimpan hasilnya ke *hyperparameter-search/logreg-search.csv*. Sel ketiga memilih kombinasi teratas dan menetapkannya pada variabel *LOGREG_LEARNING_RATE*, *LOGREG_BATCH_SIZE*, *LOGREG_L2*, dan *LOGREG_EPOCHS*, yaitu jumlah *epoch* yang dijalankan pada kombinasi tersebut (Langkah 6).

```python
def search_logreg(
    train_x,
    train_y,
    dev_x,
    dev_y,
    combinations: list[tuple] | None = None,
    train_samples: int | None = LOGREG_SEARCH_TRAIN_SAMPLES,
    dev_samples: int | None = LOGREG_SEARCH_DEV_SAMPLES,
    dev_min_per_class: int = SEARCH_DEV_MIN_PER_CLASS,
    random_state: int = RANDOM_STATE,
    score_column: str = SCORE_COLUMN,
) -> pd.DataFrame:
    if combinations is None:
        combinations = build_grid(list_of_learning_rates, list_of_batch_sizes, list_of_l2)

    print(f"Logistic Regression search over {len(combinations)} combinations")
    free_gpu_memory()

    print("Preparing search subsamples...")
    search_data = (
        *stratified_subsample(train_x, train_y, train_samples, random_state),
        *stratified_subsample(dev_x, dev_y, dev_samples, random_state, min_per_class=dev_min_per_class),
    )

    return run_search(
        combinations,
        ("learning_rate", "batch_size", "l2"),
        fit_and_score_logreg,
        search_data,
        score_column,
    )

logreg_search_results = search_logreg(train_x, train_y, dev_x, dev_y)
save_search_results(logreg_search_results, "logreg-search.csv", PATH_FOLDER_SEARCH_RESULT)
logreg_search_results

best_logreg = get_best_hyperparameter(logreg_search_results, SCORE_COLUMN)
LOGREG_LEARNING_RATE = float(best_logreg["learning_rate"])
LOGREG_BATCH_SIZE = int(best_logreg["batch_size"])
LOGREG_L2 = float(best_logreg["l2"])
LOGREG_EPOCHS = int(best_logreg["epochs_run"])
```

Kode Program 5.104 Pencarian hiperparameter Logistic Regression dan pemilihan konfigurasi terbaik

Fungsi *dump_keras_model* pada Kode Program 5.105 menyimpan model *Keras* ke berkas *.keras* melalui berkas sementara. Fungsi *train_logreg* mengimplementasikan Langkah 7 dan 8. Fungsi ini membentuk model dengan konfigurasi terpilih, menampilkan ringkasan arsitekturnya, melatihnya pada seluruh data latih selama jumlah *epoch* hasil pencarian tanpa data validasi dan tanpa penghentian dini, dan menyimpan modelnya. Sel terakhir menjalankannya.

```python
def dump_keras_model(model, name: str, subfolder: str, folder_name: str) -> str:
    target_folder = os.path.join(folder_name, subfolder)
    os.makedirs(target_folder, exist_ok=True)
    file_path = os.path.join(target_folder, name)
    temporary_path = get_temporary_model_path(file_path)
    model.save(temporary_path)
    os.replace(temporary_path, file_path)
    return file_path

def train_logreg(
    train_x,
    train_y,
    learning_rate: float,
    batch_size: int,
    l2: float,
    epochs: int,
    train_samples: int | None = LOGREG_TRAIN_SAMPLES,
    random_state: int = RANDOM_STATE,
    subfolder: str = LOGREG_SUBFOLDER,
    folder_name: str = PATH_FOLDER_MODEL,
) -> str:
    print(
        f"Training Logistic Regression: learning_rate={learning_rate} "
        f"batch_size={batch_size} l2={l2} epochs={epochs}"
    )

    print("Preparing training data...")
    features, labels = stratified_subsample(train_x, train_y, train_samples, random_state)
    n_rows, n_features = features.shape

    model = build_logreg(n_features, learning_rate, l2, random_state=random_state)
    model.summary()

    print(f"Fitting {epochs} epochs on {n_rows:,} rows x {n_features} features")
    model.fit(features, labels, epochs=int(epochs), batch_size=int(batch_size), shuffle=True, verbose=2)

    with step("saving model"):
        file_path = dump_keras_model(
            model, get_logreg_name(learning_rate, batch_size, l2), subfolder, folder_name
        )

    print(f"Saved model to {file_path}.")
    return file_path

train_logreg(train_x, train_y, LOGREG_LEARNING_RATE, LOGREG_BATCH_SIZE, LOGREG_L2, LOGREG_EPOCHS)
```

Kode Program 5.105 Pelatihan model Logistic Regression akhir

Fungsi *logreg_predict* pada Kode Program 5.106 mengimplementasikan Langkah 9. Fungsi ini memuat model dengan *keras.models.load_model*, memprediksi data per potongan dengan *predict* berukuran *batch* 8.192 dan mengambil kelas berpeluang terbesar, menyimpan hasilnya ke folder *prediction-result/LOGREG*, kemudian melepas model dan membersihkan sesi *Keras*. Sel-sel berikutnya menetapkan nama model terpilih dan memprediksi data validasi dan data uji.

```python
def logreg_predict(
    model_name: str,
    x,
    split_name: str,
    subfolder: str = LOGREG_SUBFOLDER,
    chunk_size: int = PREDICT_CHUNK_SIZE,
    batch_size: int = LOGREG_PREDICT_BATCH,
) -> np.ndarray:
    with step(f"loading {model_name}"):
        model = keras.models.load_model(
            os.path.join(PATH_FOLDER_MODEL, subfolder, model_name)
        )

    predictions = predict_in_chunks(
        lambda rows: model.predict(
            to_features(rows), batch_size=batch_size, verbose=0
        ).argmax(axis=1),
        x,
        get_split_folder(os.path.join(PATH_FOLDER_PREDICTION, subfolder), split_name),
        "logreg",
        chunk_size,
    )

    del model
    keras.backend.clear_session()
    free_gpu_memory()
    return predictions

logreg_model_name = get_logreg_name(LOGREG_LEARNING_RATE, LOGREG_BATCH_SIZE, LOGREG_L2)
logreg_predict(logreg_model_name, dev_x, "dev")

logreg_predict(logreg_model_name, test_x, "test")
```

Kode Program 5.106 Prediksi dengan model Logistic Regression

Kode Program 5.107 menghitung metrik pada data validasi dan data uji dari prediksi yang telah tersimpan, kemudian menyusun laporan klasifikasi per kelas dan *confusion matrix* data uji.

```python
logreg_dev_folder = get_split_folder(os.path.join(PATH_FOLDER_PREDICTION, LOGREG_SUBFOLDER), "dev")
logreg_test_folder = get_split_folder(os.path.join(PATH_FOLDER_PREDICTION, LOGREG_SUBFOLDER), "test")

pd.concat(
    [
        get_evaluation_results(logreg_dev_folder, dev_y).assign(split="dev"),
        get_evaluation_results(logreg_test_folder, test_y).assign(split="test"),
    ],
    ignore_index=True,
)

logreg_test_pred = load_prediction_chunks(logreg_test_folder)
get_classification_report(pred=logreg_test_pred, true=to_labels(test_y), class_names=CLASS_NAMES)

get_confusion_matrix(pred=logreg_test_pred, true=to_labels(test_y), class_names=CLASS_NAMES)
```

Kode Program 5.107 Laporan hasil evaluasi model Logistic Regression
