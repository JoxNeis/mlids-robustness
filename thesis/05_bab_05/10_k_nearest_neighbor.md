## **5.10 Implementasi *K Nearest-Neighbor* (KNN)**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.10, yaitu pemilihan hiperparameter, pelatihan, dan prediksi model *K Nearest-Neighbor*. Kode program disusun mengikuti Langkah 1 hingga 8 pada algoritma di Subbab 4.10 dan menggunakan fungsi bantu pada Subbab 5.9.

Kode Program 5.81 mendefinisikan subfolder model (*KNN_SUBFOLDER*) dan ruang pencarian, yaitu jumlah tetangga 1, 3, 5, 7, 9, 11, dan 13, ukuran jarak *manhattan*, *euclidean*, dan *cosine*, serta pembobotan *uniform* dan *distance*. Konstanta lainnya adalah ukuran subsampel pencarian sebanyak 500.000 baris latih dan 200.000 baris validasi, serta ukuran potongan pencarian tetangga sebanyak 20.000 baris. Fungsi *get_knn_name* membentuk nama berkas model dari hiperparameternya.

```python
KNN_SUBFOLDER = "KNN"

list_of_n_neighbors = [1, 3, 5, 7, 9, 11, 13]
list_of_distance_metrics = ["manhattan", "euclidean", "cosine"]
list_of_weightings = ["uniform", "distance"]

KNN_SEARCH_TRAIN_SAMPLES = 500_000
KNN_SEARCH_DEV_SAMPLES = 200_000
KNN_SEARCH_QUERY_CHUNK = 20_000

def get_knn_name(n_neighbors: int, metric: str, weighting: str) -> str:
    return f"knn-k-{n_neighbors}-m-{metric}-w-{weighting}.pkl"
```

Kode Program 5.81 Pengaturan model KNN

Fungsi *knn_vote* pada Kode Program 5.82 mengimplementasikan Langkah 4. Untuk setiap baris, bobot setiap tetangga ditetapkan satu pada pembobotan *uniform* atau kebalikan jaraknya pada pembobotan *distance*. Apabila terdapat tetangga berjarak nol, hanya tetangga tersebut yang diberi bobot. Bobot dijumlahkan menurut kelas dengan *numpy.bincount* pada indeks gabungan baris dan kelas, dan kelas dengan total bobot terbesar ditetapkan sebagai prediksi.

```python
def knn_vote(
    neighbor_labels: np.ndarray,
    neighbor_distances: np.ndarray,
    n_classes: int,
    weighting: str,
) -> np.ndarray:
    n_query, n_neighbors = neighbor_labels.shape

    if weighting == "uniform":
        weights = np.ones((n_query, n_neighbors), dtype=np.float64)
    else:
        with np.errstate(divide="ignore"):
            weights = 1.0 / neighbor_distances.astype(np.float64)

        exact_match = ~np.isfinite(weights)
        rows_with_exact_match = exact_match.any(axis=1)
        if rows_with_exact_match.any():
            weights[rows_with_exact_match] = exact_match[rows_with_exact_match].astype(np.float64)

    row_offset = np.arange(n_query, dtype=np.int64)[:, None] * n_classes
    flat_index = (row_offset + neighbor_labels).ravel()
    scores = np.bincount(flat_index, weights=weights.ravel(), minlength=n_query * n_classes)

    return scores.reshape(n_query, n_classes).argmax(axis=1).astype(np.int32)
```

Kode Program 5.82 Pemungutan suara tetangga terdekat

Fungsi *kneighbors_in_chunks* pada Kode Program 5.83 mengimplementasikan Langkah 3. Fungsi ini memanggil *kneighbors* pada indeks tetangga untuk setiap potongan sampel validasi, mengonversi jarak dan indeks hasilnya menjadi larik *NumPy*, dan menampilkan bilah kemajuan beserta perkiraan sisa waktu.

```python
def kneighbors_in_chunks(
    index, query: np.ndarray, n_neighbors: int, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    total = query.shape[0]
    n_chunks = math.ceil(total / chunk_size)
    distances, indices = [], []
    started = time.time()

    for number, start in enumerate(range(0, total, chunk_size), start=1):
        chunk_distances, chunk_indices = index.kneighbors(
            query[start : start + chunk_size], n_neighbors=n_neighbors
        )
        distances.append(to_numpy_2d(chunk_distances).astype(np.float32, copy=False))
        indices.append(to_numpy_2d(chunk_indices).astype(np.int64, copy=False))

        elapsed = time.time() - started
        progress_bar(
            number, n_chunks, f"neighbours (elapsed {elapsed:,.0f}s, eta {elapsed / number * (n_chunks - number):,.0f}s)"
        )

    return np.vstack(distances), np.vstack(indices)
```

Kode Program 5.83 Pencarian tetangga terdekat per potongan

Fungsi *search_knn* pada Kode Program 5.84 mengimplementasikan Langkah 1 hingga 5 serta pengurutan hasil pada Langkah 6. Fungsi ini mengambil subsampel berstrata, kemudian untuk setiap ukuran jarak membentuk indeks *NearestNeighbors* dengan jumlah tetangga terbesar dan mencari tetangga seluruh sampel validasi satu kali. Prediksi dan metrik untuk setiap nilai *k* dan pembobotan dihitung dari *k* kolom pertama daftar tetangga tersebut, sehingga indeks tidak perlu dibentuk ulang untuk setiap kombinasi. Hasilnya diurutkan menurut skor terbaik.

```python
def search_knn(
    train_x,
    train_y,
    dev_x,
    dev_y,
    list_n_neighbors: list[int] = list_of_n_neighbors,
    list_metrics: list[str] = list_of_distance_metrics,
    list_weightings: list[str] = list_of_weightings,
    train_samples: int | None = KNN_SEARCH_TRAIN_SAMPLES,
    dev_samples: int | None = KNN_SEARCH_DEV_SAMPLES,
    dev_min_per_class: int = SEARCH_DEV_MIN_PER_CLASS,
    chunk_size: int = KNN_SEARCH_QUERY_CHUNK,
    random_state: int = RANDOM_STATE,
    score_column: str = SCORE_COLUMN,
) -> pd.DataFrame:
    print("Preparing search subsamples...")
    search_train_x, search_train_y = stratified_subsample(
        train_x, train_y, train_samples, random_state
    )
    search_dev_x, search_dev_y = stratified_subsample(
        dev_x, dev_y, dev_samples, random_state, min_per_class=dev_min_per_class
    )

    n_classes = int(max(search_train_y.max(), search_dev_y.max())) + 1
    max_neighbors = max(list_n_neighbors)
    rows = []

    for metric in list_metrics:
        print(f"Fitting neighbour index (metric={metric})...")
        index = NearestNeighbors(n_neighbors=max_neighbors, metric=metric)
        index.fit(search_train_x)

        neighbor_distances, neighbor_indices = kneighbors_in_chunks(
            index, search_dev_x, max_neighbors, chunk_size
        )
        neighbor_labels = search_train_y[neighbor_indices]

        for n_neighbors in list_n_neighbors:
            for weighting in list_weightings:
                pred = knn_vote(
                    neighbor_labels[:, :n_neighbors],
                    neighbor_distances[:, :n_neighbors],
                    n_classes,
                    weighting,
                )
                scores = evaluate(pred=pred, true=search_dev_y).iloc[0].to_dict()
                rows.append(
                    {"n_neighbors": n_neighbors, "metric": metric, "weights": weighting, **scores}
                )
                print(
                    f"  k={n_neighbors:>3} metric={metric:<10} weights={weighting:<8}"
                    f" {score_column}={scores[score_column]:.4f}"
                )

        del index, neighbor_distances, neighbor_indices, neighbor_labels
        free_gpu_memory()

    return pd.DataFrame(rows).sort_values(score_column, ascending=False, ignore_index=True)
```

Kode Program 5.84 Pencarian hiperparameter KNN tanpa pelatihan ulang

Kode Program 5.85 mengimplementasikan Langkah 6. Sel pertama menjalankan pencarian dan menyimpan tabel hasilnya ke *hyperparameter-search/knn-search.csv*. Sel kedua memilih kombinasi teratas dan menetapkannya pada variabel *KNN_N_NEIGHBORS*, *KNN_METRIC*, dan *KNN_WEIGHTING*.

```python
knn_search_results = search_knn(train_x, train_y, dev_x, dev_y)
save_search_results(knn_search_results, "knn-search.csv", PATH_FOLDER_SEARCH_RESULT)
knn_search_results

best_knn = get_best_hyperparameter(knn_search_results, SCORE_COLUMN)
KNN_N_NEIGHBORS = int(best_knn["n_neighbors"])
KNN_METRIC = str(best_knn["metric"])
KNN_WEIGHTING = str(best_knn["weights"])
```

Kode Program 5.85 Menjalankan pencarian dan memilih konfigurasi terbaik KNN

Fungsi *train_knn* pada Kode Program 5.86 mengimplementasikan Langkah 7. Fungsi ini membentuk *KNeighborsClassifier* dengan konfigurasi terpilih, melatihnya pada seluruh data latih dengan *fit*, yaitu pembentukan indeks, dan menyimpannya dengan *dump_trained_model* menggunakan nama yang memuat hiperparameternya. Sel kedua menjalankannya.

```python
def train_knn(
    train_x,
    train_y,
    n_neighbors: int,
    metric: str,
    weighting: str,
    subfolder: str = KNN_SUBFOLDER,
    folder_name: str = PATH_FOLDER_MODEL,
) -> str:
    print(f"Training KNN: k={n_neighbors} metric={metric} weights={weighting}")

    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weighting)
    with step(f"indexing {len(train_y):,} rows"):
        model.fit(to_features(train_x), to_labels(train_y))

    with step("saving model"):
        file_path = dump_trained_model(
            model, get_knn_name(n_neighbors, metric, weighting), subfolder, folder_name
        )

    print(f"Saved model to {file_path}.")
    return file_path

train_knn(train_x, train_y, KNN_N_NEIGHBORS, KNN_METRIC, KNN_WEIGHTING)
```

Kode Program 5.86 Pelatihan model KNN akhir

Fungsi *knn_predict* pada Kode Program 5.87 mengimplementasikan Langkah 8. Fungsi ini memuat model, memprediksi data per potongan dengan *predict_in_chunks* (Kode Program 5.76) ke folder *prediction-result/KNN*, kemudian melepas model dan membebaskan memori GPU. Sel-sel berikutnya menetapkan nama model terpilih dan memprediksi data validasi dan data uji.

```python
def knn_predict(
    model_name: str,
    x,
    split_name: str,
    subfolder: str = KNN_SUBFOLDER,
    chunk_size: int = PREDICT_CHUNK_SIZE,
) -> np.ndarray:
    with step(f"loading {model_name}"):
        model = load_trained_model(model_name, subfolder, PATH_FOLDER_MODEL)

    predictions = predict_in_chunks(
        lambda rows: to_numpy(model.predict(to_features(rows))),
        x,
        get_split_folder(os.path.join(PATH_FOLDER_PREDICTION, subfolder), split_name),
        "knn",
        chunk_size,
    )

    del model
    free_gpu_memory()
    return predictions

knn_model_name = get_knn_name(KNN_N_NEIGHBORS, KNN_METRIC, KNN_WEIGHTING)
knn_predict(knn_model_name, dev_x, "dev")

knn_predict(knn_model_name, test_x, "test")
```

Kode Program 5.87 Prediksi dengan model KNN

Kode Program 5.88 menghitung metrik pada data validasi dan data uji dari prediksi yang telah tersimpan dengan *get_evaluation_results*, kemudian menyusun laporan klasifikasi per kelas dan *confusion matrix* data uji, sebagaimana Langkah 8 pada Subbab 4.9.

```python
knn_dev_folder = get_split_folder(os.path.join(PATH_FOLDER_PREDICTION, KNN_SUBFOLDER), "dev")
knn_test_folder = get_split_folder(os.path.join(PATH_FOLDER_PREDICTION, KNN_SUBFOLDER), "test")

pd.concat(
    [
        get_evaluation_results(knn_dev_folder, dev_y).assign(split="dev"),
        get_evaluation_results(knn_test_folder, test_y).assign(split="test"),
    ],
    ignore_index=True,
)

knn_test_pred = load_prediction_chunks(knn_test_folder)
get_classification_report(pred=knn_test_pred, true=to_labels(test_y), class_names=CLASS_NAMES)

get_confusion_matrix(pred=knn_test_pred, true=to_labels(test_y), class_names=CLASS_NAMES)
```

Kode Program 5.88 Laporan hasil evaluasi model KNN
