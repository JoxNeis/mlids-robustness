## **5.9 Implementasi Pelatihan dan Evaluasi Model**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.9, yaitu kerangka pelatihan dan evaluasi yang digunakan bersama oleh kelima algoritma. Kode program disusun mengikuti Langkah 1 hingga 9 pada algoritma di Subbab 4.9. Fungsi bantu pada subbab ini digunakan kembali oleh Subbab 5.10 hingga 5.17. Kode untuk membandingkan kelima model (Langkah 9) merupakan pengecualian dalam urutan penyajian: kode tersebut dijalankan setelah kelima model selesai dilatih pada Subbab 5.10 hingga 5.14, karena menggunakan konstanta dan hasil pencarian hiperparameter yang didefinisikan pada subbab tersebut.

Kode Program 5.68 mengimplementasikan Langkah 1. Fungsi *load_split_frame* membaca seluruh berkas Parquet suatu himpunan menjadi satu tabel *Polars* menggunakan *scan_parquet* dan mesin *streaming*. Fungsi *split_features_and_labels* memisahkan kolom label dari fitur. Fungsi *load_features_and_labels* memuat fitur dan label dari dua folder yang berbeda dan menghentikan proses dengan galat apabila jumlah barisnya tidak sama. Fungsi *describe_dataset* mencetak jumlah baris dan kolom, tipe data, ukuran memori, serta jumlah kelas himpunan yang dimuat.

```python
def load_split_frame(folder_name: str, split_name: str) -> pl.DataFrame:
    file_paths = get_split_file_paths(folder_name, split_name)
    return pl.scan_parquet(file_paths).collect(engine="streaming")

def split_features_and_labels(
    frame: pl.DataFrame, label_column: str = LABEL_COLUMN
) -> tuple[pl.DataFrame, pl.Series]:
    return frame.drop(label_column), frame[label_column]

def load_features_and_labels(
    features_folder_name: str,
    labels_folder_name: str,
    split_name: str,
    label_column: str = LABEL_COLUMN,
) -> tuple[pl.DataFrame, pl.Series]:
    features = load_split_frame(features_folder_name, split_name)
    labels = load_split_frame(labels_folder_name, split_name)[label_column]

    if features.height != labels.len():
        raise ValueError(
            f"{split_name}: {features.height:,} feature rows against {labels.len():,} labels — "
            "the two trees are no longer row-for-row."
        )
    return features, labels

def describe_dataset(name: str, features: pl.DataFrame, labels: pl.Series) -> None:
    print(
        f"{name}: {features.height:,} rows x {features.width} columns "
        f"({features.dtypes[0]}, {features.estimated_size('mb'):.0f} MB) | "
        f"labels {labels.dtype} over {labels.n_unique()} classes"
    )
```

Kode Program 5.68 Fungsi pemuatan himpunan data

Kode Program 5.69 memuat data latih dari folder *data-smote*, tempat fitur dan label berada pada berkas yang sama, serta data validasi dan data uji dari folder *data-ipca* untuk fitur dan folder *data-encoded-label* untuk label.

```python
train_x, train_y = split_features_and_labels(load_split_frame(PATH_FOLDER_SMOTE, FIT_SPLIT))
describe_dataset("train", train_x, train_y)

dev_x, dev_y = load_features_and_labels(PATH_FOLDER_IPCA, PATH_FOLDER_ENCODED_LABEL, "dev")
describe_dataset("dev", dev_x, dev_y)

test_x, test_y = load_features_and_labels(PATH_FOLDER_IPCA, PATH_FOLDER_ENCODED_LABEL, "test")
describe_dataset("test", test_x, test_y)
```

Kode Program 5.69 Pemuatan data latih, data validasi, dan data uji

Kode Program 5.70 mendefinisikan folder model (*PATH_FOLDER_MODEL*), folder hasil prediksi (*PATH_FOLDER_PREDICTION*), dan folder hasil pencarian hiperparameter (*PATH_FOLDER_SEARCH_RESULT*). Konstanta lainnya adalah ukuran potongan prediksi sebesar 10.000 baris (*PREDICT_CHUNK_SIZE*), kolom skor untuk pemilihan model, yaitu *f1_macro* (*SCORE_COLUMN*), jumlah minimum sampel per kelas pada subsampel validasi sebanyak 200 (*SEARCH_DEV_MIN_PER_CLASS*), dan daftar nama kelas (*CLASS_NAMES*) yang dimuat dari berkas pemetaan.

```python
PATH_FOLDER_MODEL = "trained-model"
PATH_FOLDER_PREDICTION = "prediction-result"
PATH_FOLDER_SEARCH_RESULT = "hyperparameter-search"

PREDICT_CHUNK_SIZE = 10_000

SCORE_COLUMN = "f1_macro"
SEARCH_DEV_MIN_PER_CLASS = 200

CLASS_NAMES = load_label_classes(PATH_LABEL_ENCODER)
print(f"{len(CLASS_NAMES)} classes: {CLASS_NAMES[0]} ... {CLASS_NAMES[-1]}")
```

Kode Program 5.70 Pengaturan pelatihan dan evaluasi

Kode Program 5.71 memuat empat fungsi konversi. Fungsi *to_features* dan *to_labels* mengubah tabel atau larik menjadi larik *float32* yang bersambungan di memori untuk fitur dan larik bilangan bulat 32-bit untuk label. Fungsi *to_numpy* dan *to_numpy_2d* mengubah keluaran model, baik dari *pandas*, *Polars*, maupun larik GPU, menjadi larik *NumPy* satu dimensi dan dua dimensi.

```python
def to_features(data) -> np.ndarray:
    if isinstance(data, (pl.DataFrame, pl.Series)):
        data = data.to_numpy()
    return np.ascontiguousarray(data, dtype=np.float32)

def to_labels(data) -> np.ndarray:
    if isinstance(data, (pl.DataFrame, pl.Series)):
        data = data.to_numpy()
    return np.asarray(data).ravel().astype(np.int32)

def to_numpy(data) -> np.ndarray:
    if hasattr(data, "to_numpy"):
        return np.asarray(data.to_numpy()).ravel()
    if hasattr(data, "get"):
        return data.get().ravel()
    return np.asarray(data).ravel()

def to_numpy_2d(data) -> np.ndarray:
    if hasattr(data, "to_numpy"):
        return np.asarray(data.to_numpy())
    if hasattr(data, "get"):
        return np.asarray(data.get())
    return np.asarray(data)
```

Kode Program 5.71 Fungsi konversi tipe data untuk fitur, label, dan hasil prediksi

Fungsi *free_gpu_memory* pada Kode Program 5.72 menjalankan pengumpulan sampah dan, apabila GPU digunakan, membebaskan seluruh blok memori yang ditahan oleh *CuPy*. Pengelola konteks *step* mencetak awal dan hasil, yaitu *done* atau *failed*, dari suatu langkah yang berjalan lama, dan *progress_bar* mencetak bilah kemajuan pada baris yang sama.

```python
def free_gpu_memory() -> None:
    gc.collect()
    if not use_cuda:
        return
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

@contextmanager
def step(message: str):
    print(f"  {message} ...", end="", flush=True)
    try:
        yield
    except BaseException:
        print(" failed")
        raise
    print(" done")

def progress_bar(done: int, total: int, label: str = "", width: int = 30) -> None:
    filled = width if total <= 0 else int(width * done / total)
    line = f"  [{'#' * filled}{'-' * (width - filled)}] {done:,}/{total:,}"
    if label:
        line += f" {label}"
    print(line.ljust(100), end="\n" if done >= total else "\r", flush=True)
```

Kode Program 5.72 Fungsi pembebasan memori GPU, pencatatan langkah, dan bilah kemajuan

Fungsi *stratified_subsample* pada Kode Program 5.73 mengimplementasikan Langkah 2. Fungsi ini menghitung kuota setiap kelas sebanding dengan proporsinya, menaikkannya sampai batas bawah *min_per_class* yang dibatasi oleh jumlah sampel kelas tersebut, dan memilih baris pada setiap kelas secara acak tanpa pengembalian menggunakan *numpy.random.default_rng* dengan *random state* yang ditetapkan. Fungsi ini mengembalikan fitur bertipe *float32* beserta labelnya, dan mengembalikan seluruh baris tanpa subsampel apabila jumlah sampel yang diminta tidak ditetapkan atau tidak lebih kecil daripada jumlah baris.

```python
def stratified_subsample(
    x, y, n_samples: int | None, random_state: int, min_per_class: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    labels = to_labels(y)
    total = labels.shape[0]

    if n_samples is None or n_samples >= total:
        return to_features(x), labels

    generator = np.random.default_rng(random_state)
    classes, counts = np.unique(labels, return_counts=True)

    quota = np.floor(counts / total * n_samples).astype(np.int64)
    quota = np.maximum(quota, np.minimum(min_per_class, counts))
    quota = np.minimum(quota, counts)

    selected = []
    for class_id, class_quota in zip(classes, quota):
        class_rows = np.flatnonzero(labels == class_id)
        if class_quota < class_rows.size:
            class_rows = generator.choice(class_rows, size=int(class_quota), replace=False)
        selected.append(class_rows)

    keep = np.zeros(total, dtype=bool)
    keep[np.sort(np.concatenate(selected))] = True
    subset = x.filter(pl.Series(keep)) if isinstance(x, pl.DataFrame) else x[keep]

    print(f"  subsampled {total:,} -> {int(keep.sum()):,} rows across {classes.size} classes")
    return to_features(subset), labels[keep]
```

Kode Program 5.73 Pengambilan subsampel berstrata

Kode Program 5.74 mengimplementasikan Langkah 3 dan 4. Fungsi *build_grid* menyusun seluruh kombinasi nilai hiperparameter sebagai hasil kali kartesian. Fungsi *run_search* menjalankan fungsi penilai untuk setiap kombinasi, mencetak kemajuan, dan mencatat kombinasi yang gagal beserta pesan galatnya tanpa menghentikan pencarian, kemudian mengurutkan hasil menurut kolom skor dari yang tertinggi. Fungsi *save_search_results* menyimpan tabel hasil ke berkas CSV. Fungsi *get_best_hyperparameter* mengembalikan baris dengan skor tertinggi dan menghentikan proses dengan galat apabila tidak ada kombinasi yang berhasil dinilai.

```python
def build_grid(*value_lists) -> list[tuple]:
    return [tuple(combination) for combination in itertools.product(*value_lists)]

def run_search(
    combinations: list[tuple],
    parameter_names: tuple[str, ...],
    fit_and_score,
    search_data: tuple,
    score_column: str,
) -> pd.DataFrame:
    rows = []
    total = len(combinations)

    for number, combination in enumerate(combinations, start=1):
        parameters = dict(zip(parameter_names, combination))
        described = " ".join(f"{name}={value}" for name, value in parameters.items())
        print(f"[{number}/{total}] {described}")

        try:
            measurements = fit_and_score(*combination, *search_data)
        except Exception as error:
            reason = f"{type(error).__name__}: {str(error).splitlines()[0][:120]}"
            print(f"  !! failed, skipped: {reason}")
            free_gpu_memory()
            rows.append({**parameters, "error": reason})
            continue

        rows.append({**parameters, **measurements})
        print(f"  {score_column}={measurements[score_column]:.4f}")

    results = pd.DataFrame(rows)
    failed = int(results["error"].notna().sum()) if "error" in results else 0
    print(f"Scored {total - failed} of {total} combinations" + (f", {failed} failed." if failed else "."))

    if score_column not in results:
        return results
    return results.sort_values(score_column, ascending=False, ignore_index=True)

def save_search_results(results: pd.DataFrame, file_name: str, folder_name: str) -> str:
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, file_name)
    results.to_csv(file_path, index=False)
    print(f"Saved search results to {file_path}.")
    return file_path

def get_best_hyperparameter(results: pd.DataFrame, score_column: str) -> dict:
    if score_column not in results:
        raise ValueError(
            f"None of the {len(results)} combinations in this grid was scored — every one of them "
            "failed, so there is nothing to choose between. The `error` column says why."
        )

    best = results.sort_values(score_column, ascending=False).iloc[0]
    print(f"Best by {score_column}:")
    for name, value in best.items():
        print(f"  {name:>18} = {value}")
    return best.to_dict()
```

Kode Program 5.74 Fungsi pencarian hiperparameter dan pemilihan konfigurasi terbaik

Kode Program 5.75 mengimplementasikan Langkah 6 untuk model yang disimpan dengan *joblib*. Fungsi *get_temporary_model_path* membentuk nama berkas sementara dengan sisipan *.tmp*. Fungsi *dump_trained_model* menulis model ke berkas sementara tersebut kemudian menggantinya menjadi nama akhir secara atomik dengan *os.replace*. Fungsi *load_trained_model* memuat model dari folder model.

```python
def get_temporary_model_path(file_path: str) -> str:
    stem, extension = os.path.splitext(file_path)
    return f"{stem}.tmp{extension}"

def dump_trained_model(model, name: str, subfolder: str, folder_name: str) -> str:
    target_folder = os.path.join(folder_name, subfolder)
    os.makedirs(target_folder, exist_ok=True)
    file_path = os.path.join(target_folder, name)
    temporary_path = get_temporary_model_path(file_path)
    joblib.dump(model, temporary_path)
    os.replace(temporary_path, file_path)
    return file_path

def load_trained_model(name: str, subfolder: str, folder_name: str):
    return joblib.load(os.path.join(folder_name, subfolder, name))
```

Kode Program 5.75 Penyimpanan dan pemuatan model terlatih

Kode Program 5.76 mengimplementasikan Langkah 7. Fungsi *save_array_atomically* menulis larik ke berkas *.npy* melalui berkas sementara. Fungsi *predict_in_chunks* membagi data menjadi potongan berukuran *chunk_size*, memprediksi setiap potongan hanya apabila berkas hasilnya belum ada, menyimpan hasilnya, menampilkan bilah kemajuan beserta jumlah potongan yang dihitung dan yang diambil dari *cache*, dan menggabungkan seluruh potongan sesuai urutannya. Fungsi *load_prediction_chunks* menggabungkan potongan yang telah tersimpan tanpa memprediksi ulang.

```python
def save_array_atomically(array: np.ndarray, file_path: str) -> None:
    temporary_path = file_path + ".tmp"
    with open(temporary_path, "wb") as file:
        np.save(file, array)
    os.replace(temporary_path, file_path)

def predict_in_chunks(
    predict_chunk, x, cache_folder: str, prefix: str, chunk_size: int
) -> np.ndarray:
    os.makedirs(cache_folder, exist_ok=True)
    total = len(x)
    if total == 0:
        print("Nothing to predict (0 rows).")
        return np.empty(0, dtype=np.int32)

    n_chunks = math.ceil(total / chunk_size)
    print(f"Predicting {total:,} rows in {n_chunks:,} chunks of {chunk_size:,} into {cache_folder}/")

    file_paths, computed, cached = [], 0, 0
    for number, start in enumerate(range(0, total, chunk_size), start=1):
        file_path = os.path.join(cache_folder, f"{prefix}_{number:05d}.npy")
        file_paths.append(file_path)

        if os.path.exists(file_path):
            cached += 1
        else:
            rows = (
                x.slice(start, chunk_size)
                if isinstance(x, (pl.DataFrame, pl.Series))
                else x[start : start + chunk_size]
            )
            save_array_atomically(np.asarray(predict_chunk(rows), dtype=np.int32), file_path)
            computed += 1

        progress_bar(number, n_chunks, f"chunks ({computed:,} computed, {cached:,} cached)")

    return np.concatenate([np.load(file_path) for file_path in file_paths])

def load_prediction_chunks(cache_folder: str) -> np.ndarray:
    if not os.path.isdir(cache_folder):
        raise FileNotFoundError(cache_folder)

    file_names = get_all_file_names_in_folder(cache_folder, "npy")
    if not file_names:
        raise FileNotFoundError(f"No prediction chunks in {cache_folder}/.")

    return np.concatenate(
        [np.load(os.path.join(cache_folder, file_name)) for file_name in file_names]
    )
```

Kode Program 5.76 Prediksi per potongan dengan penyimpanan hasil antara

Kode Program 5.77 mengimplementasikan Langkah 8. Fungsi *get_present_labels* mengumpulkan kelas yang muncul pada label sebenarnya maupun prediksi. Fungsi *evaluate* menghitung akurasi serta presisi, *recall*, dan F1-*score* rata-rata makro dengan *zero_division=0*. Fungsi *get_classification_report* menyusun presisi, *recall*, F1-*score*, dan jumlah sampel setiap kelas, dan *get_confusion_matrix* menyusun *confusion matrix* dengan nama kelas pada baris dan kolomnya. Fungsi *get_evaluation_results* memuat prediksi tersimpan dari suatu folder, memeriksa bahwa jumlahnya sama dengan jumlah label, dan mengembalikan metrik hasil *evaluate*.

```python
def get_present_labels(true, pred) -> list[int]:
    return sorted({int(x) for x in np.unique(true)} | {int(x) for x in np.unique(pred)})

def evaluate(pred, true, labels: list[int] | None = None) -> pd.DataFrame:
    if labels is None:
        labels = get_present_labels(true, pred)

    return pd.DataFrame(
        [
            {
                "accuracy": accuracy_score(true, pred),
                "precision_macro": precision_score(
                    true, pred, labels=labels, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    true, pred, labels=labels, average="macro", zero_division=0
                ),
                "f1_macro": f1_score(
                    true, pred, labels=labels, average="macro", zero_division=0
                ),
            }
        ]
    )

def get_classification_report(pred, true, class_names: list[str]) -> pd.DataFrame:
    labels = get_present_labels(true, pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=labels, average=None, zero_division=0
    )
    return pd.DataFrame(
        {
            "class": [class_names[label] for label in labels],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

def get_confusion_matrix(pred, true, class_names: list[str]) -> pd.DataFrame:
    labels = get_present_labels(true, pred)
    present_names = [class_names[label] for label in labels]
    return pd.DataFrame(
        confusion_matrix(true, pred, labels=labels), index=present_names, columns=present_names
    )

def get_evaluation_results(
    cache_folder: str, true_labels, labels: list[int] | None = None
) -> pd.DataFrame:
    pred = load_prediction_chunks(cache_folder)
    true = to_labels(true_labels)

    if len(pred) != len(true):
        raise ValueError(
            f"{len(pred):,} predictions against {len(true):,} labels in {cache_folder}/ — "
            "the cache belongs to a different split, or was written by a different run."
        )
    return evaluate(pred=pred, true=true, labels=labels)
```

Kode Program 5.77 Fungsi penghitungan metrik evaluasi

Kode Program 5.78 mengimplementasikan bagian pertama Langkah 9. Kamus *MODEL_SUBFOLDERS* memetakan nama setiap model ke subfolder penyimpanan model dan prediksinya, dan *get_prediction_folder* membentuk lokasi folder prediksi untuk suatu model dan himpunan. Fungsi *compare_models* menghitung metrik setiap model pada setiap himpunan dari prediksi tersimpan, melewati model yang prediksinya belum tersedia, dan mengurutkan hasilnya menurut himpunan dan skor. Sel terakhir membandingkan kelima model pada data validasi dan data uji.

```python
MODEL_SUBFOLDERS = {
    "KNN": KNN_SUBFOLDER,
    "SVM": SVM_SUBFOLDER,
    "Random Forest": RF_SUBFOLDER,
    "Softmax": LOGREG_SUBFOLDER,
    "XGBoost": XGB_SUBFOLDER,
}

def get_prediction_folder(subfolder: str, split_name: str) -> str:
    return get_split_folder(os.path.join(PATH_FOLDER_PREDICTION, subfolder), split_name)

def compare_models(
    subfolders: dict[str, str],
    true_labels_per_split: dict[str, object],
    score_column: str = SCORE_COLUMN,
) -> pd.DataFrame:
    rows = []
    for model_name, subfolder in subfolders.items():
        for split_name, true_labels in true_labels_per_split.items():
            folder = get_prediction_folder(subfolder, split_name)
            try:
                scores = get_evaluation_results(folder, true_labels).iloc[0]
            except FileNotFoundError:
                print(f"{model_name} / {split_name}: no cached predictions, skipped.")
                continue
            rows.append({"model": model_name, "split": split_name, **scores.to_dict()})

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    return results.sort_values(
        ["split", score_column], ascending=[True, False], ignore_index=True
    )

model_comparison = compare_models(MODEL_SUBFOLDERS, {"dev": dev_y, "test": test_y})
model_comparison
```

Kode Program 5.78 Perbandingan metrik keseluruhan kelima model

Kode Program 5.79 mengimplementasikan bagian kedua Langkah 9. Fungsi *compare_per_class_f1* menyusun tabel F1-*score* setiap kelas untuk setiap model dari laporan klasifikasi, dan *plot_per_class_f1* menggambarkannya sebagai peta panas dengan nilai pada setiap sel. Dua sel terakhir menjalankan keduanya pada data uji.

```python
def compare_per_class_f1(
    subfolders: dict[str, str],
    true_labels,
    split_name: str,
    class_names: list[str],
) -> pd.DataFrame:
    true = to_labels(true_labels)
    columns = {}

    for model_name, subfolder in subfolders.items():
        folder = get_prediction_folder(subfolder, split_name)
        try:
            pred = load_prediction_chunks(folder)
        except FileNotFoundError:
            print(f"{model_name}: no cached predictions for {split_name}, skipped.")
            continue

        report = get_classification_report(pred=pred, true=true, class_names=class_names)
        columns[model_name] = report.set_index("class")["f1"]

    return pd.DataFrame(columns).reindex(class_names)

def plot_per_class_f1(per_class_f1: pd.DataFrame, split_name: str = "test") -> None:
    surface, ink, muted = "#fcfcfb", "#0b0b0b", "#52514e"
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    colours = matplotlib.colors.LinearSegmentedColormap.from_list("f1", ramp)

    values = per_class_f1.to_numpy(dtype=float)
    figure, axes = plt.subplots(figsize=(1.6 * per_class_f1.shape[1] + 3, 0.42 * len(per_class_f1) + 2))
    figure.patch.set_facecolor(surface)

    axes.imshow(values, cmap=colours, vmin=0.0, vmax=1.0, aspect="auto")

    axes.set_xticks(range(per_class_f1.shape[1]))
    axes.set_xticklabels([str(c) for c in per_class_f1.columns], fontsize=9)
    axes.set_yticks(range(len(per_class_f1)))
    axes.set_yticklabels([str(i) for i in per_class_f1.index], fontsize=9)
    axes.set_title(f"Per-class F1 on the {split_name} split", color=ink, fontsize=11, loc="left", pad=12)
    axes.tick_params(colors=muted, length=0)
    for side in ("top", "right", "bottom", "left"):
        axes.spines[side].set_visible(False)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isnan(value):
                continue
            axes.text(
                column, row, f"{value:.2f}",
                ha="center", va="center", fontsize=9,
                color="#ffffff" if value > 0.55 else ink,
            )

    figure.tight_layout()
    plt.show()

per_class_f1 = compare_per_class_f1(MODEL_SUBFOLDERS, test_y, "test", CLASS_NAMES)
per_class_f1.round(3)

plot_per_class_f1(per_class_f1)
```

Kode Program 5.79 Perbandingan F1-score per kelas pada data uji

Kode Program 5.80 mengimplementasikan bagian ketiga Langkah 9. Fungsi *compare_search_cost* mengambil, dari tabel hasil pencarian setiap model, baris dengan skor tertinggi beserta waktu pelatihan, waktu prediksi, jumlah *support vector*, jumlah pohon, dan jumlah *epoch* yang tersedia. Sel kedua menjalankannya pada hasil pencarian kelima model.

```python
def compare_search_cost(
    search_results: dict[str, pd.DataFrame], score_column: str = SCORE_COLUMN
) -> pd.DataFrame:
    rows = []
    for model_name, results in search_results.items():
        if results is None or results.empty or score_column not in results:
            continue

        best = results.sort_values(score_column, ascending=False).iloc[0]
        rows.append(
            {
                "model": model_name,
                score_column: best[score_column],
                "fit_seconds": best.get("fit_seconds", float("nan")),
                "predict_seconds": best.get("predict_seconds", float("nan")),
                "n_support": best.get("n_support", float("nan")),
                "n_trees_used": best.get("n_trees_used", float("nan")),
                "epochs_run": best.get("epochs_run", float("nan")),
            }
        )

    return pd.DataFrame(rows).sort_values(score_column, ascending=False, ignore_index=True)

compare_search_cost(
    {
        "KNN": knn_search_results,
        "SVM": svm_search_results,
        "Random Forest": rf_search_results,
        "Softmax": logreg_search_results,
        "XGBoost": xgb_search_results,
    }
)
```

Kode Program 5.80 Perbandingan biaya komputasi konfigurasi terbaik
