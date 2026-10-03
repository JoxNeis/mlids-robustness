## **5.15 Implementasi Pengujian dan Pengukuran Ketahanan Model**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.15, yaitu prediksi kesepuluh model pada data uji bersih dan dua belas skenario gangguan, peringkasan hasil prediksi, penghitungan metrik pada setiap populasi, serta pengukuran penurunan kinerja dan peringkat ketahanan. Kode program disusun mengikuti Langkah 1 hingga 9 pada algoritma di Subbab 4.15, dilanjutkan dengan kode untuk menyajikan hasil pengujian.

### **5.15.1 Prediksi pada Setiap Skenario**

Kode Program 5.82 menetapkan pengaturan pengujian. Kamus *SMOTE_SETTINGS* dan *EVALUATED_MODELS* menetapkan dua pengaturan SMOTE dan lima algoritma yang diuji beserta nama tampilannya. Konstanta *CLEAN_SCENARIO* menetapkan nama data uji bersih, *PREDICTION_BATCH_SIZE* menetapkan ukuran kelompok prediksi sebanyak 250.000 baris, *SAVE_CONFIDENCE* menentukan apakah tingkat kepercayaan dicatat, dan *OVERWRITE_PREDICTIONS* menentukan apakah prediksi yang telah tersimpan dihitung ulang. Konstanta berikutnya menetapkan kunci ringkasan pengujian pada metadata file prediksi dan nama kolom pada file prediksi.

```python
SMOTE_SETTINGS = {
    "with-smote": "With SMOTE",
    "wout-smote": "Without SMOTE",
}

EVALUATED_MODELS = {
    "knn": "KNN",
    "svm": "SVM",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "logreg": "Logistic Regression",
}

CLEAN_SCENARIO = "clean"
PREDICTION_BATCH_SIZE = 250_000
SAVE_CONFIDENCE = True
OVERWRITE_PREDICTIONS = False
PREDICTION_SUMMARY_KEY = "prediction_summary"
DATASET_FINGERPRINT_KEY = "dataset_fingerprint"
PREDICTION_BATCH_SIZE_KEY = "prediction_batch_size"

SMOTE_SETTING_COLUMN = "smote_setting"
MODEL_COLUMN = "model"
PREDICTION_COLUMN = "prediction"
CLEAN_PREDICTION_COLUMN = "clean_prediction"
CONFIDENCE_COLUMN = "confidence"
TRUE_CLASS_PROBABILITY_COLUMN = "true_class_probability"
DISPLACEMENT_COLUMN = "displacement"
```

Kode Program 5.82 Pengaturan pengujian ketahanan

Kode Program 5.83 memuat nama kelas dari *LabelEncoder* (Subbab 5.6) dengan urutan yang sama dengan kodenya. Daftar ini digunakan untuk menentukan kode kelas *Benign* dan untuk memberi nama pada tabel dan gambar hasil.

```python
class_names = load_label_encoder().classes_.tolist()
end_status(f"{len(class_names)} classes: {', '.join(class_names)}")
```

Kode Program 5.83 Pemuatan nama kelas

Kode Program 5.84 mengimplementasikan Langkah 1. Fungsi *list_evaluation_scenarios* menyusun tiga belas data uji, yaitu data uji bersih dengan intensitas nol diikuti dua belas skenario gangguan dari Subbab 5.14. Fungsi *list_scenario_names* mengembalikan nama data uji tersebut, dengan pilihan untuk mengecualikan data uji bersih. Fungsi *get_scenario_details* mengembalikan jenis gangguan dan intensitas sebuah skenario, sedangkan *get_scenario_folder_path* mengembalikan lokasi folder data uji, yaitu folder data uji hasil pembagian untuk data uji bersih dan folder skenario untuk skenario gangguan.

```python
def list_evaluation_scenarios() -> list[tuple[str, float, str]]:
    scenarios = [(CLEAN_SCENARIO, 0.0, CLEAN_SCENARIO)]
    scenarios.extend(list_noise_scenarios())
    return scenarios

def list_scenario_names(include_clean: bool = True) -> list[str]:
    scenarios = []
    for _, _, scenario in list_evaluation_scenarios():
        if include_clean or scenario != CLEAN_SCENARIO:
            scenarios.append(scenario)
    return scenarios

def get_scenario_details(scenario: str) -> tuple[str, float]:
    for noise_type, intensity, scenario_name in list_evaluation_scenarios():
        if scenario_name == scenario:
            return noise_type, intensity
    raise ValueError(f"Unknown scenario: {scenario}")

def get_scenario_folder_path(scenario: str) -> str:
    if scenario == CLEAN_SCENARIO:
        return PATH_FOLDER_SPLITTED_DATASET_TEST
    return os.path.join(PATH_FOLDER_NOISY_DATASET_TEST, scenario)
```

Kode Program 5.84 Daftar dan lokasi data uji

Fungsi *scan_scenario_split* pada Kode Program 5.85 membaca fitur sebuah data uji secara *lazy* dan memuat labelnya. Apabila file fitur memiliki kolom *is_noise*, kolom tersebut dimuat sebagai penanda baris terganggu dan dibuang dari fitur, sedangkan pada data uji bersih seluruh baris ditandai tidak terganggu. Proses dihentikan dengan galat apabila jumlah baris fitur dan label berbeda.

```python
def scan_scenario_split(scenario: str) -> tuple[pl.LazyFrame, np.ndarray, np.ndarray]:
    folder_path = get_scenario_folder_path(scenario)
    features = pl.scan_parquet(os.path.join(folder_path, FEATURES_FILE_NAME))
    labels = load_labels(folder_path).to_numpy()
    if IS_NOISE_COLUMN in features.collect_schema().names():
        is_noise = features.select(IS_NOISE_COLUMN).collect().to_series().to_numpy()
        features = features.drop(IS_NOISE_COLUMN)
    else:
        is_noise = np.zeros(len(labels), dtype=bool)
    if count_rows(features) != len(labels):
        raise ValueError(f"{folder_path}: {count_rows(features):,} feature rows but {len(labels):,} labels")
    return features, labels, is_noise
```

Kode Program 5.85 Pembacaan data uji sebuah skenario

Kode Program 5.86 mengimplementasikan Langkah 3. Fungsi *get_trained_model_path* membentuk lokasi file model, dan *load_trained_pipeline* memuat *pipeline* tersebut atau melewatinya dengan pemberitahuan apabila model belum dilatih. Fungsi *split_trained_pipeline* memisahkan *pipeline* menjadi tahap praproses dan model klasifikasi, dengan membuang tahap yang memiliki fungsi *fit_resample*, yaitu SMOTE, karena tahap tersebut hanya berlaku pada pelatihan. Fungsi *summarize_pipeline_settings* mencatat nama kelas model, jumlah komponen utama, persentase varians yang dipertahankan, dan jumlah tetangga SMOTE.

```python
def get_trained_model_path(smote_setting: str, model_name: str) -> str:
    return os.path.join(PATH_FOLDER_TRAINED_MODEL, smote_setting, f"{model_name}.pkl")

def load_trained_pipeline(smote_setting: str, model_name: str) -> Pipeline | None:
    model_path = get_trained_model_path(smote_setting, model_name)
    if not os.path.exists(model_path):
        end_status(f"No trained model at {model_path}, skipped")
        return None

    print_status(f"Loading {model_path}")
    return load(model_path)

def split_trained_pipeline(pipeline: Pipeline) -> tuple[Pipeline, Any]:
    preprocessing_steps = []
    for step_name, step in pipeline.steps[:-1]:
        if not hasattr(step, "fit_resample"):
            preprocessing_steps.append((step_name, step))
    return Pipeline(preprocessing_steps), pipeline.steps[-1][1]

def summarize_pipeline_settings(pipeline: Pipeline) -> dict:
    pca = pipeline.named_steps["pca"]
    smote = pipeline.named_steps.get("smote")
    return {
        "classifier": type(pipeline.steps[-1][1]).__name__,
        "pca_components": int(pca.n_components_),
        "pca_variance_percentage": float(pca.explained_variance_ratio_.sum() * 100),
        "smote_k_neighbors": None if smote is None else int(smote.k_neighbors),
    }
```

Kode Program 5.86 Pemuatan dan pemisahan pipeline terlatih

Kode Program 5.87 memuat fungsi pengelolaan file prediksi. Fungsi *get_predictions_path* membentuk lokasi file prediksi dengan subfolder bernama *smote_setting=…*, *model=…*, dan *scenario=…*. Fungsi *save_predictions* menyimpan tabel prediksi sebagai file Parquet dengan ringkasan pengujian dalam bentuk JSON pada metadata skema file, ditulis ke file sementara terlebih dahulu kemudian diganti namanya dengan *os.replace*. Fungsi *scan_predictions* membaca file prediksi secara *lazy*, sedangkan *read_prediction_summary* membaca ringkasan pengujian dari metadata tanpa membaca isi file.

```python
def get_predictions_path(smote_setting: str, model_name: str, scenario: str) -> str:
    return os.path.join(
        PATH_FOLDER_PREDICTIONS,
        f"{SMOTE_SETTING_COLUMN}={smote_setting}",
        f"{MODEL_COLUMN}={model_name}",
        f"{SCENARIO_COLUMN}={scenario}",
        PREDICTIONS_FILE_NAME,
    )

def save_predictions(predictions: pd.DataFrame, summary: dict, predictions_path: str) -> None:
    table = pa.Table.from_pandas(predictions, preserve_index=False)
    metadata = {**table.schema.metadata, PREDICTION_SUMMARY_KEY.encode(): json.dumps(summary).encode()}
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    # written under a temporary name first, so an interrupted run never leaves a file that looks finished
    temporary_path = f"{predictions_path}.tmp"
    pq.write_table(table.replace_schema_metadata(metadata), temporary_path)
    os.replace(temporary_path, predictions_path)

def scan_predictions(smote_setting: str, model_name: str, scenario: str) -> pl.LazyFrame:
    return pl.scan_parquet(get_predictions_path(smote_setting, model_name, scenario))

def read_prediction_summary(smote_setting: str, model_name: str, scenario: str) -> dict:
    metadata = pq.read_schema(get_predictions_path(smote_setting, model_name, scenario)).metadata or {}
    return json.loads(metadata.get(PREDICTION_SUMMARY_KEY.encode(), b"{}"))
```

Kode Program 5.87 Penyimpanan dan pembacaan file prediksi

Kode Program 5.88 mengimplementasikan Langkah 2. Fungsi *fingerprint_file* menghitung sidik jari BLAKE2b dari isi sebuah file dan menyimpannya pada kamus *file_fingerprints* berdasarkan lokasi, ukuran, dan waktu perubahan file, sehingga file yang sama tidak di-*hash* berulang kali. Fungsi *is_scenario_predicted* menyatakan sebuah pengujian telah selesai apabila file prediksinya ada, model tidak lebih baru daripada file prediksi, ukuran kelompok pada ringkasan sama dengan *PREDICTION_BATCH_SIZE*, dan sidik jari data uji pada ringkasan sama dengan sidik jari file data uji saat ini. Fungsi *list_pending_scenarios* mengembalikan seluruh data uji apabila data uji bersih belum selesai, dan hanya data uji yang belum selesai apabila sebaliknya.

```python
file_fingerprints = {}


def fingerprint_file(file_path: str) -> str:
    key = (file_path, os.path.getsize(file_path), os.path.getmtime(file_path))
    if key not in file_fingerprints:
        print_status(f"Fingerprinting {file_path}")
        with open(file_path, "rb") as file:
            file_fingerprints[key] = hashlib.file_digest(file, "blake2b").hexdigest()
    return file_fingerprints[key]

def is_scenario_predicted(
    smote_setting: str,
    model_name: str,
    scenario: str,
    overwrite: bool = OVERWRITE_PREDICTIONS,
    batch_size: int = PREDICTION_BATCH_SIZE,
) -> bool:
    predictions_path = get_predictions_path(smote_setting, model_name, scenario)
    features_path = os.path.join(get_scenario_folder_path(scenario), FEATURES_FILE_NAME)
    if overwrite or not os.path.exists(predictions_path) or not os.path.exists(features_path):
        return False

    model_path = get_trained_model_path(smote_setting, model_name)
    if os.path.exists(model_path) and os.path.getmtime(model_path) > os.path.getmtime(predictions_path):
        return False
    summary = read_prediction_summary(smote_setting, model_name, scenario)
    # the batch size changes the last bits of the components, so every scenario of a model needs the same one
    if summary.get(PREDICTION_BATCH_SIZE_KEY) != batch_size:
        return False
    return summary.get(DATASET_FINGERPRINT_KEY) == fingerprint_file(features_path)

def list_pending_scenarios(smote_setting: str, model_name: str) -> list[str]:
    scenarios = list_scenario_names()
    if not is_scenario_predicted(smote_setting, model_name, CLEAN_SCENARIO):
        return scenarios

    pending_scenarios = []
    for scenario in scenarios:
        if not is_scenario_predicted(smote_setting, model_name, scenario):
            pending_scenarios.append(scenario)
    return pending_scenarios
```

Kode Program 5.88 Pemeriksaan pengujian yang perlu dijalankan

Kode Program 5.89 memuat fungsi prediksi satu kelompok. Fungsi *has_probabilities* menentukan apakah model menyediakan peluang kelas, yang pada *scikit-learn* ditandai dengan keberadaan fungsi *predict_proba*. Fungsi *predict_batch* menghitung peluang seluruh kelas satu kali apabila tersedia, kemudian menetapkan prediksi sebagai kelas dengan peluang terbesar, tingkat kepercayaan sebagai peluang terbesar tersebut, dan peluang kelas sebenarnya dari kolom yang sesuai dengan label. Komentar pada fungsi tersebut mencatat bahwa prediksi KNN, *Random Forest*, *XGBoost*, dan *Logistic Regression* sama dengan kelas berpeluang terbesar, sehingga satu perhitungan menghasilkan keduanya. Apabila peluang tidak tersedia, yaitu pada SVM, prediksi dihitung dengan *predict* dan kolom peluang diisi NaN. Jumlah kolom peluang diperiksa harus sama dengan jumlah kelas.

```python
def has_probabilities(model: Any) -> bool:
    # SVC only has predict_proba when it is fitted with probability=True, and LinearSVC never has it
    return hasattr(model, "predict_proba")

def predict_batch(
    model: Any,
    components: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    save_confidence: bool = SAVE_CONFIDENCE,
) -> pd.DataFrame:
    if save_confidence and has_probabilities(model):
        probabilities = np.asarray(model.predict_proba(components), dtype=np.float32)
        if probabilities.shape[1] != class_count:
            raise ValueError(f"predict_proba gave {probabilities.shape[1]} classes instead of {class_count}")
        # predict of KNN, Random Forest, XGBoost and Logistic Regression is the argmax of predict_proba,
        # so one pass gives both and the neighbour search of KNN runs once
        return pd.DataFrame({
            PREDICTION_COLUMN: probabilities.argmax(axis=1).astype(np.int64),
            CONFIDENCE_COLUMN: probabilities.max(axis=1),
            TRUE_CLASS_PROBABILITY_COLUMN: probabilities[np.arange(len(labels)), labels],
        })

    no_probabilities = np.full(len(components), np.nan, dtype=np.float32)
    return pd.DataFrame({
        PREDICTION_COLUMN: np.asarray(model.predict(components), dtype=np.int64),
        CONFIDENCE_COLUMN: no_probabilities,
        TRUE_CLASS_PROBABILITY_COLUMN: no_probabilities,
    })
```

Kode Program 5.89 Prediksi satu kelompok baris

Fungsi *predict_scenario* pada Kode Program 5.90 mengimplementasikan Langkah 4 dan 5 untuk satu data uji. Fungsi ini menghitung sidik jari data uji, membaca data uji tersebut beserta data uji bersih, kemudian memproses keduanya per kelompok 250.000 baris dengan batas yang sama. Pada setiap kelompok, data uji dan data uji bersih ditransformasikan dengan tahap praproses yang sama, data uji diprediksi dengan *predict_batch*, dan perpindahan setiap baris dihitung sebagai norma Euclidean selisih komponen utama keduanya sesuai Persamaan 4.15. Waktu praproses dan waktu prediksi dijumlahkan dari seluruh kelompok. Hasil seluruh kelompok digabung, kemudian ditambah label, prediksi bersih, dan penanda baris terganggu. Prediksi bersih diambil dari prediksi data uji bersih yang diberikan, atau dari prediksi itu sendiri apabila data uji yang diprediksi adalah data uji bersih. Ringkasan pengujian disimpan bersama prediksi, dan F1-*score* makro serta laju perubahan prediksi ditampilkan sebagai pemeriksaan cepat.

```python
def predict_scenario(
    smote_setting: str,
    model_name: str,
    scenario: str,
    preprocessing: Pipeline,
    model: Any,
    clean_predictions: np.ndarray | None,
    pipeline_settings: dict,
    class_count: int,
    batch_size: int = PREDICTION_BATCH_SIZE,
) -> np.ndarray:
    title = f"{EVALUATED_MODELS[model_name]} ({SMOTE_SETTINGS[smote_setting]}), {scenario}"
    dataset_fingerprint = fingerprint_file(os.path.join(get_scenario_folder_path(scenario), FEATURES_FILE_NAME))
    features, labels, is_noise = scan_scenario_split(scenario)
    clean_features, _, _ = scan_scenario_split(CLEAN_SCENARIO)

    batches = []
    transform_seconds = 0.0
    predict_seconds = 0.0
    for row_offset, batch in read_in_batches(features, f"{title}: predicting", batch_size):
        started = time.time()
        components = preprocessing.transform(batch.to_pandas())
        transform_seconds += time.time() - started
        clean_batch = clean_features.slice(row_offset, batch_size).collect()
        clean_components = preprocessing.transform(clean_batch.to_pandas())

        started = time.time()
        batch_labels = labels[row_offset:row_offset + len(batch)]
        batch_predictions = predict_batch(model, components, batch_labels, class_count)
        predict_seconds += time.time() - started
        displacements = np.linalg.norm(components - clean_components, axis=1).astype(np.float32)
        batch_predictions[DISPLACEMENT_COLUMN] = displacements
        batches.append(batch_predictions)
    predictions = pd.concat(batches, ignore_index=True)

    if clean_predictions is None:
        clean_predictions = predictions[PREDICTION_COLUMN].to_numpy()
    predictions.insert(0, LABEL_COLUMN, labels)
    predictions.insert(2, CLEAN_PREDICTION_COLUMN, clean_predictions)
    predictions.insert(3, IS_NOISE_COLUMN, is_noise)

    summary = {
        SMOTE_SETTING_COLUMN: smote_setting,
        MODEL_COLUMN: model_name,
        SCENARIO_COLUMN: scenario,
        ROW_COUNT_COLUMN: len(predictions),
        "noisy_rows": int(is_noise.sum()),
        "transform_seconds": transform_seconds,
        "predict_seconds": predict_seconds,
        "has_confidence": bool(predictions[CONFIDENCE_COLUMN].notna().any()),
        DATASET_FINGERPRINT_KEY: dataset_fingerprint,
        PREDICTION_BATCH_SIZE_KEY: batch_size,
        **pipeline_settings,
    }
    save_predictions(predictions, summary, get_predictions_path(smote_setting, model_name, scenario))

    f1_macro = f1_score(labels, predictions[PREDICTION_COLUMN], average="macro", zero_division=0)
    flip_rate = np.mean(predictions[PREDICTION_COLUMN].to_numpy() != clean_predictions)
    end_status(
        f"{title}: predicted in {format_duration(predict_seconds)}, f1_macro {f1_macro:.4f}, "
        f"{flip_rate:.2%} of the rows changed prediction"
    )
    return predictions[PREDICTION_COLUMN].to_numpy()
```

Kode Program 5.90 Prediksi satu data uji

Kode Program 5.91 menyatukan seluruh prediksi. Fungsi *free_memory* menjalankan pengumpul sampah (*garbage collector*) Python untuk membebaskan memori di antara pengujian. Fungsi *predict_every_scenario* menentukan data uji yang belum selesai untuk satu model, memuat dan memisahkan *pipeline*-nya, membaca prediksi data uji bersih yang telah tersimpan apabila data uji bersih tidak perlu diulang, kemudian menjalankan *predict_scenario* untuk setiap data uji secara berurutan, dimulai dari data uji bersih. Skenario yang folder datanya belum tersedia dilewati dengan pemberitahuan. Sel terakhir menjalankan prediksi untuk kelima algoritma pada kedua pengaturan SMOTE.

```python
def free_memory() -> None:
    gc.collect()

def predict_every_scenario(smote_setting: str, model_name: str, class_count: int) -> None:
    title = f"{EVALUATED_MODELS[model_name]} ({SMOTE_SETTINGS[smote_setting]})"
    scenarios = list_pending_scenarios(smote_setting, model_name)
    if not scenarios:
        end_status(f"{title}: every scenario is already predicted, skipped")
        return
    pipeline = load_trained_pipeline(smote_setting, model_name)
    if pipeline is None:
        return
    preprocessing, model = split_trained_pipeline(pipeline)
    pipeline_settings = summarize_pipeline_settings(pipeline)

    clean_predictions = None
    if CLEAN_SCENARIO not in scenarios:
        clean_predictions = scan_predictions(smote_setting, model_name, CLEAN_SCENARIO).select(PREDICTION_COLUMN)
        clean_predictions = clean_predictions.collect().to_series().to_numpy()
    end_status(f"{title}: {len(scenarios)} scenarios to predict")

    for scenario in scenarios:
        scenario_folder_path = get_scenario_folder_path(scenario)
        if not os.path.exists(os.path.join(scenario_folder_path, FEATURES_FILE_NAME)):
            end_status(f"{title}, {scenario}: no dataset in {scenario_folder_path}, skipped")
            continue
        predictions = predict_scenario(
            smote_setting,
            model_name,
            scenario,
            preprocessing,
            model,
            clean_predictions,
            pipeline_settings,
            class_count,
        )
        if scenario == CLEAN_SCENARIO:
            clean_predictions = predictions
        free_memory()

for smote_setting in SMOTE_SETTINGS:
    for model_name in EVALUATED_MODELS:
        predict_every_scenario(smote_setting, model_name, len(class_names))
        free_memory()
```

Kode Program 5.91 Prediksi seluruh model pada seluruh data uji

### **5.15.2 Peringkasan Hasil Prediksi dan Penghitungan Metrik**

Kode Program 5.92 menetapkan pengaturan perbandingan model. Kamus *POPULATIONS* menetapkan tiga populasi, yaitu *all*, *intact*, dan *noisy*. Konstanta berikutnya menetapkan nama kelas *Benign*, ambang kesalahan yakin sebesar 0,9, batas bawah peluang 10⁻⁷ untuk *log loss*, jumlah desil perpindahan, serta pilihan penyimpanan dan penampilan matriks konfusi. Daftar *OUTCOME_COLUMNS* dan *COMBINATION_COLUMNS* menetapkan kolom pengelompokan tabel hasil dan kolom pengenal setiap pengujian. Kamus *METRIC_LABELS* menetapkan nama tampilan setiap metrik, *ROBUSTNESS_METRIC* menetapkan F1-*score* makro sebagai metrik utama, *ROBUSTNESS_METRICS* menetapkan sepuluh metrik ketahanan, *COMPARED_METRICS* menetapkan empat metrik pada perbandingan data uji bersih, dan *INTENSITY_PLOTS* menetapkan pasangan metrik dan populasi yang digambarkan terhadap intensitas.

```python
POPULATIONS = {
    "all": "All Rows",
    "intact": "Intact Rows",
    "noisy": "Noisy Rows",
}
BENIGN_CLASS = "Benign"
CONFIDENT_ERROR_THRESHOLD = 0.9
PROBABILITY_FLOOR = 1e-7
DISPLACEMENT_BIN_COUNT = 10
SAVE_MATRIX_PLOTS = True
SHOW_NOISY_MATRICES = False

OUTCOME_COLUMNS = [IS_NOISE_COLUMN, LABEL_COLUMN, CLEAN_PREDICTION_COLUMN, PREDICTION_COLUMN]
COMBINATION_COLUMNS = [SMOTE_SETTING_COLUMN, MODEL_COLUMN, SCENARIO_COLUMN, NOISE_TYPE_COLUMN, INTENSITY_COLUMN]
CONFIDENCE_SUM_COLUMN = "confidence_sum"
CONFIDENT_ROWS_COLUMN = "confident_rows"
LOG_LOSS_SUM_COLUMN = "log_loss_sum"
DISPLACEMENT_SUM_COLUMN = "displacement_sum"
POPULATION_COLUMN = "population"
METRIC_COLUMN = "metric"
VALUE_COLUMN = "value"
CLEAN_VALUE_COLUMN = "clean_value"
DROP_COLUMN = "drop"
RETENTION_COLUMN = "retention"
BIN_COLUMN = "bin"

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)",
    "f1_macro": "F1-score (macro)",
    "precision_weighted": "Precision (weighted)",
    "recall_weighted": "Recall (weighted)",
    "f1_weighted": "F1-score (weighted)",
    "mcc": "Matthews Correlation",
    "cohen_kappa": "Cohen's Kappa",
    "detection_rate": "Attack Detection Rate",
    "attack_precision": "Attack Alarm Precision",
    "attack_f1": "Attack F1-score",
    "false_alarm_rate": "False Alarm Rate",
    "flip_rate": "Prediction Flip Rate",
    "evasion_rate": "Attack Evasion Rate",
    "mean_confidence": "Mean Confidence",
    "confident_error_share": "Confident Error Share",
}
ROBUSTNESS_METRIC = "f1_macro"
ROBUSTNESS_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "mcc",
    "cohen_kappa",
    "detection_rate",
    "attack_f1",
]
COMPARED_METRICS = {
    "accuracy": "Accuracy",
    "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)",
    "f1_macro": "F1-score (macro)",
}
INTENSITY_PLOTS = [
    ("f1_macro", "all"),
    ("f1_macro", "noisy"),
    ("accuracy", "all"),
    ("mcc", "all"),
    ("detection_rate", "noisy"),
    ("false_alarm_rate", "noisy"),
    ("flip_rate", "noisy"),
    ("mean_confidence", "noisy"),
]
```

Kode Program 5.92 Pengaturan perbandingan model

Kode Program 5.93 mengimplementasikan Langkah 6. Fungsi *add_combination_columns* menambahkan kolom pengaturan SMOTE, model, skenario, jenis gangguan, dan intensitas pada sebuah tabel. Fungsi *count_outcomes* membaca file prediksi secara *lazy* dan mengelompokkannya menurut penanda baris terganggu, label, prediksi bersih, dan prediksi pada skenario. Untuk setiap kelompok dihitung jumlah baris, jumlah tingkat kepercayaan, jumlah baris dengan tingkat kepercayaan sedikitnya 0,9, jumlah *log loss* dari peluang kelas sebenarnya yang dibatasi minimum 10⁻⁷, dan jumlah perpindahan. Agregasi dijalankan dengan mesin *streaming* *Polars*. Perbandingan tingkat kepercayaan dengan ambang dilakukan dalam tipe *float32* sesuai tipe yang tersimpan, sehingga tingkat kepercayaan yang tepat bernilai 0,9 tetap terhitung sebagai kesalahan yakin. Apabila model tidak menyediakan tingkat kepercayaan, kolom yang berkaitan diisi NaN.

```python
def add_combination_columns(table: pd.DataFrame, smote_setting: str, model_name: str, scenario: str) -> pd.DataFrame:
    noise_type, intensity = get_scenario_details(scenario)
    combination = {
        SMOTE_SETTING_COLUMN: smote_setting,
        MODEL_COLUMN: model_name,
        SCENARIO_COLUMN: scenario,
        NOISE_TYPE_COLUMN: noise_type,
        INTENSITY_COLUMN: intensity,
    }
    table = table.copy()
    for position, (column, value) in enumerate(combination.items()):
        table.insert(position, column, value)
    return table

def count_outcomes(
    predictions: pl.LazyFrame,
    confident_threshold: float = CONFIDENT_ERROR_THRESHOLD,
    probability_floor: float = PROBABILITY_FLOOR,
) -> pd.DataFrame:
    confidence = pl.col(CONFIDENCE_COLUMN).fill_nan(None)
    true_class_probability = pl.col(TRUE_CLASS_PROBABILITY_COLUMN).cast(pl.Float64).fill_nan(None)
    outcomes = predictions.group_by(OUTCOME_COLUMNS).agg(
        pl.len().cast(pl.Int64).alias(ROW_COUNT_COLUMN),
        confidence.cast(pl.Float64).sum().alias(CONFIDENCE_SUM_COLUMN),
        # compared in float32 like the stored confidence, so a confidence of 0.9 counts as confident
        (confidence >= confident_threshold).sum().cast(pl.Float64).alias(CONFIDENT_ROWS_COLUMN),
        (-true_class_probability.clip(lower_bound=probability_floor).log()).sum().alias(LOG_LOSS_SUM_COLUMN),
        pl.col(DISPLACEMENT_COLUMN).cast(pl.Float64).sum().alias(DISPLACEMENT_SUM_COLUMN),
    )
    outcomes = outcomes.sort(OUTCOME_COLUMNS).collect(engine="streaming").to_pandas()
    if not predictions.select(confidence.is_not_null().any()).collect().item():
        outcomes[[CONFIDENCE_SUM_COLUMN, CONFIDENT_ROWS_COLUMN, LOG_LOSS_SUM_COLUMN]] = np.nan
    return outcomes
```

Kode Program 5.93 Peringkasan file prediksi menjadi tabel hasil

Fungsi *bin_displacement* pada Kode Program 5.94 membaca hanya baris terganggu dari file prediksi, membagi baris tersebut menjadi sepuluh desil menurut perpindahannya menggunakan *pandas.qcut*, dan menghitung jumlah baris, rata-rata perpindahan, laju perubahan prediksi, dan laju kesalahan pada setiap desil. Desil dengan batas yang sama digabungkan, dan data uji bersih yang tidak memiliki baris terganggu menghasilkan tabel kosong.

```python
def bin_displacement(predictions: pl.LazyFrame, bin_count: int = DISPLACEMENT_BIN_COUNT) -> pd.DataFrame:
    columns = [LABEL_COLUMN, PREDICTION_COLUMN, CLEAN_PREDICTION_COLUMN, DISPLACEMENT_COLUMN]
    noisy_rows = predictions.filter(pl.col(IS_NOISE_COLUMN)).select(columns).collect().to_pandas()
    if noisy_rows.empty:
        return pd.DataFrame()

    noisy_rows = noisy_rows.assign(**{
        BIN_COLUMN: pd.qcut(noisy_rows[DISPLACEMENT_COLUMN], bin_count, labels=False, duplicates="drop"),
        "is_flipped": noisy_rows[PREDICTION_COLUMN] != noisy_rows[CLEAN_PREDICTION_COLUMN],
        "is_error": noisy_rows[PREDICTION_COLUMN] != noisy_rows[LABEL_COLUMN],
    })
    bins = noisy_rows.groupby(BIN_COLUMN).agg(**{
        ROW_COUNT_COLUMN: (DISPLACEMENT_COLUMN, "size"),
        "mean_displacement": (DISPLACEMENT_COLUMN, "mean"),
        "flip_rate": ("is_flipped", "mean"),
        "error_rate": ("is_error", "mean"),
    })
    return bins.reset_index()
```

Kode Program 5.94 Desil perpindahan pada baris terganggu

Kode Program 5.95 menjalankan peringkasan pada seluruh file prediksi. Fungsi *list_predicted_combinations* mendaftar pasangan pengaturan SMOTE, model, dan data uji yang telah memiliki file prediksi. Fungsi *count_every_outcome* meringkas setiap file menjadi tabel hasil dan desil perpindahan, menggabungkan seluruhnya, dan menyimpannya pada file *outcome-counts.parquet* dan *displacement-bins.csv*. Sel terakhir menjalankan peringkasan.

```python
def list_predicted_combinations() -> list[tuple[str, str, str]]:
    combinations = []
    for smote_setting in SMOTE_SETTINGS:
        for model_name in EVALUATED_MODELS:
            for scenario in list_scenario_names():
                if os.path.exists(get_predictions_path(smote_setting, model_name, scenario)):
                    combinations.append((smote_setting, model_name, scenario))
    return combinations

def count_every_outcome(evaluation_folder_path: str = PATH_FOLDER_EVALUATION) -> tuple[pd.DataFrame, pd.DataFrame]:
    combinations = list_predicted_combinations()
    if not combinations:
        raise FileNotFoundError(f"No predictions in {PATH_FOLDER_PREDICTIONS}")

    outcome_tables = []
    displacement_tables = []
    started = time.time()
    for combination_index, (smote_setting, model_name, scenario) in enumerate(combinations, start=1):
        predictions = scan_predictions(smote_setting, model_name, scenario)
        outcomes = count_outcomes(predictions)
        outcome_tables.append(add_combination_columns(outcomes, smote_setting, model_name, scenario))
        displacement_bins = bin_displacement(predictions)
        if not displacement_bins.empty:
            displacement_tables.append(add_combination_columns(displacement_bins, smote_setting, model_name, scenario))
        print_progress(combination_index, len(combinations), "Counting the outcomes of predictions", started=started)

    outcomes = pd.concat(outcome_tables, ignore_index=True)
    displacement_bins = pd.DataFrame()
    if displacement_tables:
        displacement_bins = pd.concat(displacement_tables, ignore_index=True)
    os.makedirs(evaluation_folder_path, exist_ok=True)
    outcomes.to_parquet(os.path.join(evaluation_folder_path, OUTCOME_COUNTS_FILE_NAME), index=False)
    displacement_bins.to_csv(os.path.join(evaluation_folder_path, DISPLACEMENT_BINS_FILE_NAME), index=False)
    end_status(f"Counted the outcomes of {len(combinations)} predictions into {len(outcomes):,} rows")
    return outcomes, displacement_bins

outcomes, displacement_bins = count_every_outcome()
display(outcomes.head(10))
```

Kode Program 5.95 Peringkasan seluruh file prediksi

Kode Program 5.96 memuat dua fungsi bantu penghitungan metrik. Fungsi *mean_per_row* menghitung rata-rata per baris dari jumlah yang tersimpan pada tabel hasil, yaitu jumlah nilai pada kelompok terpilih dibagi jumlah baris kelompok tersebut, dan mengembalikan NaN apabila tidak ada baris yang terpilih. Fungsi *select_population* memilih kelompok pada tabel hasil yang termasuk populasi *noisy*, *intact*, atau seluruhnya.

```python
def mean_per_row(sums: np.ndarray, rows: np.ndarray, is_selected: np.ndarray) -> float:
    selected_rows = rows[is_selected].sum()
    if selected_rows == 0:
        return np.nan
    return float(sums[is_selected].sum() / selected_rows)

def select_population(outcomes: pd.DataFrame, population: str) -> pd.DataFrame:
    if population == "noisy":
        return outcomes[outcomes[IS_NOISE_COLUMN]]
    if population == "intact":
        return outcomes[~outcomes[IS_NOISE_COLUMN]]
    return outcomes
```

Kode Program 5.96 Fungsi bantu penghitungan metrik

Fungsi *score_predictions* pada Kode Program 5.97 mengimplementasikan metrik kinerja klasifikasi dan deteksi serangan pada Tabel 4.4. Seluruh metrik dihitung dengan fungsi *sklearn.metrics* menggunakan jumlah baris setiap kelompok sebagai bobot sampel (*sample_weight*). Rata-rata makro dihitung pada kelas yang muncul pada label maupun prediksi di populasi tersebut, sebagaimana ditunjukkan oleh komentar pada fungsi. Metrik deteksi serangan dihitung dengan mengubah label dan prediksi menjadi dua kelas, yaitu serangan dan bukan serangan. Fungsi ini juga menghitung laju alarm palsu serta proporsi tiga kelompok kesalahan, yaitu serangan yang diprediksi *Benign*, serangan yang tertukar dengan serangan lain, dan *Benign* yang diprediksi serangan, yang digunakan pada Subbab 5.16. Parameter *prediction_column* memungkinkan fungsi yang sama digunakan untuk prediksi bersih.

```python
def score_predictions(outcomes: pd.DataFrame, benign: int, prediction_column: str = PREDICTION_COLUMN) -> dict:
    labels = outcomes[LABEL_COLUMN].to_numpy()
    predictions = outcomes[prediction_column].to_numpy()
    rows = outcomes[ROW_COUNT_COLUMN].to_numpy()
    # a rare class can be absent from a population, and counting it would add an F1-score of 0 to the macro average
    class_codes = np.union1d(labels, predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, labels=class_codes, average="macro", sample_weight=rows, zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, labels=class_codes, average="weighted", sample_weight=rows, zero_division=0
    )
    is_attack = labels != benign
    is_alarm = predictions != benign
    attack_precision, detection_rate, attack_f1, _ = precision_recall_fscore_support(
        is_attack, is_alarm, average="binary", sample_weight=rows, zero_division=0
    )
    is_error = labels != predictions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        balanced_accuracy = balanced_accuracy_score(labels, predictions, sample_weight=rows)
    return {
        "accuracy": accuracy_score(labels, predictions, sample_weight=rows),
        "balanced_accuracy": balanced_accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "mcc": matthews_corrcoef(labels, predictions, sample_weight=rows),
        "cohen_kappa": cohen_kappa_score(labels, predictions, labels=class_codes, sample_weight=rows),
        "detection_rate": detection_rate,
        "attack_precision": attack_precision,
        "attack_f1": attack_f1,
        "false_alarm_rate": mean_per_row(rows * is_alarm, rows, ~is_attack),
        "missed_attack_share": mean_per_row(rows * (is_attack & ~is_alarm), rows, is_error),
        "attack_confusion_share": mean_per_row(rows * (is_attack & is_alarm), rows, is_error),
        "false_alarm_share": mean_per_row(rows * ~is_attack, rows, is_error),
    }
```

Kode Program 5.97 Metrik kinerja klasifikasi dan deteksi serangan

Kode Program 5.98 mengimplementasikan metrik stabilitas prediksi dan kepercayaan pada Tabel 4.4. Fungsi *score_stability* menghitung laju perubahan prediksi, laju rusak, laju pulih, laju lolos serangan pada serangan yang terdeteksi pada data bersih, dan laju alarm palsu baru pada lalu lintas *Benign* yang diprediksi *Benign* pada data bersih. Fungsi *score_confidence* menghitung rata-rata tingkat kepercayaan pada seluruh baris dan pada kesalahan, proporsi kesalahan yakin, *log loss*, dan rata-rata perpindahan dari jumlah yang tersimpan pada tabel hasil.

```python
def score_stability(outcomes: pd.DataFrame, benign: int) -> dict:
    labels = outcomes[LABEL_COLUMN].to_numpy()
    predictions = outcomes[PREDICTION_COLUMN].to_numpy()
    clean_predictions = outcomes[CLEAN_PREDICTION_COLUMN].to_numpy()
    rows = outcomes[ROW_COUNT_COLUMN].to_numpy()
    every_row = np.ones(len(rows), dtype=bool)
    was_correct = clean_predictions == labels
    is_correct = predictions == labels
    is_attack = labels != benign
    return {
        "flip_rate": mean_per_row(rows * (predictions != clean_predictions), rows, every_row),
        "broken_rate": mean_per_row(rows * (was_correct & ~is_correct), rows, every_row),
        "fixed_rate": mean_per_row(rows * (~was_correct & is_correct), rows, every_row),
        "evasion_rate": mean_per_row(rows * (predictions == benign), rows, is_attack & (clean_predictions != benign)),
        "induced_false_alarm_rate": mean_per_row(
            rows * (predictions != benign), rows, ~is_attack & (clean_predictions == benign)
        ),
    }

def score_confidence(outcomes: pd.DataFrame) -> dict:
    rows = outcomes[ROW_COUNT_COLUMN].to_numpy()
    every_row = np.ones(len(rows), dtype=bool)
    is_error = (outcomes[LABEL_COLUMN] != outcomes[PREDICTION_COLUMN]).to_numpy()
    confidence_sums = outcomes[CONFIDENCE_SUM_COLUMN].to_numpy()
    return {
        "mean_confidence": mean_per_row(confidence_sums, rows, every_row),
        "mean_error_confidence": mean_per_row(confidence_sums, rows, is_error),
        "confident_error_share": mean_per_row(outcomes[CONFIDENT_ROWS_COLUMN].to_numpy(), rows, is_error),
        "log_loss": mean_per_row(outcomes[LOG_LOSS_SUM_COLUMN].to_numpy(), rows, every_row),
        "mean_displacement": mean_per_row(outcomes[DISPLACEMENT_SUM_COLUMN].to_numpy(), rows, every_row),
    }
```

Kode Program 5.98 Metrik stabilitas prediksi dan tingkat kepercayaan

Kode Program 5.99 mengimplementasikan Langkah 7. Fungsi *score_population* menggabungkan seluruh metrik sebuah populasi, menghitung kesenjangan kepercayaan, dan menambahkan metrik prediksi bersih pada baris yang sama dengan awalan *clean_*. Fungsi *measure_every_population* mengelompokkan tabel hasil menurut pengujian, menilai setiap populasi yang tidak kosong, dan menyusun hasilnya menjadi satu tabel. Sel terakhir menjalankannya dan menyimpan hasilnya pada file *metrics.csv*.

```python
def score_population(outcomes: pd.DataFrame, benign: int) -> dict:
    scores = {
        ROW_COUNT_COLUMN: int(outcomes[ROW_COUNT_COLUMN].sum()),
        **score_predictions(outcomes, benign),
        **score_stability(outcomes, benign),
        **score_confidence(outcomes),
    }
    scores["confidence_gap"] = scores["mean_confidence"] - scores["accuracy"]
    for metric, value in score_predictions(outcomes, benign, CLEAN_PREDICTION_COLUMN).items():
        scores[f"clean_{metric}"] = value
    return scores

def measure_every_population(outcomes: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    benign = class_names.index(BENIGN_CLASS)
    groups = outcomes.groupby(COMBINATION_COLUMNS, sort=False)
    records = []
    started = time.time()
    for group_index, (combination, combination_outcomes) in enumerate(groups, start=1):
        for population in POPULATIONS:
            population_outcomes = select_population(combination_outcomes, population)
            if population_outcomes.empty:
                continue
            record = dict(zip(COMBINATION_COLUMNS, combination))
            record[POPULATION_COLUMN] = population
            record.update(score_population(population_outcomes, benign))
            records.append(record)
        print_progress(group_index, groups.ngroups, "Scoring every population", started=started)
    end_status(f"Scored {len(records)} populations of {groups.ngroups} predictions")
    return pd.DataFrame(records)

metrics = measure_every_population(outcomes, class_names)
metrics.to_csv(os.path.join(PATH_FOLDER_EVALUATION, METRICS_FILE_NAME), index=False)
display(metrics.round(4))
```

Kode Program 5.99 Penghitungan metrik pada setiap populasi

### **5.15.3 Penurunan Kinerja dan Peringkat Ketahanan**

Fungsi *measure_degradation* pada Kode Program 5.100 mengimplementasikan Langkah 8. Untuk setiap metrik ketahanan, fungsi ini menyusun tabel yang memuat nilai metrik dan nilai bersihnya pada setiap pengujian dan populasi, kemudian menghitung penurunan mutlak sesuai Persamaan 4.16 dan retensi sesuai Persamaan 4.17. Retensi yang tidak terhingga akibat nilai bersih nol diganti NaN. Sel kedua menjalankannya, menyimpan hasilnya pada file *degradation.csv*, dan menampilkan F1-*score* makro setiap model pada setiap data uji untuk populasi seluruh baris dan baris terganggu.

```python
def measure_degradation(metrics: pd.DataFrame, robustness_metrics: list[str] = ROBUSTNESS_METRICS) -> pd.DataFrame:
    tables = []
    for metric in robustness_metrics:
        table = metrics[COMBINATION_COLUMNS + [POPULATION_COLUMN]].copy()
        table[METRIC_COLUMN] = metric
        table[CLEAN_VALUE_COLUMN] = metrics[f"clean_{metric}"]
        table[VALUE_COLUMN] = metrics[metric]
        tables.append(table)

    degradation = pd.concat(tables, ignore_index=True)
    degradation[DROP_COLUMN] = degradation[CLEAN_VALUE_COLUMN] - degradation[VALUE_COLUMN]
    retention = degradation[VALUE_COLUMN] / degradation[CLEAN_VALUE_COLUMN]
    degradation[RETENTION_COLUMN] = retention.replace([np.inf, -np.inf], np.nan)
    return degradation

degradation = measure_degradation(metrics)
degradation.to_csv(os.path.join(PATH_FOLDER_EVALUATION, DEGRADATION_FILE_NAME), index=False)
end_status(f"Measured the degradation of {len(ROBUSTNESS_METRICS)} metrics on {len(metrics)} populations")
for population in ["all", "noisy"]:
    is_shown = (degradation[METRIC_COLUMN] == ROBUSTNESS_METRIC) & (degradation[POPULATION_COLUMN] == population)
    display(
        degradation[is_shown].pivot_table(
            index=[SMOTE_SETTING_COLUMN, MODEL_COLUMN],
            columns=SCENARIO_COLUMN,
            values=VALUE_COLUMN,
        ).reindex(columns=list_scenario_names()).round(4)
    )
```

Kode Program 5.100 Penurunan kinerja dan retensi

Kode Program 5.101 memuat fungsi bantu penyajian. Fungsi *pivot_models* menyusun tabel dengan satu kolom untuk setiap pasangan model dan pengaturan SMOTE dan satu baris untuk setiap kunci, misalnya skenario atau kelas. Fungsi *plot_scenario_heatmap* menggunakan *pivot_models* untuk menggambar sebuah besaran pada setiap pasangan model dan skenario gangguan sebagai peta panas beranotasi untuk populasi tertentu.

```python
def pivot_models(table: pd.DataFrame, key_column: str, value_column: str, keys: list[str]) -> pd.DataFrame:
    pivot = {}
    for smote_setting, smote_label in SMOTE_SETTINGS.items():
        for model_name, model_label in EVALUATED_MODELS.items():
            is_model = (table[SMOTE_SETTING_COLUMN] == smote_setting) & (table[MODEL_COLUMN] == model_name)
            if is_model.any():
                values = table[is_model].set_index(key_column)[value_column]
                pivot[f"{model_label} ({smote_label})"] = values.reindex(keys)
    return pd.DataFrame(pivot, index=keys)

def plot_scenario_heatmap(
    table: pd.DataFrame,
    value_column: str,
    population: str,
    title: str,
    colorbar_label: str,
    save_path: str,
    colormap: mcolors.Colormap = SEQUENTIAL_COLORMAP,
    value_range: tuple[float | None, float | None] = (None, None),
) -> None:
    rows = table[table[POPULATION_COLUMN] == population]
    plot_heatmap(
        pivot_models(rows, SCENARIO_COLUMN, value_column, list_scenario_names(include_clean=False)).T,
        title,
        colormap=colormap,
        value_range=value_range,
        annotate=True,
        annotation_format=".2f",
        xlabel="Scenario",
        ylabel="Model",
        colorbar_label=colorbar_label,
        figsize=(14, 7),
        save_path=save_path,
    )
```

Kode Program 5.101 Penyusunan tabel model dan peta panas skenario

Kode Program 5.102 menyajikan kinerja pada data uji bersih. Sel pertama memilih metrik populasi seluruh baris pada data uji bersih dan menampilkannya terurut menurut F1-*score* makro. Fungsi *plot_model_comparison* menggambar satu diagram batang untuk setiap metrik pada *COMPARED_METRICS*, dengan batang berdampingan untuk kedua pengaturan SMOTE pada setiap model. Sel terakhir menggambar dan menyimpan perbandingan tersebut.

```python
clean_metrics = metrics[(metrics[SCENARIO_COLUMN] == CLEAN_SCENARIO) & (metrics[POPULATION_COLUMN] == "all")]
end_status(f"Compared {len(clean_metrics)} trained models on the clean test split")
display(clean_metrics.sort_values(ROBUSTNESS_METRIC, ascending=False, ignore_index=True).round(4))

def plot_model_comparison(
    model_comparison: pd.DataFrame,
    metrics: dict[str, str] = COMPARED_METRICS,
    bar_width: float = 0.3,
    bar_gap: float = 0.02,
    figsize: tuple[int, int] = (14, 10),
    title: str = "Model Comparison on the Test Split",
    save_path: str | None = None,
) -> None:
    positions = np.arange(len(EVALUATED_MODELS))
    row_count = (len(metrics) + 1) // 2
    fig, axes = plt.subplots(row_count, 2, figsize=figsize, sharey=True, squeeze=False, layout="constrained")

    for ax, (metric, metric_label) in zip(axes.flat, metrics.items()):
        scores = model_comparison.pivot(index=MODEL_COLUMN, columns=SMOTE_SETTING_COLUMN, values=metric)
        scores = scores.reindex(index=list(EVALUATED_MODELS), columns=list(SMOTE_SETTINGS))
        for setting_index, (smote_setting, smote_label) in enumerate(SMOTE_SETTINGS.items()):
            offset = (setting_index - (len(SMOTE_SETTINGS) - 1) / 2) * (bar_width + bar_gap)
            ax.bar(
                positions + offset,
                scores[smote_setting],
                bar_width,
                label=smote_label,
                color=SERIES_COLORS[setting_index],
                zorder=3,
            )
        ax.set_title(metric_label)
        ax.set_xticks(positions)
        ax.set_xticklabels(list(EVALUATED_MODELS.values()), rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)

    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    fig.suptitle(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()

plot_model_comparison(clean_metrics, save_path=os.path.join(PATH_FOLDER_EVALUATION, MODEL_COMPARISON_PLOT_FILE_NAME))
```

Kode Program 5.102 Perbandingan kinerja pada data uji bersih

Fungsi *plot_metric_against_intensity* pada Kode Program 5.103 menggambar sebuah metrik terhadap intensitas gangguan dalam kisi grafik dengan satu baris untuk setiap pengaturan SMOTE dan satu kolom untuk setiap jenis gangguan. Setiap model digambarkan sebagai satu garis dengan warna dan penanda tersendiri, dan skor data uji bersih ditempatkan sebagai titik pada intensitas 0%. Sel kedua menggambar dan menyimpan grafik untuk setiap pasangan metrik dan populasi pada *INTENSITY_PLOTS*.

```python
def plot_metric_against_intensity(
    metrics: pd.DataFrame,
    metric: str,
    population: str = "all",
    figsize: tuple[int, int] = (18, 9),
    save_path: str | None = None,
) -> None:
    noise_types = list(NOISE_MAGNITUDES)
    intensity_ticks = [0]
    for intensity in NOISE_INTENSITIES:
        intensity_ticks.append(round(intensity * 100))

    is_clean = (metrics[SCENARIO_COLUMN] == CLEAN_SCENARIO) & (metrics[POPULATION_COLUMN] == "all")
    is_population = metrics[POPULATION_COLUMN] == population
    fig, axes = plt.subplots(
        len(SMOTE_SETTINGS),
        len(noise_types),
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    legend_handles = {}
    for row_index, (smote_setting, smote_label) in enumerate(SMOTE_SETTINGS.items()):
        for column_index, noise_type in enumerate(noise_types):
            ax = axes[row_index, column_index]
            is_noise_type = is_population & (metrics[NOISE_TYPE_COLUMN] == noise_type)
            for model_index, (model_name, model_label) in enumerate(EVALUATED_MODELS.items()):
                is_model = (metrics[SMOTE_SETTING_COLUMN] == smote_setting) & (metrics[MODEL_COLUMN] == model_name)
                rows = metrics[is_model & (is_noise_type | is_clean)].sort_values(INTENSITY_COLUMN)
                if len(rows) < 2:
                    continue
                (line,) = ax.plot(
                    rows[INTENSITY_COLUMN] * 100,
                    rows[metric],
                    color=SERIES_COLORS[model_index],
                    marker=SERIES_MARKERS[model_index],
                    markersize=8,
                    markeredgecolor=EMPTY_CELL_COLOR,
                    markeredgewidth=1.5,
                    linewidth=2,
                    zorder=3,
                )
                legend_handles.setdefault(model_label, line)
            ax.set_title(f"{noise_type.capitalize()} ({smote_label})")
            ax.set_xticks(intensity_ticks)
            ax.grid(color=GRID_COLOR, linewidth=0.8, zorder=0)

    metric_label = METRIC_LABELS.get(metric, metric)
    for ax in axes[-1, :]:
        ax.set_xlabel("Noisy rows (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel(metric_label)
    handles = []
    labels = []
    for model_label in EVALUATED_MODELS.values():
        if model_label in legend_handles:
            handles.append(legend_handles[model_label])
            labels.append(model_label)
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    fig.suptitle(f"{metric_label} Under Feature Noise, {POPULATIONS[population]} (0% is the clean test split)")

    if save_path is not None:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()

for metric, population in INTENSITY_PLOTS:
    plot_metric_against_intensity(
        metrics,
        metric,
        population,
        save_path=os.path.join(PATH_FOLDER_EVALUATION, f"{metric}-{population}-against-intensity.png"),
    )
```

Kode Program 5.103 Grafik metrik terhadap intensitas gangguan

Kode Program 5.104 menggambar peta panas retensi F1-*score* makro untuk setiap pasangan model dan skenario pada populasi seluruh baris dan baris terganggu.

```python
for population in ["all", "noisy"]:
    plot_scenario_heatmap(
        degradation[degradation[METRIC_COLUMN] == ROBUSTNESS_METRIC],
        RETENTION_COLUMN,
        population,
        f"{METRIC_LABELS[ROBUSTNESS_METRIC]} Retention Under Feature Noise, {POPULATIONS[population]}",
        "Retention (noisy / clean)",
        os.path.join(PATH_FOLDER_EVALUATION, f"{ROBUSTNESS_METRIC}-{population}-retention.png"),
    )
```

Kode Program 5.104 Peta panas retensi

Fungsi *rank_robustness* pada Kode Program 5.105 mengimplementasikan Langkah 9. Fungsi ini mengelompokkan retensi pada skenario gangguan menurut metrik, populasi, pengaturan SMOTE, dan model, kemudian menghitung jumlah skenario, rata-rata nilai bersih dan nilai pada skenario, rata-rata retensi sesuai Persamaan 4.18, retensi terburuk beserta skenarionya, rata-rata dan nilai terbesar penurunan mutlak, serta rata-rata retensi untuk setiap jenis gangguan. Peringkat disusun menurut rata-rata retensi di dalam setiap pengaturan SMOTE (*rank*) dan pada gabungan kedua pengaturan (*overall_rank*). Sel kedua menyimpan peringkat pada file *robustness-ranking.csv*, kemudian menampilkan dan menggambar rata-rata serta retensi terburuk F1-*score* makro untuk populasi seluruh baris dan baris terganggu.

```python
def rank_robustness(degradation: pd.DataFrame) -> pd.DataFrame:
    is_noise_scenario = degradation[SCENARIO_COLUMN] != CLEAN_SCENARIO
    scored = degradation[is_noise_scenario].dropna(subset=[RETENTION_COLUMN])
    group_columns = [METRIC_COLUMN, POPULATION_COLUMN, SMOTE_SETTING_COLUMN, MODEL_COLUMN]
    records = []
    for group, rows in scored.groupby(group_columns, sort=False):
        worst = rows.loc[rows[RETENTION_COLUMN].idxmin()]
        record = dict(zip(group_columns, group))
        record.update({
            "scenarios": len(rows),
            "mean_clean_value": rows[CLEAN_VALUE_COLUMN].mean(),
            "mean_value": rows[VALUE_COLUMN].mean(),
            "mean_retention": rows[RETENTION_COLUMN].mean(),
            "worst_retention": worst[RETENTION_COLUMN],
            "worst_scenario": worst[SCENARIO_COLUMN],
            "mean_drop": rows[DROP_COLUMN].mean(),
            "max_drop": rows[DROP_COLUMN].max(),
        })
        for noise_type, noise_rows in rows.groupby(NOISE_TYPE_COLUMN, sort=False):
            record[f"{noise_type}_retention"] = noise_rows[RETENTION_COLUMN].mean()
        records.append(record)

    ranking = pd.DataFrame(records)
    within_setting = ranking.groupby([METRIC_COLUMN, POPULATION_COLUMN, SMOTE_SETTING_COLUMN])["mean_retention"]
    across_settings = ranking.groupby([METRIC_COLUMN, POPULATION_COLUMN])["mean_retention"]
    ranking.insert(0, "rank", within_setting.rank(ascending=False, method="min").astype(int))
    ranking.insert(1, "overall_rank", across_settings.rank(ascending=False, method="min").astype(int))
    return ranking.sort_values([METRIC_COLUMN, POPULATION_COLUMN, SMOTE_SETTING_COLUMN, "rank"], ignore_index=True)

ranking = rank_robustness(degradation)
ranking.to_csv(os.path.join(PATH_FOLDER_EVALUATION, ROBUSTNESS_RANKING_FILE_NAME), index=False)
for population in ["all", "noisy"]:
    is_shown = (ranking[METRIC_COLUMN] == ROBUSTNESS_METRIC) & (ranking[POPULATION_COLUMN] == population)
    display(ranking[is_shown].round(4))
    plot_model_comparison(
        ranking[is_shown],
        metrics={"mean_retention": "Mean Retention", "worst_retention": "Worst Retention"},
        figsize=(14, 6),
        title=f"{METRIC_LABELS[ROBUSTNESS_METRIC]} Retention Under Feature Noise, {POPULATIONS[population]}",
        save_path=os.path.join(PATH_FOLDER_EVALUATION, f"{ROBUSTNESS_METRIC}-{population}-ranking.png"),
    )
```

Kode Program 5.105 Peringkat ketahanan model
