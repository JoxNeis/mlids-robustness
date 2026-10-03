## **5.14 Implementasi Simulasi Injeksi *Feature Noise***

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.14, yaitu pembentukan dua belas skenario gangguan pada data uji beserta pemeriksaannya. Kode program disusun mengikuti Langkah 1 hingga 7 pada algoritma di Subbab 4.14.

Kode Program 5.73 menetapkan pengaturan simulasi gangguan. Kamus *NOISE_MAGNITUDES* menetapkan jenis gangguan beserta parameter besarannya, yaitu 1,0 untuk *gaussian* dan *uniform*, 0,5 untuk *multiplicative*, dan tanpa parameter untuk *missing*, sesuai Tabel 4.7. Daftar *NOISE_INTENSITIES* menetapkan tiga intensitas, yaitu 10%, 20%, dan 30%, dan *NOISE_SEED* menggunakan *random state* 42 sebagai *seed* dasar. Konstanta berikutnya menetapkan nama kolom penanda baris terganggu (*is_noise*) dan nama kolom pada tabel hasil.

```python
NOISE_MAGNITUDES = {
    "gaussian": 1.0,
    "uniform": 1.0,
    "multiplicative": 0.5,
    "missing": None,
}
NOISE_INTENSITIES = [0.1, 0.2, 0.3]
NOISE_SEED = RANDOM_STATE

IS_NOISE_COLUMN = "is_noise"
SCENARIO_COLUMN = "scenario"
NOISE_TYPE_COLUMN = "noise_type"
INTENSITY_COLUMN = "intensity"
STANDARD_DEVIATION_COLUMN = "standard_deviation"
MEDIAN_COLUMN = "median"
```

Kode Program 5.73 Pengaturan simulasi gangguan

Kode Program 5.74 membaca fitur data uji secara *lazy* dan menampilkan jumlah baris serta jumlah fiturnya. Data uji tidak dimuat ke memori pada tahap ini, karena gangguan diberikan per kelompok baris.

```python
test_features = pl.scan_parquet(os.path.join(PATH_FOLDER_SPLITTED_DATASET_TEST, FEATURES_FILE_NAME))
end_status(f"Scanned {count_rows(test_features):,} test rows x {len(test_features.collect_schema())} features")
```

Kode Program 5.74 Pembacaan data uji secara lazy

Fungsi *compute_feature_statistics* pada Kode Program 5.75 mengimplementasikan Langkah 1. Fungsi ini menghitung simpangan baku dengan pembagi *n* (*ddof=0*), sama seperti *StandardScaler*, dan median setiap fitur pada data latih. Perhitungan dilakukan satu fitur pada setiap kali pembacaan secara *lazy*, sehingga hanya satu kolom data latih yang dimuat ke memori pada satu waktu. Sel kedua menjalankan perhitungan dan menampilkan statistik seluruh fitur.

```python
def compute_feature_statistics(train_folder_path: str = PATH_FOLDER_SPLITTED_DATASET_TRAIN) -> pd.DataFrame:
    features = pl.scan_parquet(os.path.join(train_folder_path, FEATURES_FILE_NAME))
    feature_columns = features.collect_schema().names()
    records = []
    started = time.time()
    for feature_index, feature in enumerate(feature_columns, start=1):
        values = pl.col(feature).cast(pl.Float64)
        statistics = features.select(
            values.std(ddof=0).alias(STANDARD_DEVIATION_COLUMN),  # ddof=0, like StandardScaler
            values.median().alias(MEDIAN_COLUMN),
        )
        records.append(statistics.collect().row(0, named=True))
        print_progress(feature_index, len(feature_columns), "Computing the train feature statistics", started=started)
    return pd.DataFrame(records, index=feature_columns)

feature_statistics = compute_feature_statistics()
end_status(f"Computed the standard deviation and median of {len(feature_statistics)} features")
with pd.option_context("display.max_rows", None):
    display(feature_statistics)
```

Kode Program 5.75 Simpangan baku dan median fitur data latih

Kode Program 5.76 mengimplementasikan Langkah 2 dan 3. Fungsi *get_noise_scenario_name* membentuk nama skenario dari jenis gangguan dan intensitasnya, misalnya *gaussian-20*. Fungsi *list_noise_scenarios* menyusun dua belas skenario dari empat jenis gangguan dan tiga intensitas. Fungsi *make_scenario_rng* membentuk generator bilangan acak *numpy.random.default_rng* dengan *seed* berupa hasil *hash* BLAKE2b sepanjang 8 *byte* atas teks yang memuat *seed* dasar dan nama skenario.

```python
def get_noise_scenario_name(noise_type: str, intensity: float) -> str:
    return f"{noise_type}-{round(intensity * 100):02d}"

def list_noise_scenarios() -> list[tuple[str, float, str]]:
    scenarios = []
    for noise_type in NOISE_MAGNITUDES:
        for intensity in NOISE_INTENSITIES:
            scenarios.append((noise_type, intensity, get_noise_scenario_name(noise_type, intensity)))
    return scenarios

def make_scenario_rng(scenario: str, seed: int = NOISE_SEED) -> np.random.Generator:
    digest = hashlib.blake2b(f"{seed}/{scenario}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))
```

Kode Program 5.76 Penamaan skenario dan generator bilangan acak

Kode Program 5.77 memuat fungsi pembangkit gangguan untuk setiap jenis gangguan, yang menerima nilai pada baris terganggu dan mengembalikan nilai penggantinya. Fungsi *add_gaussian_noise* menambahkan bilangan acak berdistribusi normal dengan simpangan baku sebesar parameter besaran dikalikan simpangan baku setiap fitur (Persamaan 4.10). Fungsi *add_uniform_noise* menambahkan bilangan acak berdistribusi seragam pada rentang simetris dengan setengah lebar sebesar parameter besaran dikalikan simpangan baku dan $\sqrt{3}$ (Persamaan 4.11). Fungsi *add_multiplicative_noise* mengalikan nilai dengan bilangan acak berdistribusi normal dengan rata-rata satu dan simpangan baku sebesar parameter besaran (Persamaan 4.12). Fungsi *add_missing_values* mengganti seluruh nilai dengan median data latih setiap fitur (Persamaan 4.13). Kamus *NOISE_INJECTORS* memetakan nama jenis gangguan ke fungsi pembangkitnya.

```python
def add_gaussian_noise(
    values: np.ndarray,
    magnitude: float,
    feature_statistics: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    standard_deviations = magnitude * feature_statistics[STANDARD_DEVIATION_COLUMN].to_numpy()
    return values + rng.normal(0.0, standard_deviations, values.shape)

def add_uniform_noise(
    values: np.ndarray,
    magnitude: float,
    feature_statistics: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    half_widths = magnitude * feature_statistics[STANDARD_DEVIATION_COLUMN].to_numpy() * np.sqrt(3)  # the same variance as the gaussian noise
    return values + rng.uniform(-half_widths, half_widths, values.shape)

def add_multiplicative_noise(
    values: np.ndarray,
    magnitude: float,
    feature_statistics: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    return values * rng.normal(1.0, magnitude, values.shape)

def add_missing_values(
    values: np.ndarray,
    magnitude: float,
    feature_statistics: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    # the pipeline cannot take NaN, so the blanked values are imputed with the train medians right away
    return np.broadcast_to(feature_statistics[MEDIAN_COLUMN].to_numpy(), values.shape)

NOISE_INJECTORS = {
    "gaussian": add_gaussian_noise,
    "uniform": add_uniform_noise,
    "multiplicative": add_multiplicative_noise,
    "missing": add_missing_values,
}
```

Kode Program 5.77 Fungsi pembangkit setiap jenis gangguan

Kode Program 5.78 mengimplementasikan Langkah 4 dan bagian pertama Langkah 5. Fungsi *select_noisy_rows* memilih sebanyak pembulatan intensitas dikalikan jumlah baris secara acak tanpa pengembalian dan mengembalikan penanda baris terganggu. Fungsi *add_noise_to_features* menerapkan gangguan pada satu kelompok baris. Nilai fitur dikonversi ke *float64* agar perhitungan gangguan tidak kehilangan presisi, baris yang ditandai diganti dengan hasil fungsi pembangkit gangguan, kemudian hasilnya dikembalikan ke tipe *float32* dan ditambah kolom *is_noise*. Parameter *orient="row"* memastikan matriks nilai selalu dibaca per baris, termasuk pada kelompok terakhir yang jumlah barisnya dapat kebetulan sama dengan jumlah fitur.

```python
def select_noisy_rows(row_count: int, intensity: float, rng: np.random.Generator) -> np.ndarray:
    is_noise = np.zeros(row_count, dtype=bool)
    is_noise[rng.choice(row_count, round(intensity * row_count), replace=False)] = True
    return is_noise

def add_noise_to_features(
    features: pl.DataFrame,
    is_noise: np.ndarray,
    noise_type: str,
    feature_statistics: pd.DataFrame,
    rng: np.random.Generator,
) -> pl.DataFrame:
    values = features.to_numpy().astype(np.float64)
    add_noise = NOISE_INJECTORS[noise_type]
    values[is_noise] = add_noise(values[is_noise], NOISE_MAGNITUDES[noise_type], feature_statistics, rng)

    noisy_features = pl.DataFrame(values.astype(NUMERIC_DATA_TYPE), schema=features.columns, orient="row")
    return noisy_features.with_columns(pl.Series(IS_NOISE_COLUMN, is_noise))
```

Kode Program 5.78 Pemilihan baris terganggu dan pemberian gangguan pada satu kelompok

Fungsi *make_noisy_dataset* pada Kode Program 5.79 menyatukan Langkah 3 hingga 6 untuk satu skenario. Fungsi ini memeriksa bahwa kolom data uji sama dengan kolom statistik data latih, membentuk generator bilangan acak skenario, dan memilih baris terganggu pada seluruh data uji. Data uji kemudian dibaca per kelompok 500.000 baris dengan *read_in_batches* (pendahuluan Bab 5), dan setiap kelompok diberi gangguan dengan potongan penanda yang sesuai lalu ditulis ke satu file *features.parquet* menggunakan *ParquetWriter* dari *PyArrow*. Karena setiap kelompok menggunakan generator yang sama secara berurutan, hasilnya identik dengan pembangkitan gangguan pada seluruh data uji sekaligus. file label data uji disalin ke folder skenario, dan file fitur yang baru ditulis dikembalikan sebagai *LazyFrame*.

```python
def make_noisy_dataset(
    noise_type: str,
    intensity: float,
    scenario: str,
    test_features: pl.LazyFrame,
    feature_statistics: pd.DataFrame,
    output_folder_path: str = PATH_FOLDER_NOISY_DATASET_TEST,
    test_folder_path: str = PATH_FOLDER_SPLITTED_DATASET_TEST,
    batch_size: int = BATCH_SIZE,
) -> pl.LazyFrame:
    if test_features.collect_schema().names() != feature_statistics.index.tolist():
        raise ValueError("The test features and the train feature statistics have different columns")

    rng = make_scenario_rng(scenario)
    is_noise = select_noisy_rows(count_rows(test_features), intensity, rng)

    scenario_folder_path = os.path.join(output_folder_path, scenario)
    os.makedirs(scenario_folder_path, exist_ok=True)
    features_path = os.path.join(scenario_folder_path, FEATURES_FILE_NAME)
    prefix = f"{scenario}: adding {noise_type} noise to {intensity:.0%} of the rows"
    writer = None
    # the batches draw from one rng in row order, so the noise is the same as one draw over the whole split
    for row_offset, batch in read_in_batches(test_features, prefix, batch_size):
        batch_is_noise = is_noise[row_offset:row_offset + len(batch)]
        table = add_noise_to_features(batch, batch_is_noise, noise_type, feature_statistics, rng).to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(features_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()

    shutil.copyfile(
        os.path.join(test_folder_path, ENCODED_LABELS_FILE_NAME),
        os.path.join(scenario_folder_path, ENCODED_LABELS_FILE_NAME),
    )
    return pl.scan_parquet(features_path)
```

Kode Program 5.79 Pembentukan dan penyimpanan satu skenario gangguan

Fungsi *measure_noise* pada Kode Program 5.80 mengimplementasikan Langkah 7. Fungsi ini membaca data skenario dan data uji bersih per kelompok secara berpasangan, kemudian mengumpulkan jumlah baris, jumlah sel, jumlah baris terganggu, jumlah baris dan sel yang berubah, serta jumlah baris tidak terganggu yang berubah. Besar gangguan yang teramati dihitung dari selisih nilai pada sel yang berubah, yang dibagi dengan simpangan baku fitur pada *gaussian* dan *uniform* atau dengan nilai asli pada *multiplicative*. Simpangan baku selisih tersebut dihitung secara bertahap dengan *StandardScaler.partial_fit*, yang memperbarui rata-rata dan varians dari setiap kelompok tanpa menyimpan seluruh selisih di memori. Hasilnya dikembalikan dalam bentuk persentase dan besar gangguan yang teramati.

```python
def measure_noise(
    clean_features: pl.LazyFrame,
    noisy_features: pl.LazyFrame,
    noise_type: str,
    feature_statistics: pd.DataFrame,
    prefix: str = "Measuring the noise",
    batch_size: int = BATCH_SIZE,
) -> dict:
    standard_deviations = feature_statistics[STANDARD_DEVIATION_COLUMN].to_numpy()
    row_count = 0
    cell_count = 0
    noisy_row_count = 0
    changed_row_count = 0
    changed_cell_count = 0
    changed_intact_row_count = 0
    magnitude_statistics = StandardScaler()  # partial_fit keeps a running variance over the batches
    for row_offset, noisy_batch in read_in_batches(noisy_features, prefix, batch_size):
        is_noise = noisy_batch[IS_NOISE_COLUMN].to_numpy()
        noisy_values = noisy_batch.drop(IS_NOISE_COLUMN).to_numpy().astype(np.float64)
        clean_values = clean_features.slice(row_offset, batch_size).collect().to_numpy().astype(np.float64)
        is_changed = noisy_values != clean_values

        row_count += len(is_noise)
        cell_count += is_changed.size
        noisy_row_count += int(is_noise.sum())
        changed_row_count += int(is_changed.any(axis=1).sum())
        changed_cell_count += int(is_changed.sum())
        changed_intact_row_count += int(is_changed[~is_noise].any(axis=1).sum())

        differences = noisy_values[is_noise] - clean_values[is_noise]
        with np.errstate(divide="ignore", invalid="ignore"):
            if noise_type == "multiplicative":
                relative_differences = differences / clean_values[is_noise]
            else:
                relative_differences = differences / standard_deviations
        relative_differences = relative_differences[is_changed[is_noise]]
        if noise_type != "missing" and relative_differences.size > 0:
            magnitude_statistics.partial_fit(relative_differences.reshape(-1, 1))

    observed_magnitude = np.nan
    if hasattr(magnitude_statistics, "var_"):
        observed_magnitude = float(np.sqrt(magnitude_statistics.var_[0]))
    return {
        "noisy_rows_percentage": noisy_row_count / row_count * 100,
        "changed_rows_percentage": changed_row_count / row_count * 100,
        "changed_cells_percentage": changed_cell_count / cell_count * 100,
        "changed_intact_rows": changed_intact_row_count,
        "observed_magnitude": observed_magnitude,
    }
```

Kode Program 5.80 Pemeriksaan hasil gangguan

Kode Program 5.81 menjalankan pembentukan dan pemeriksaan untuk kedua belas skenario secara berurutan, menyusun hasil pemeriksaannya bersama nama skenario, jenis gangguan, intensitas, dan parameter besarannya, kemudian menyimpan tabel tersebut pada file *noise-evaluation/noise-check.csv*.

```python
noise_check = []
for noise_type, intensity, scenario in list_noise_scenarios():
    noisy_features = make_noisy_dataset(noise_type, intensity, scenario, test_features, feature_statistics)
    record = {
        SCENARIO_COLUMN: scenario,
        NOISE_TYPE_COLUMN: noise_type,
        INTENSITY_COLUMN: intensity,
        "magnitude": NOISE_MAGNITUDES[noise_type],
        **measure_noise(test_features, noisy_features, noise_type, feature_statistics, f"{scenario}: measuring the noise"),
    }
    noise_check.append(record)
    end_status(f"{scenario}: stored in {os.path.join(PATH_FOLDER_NOISY_DATASET_TEST, scenario)}")

noise_check = pd.DataFrame(noise_check)
os.makedirs(PATH_FOLDER_NOISE_EVALUATION, exist_ok=True)
noise_check.to_csv(os.path.join(PATH_FOLDER_NOISE_EVALUATION, NOISE_CHECK_FILE_NAME), index=False)
display(noise_check.round(4))
del noisy_features
```

Kode Program 5.81 Menjalankan simulasi gangguan pada seluruh skenario
