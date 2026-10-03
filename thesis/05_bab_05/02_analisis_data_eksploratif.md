## **5.2 Implementasi Analisis Data Eksploratif**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.2, yaitu analisis data eksploratif pada data di folder *data-pipeline/01-raw-parquet*. Kode program disusun mengikuti Langkah 1 hingga 7 pada algoritma di Subbab 4.2, didahului oleh pengaturan serta fungsi bantu pengelompokan dan visualisasi yang juga digunakan pada tahap-tahap berikutnya.

Kode Program 5.18 menetapkan pengaturan tahap EDA, yaitu nama kolom pada tabel hasil analisis, pola ekspresi reguler *DAY_REGEX* untuk mengambil tanggal pengambilan data dari awal nama berkas, ambang korelasi tinggi sebesar 0,95 (*CORRELATION_THRESHOLD*), dan jumlah proses paralel sebanyak delapan (*EDA_N_JOBS*).

```python
FILE_COLUMN = "file"
DAY_COLUMN = "day"
FEATURE_COLUMN = "feature"
PERCENTAGE_COLUMN = "percentage"
MISSING_COUNT_COLUMN = "missing"
NAN_COUNT_COLUMN = "nan"
INFINITE_COUNT_COLUMN = "infinite"
UNIQUE_COUNT_COLUMN = "unique"
CORRELATION_COLUMN = "correlation"

DAY_REGEX = r"^(\d{4}-\d{2}-\d{2})"
CORRELATION_THRESHOLD = 0.95
EDA_N_JOBS = 8
```

Kode Program 5.18 Pengaturan tahap analisis data eksploratif

Kode Program 5.19 memuat fungsi bantu pengelompokan per hari. Fungsi *extract_column* mengambil bagian teks dari sebuah kolom menggunakan ekspresi reguler dan menyimpannya sebagai kolom baru, sedangkan *group_and_sum* menggunakan fungsi tersebut untuk mengambil tanggal dari nama berkas, kemudian menjumlahkan kolom numerik per tanggal. Fungsi ini digunakan untuk merangkum hasil per berkas menjadi hasil per hari pada tahap EDA, pembersihan data, dan penghapusan duplikat.

```python
def extract_column(
    dataframe: pd.DataFrame,
    source_column: str,
    target_column: str,
    regex: str,
) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe[target_column] = dataframe[source_column].str.extract(regex)[0]
    return dataframe

def group_and_sum(
    dataframe: pd.DataFrame,
    source_column: str,
    group_column: str,
    regex: str,
    value_columns: str | list[str] | None = None,
) -> pd.DataFrame:
    dataframe = extract_column(dataframe, source_column, group_column, regex)

    if value_columns is None:
        value_columns = dataframe.select_dtypes("number").columns.tolist()
    elif isinstance(value_columns, str):
        value_columns = [value_columns]

    return dataframe.groupby(group_column, as_index=False)[value_columns].sum()
```

Kode Program 5.19 Pengelompokan hasil per hari pengambilan data

Kode Program 5.20 menetapkan warna dan penanda yang digunakan pada seluruh visualisasi, yaitu warna batang, garis kisi, dan teks, delapan warna kategori untuk membedakan model atau jenis kesalahan, peta warna berurutan (*sequential*) untuk besaran yang hanya bernilai positif, serta peta warna divergen untuk besaran yang dapat bernilai positif maupun negatif, misalnya korelasi. Gambar disimpan dengan resolusi 300 dpi.

```python
BAR_COLOR = "#2a78d6"
GRID_COLOR = "#e1e0d9"
TEXT_COLOR = "#0b0b0b"
TEXT_ON_DARK_COLOR = "#ffffff"
EMPTY_CELL_COLOR = "#fcfcfb"
# categorical colors, given to the series in this fixed order
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

SEQUENTIAL_COLORMAP = mcolors.LinearSegmentedColormap.from_list(
    "sequential_blue",
    ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"],
).with_extremes(bad=EMPTY_CELL_COLOR)

DIVERGING_COLORMAP = mcolors.LinearSegmentedColormap.from_list(
    "diverging_blue_red",
    ["#2a78d6", "#f0efec", "#e34948"],
).with_extremes(bad=EMPTY_CELL_COLOR)

FIGURE_DPI = 300  # resolution of the saved figures
```

Kode Program 5.20 Warna dan penanda visualisasi

Kode Program 5.21 memuat dua fungsi visualisasi umum. Fungsi *plot_bar* menggambar diagram batang dengan pilihan skala logaritmik pada sumbu tegak. Fungsi *plot_heatmap* menggambar peta panas dari sebuah tabel, dengan pilihan skala warna logaritmik yang menyembunyikan sel bernilai nol atau negatif, rentang nilai warna, anotasi nilai pada setiap sel dengan warna teks yang menyesuaikan kegelapan sel, serta penyimpanan gambar ke berkas.

```python
def plot_bar(
    dataframe: pd.DataFrame,
    x: str,
    y: str,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[int, int] = (10, 5),
    rotation: int = 45,
    log_scale: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(dataframe[x].astype(str), dataframe[y], color=BAR_COLOR, zorder=3)
    ax.set_title(title or f"{y} per {x}")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if log_scale:
        ax.set_yscale("log")
    else:
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.show()

def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    colormap: mcolors.Colormap = SEQUENTIAL_COLORMAP,
    value_range: tuple[float | None, float | None] = (None, None),
    log_scale: bool = False,
    annotate: bool = False,
    annotation_format: str = ",.0f",
    xlabel: str = "",
    ylabel: str = "",
    colorbar_label: str = "",
    figsize: tuple[int, int] = (12, 10),
    font_size: int = 8,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    values = matrix.to_numpy(dtype=float)
    if log_scale:
        values = np.ma.masked_less_equal(values, 0)
        normalization = mcolors.LogNorm(*value_range)
    else:
        normalization = mcolors.Normalize(*value_range)

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(values, cmap=colormap, norm=normalization, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=font_size)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=font_size)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)

    if annotate:
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = values[row_index, column_index]
                if value is np.ma.masked or np.isnan(value):
                    continue
                if normalization(value) > 0.6:
                    text_color = TEXT_ON_DARK_COLOR
                else:
                    text_color = TEXT_COLOR
                ax.text(
                    column_index,
                    row_index,
                    f"{value:{annotation_format}}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=font_size,
                )

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
```

Kode Program 5.21 Fungsi diagram batang dan peta panas

Kode Program 5.22 mengimplementasikan Langkah 1. Fungsi *count_rows_in_a_single_file* membaca jumlah baris dari metadata berkas Parquet menggunakan *PyArrow* tanpa membaca isi data, dan *count_rows_in_all_files* menyusun jumlah baris setiap berkas menjadi tabel. Sel ketiga menjumlahkan tabel tersebut per hari dengan *group_and_sum*, dan sel terakhir menggambarkan hasilnya sebagai diagram batang.

```python
def count_rows_in_a_single_file(file_path: str) -> int:
    return pq.ParquetFile(file_path).metadata.num_rows

def count_rows_in_all_files(
    folder_path: str = PATH_FOLDER_PARQUET_DATASET,
    file_column: str = FILE_COLUMN,
    row_count_column: str = ROW_COUNT_COLUMN,
) -> pd.DataFrame:
    records = []
    for file_path in get_file_paths_in_folder(folder_path, "parquet"):
        record = {
            file_column: os.path.basename(file_path),
            row_count_column: count_rows_in_a_single_file(file_path),
        }
        records.append(record)
    return pd.DataFrame(records)

results = group_and_sum(count_rows_in_all_files(), FILE_COLUMN, DAY_COLUMN, DAY_REGEX, ROW_COUNT_COLUMN)
display(results)

plot_bar(results, DAY_COLUMN, ROW_COUNT_COLUMN, "Row Count Each Day", "Day", "Row Count")
del results
```

Kode Program 5.22 Penghitungan baris per hari

Kode Program 5.23 mengimplementasikan Langkah 2. Fungsi *get_classes_list_from_a_folder* membaca hanya kolom label dari setiap berkas, baik berkas Parquet maupun berkas CSV, mengumpulkan nilai uniknya, dan mengembalikan daftar kelas yang telah diurutkan. Pada berkas CSV, kolom label dicari setelah nama kolom dinormalisasi dan baris judul yang berulang dibuang. Fungsi *create_classes_list_from_a_folder* memilih jenis berkas yang tersedia pada folder dan menyimpan daftar kelas ke berkas *cache/original-classes.json*, sedangkan *load_classes_list* memuat kembali daftar tersebut. Sel terakhir menyusun daftar kelas dari data hasil konversi.

```python
def get_classes_list_from_a_folder(
    folder_path: str,
    file_extension: Literal["csv", "parquet"],
    label_column: str = LABEL_COLUMN,
) -> list[str]:
    if file_extension not in ("csv", "parquet"):
        raise ValueError("file_extension must be 'csv' or 'parquet'")

    classes = set()
    file_paths = get_file_paths_in_folder(folder_path, file_extension)
    started = time.time()
    for file_index, file_path in enumerate(file_paths, start=1):
        if file_extension == "csv":
            dataframe = pd.read_csv(
                file_path,
                usecols=lambda column: normalize_column_name(column) == label_column,
            )
            labels = dataframe.iloc[:, 0]
            labels = labels[labels != labels.name]
        else:
            labels = pd.read_parquet(file_path, columns=[label_column])[label_column]
        classes.update(labels.dropna().unique())
        print_progress(file_index, len(file_paths), "Reading classes", started=started)

    end_status(f"Found {len(classes)} classes in {len(file_paths)} files")
    return sorted(classes)

def create_classes_list_from_a_folder(
    folder_path: str,
    output_path: str = PATH_JSON_CURRENTLY_USED_CLASSES_LIST,
    label_column: str = LABEL_COLUMN,
) -> list[str]:
    if get_file_paths_in_folder(folder_path, "parquet"):
        file_extension = "parquet"
    elif get_file_paths_in_folder(folder_path, "csv"):
        file_extension = "csv"
    else:
        raise FileNotFoundError(f"No csv or parquet files found in {folder_path}")

    classes = get_classes_list_from_a_folder(folder_path, file_extension, label_column)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(classes, file, indent=4)

    return classes

def load_classes_list(file_path: str = PATH_JSON_CURRENTLY_USED_CLASSES_LIST) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

results = create_classes_list_from_a_folder(PATH_FOLDER_PARQUET_DATASET)
display(results)
del results
```

Kode Program 5.23 Penyusunan dan penyimpanan daftar kelas

Kode Program 5.24 mengimplementasikan Langkah 3. Fungsi *count_classes_in_a_single_file* membaca kolom label satu berkas dan menghitung jumlah baris setiap kelas, dengan kelas yang tidak muncul pada berkas tersebut diberi jumlah nol agar seluruh berkas memiliki kolom yang sama. Fungsi *count_classes_in_all_files* menjalankannya pada seluruh berkas secara paralel dengan delapan proses dan menggabungkan hasilnya. Sel terakhir menjumlahkan hasil tersebut per hari.

```python
def count_classes_in_a_single_file(
    file_path: str,
    label_column: str = LABEL_COLUMN,
    classes_list: list[str] | None = None,
    file_column: str = FILE_COLUMN,
) -> pd.DataFrame:
    if classes_list is None:
        classes_list = load_classes_list()

    labels = pd.read_parquet(file_path, columns=[label_column])[label_column]
    counts = labels.value_counts().reindex(classes_list, fill_value=0)

    record = {file_column: os.path.basename(file_path), **counts.to_dict()}
    return pd.DataFrame([record])

def count_classes_in_all_files(
    folder_path: str = PATH_FOLDER_PARQUET_DATASET,
    label_column: str = LABEL_COLUMN,
    classes_list: list[str] | None = None,
    n_jobs: int = EDA_N_JOBS,
) -> pd.DataFrame:
    if classes_list is None:
        classes_list = load_classes_list()

    file_paths = get_file_paths_in_folder(folder_path, "parquet")
    tasks = create_tasks_list(count_classes_in_a_single_file, file_paths, label_column, classes_list)

    results = run_tasks_in_parallel(tasks, n_jobs, "Counting classes")
    end_status(f"Counted classes in {len(tasks)} files")

    return pd.concat(results, ignore_index=True)

results = group_and_sum(count_classes_in_all_files(), FILE_COLUMN, DAY_COLUMN, DAY_REGEX)
display(results)
```

Kode Program 5.24 Penghitungan distribusi kelas per berkas dan per hari

Kode Program 5.25 menggambarkan jumlah baris setiap kelas pada setiap hari sebagai peta panas berskala logaritmik dengan anotasi nilai, sehingga hari kemunculan setiap kelas serangan dapat diamati.

```python
plot_heatmap(
    results.set_index(DAY_COLUMN).T,
    "Row Count of Each Class Each Day",
    xlabel="Day",
    ylabel="Class",
    log_scale=True,
    annotate=True,
    colorbar_label="Row Count (log scale)",
    figsize=(14, 7),
)
```

Kode Program 5.25 Peta panas jumlah baris setiap kelas per hari

Fungsi *count_total_classes* pada Kode Program 5.26 menjumlahkan jumlah baris setiap kelas dari seluruh hari, menghitung persentasenya terhadap seluruh data, dan mengurutkannya dari kelas terbesar. Sel kedua menampilkan tabel tersebut dan menggambarkannya sebagai diagram batang berskala logaritmik.

```python
def count_total_classes(
    class_counts_per_day: pd.DataFrame,
    day_column: str = DAY_COLUMN,
    class_column: str = CLASS_COLUMN,
    row_count_column: str = ROW_COUNT_COLUMN,
    percentage_column: str = PERCENTAGE_COLUMN,
) -> pd.DataFrame:
    totals = class_counts_per_day.drop(columns=day_column).sum()
    totals = totals.rename_axis(class_column).reset_index(name=row_count_column)
    totals[percentage_column] = totals[row_count_column] / totals[row_count_column].sum() * 100
    return totals.sort_values(row_count_column, ascending=False, ignore_index=True)

results = count_total_classes(results)
display(results)
plot_bar(
    results,
    CLASS_COLUMN,
    ROW_COUNT_COLUMN,
    "Row Count Each Class",
    "Class",
    "Row Count (log scale)",
    log_scale=True,
)
del results
```

Kode Program 5.26 Distribusi kelas pada seluruh dataset

Kode Program 5.27 mengimplementasikan Langkah 4. Fungsi *get_feature_columns* mengambil nama seluruh kolom selain label dari skema berkas Parquet pertama. Fungsi *describe_a_single_feature* membaca seluruh berkas secara *lazy* dengan *polars.scan_parquet*, tetapi hanya kolom satu fitur yang benar-benar dimuat, kemudian menghitung jumlah baris, jumlah nilai hilang (*null*), jumlah NaN, jumlah nilai tak hingga, jumlah nilai negatif, jumlah nilai nol, dan jumlah nilai unik, serta nilai minimum, maksimum, rata-rata, median, dan simpangan baku yang hanya dihitung dari nilai terhingga. Fungsi *describe_all_features* mengulang perhitungan tersebut untuk setiap fitur sambil menampilkan bilah kemajuan, dan sel terakhir menampilkan seluruh statistik.

```python
def get_feature_columns(
    folder_path: str = PATH_FOLDER_PARQUET_DATASET,
    label_column: str = LABEL_COLUMN,
) -> list[str]:
    first_file_path = get_file_paths_in_folder(folder_path, "parquet")[0]
    feature_columns = []
    for column in pq.read_schema(first_file_path).names:
        if column != label_column:
            feature_columns.append(column)
    return feature_columns

def describe_a_single_feature(
    feature: str,
    folder_path: str = PATH_FOLDER_PARQUET_DATASET,
    feature_column: str = FEATURE_COLUMN,
) -> dict:
    values = pl.col(feature).cast(pl.Float64)
    finite_values = values.filter(values.is_finite())

    statistics = pl.scan_parquet(os.path.join(folder_path, "*.parquet")).select(
        pl.len().alias(ROW_COUNT_COLUMN),
        values.null_count().alias(MISSING_COUNT_COLUMN),
        values.is_nan().sum().alias(NAN_COUNT_COLUMN),
        values.is_infinite().sum().alias(INFINITE_COUNT_COLUMN),
        finite_values.min().alias("min"),
        finite_values.max().alias("max"),
        finite_values.mean().alias("mean"),
        finite_values.median().alias("median"),
        finite_values.std().alias("std"),
        (values < 0).sum().alias("negative"),
        (values == 0).sum().alias("zero"),
        values.drop_nulls().n_unique().alias(UNIQUE_COUNT_COLUMN),
    )
    return {feature_column: feature, **statistics.collect().row(0, named=True)}

def describe_all_features(folder_path: str = PATH_FOLDER_PARQUET_DATASET) -> pd.DataFrame:
    feature_columns = get_feature_columns(folder_path)

    records = []
    started = time.time()
    for feature_index, feature in enumerate(feature_columns, start=1):
        records.append(describe_a_single_feature(feature, folder_path))
        print_progress(feature_index, len(feature_columns), prefix="Describing features", started=started)

    end_status(f"Described {len(feature_columns)} features")
    return pd.DataFrame(records)

descriptive_statistics = describe_all_features()
with pd.option_context("display.max_rows", None):
    display(descriptive_statistics)
```

Kode Program 5.27 Statistik deskriptif setiap fitur

Kode Program 5.28 mengimplementasikan bagian pertama Langkah 5. Fungsi *find_affected_features* menyaring fitur yang memiliki sedikitnya satu nilai pada kolom hitungan tertentu dan menghitung persentasenya terhadap jumlah baris. Ketiga sel berikutnya menjalankannya untuk nilai tak hingga, nilai hilang, dan NaN.

```python
def find_affected_features(
    descriptive_statistics: pd.DataFrame,
    count_column: str,
    feature_column: str = FEATURE_COLUMN,
    row_count_column: str = ROW_COUNT_COLUMN,
    percentage_column: str = PERCENTAGE_COLUMN,
) -> pd.DataFrame:
    is_affected = descriptive_statistics[count_column] > 0
    columns = [feature_column, count_column, row_count_column]
    affected = descriptive_statistics.loc[is_affected, columns].copy()
    affected[percentage_column] = affected[count_column] / affected[row_count_column] * 100
    affected = affected.drop(columns=row_count_column)
    return affected.sort_values(count_column, ascending=False, ignore_index=True)

results = find_affected_features(descriptive_statistics, INFINITE_COUNT_COLUMN)
display(results)
del results

results = find_affected_features(descriptive_statistics, MISSING_COUNT_COLUMN)
display(results)
del results

results = find_affected_features(descriptive_statistics, NAN_COUNT_COLUMN)
display(results)
del results
```

Kode Program 5.28 Identifikasi fitur dengan nilai tak hingga, nilai hilang, dan NaN

Kode Program 5.29 mengimplementasikan bagian kedua Langkah 5. Fungsi *find_constant_features* menyaring fitur yang jumlah nilai uniknya tidak lebih dari satu, beserta nilai minimumnya, yaitu nilai tunggal yang dimiliki fitur tersebut.

```python
def find_constant_features(
    descriptive_statistics: pd.DataFrame,
    feature_column: str = FEATURE_COLUMN,
    unique_count_column: str = UNIQUE_COUNT_COLUMN,
) -> pd.DataFrame:
    is_constant = descriptive_statistics[unique_count_column] <= 1
    columns = [feature_column, "min", unique_count_column]
    return descriptive_statistics.loc[is_constant, columns].reset_index(drop=True)

results = find_constant_features(descriptive_statistics)
display(results)
del results
```

Kode Program 5.29 Identifikasi fitur konstan

Kode Program 5.30 mengimplementasikan bagian pertama Langkah 6. Fungsi *get_varying_features* mengambil fitur yang tidak konstan. Fungsi *compute_covariance_in_a_single_file* membaca fitur tersebut dari satu berkas dalam tipe *float64*, membuang baris yang memuat nilai tidak terhingga, kemudian mengembalikan jumlah baris, vektor rata-rata, dan matriks *cross-product* dari data yang telah dikurangi rata-ratanya. Fungsi *merge_covariance_statistics* menggabungkan statistik dua kelompok data sesuai Persamaan 4.1.

```python
def get_varying_features(
    descriptive_statistics: pd.DataFrame,
    feature_column: str = FEATURE_COLUMN,
    unique_count_column: str = UNIQUE_COUNT_COLUMN,
) -> list[str]:
    is_varying = descriptive_statistics[unique_count_column] > 1
    return descriptive_statistics.loc[is_varying, feature_column].tolist()

def compute_covariance_in_a_single_file(
    file_path: str,
    feature_columns: list[str],
) -> tuple[int, np.ndarray, np.ndarray]:
    features = pd.read_parquet(file_path, columns=feature_columns).to_numpy(dtype=np.float64)
    features = features[np.isfinite(features).all(axis=1)]

    row_count = len(features)
    if row_count == 0:
        feature_count = len(feature_columns)
        return 0, np.zeros(feature_count), np.zeros((feature_count, feature_count))

    mean = features.mean(axis=0)
    features -= mean
    return row_count, mean, features.T @ features

def merge_covariance_statistics(first: tuple, second: tuple) -> tuple:
    first_row_count, first_mean, first_cross_products = first
    second_row_count, second_mean, second_cross_products = second

    row_count = first_row_count + second_row_count
    if row_count == 0:
        return first

    delta = second_mean - first_mean
    mean = first_mean + delta * second_row_count / row_count
    correction = np.outer(delta, delta) * first_row_count * second_row_count / row_count
    cross_products = first_cross_products + second_cross_products + correction
    return row_count, mean, cross_products
```

Kode Program 5.30 Statistik kovarians per berkas dan penggabungannya

Kode Program 5.31 mengimplementasikan bagian kedua Langkah 6. Fungsi *compute_correlation_in_all_files* menghitung statistik kovarians setiap berkas secara paralel dengan delapan proses, menggabungkannya secara berurutan, kemudian mengubah matriks *cross-product* gabungan menjadi matriks korelasi Pearson sesuai Persamaan 4.2. Fungsi *find_highly_correlated_pairs* menelusuri setengah bagian atas matriks korelasi dan mendaftar pasangan fitur yang nilai mutlak korelasinya sedikitnya 0,95, diurutkan dari korelasi mutlak terbesar.

```python
def compute_correlation_in_all_files(
    feature_columns: list[str],
    folder_path: str = PATH_FOLDER_PARQUET_DATASET,
    n_jobs: int = EDA_N_JOBS,
) -> pd.DataFrame:
    file_paths = get_file_paths_in_folder(folder_path, "parquet")
    tasks = create_tasks_list(compute_covariance_in_a_single_file, file_paths, feature_columns)

    results = run_tasks_in_parallel(tasks, n_jobs, f"Correlating {len(feature_columns)} features")
    end_status(f"Correlated {len(feature_columns)} features in {len(tasks)} files")

    total = results[0]
    for result in results[1:]:
        total = merge_covariance_statistics(total, result)

    _, _, cross_products = total
    standard_deviations = np.sqrt(np.diag(cross_products))
    correlation = cross_products / np.outer(standard_deviations, standard_deviations)
    return pd.DataFrame(correlation, index=feature_columns, columns=feature_columns)

def find_highly_correlated_pairs(
    correlation: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
    feature_column: str = FEATURE_COLUMN,
    correlation_column: str = CORRELATION_COLUMN,
) -> pd.DataFrame:
    other_feature_column = f"other_{feature_column}"
    features = correlation.columns.tolist()

    records = []
    for first_index in range(len(features)):
        for second_index in range(first_index + 1, len(features)):
            value = correlation.iat[first_index, second_index]
            if abs(value) >= threshold:
                record = {
                    feature_column: features[first_index],
                    other_feature_column: features[second_index],
                    correlation_column: value,
                }
                records.append(record)

    pairs = pd.DataFrame(records, columns=[feature_column, other_feature_column, correlation_column])
    return pairs.sort_values(correlation_column, key=abs, ascending=False, ignore_index=True)
```

Kode Program 5.31 Matriks korelasi dan pasangan fitur berkorelasi tinggi

Kode Program 5.32 mengimplementasikan Langkah 7 untuk korelasi. Sel pertama menghitung matriks korelasi fitur yang tidak konstan dan menggambarkannya sebagai peta panas dengan peta warna divergen pada rentang −1 hingga 1, sedangkan sel kedua menampilkan pasangan fitur yang berkorelasi tinggi.

```python
correlation = compute_correlation_in_all_files(get_varying_features(descriptive_statistics))
plot_heatmap(
    correlation,
    "Pearson Correlation Between Features",
    colormap=DIVERGING_COLORMAP,
    value_range=(-1, 1),
    colorbar_label="Correlation",
    figsize=(16, 14),
    font_size=6,
)

results = find_highly_correlated_pairs(correlation)
with pd.option_context("display.max_rows", None):
    display(results)
del results, correlation, descriptive_statistics
```

Kode Program 5.32 Menjalankan analisis korelasi
