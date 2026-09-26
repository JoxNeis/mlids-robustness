## **5.4 Implementasi Imputasi Nilai Hilang (*Missing Value Imputation*)**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.4, yaitu imputasi nilai hilang dengan median data latih. Kode program disusun mengikuti Langkah 1 hingga 7 pada algoritma di Subbab 4.4, dilanjutkan dengan kode untuk memverifikasi hasil imputasi.

Kode Program 5.24 memuat dua fungsi bantu yang digunakan pada tahap ini dan tahap-tahap berikutnya. Fungsi *get_split_file_paths* mengembalikan lokasi lengkap seluruh berkas Parquet pada suatu himpunan. Fungsi *process_all_splits* menjalankan suatu fungsi pemroses berkas pada setiap berkas di seluruh himpunan: fungsi ini membuat folder tujuan, memproses himpunan secara berurutan, memproses berkas di dalam satu himpunan secara paralel melalui *run_tasks_in_parallel*, dan mencetak jumlah berkas yang ditulis pada setiap himpunan. Argumen tambahan diteruskan apa adanya ke fungsi pemroses, sehingga fungsi yang sama dapat digunakan oleh imputasi, pengodean label, penskalaan, dan proyeksi.

```python
def get_split_file_paths(folder_name: str, split_name: str) -> list[str]:
    split_folder = get_split_folder(folder_name, split_name)
    return [
        os.path.join(split_folder, file_name)
        for file_name in get_all_file_names_in_folder(split_folder, "parquet")
    ]

def process_all_splits(
    source_folder_name: str,
    target_folder_name: str,
    split_names: tuple[str, ...],
    process_file,
    *arguments,
) -> None:
    create_split_folders(target_folder_name, split_names)

    for split_name in split_names:
        source_split_folder = get_split_folder(source_folder_name, split_name)
        target_split_folder = get_split_folder(target_folder_name, split_name)
        file_names = get_all_file_names_in_folder(source_split_folder, "parquet")

        run_tasks_in_parallel(
            delayed(process_file)(
                source_split_folder, target_split_folder, file_name, *arguments
            )
            for file_name in file_names
        )

        print(f"{split_name}: {len(file_names)} files written to {target_split_folder}/.")
```

Kode Program 5.24 Fungsi bantu pengambilan daftar berkas dan pemrosesan seluruh himpunan

Kode Program 5.25 mendefinisikan lokasi folder *cache* dan berkas median (*PATH_MEDIANS*), folder tujuan *data-imputed*, nama himpunan yang digunakan untuk mempelajari statistik (*FIT_SPLIT*, yaitu *train*), dan nilai bawaan 0,0 bagi fitur yang tidak memiliki nilai valid (*MEDIAN_FALLBACK*). Konstanta *PATH_CACHE* dan *FIT_SPLIT* digunakan kembali oleh tahap-tahap berikutnya.

```python
PATH_CACHE = "cache"
PATH_MEDIANS = os.path.join(PATH_CACHE, "median-imputer.json")

PATH_FOLDER_IMPUTED = "data-imputed"

FIT_SPLIT = "train"

MEDIAN_FALLBACK = 0.0
```

Kode Program 5.25 Pengaturan tahap imputasi

Kode Program 5.26 mengimplementasikan Langkah 2 dan 3. Fungsi *compute_medians* memindai seluruh berkas data latih secara *lazy* dengan *polars.scan_parquet*, memilih kolom bertipe numerik, dan menghitung median setiap kolom dengan mesin *streaming*. Fungsi *replace_empty_medians* mengganti median yang tidak dapat dihitung, yaitu bernilai *None*, dengan nilai bawaan, dan menampilkan peringatan yang memuat nama fitur yang terdampak.

```python
def compute_medians(file_paths: list[str]) -> dict[str, float | None]:
    lazy_frame = pl.scan_parquet(file_paths)
    schema = lazy_frame.collect_schema()

    numeric_columns = []
    for column, dtype in zip(schema.names(), schema.dtypes()):
        if dtype.is_numeric():
            numeric_columns.append(column)

    medians = lazy_frame.select(pl.col(numeric_columns).median()).collect(engine="streaming")
    return medians.row(0, named=True)

def replace_empty_medians(
    medians: dict[str, float | None], fallback: float = MEDIAN_FALLBACK
) -> dict[str, float]:
    complete_medians = {}
    empty_columns = []

    for column, median in medians.items():
        if median is None:
            empty_columns.append(column)
            complete_medians[column] = fallback
        else:
            complete_medians[column] = float(median)

    if empty_columns:
        print(
            f"Warning: {len(empty_columns)} columns had no value to take a median from and were "
            f"given {fallback}: {empty_columns}"
        )
    return complete_medians
```

Kode Program 5.26 Penghitungan median setiap fitur

Fungsi *save_medians* pada Kode Program 5.27 menyimpan median sebagai berkas JSON (Langkah 4), dan *load_medians* memuatnya kembali pada saat imputasi diterapkan.

```python
def save_medians(medians: dict[str, float], output_file: str) -> str:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as file:
        json.dump(medians, file, indent=4)
    return output_file

def load_medians(input_file: str) -> dict[str, float]:
    with open(input_file) as file:
        return json.load(file)
```

Kode Program 5.27 Penyimpanan dan pemuatan median

Fungsi *create_median_imputer* pada Kode Program 5.28 merangkai Langkah 1 hingga 4: mengambil daftar berkas data latih, menghitung median, menangani fitur tanpa nilai valid, dan menyimpan hasilnya, kemudian mencetak jumlah median, jumlah berkas sumber, dan lokasi berkas. Sel kedua menjalankannya pada folder *data-split-feature* dan menampilkan median setiap fitur.

```python
def create_median_imputer(
    source_folder_name: str, output_file: str, split_name: str = FIT_SPLIT
) -> dict[str, float]:
    file_paths = get_split_file_paths(source_folder_name, split_name)
    medians = compute_medians(file_paths)
    medians = replace_empty_medians(medians)
    save_medians(medians, output_file)

    print(
        f"Fitted {len(medians)} medians on the {split_name} split "
        f"({len(file_paths)} files) and saved them to {output_file}."
    )
    return medians

medians = create_median_imputer(PATH_FOLDER_SPLIT_FEATURE, PATH_MEDIANS)
pd.Series(medians, name="median")
```

Kode Program 5.28 Pembentukan pengimputasi median dari data latih

Kode Program 5.29 mengimplementasikan Langkah 5 hingga 7 untuk satu berkas. Fungsi *fill_missing_values* mengisi setiap *NaN* dengan median kolomnya menggunakan *DataFrame.fillna* berargumen kamus median. Fungsi *impute_file* membaca berkas, memanggil fungsi tersebut, dan menulis hasilnya ke folder tujuan.

```python
def fill_missing_values(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    return df.fillna(medians)

def impute_file(
    source_folder_name: str, target_folder_name: str, file_name: str, medians: dict[str, float]
) -> str:
    df = read_parquet_file(source_folder_name, file_name)
    df = fill_missing_values(df, medians)
    return write_dataframe_to_parquet(df, target_folder_name, file_name)
```

Kode Program 5.29 Pengisian nilai hilang pada satu berkas

Fungsi *impute_all_splits* pada Kode Program 5.30 memuat median dari berkas JSON dan menerapkannya pada himpunan yang ditentukan melalui *process_all_splits*, yang bawaannya adalah ketiga himpunan. Parameter *split_names* dapat dibatasi dan digunakan demikian pada pembentukan skenario gangguan (Subbab 5.15).

```python
def impute_all_splits(
    source_folder_name: str,
    target_folder_name: str,
    medians_file: str = PATH_MEDIANS,
    split_names: tuple[str, ...] = SPLIT_NAMES,
) -> None:
    medians = load_medians(medians_file)
    process_all_splits(
        source_folder_name, target_folder_name, split_names, impute_file, medians
    )
```

Kode Program 5.30 Pengisian nilai hilang pada seluruh himpunan

Kode Program 5.31 menjalankan imputasi dari folder *data-split-feature* ke folder *data-imputed*.

```python
impute_all_splits(PATH_FOLDER_SPLIT_FEATURE, PATH_FOLDER_IMPUTED)
```

Kode Program 5.31 Menjalankan imputasi

Kode Program 5.32 mengimplementasikan verifikasi hasil imputasi. Fungsi *count_missing_values_in_file* dan *count_missing_values_in_split* menghitung jumlah nilai hilang pada setiap kolom, dijumlahkan dari seluruh berkas satu himpunan secara paralel. Fungsi *summarize_missing_values* menyusunnya menjadi tabel untuk setiap himpunan yang hanya memuat kolom yang memiliki nilai hilang. Dua pemanggilan terakhir memeriksa folder *data-split-feature* (sebelum imputasi) dan folder *data-imputed* (setelah imputasi).

```python
def count_missing_values_in_file(folder_name: str, file_name: str) -> pd.Series:
    df = read_parquet_file(folder_name, file_name)
    return df.isna().sum()

def count_missing_values_in_split(folder_name: str, split_name: str) -> pd.Series:
    split_folder = get_split_folder(folder_name, split_name)
    file_names = get_all_file_names_in_folder(split_folder, "parquet")

    counts_per_file: list[pd.Series] = run_tasks_in_parallel(
        delayed(count_missing_values_in_file)(split_folder, file_name)
        for file_name in file_names
    )
    return sum(counts_per_file, start=pd.Series(dtype="int64"))

def summarize_missing_values(
    folder_name: str, split_names: tuple[str, ...] = SPLIT_NAMES
) -> pd.DataFrame:
    counts_per_split = {
        split_name: count_missing_values_in_split(folder_name, split_name)
        for split_name in split_names
    }

    table = pd.DataFrame(counts_per_split)
    table["total"] = table[list(split_names)].sum(axis=1)

    print(f"{int(table['total'].sum()):,} missing cells in {folder_name}/.")
    return table[table["total"] > 0]

summarize_missing_values(PATH_FOLDER_SPLIT_FEATURE)

summarize_missing_values(PATH_FOLDER_IMPUTED)
```

Kode Program 5.32 Verifikasi jumlah nilai hilang sebelum dan sesudah imputasi
