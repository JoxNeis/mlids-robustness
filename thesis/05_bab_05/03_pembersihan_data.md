## **5.3 Implementasi Pembersihan Data**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.3, yaitu penghapusan fitur konstan serta penghapusan baris yang memuat nilai hilang atau nilai tak hingga. Kode program disusun mengikuti Langkah 1 hingga 6 pada algoritma di Subbab 4.3.

Kode Program 5.33 menetapkan pengaturan tahap pembersihan. Daftar *CONSTANT_FEATURES* memuat delapan fitur konstan hasil EDA (Subbab 5.2), yang ditetapkan sebagai konstanta agar seluruh berkas kehilangan kolom yang sama. Konstanta berikutnya menetapkan nama kolom pada tabel ringkasan pembersihan, dan *CLEANING_N_JOBS* menetapkan jumlah proses paralel sebanyak delapan.

```python
CONSTANT_FEATURES = [
    "bwd_blk_rate_avg",
    "bwd_byts_b_avg",
    "bwd_pkts_b_avg",
    "bwd_psh_flags",
    "bwd_urg_flags",
    "fwd_blk_rate_avg",
    "fwd_byts_b_avg",
    "fwd_pkts_b_avg",
]

ROWS_BEFORE_COLUMN = "rows_before"
MISSING_ROWS_COLUMN = "missing_rows_removed"
INFINITE_ROWS_COLUMN = "infinite_rows_removed"
ROWS_AFTER_COLUMN = "rows_after"

CLEANING_N_JOBS = 8
```

Kode Program 5.33 Pengaturan tahap pembersihan data

Kode Program 5.34 mengimplementasikan Langkah 2 hingga 4. Fungsi *remove_constant_features* menghapus kolom fitur konstan. Fungsi *remove_missing_value_rows* menghapus setiap baris yang memiliki sedikitnya satu nilai kosong menggunakan *DataFrame.dropna*. Fungsi *remove_infinite_value_rows* memilih kolom bertipe numerik, menandai baris yang memiliki sedikitnya satu nilai +∞ atau −∞ menggunakan *numpy.isinf*, dan menghapus baris tersebut.

```python
def remove_constant_features(
    dataframe: pd.DataFrame,
    constant_features: list[str] = CONSTANT_FEATURES,
) -> pd.DataFrame:
    return dataframe.drop(columns=constant_features)

def remove_missing_value_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.dropna()

def remove_infinite_value_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = dataframe.select_dtypes("number").columns
    has_infinite_value = np.isinf(dataframe[numeric_columns]).any(axis=1)
    return dataframe[~has_infinite_value]
```

Kode Program 5.34 Penghapusan fitur konstan, nilai hilang, dan nilai tak hingga

Fungsi *clean_a_single_file* pada Kode Program 5.35 menerapkan ketiga langkah tersebut secara berurutan pada satu berkas, kemudian menulis hasilnya ke folder *data-pipeline/02-cleaned-parquet* dengan nama berkas yang sama (Langkah 5). Jumlah baris dicatat sebelum pembersihan, setelah penghapusan nilai hilang, dan setelah penghapusan nilai tak hingga, sehingga fungsi ini mengembalikan jumlah baris yang dihapus oleh setiap langkah.

```python
def clean_a_single_file(
    file_path: str,
    output_folder_path: str = PATH_FOLDER_CLEANED_DATASET,
    constant_features: list[str] = CONSTANT_FEATURES,
    file_column: str = FILE_COLUMN,
) -> dict:
    dataframe = pd.read_parquet(file_path)
    row_count_before = len(dataframe)

    dataframe = remove_constant_features(dataframe, constant_features)
    dataframe = remove_missing_value_rows(dataframe)
    row_count_after_missing = len(dataframe)
    dataframe = remove_infinite_value_rows(dataframe)
    row_count_after = len(dataframe)

    file_name = os.path.basename(file_path)
    os.makedirs(output_folder_path, exist_ok=True)
    dataframe.to_parquet(os.path.join(output_folder_path, file_name), index=False)

    return {
        file_column: file_name,
        ROWS_BEFORE_COLUMN: row_count_before,
        MISSING_ROWS_COLUMN: row_count_before - row_count_after_missing,
        INFINITE_ROWS_COLUMN: row_count_after_missing - row_count_after,
        ROWS_AFTER_COLUMN: row_count_after,
    }
```

Kode Program 5.35 Pembersihan satu berkas

Fungsi *clean_all_files* pada Kode Program 5.36 mendaftar seluruh berkas pada folder *data-pipeline/01-raw-parquet* (Langkah 1), menjalankan *clean_a_single_file* pada setiap berkas secara paralel dengan delapan proses, dan menyusun catatan setiap berkas menjadi tabel ringkasan. Jumlah baris sebelum dan sesudah pembersihan pada seluruh berkas ditampilkan di akhir. Sel kedua menjalankan pembersihan dan menjumlahkan tabel ringkasan per hari dengan *group_and_sum* (Langkah 6).

```python
def clean_all_files(
    input_folder_path: str = PATH_FOLDER_PARQUET_DATASET,
    output_folder_path: str = PATH_FOLDER_CLEANED_DATASET,
    n_jobs: int = CLEANING_N_JOBS,
) -> pd.DataFrame:
    file_paths = get_file_paths_in_folder(input_folder_path, "parquet")
    tasks = create_tasks_list(clean_a_single_file, file_paths, output_folder_path)

    records = run_tasks_in_parallel(tasks, n_jobs, "Cleaning files")
    summary = pd.DataFrame(records)

    rows_before = summary[ROWS_BEFORE_COLUMN].sum()
    rows_after = summary[ROWS_AFTER_COLUMN].sum()
    end_status(f"Cleaned {len(tasks)} files: {rows_before:,} rows -> {rows_after:,} rows")
    return summary

results = group_and_sum(clean_all_files(), FILE_COLUMN, DAY_COLUMN, DAY_REGEX)
display(results)
del results
```

Kode Program 5.36 Pembersihan seluruh berkas secara paralel
