## **5.2 Implementasi Pembersihan Data Bernilai Tak Hingga (*Infinite Values*)**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.2, yaitu penggantian nilai tak hingga menjadi *NaN*. Kode program disusun mengikuti Langkah 1 hingga 5 pada algoritma di Subbab 4.2, dilanjutkan dengan kode untuk memverifikasi hasil penggantian.

Kode Program 5.10 mendefinisikan folder tujuan *data-no-inf* dan dua fungsi. Fungsi *replace_infinite_values* memilih kolom bertipe numerik menggunakan *DataFrame.select_dtypes* (Langkah 3), kemudian mengganti nilai +∞ dan −∞ dengan *NaN* pada seluruh kolom tersebut sekaligus menggunakan *DataFrame.replace* (Langkah 4). Fungsi *replace_infinite_values_in_file* membaca satu berkas (Langkah 2), memanggil fungsi tersebut, dan menulis hasilnya ke folder tujuan dengan nama berkas yang sama (Langkah 5).

```python
PATH_FOLDER_NO_INF = "data-no-inf"

def replace_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = df.select_dtypes(include=np.number).columns
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return df

def replace_infinite_values_in_file(
    source_folder_name: str, target_folder_name: str, file_name: str
) -> str:
    df = read_parquet_file(source_folder_name, file_name)
    df = replace_infinite_values(df)
    return write_dataframe_to_parquet(df, target_folder_name, file_name)
```

Kode Program 5.10 Penggantian nilai tak hingga pada satu berkas

Fungsi *replace_infinite_values_in_folder* pada Kode Program 5.11 mendaftar berkas Parquet pada folder sumber (Langkah 1), membuat folder tujuan, dan menjalankan penggantian untuk setiap berkas secara paralel melalui *run_tasks_in_parallel*, kemudian mencetak jumlah berkas yang ditulis ulang.

```python
def replace_infinite_values_in_folder(source_folder_name: str, target_folder_name: str) -> None:
    file_names = get_all_file_names_in_folder(source_folder_name, "parquet")
    os.makedirs(target_folder_name, exist_ok=True)

    written_files = run_tasks_in_parallel(
        delayed(replace_infinite_values_in_file)(source_folder_name, target_folder_name, file_name)
        for file_name in file_names
    )

    print(f"Rewrote {len(written_files)} Parquet files into {target_folder_name}/.")
```

Kode Program 5.11 Penggantian nilai tak hingga pada seluruh berkas secara paralel

Kode Program 5.12 menjalankan penggantian dari folder *data-raw* ke folder *data-no-inf*.

```python
replace_infinite_values_in_folder(PATH_FOLDER_RAW, PATH_FOLDER_NO_INF)
```

Kode Program 5.12 Menjalankan penggantian nilai tak hingga

Kode Program 5.13 mengimplementasikan verifikasi hasil penggantian. Fungsi *count_infinite_values_in_file* menghitung jumlah nilai tak hingga pada setiap kolom numerik satu berkas dengan *numpy.isinf*. Fungsi *count_infinite_values* menjumlahkan hasilnya dari seluruh berkas secara paralel, mencetak total nilai tak hingga, dan hanya mengembalikan kolom yang memuat nilai tak hingga.

```python
def count_infinite_values_in_file(source_folder_name: str, file_name: str) -> pd.Series:
    df = read_parquet_file(source_folder_name, file_name)
    numeric_columns = df.select_dtypes(include=np.number)
    return np.isinf(numeric_columns).sum()

def count_infinite_values(source_folder_name: str) -> pd.Series:
    file_names = get_all_file_names_in_folder(source_folder_name, "parquet")

    counts_per_file: list[pd.Series] = run_tasks_in_parallel(
        delayed(count_infinite_values_in_file)(source_folder_name, file_name)
        for file_name in file_names
    )

    counts_per_column = sum(counts_per_file, start=pd.Series(dtype="int64"))
    print(
        f"{int(counts_per_column.sum()):,} infinite values "
        f"in {len(file_names)} Parquet files in {source_folder_name}/."
    )
    return counts_per_column[counts_per_column > 0]
```

Kode Program 5.13 Penghitungan nilai tak hingga untuk verifikasi

Kode Program 5.14 menjalankan penghitungan tersebut pada folder *data-raw* (sebelum penggantian) dan pada folder *data-no-inf* (setelah penggantian). Setelah penggantian, tidak ada kolom yang diharapkan masih memuat nilai tak hingga.

```python
count_infinite_values(PATH_FOLDER_RAW)

count_infinite_values(PATH_FOLDER_NO_INF)
```

Kode Program 5.14 Verifikasi sebelum dan sesudah penggantian nilai tak hingga
