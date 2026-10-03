## **5.4 Implementasi Penghapusan Data Duplikat**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.4, yaitu penghapusan baris duplikat pada seluruh dataset menggunakan sidik jari *hash*. Kode program disusun mengikuti Langkah 1 hingga 7 pada algoritma di Subbab 4.4.

Kode Program 5.37 menetapkan nama kolom jumlah baris duplikat pada tabel ringkasan, dua *seed hash* yang digunakan untuk membentuk sidik jari, yaitu 0 dan 1 (*ROW_HASH_SEEDS*), serta jumlah proses paralel sebanyak delapan.

```python
DUPLICATE_ROWS_COLUMN = "duplicate_rows_removed"

ROW_HASH_SEEDS = (0, 1)
DEDUPLICATION_N_JOBS = 8
```

Kode Program 5.37 Pengaturan tahap penghapusan duplikat

Kode Program 5.38 mengimplementasikan Langkah 2. Fungsi *hash_rows_in_a_single_file* membaca satu file dengan *Polars*, menggabungkan seluruh kolom setiap baris menjadi satu nilai *struct*, kemudian menghitung *hash* *struct* tersebut dengan setiap *seed*. Hasilnya adalah matriks dengan satu baris untuk setiap baris data dan dua kolom *hash* 64-bit. Fungsi *hash_rows_in_all_files* menjalankannya pada seluruh file secara paralel dan mengembalikan matriks *hash* setiap file sesuai urutan file.

```python
def hash_rows_in_a_single_file(
    file_path: str,
    hash_seeds: tuple[int, ...] = ROW_HASH_SEEDS,
) -> np.ndarray:
    row = pl.struct(pl.all())
    hash_columns = []
    for hash_seed in hash_seeds:
        hash_columns.append(row.hash(hash_seed).alias(f"hash_{hash_seed}"))
    return pl.read_parquet(file_path).select(hash_columns).to_numpy()

def hash_rows_in_all_files(
    file_paths: list[str],
    n_jobs: int = DEDUPLICATION_N_JOBS,
) -> list[np.ndarray]:
    tasks = create_tasks_list(hash_rows_in_a_single_file, file_paths)
    return run_tasks_in_parallel(tasks, n_jobs, "Hashing rows")
```

Kode Program 5.38 Pembentukan sidik jari setiap baris

Kode Program 5.39 mengimplementasikan Langkah 3 dan 4. Fungsi *find_duplicate_rows* menggabungkan matriks *hash* seluruh file sesuai urutannya, kemudian menandai setiap baris yang pasangan *hash*-nya telah muncul pada baris sebelumnya menggunakan *DataFrame.duplicated* dengan parameter *keep="first"*, sehingga kemunculan pertama tidak ditandai. Fungsi *get_row_offsets* menghitung posisi awal baris setiap file di dalam urutan gabungan, yaitu jumlah kumulatif baris pada file-file sebelumnya.

```python
def find_duplicate_rows(row_hashes: list[np.ndarray]) -> np.ndarray:
    hashes = pd.DataFrame(np.concatenate(row_hashes))
    return hashes.duplicated(keep="first").to_numpy()

def get_row_offsets(file_paths: list[str], per_file_rows: list[np.ndarray]) -> dict[str, int]:
    row_offsets = {}
    row_offset = 0
    for file_path, file_rows in zip(file_paths, per_file_rows):
        row_offsets[os.path.basename(file_path)] = row_offset
        row_offset += len(file_rows)
    return row_offsets
```

Kode Program 5.39 Penandaan baris duplikat dan posisi awal setiap file

Fungsi *remove_duplicate_rows_in_a_single_file* pada Kode Program 5.40 mengimplementasikan Langkah 5 dan 6 untuk satu file. Fungsi ini membaca file, memotong penanda duplikat sesuai posisi awal dan jumlah baris file tersebut, menghapus baris yang ditandai, dan menulis hasilnya ke folder *data-pipeline/03-deduplicated-parquet* dengan nama file yang sama. Jumlah baris sebelum dan sesudah penghapusan dikembalikan sebagai catatan.

```python
def remove_duplicate_rows_in_a_single_file(
    file_path: str,
    is_duplicate: np.ndarray,
    row_offsets: dict[str, int],
    output_folder_path: str = PATH_FOLDER_DEDUPLICATED_DATASET,
    file_column: str = FILE_COLUMN,
) -> dict:
    file_name = os.path.basename(file_path)
    dataframe = pd.read_parquet(file_path)
    row_count_before = len(dataframe)

    row_offset = row_offsets[file_name]
    is_duplicate_in_file = is_duplicate[row_offset:row_offset + row_count_before]
    dataframe = dataframe[~is_duplicate_in_file]
    row_count_after = len(dataframe)

    os.makedirs(output_folder_path, exist_ok=True)
    dataframe.to_parquet(os.path.join(output_folder_path, file_name), index=False)

    return {
        file_column: file_name,
        ROWS_BEFORE_COLUMN: row_count_before,
        DUPLICATE_ROWS_COLUMN: row_count_before - row_count_after,
        ROWS_AFTER_COLUMN: row_count_after,
    }
```

Kode Program 5.40 Penghapusan baris duplikat pada satu file

Fungsi *remove_duplicate_rows_in_all_files* pada Kode Program 5.41 menyatukan seluruh langkah. Fungsi ini mendaftar file pada folder *data-pipeline/02-cleaned-parquet* (Langkah 1), membentuk sidik jari, menandai duplikat, dan menghitung posisi awal setiap file. Matriks *hash* dihapus dari memori setelah penanda terbentuk, kemudian penghapusan pada setiap file dijalankan secara paralel, dan jumlah baris sebelum dan sesudah penghapusan ditampilkan di akhir.

```python
def remove_duplicate_rows_in_all_files(
    input_folder_path: str = PATH_FOLDER_CLEANED_DATASET,
    output_folder_path: str = PATH_FOLDER_DEDUPLICATED_DATASET,
    n_jobs: int = DEDUPLICATION_N_JOBS,
) -> pd.DataFrame:
    file_paths = get_file_paths_in_folder(input_folder_path, "parquet")

    row_hashes = hash_rows_in_all_files(file_paths, n_jobs)
    print_status("Finding duplicate rows")
    is_duplicate = find_duplicate_rows(row_hashes)
    row_offsets = get_row_offsets(file_paths, row_hashes)
    del row_hashes

    tasks = create_tasks_list(
        remove_duplicate_rows_in_a_single_file,
        file_paths,
        is_duplicate,
        row_offsets,
        output_folder_path,
    )
    records = run_tasks_in_parallel(tasks, n_jobs, f"Removing {is_duplicate.sum():,} duplicate rows")
    summary = pd.DataFrame(records)

    rows_before = summary[ROWS_BEFORE_COLUMN].sum()
    rows_after = summary[ROWS_AFTER_COLUMN].sum()
    end_status(f"Removed duplicates from {len(tasks)} files: {rows_before:,} rows -> {rows_after:,} rows")
    return summary
```

Kode Program 5.41 Penghapusan baris duplikat pada seluruh file

Kode Program 5.42 mengimplementasikan Langkah 7. Sel pertama menjalankan penghapusan duplikat dan menjumlahkan tabel ringkasan per hari, sedangkan sel kedua menghitung ulang distribusi kelas pada data hasil penghapusan duplikat menggunakan fungsi dari Subbab 5.2.

```python
results = group_and_sum(remove_duplicate_rows_in_all_files(), FILE_COLUMN, DAY_COLUMN, DAY_REGEX)
display(results)
del results

results = group_and_sum(
    count_classes_in_all_files(PATH_FOLDER_DEDUPLICATED_DATASET),
    FILE_COLUMN,
    DAY_COLUMN,
    DAY_REGEX,
)
results = count_total_classes(results)
display(results)
del results
```

Kode Program 5.42 Menjalankan penghapusan duplikat dan menghitung distribusi kelas
