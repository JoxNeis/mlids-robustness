## **5.5 Implementasi Pembagian Dataset**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.5, yaitu pembagian data hasil penghapusan duplikat menjadi data latih dan data uji secara berstrata. Kode program disusun mengikuti Langkah 1 hingga 6 pada algoritma di Subbab 4.5.

Kode Program 5.43 menetapkan ukuran data uji sebesar 20% (*TEST_SIZE*) dan nama kolom pada tabel distribusi kelas setiap himpunan.

```python
TEST_SIZE = 0.2

TRAIN_ROWS_COLUMN = "train_rows"
TEST_ROWS_COLUMN = "test_rows"
TEST_PERCENTAGE_COLUMN = "test_percentage"
```

Kode Program 5.43 Pengaturan tahap pembagian dataset

Fungsi *load_all_files* pada Kode Program 5.44 mengimplementasikan Langkah 1. Fungsi ini membaca seluruh file Parquet pada sebuah folder secara *lazy* sebagai satu tabel menggunakan *polars.scan_parquet*, dengan urutan file yang telah diurutkan, dan menambahkan kolom nomor baris. Tidak ada data yang dimuat ke memori pada tahap ini.

```python
def load_all_files(folder_path: str, row_index_column: str = ROW_INDEX_COLUMN) -> pl.LazyFrame:
    file_paths = get_file_paths_in_folder(folder_path, "parquet")
    return pl.scan_parquet(file_paths).with_row_index(row_index_column)
```

Kode Program 5.44 Pembacaan seluruh file sebagai satu tabel secara lazy

Fungsi *assign_test_rows* pada Kode Program 5.45 mengimplementasikan Langkah 2 dan 3. Kolom label diubah menjadi kode kategori dengan tipe *Enum* yang urutan kategorinya sama dengan daftar kelas hasil EDA, kemudian hanya kode tersebut yang dimuat ke memori. Fungsi *train_test_split* dari *scikit-learn* membagi nomor baris dengan ukuran data uji 20%, stratifikasi berdasarkan kode label, dan *random state* 42. Nomor baris data uji dikembalikan sebagai *Series* dengan tipe yang sama dengan kolom nomor baris.

```python
def assign_test_rows(
    dataset: pl.LazyFrame,
    label_column: str = LABEL_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> pl.Series:
    label_codes = pl.col(label_column).cast(pl.Enum(load_classes_list())).to_physical()
    labels = dataset.select(label_codes).collect().to_series().to_numpy()

    row_indices = np.arange(len(labels))
    _, test_row_indices = train_test_split(
        row_indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    return pl.Series(ROW_INDEX_COLUMN, test_row_indices).cast(pl.get_index_type())
```

Kode Program 5.45 Penentuan baris data uji secara berstrata

Fungsi *write_features_and_labels* pada Kode Program 5.46 mengimplementasikan Langkah 5. Fungsi ini membuat folder himpunan, membuang kolom nomor baris, memisahkan fitur dan label, kemudian menulis keduanya ke file *features.parquet* dan *labels.parquet* menggunakan *sink_parquet*, yang mengalirkan data dari file sumber ke file tujuan tanpa memuat seluruh data ke memori.

```python
def write_features_and_labels(
    split: pl.LazyFrame,
    split_folder_path: str,
    label_column: str = LABEL_COLUMN,
    row_index_column: str = ROW_INDEX_COLUMN,
) -> None:
    os.makedirs(split_folder_path, exist_ok=True)

    features = split.drop(row_index_column, label_column)
    labels = split.select(label_column)
    features.sink_parquet(os.path.join(split_folder_path, FEATURES_FILE_NAME))
    labels.sink_parquet(os.path.join(split_folder_path, LABELS_FILE_NAME))
```

Kode Program 5.46 Penulisan fitur dan label sebuah himpunan

Fungsi *split_dataset* pada Kode Program 5.47 menyatukan seluruh langkah. Fungsi ini membaca data hasil penghapusan duplikat, menghitung jumlah baris, dan menentukan nomor baris data uji. Kondisi *is_in* kemudian digunakan untuk mengalirkan baris yang nomornya tidak termasuk daftar data uji ke folder *train* dan baris lainnya ke folder *test* (Langkah 4). Jumlah baris data latih dan data uji ditampilkan di akhir. Sel kedua menjalankan pembagian.

```python
def split_dataset(
    input_folder_path: str = PATH_FOLDER_DEDUPLICATED_DATASET,
    train_folder_path: str = PATH_FOLDER_SPLITTED_DATASET_TRAIN,
    test_folder_path: str = PATH_FOLDER_SPLITTED_DATASET_TEST,
    row_index_column: str = ROW_INDEX_COLUMN,
) -> None:
    dataset = load_all_files(input_folder_path, row_index_column)
    row_count = dataset.select(pl.len()).collect().item()

    print_status(f"Assigning {row_count:,} rows to train and test")
    test_row_indices = assign_test_rows(dataset)
    is_test_row = pl.col(row_index_column).is_in(test_row_indices.implode())

    print_status("Writing the train split")
    write_features_and_labels(dataset.filter(~is_test_row), train_folder_path)
    print_status("Writing the test split")
    write_features_and_labels(dataset.filter(is_test_row), test_folder_path)

    test_row_count = len(test_row_indices)
    train_row_count = row_count - test_row_count
    end_status(f"Split {row_count:,} rows: {train_row_count:,} train rows, {test_row_count:,} test rows")

split_dataset()
```

Kode Program 5.47 Pembagian dataset

Kode Program 5.48 mengimplementasikan Langkah 6. Fungsi *count_classes_in_split* menghitung jumlah baris setiap kelas dari file label sebuah himpunan menggunakan agregasi *group_by* pada *Polars*. Fungsi *summarize_split_classes* menyusun jumlah tersebut untuk data latih dan data uji dalam satu tabel, menghitung persentase data uji terhadap jumlah keduanya pada setiap kelas, dan mengurutkannya menurut jumlah baris data latih. Sel terakhir menampilkan tabel tersebut.

```python
def count_classes_in_split(
    split_folder_path: str,
    labels_file_name: str = LABELS_FILE_NAME,
    label_column: str = LABEL_COLUMN,
) -> pd.Series:
    labels_file_path = os.path.join(split_folder_path, labels_file_name)
    class_counts = pl.scan_parquet(labels_file_path).group_by(label_column).len().collect()
    return class_counts.to_pandas().set_index(label_column)["len"]

def summarize_split_classes(
    train_folder_path: str = PATH_FOLDER_SPLITTED_DATASET_TRAIN,
    test_folder_path: str = PATH_FOLDER_SPLITTED_DATASET_TEST,
    class_column: str = CLASS_COLUMN,
) -> pd.DataFrame:
    summary = pd.DataFrame({
        TRAIN_ROWS_COLUMN: count_classes_in_split(train_folder_path),
        TEST_ROWS_COLUMN: count_classes_in_split(test_folder_path),
    })
    total_rows = summary[TRAIN_ROWS_COLUMN] + summary[TEST_ROWS_COLUMN]
    summary[TEST_PERCENTAGE_COLUMN] = summary[TEST_ROWS_COLUMN] / total_rows * 100

    summary = summary.rename_axis(class_column).reset_index()
    return summary.sort_values(TRAIN_ROWS_COLUMN, ascending=False, ignore_index=True)

results = summarize_split_classes()
display(results)
del results
```

Kode Program 5.48 Pemeriksaan distribusi kelas pada data latih dan data uji
