## **5.3 Implementasi Pembagian Dataset (*Data Splitting*)**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.3, yaitu pembagian dataset menjadi data latih, data validasi, dan data uji secara berstrata. Kode program disusun mengikuti Langkah 1 hingga 6 pada algoritma di Subbab 4.3, dilanjutkan dengan kode untuk memverifikasi proporsi kelas pada setiap himpunan.

Kode Program 5.15 memuat tiga fungsi bantu. Fungsi *filter_file_names* memilih nama berkas yang memuat suatu kata, yang pada tahap ini adalah tanggal pengambilan data (Langkah 1). Fungsi *get_split_folder* membentuk lokasi subfolder suatu himpunan, dan *create_split_folders* membuat subfolder untuk seluruh himpunan apabila belum tersedia (Langkah 2).

```python
def filter_file_names(file_names: list[str], filter_word: str) -> list[str]:
    matching_file_names = []
    for file_name in file_names:
        if filter_word in file_name:
            matching_file_names.append(file_name)
    return matching_file_names

def get_split_folder(folder_name: str, split_name: str) -> str:
    return os.path.join(folder_name, split_name)

def create_split_folders(folder_name: str, split_names: tuple[str, ...]) -> None:
    for split_name in split_names:
        os.makedirs(get_split_folder(folder_name, split_name), exist_ok=True)
```

Kode Program 5.15 Fungsi bantu penyaringan berkas dan pembuatan folder himpunan

Kode Program 5.16 mendefinisikan pengaturan pembagian. Konstanta yang ditetapkan adalah folder tujuan fitur dan label (*PATH_FOLDER_SPLIT_FEATURE* dan *PATH_FOLDER_SPLIT_LABEL*), nama ketiga himpunan (*SPLIT_NAMES*), ukuran data validasi dan data uji masing-masing 0,20 dengan ukuran data latih sebagai sisanya (*DEV_SIZE*, *TEST_SIZE*, dan *TRAIN_SIZE*), *random state* 42 (*RANDOM_STATE*), serta kesepuluh tanggal pengambilan data pada *COLLECTION_DAYS*, yaitu 14 Februari hingga 2 Maret 2018.

```python
PATH_FOLDER_SPLIT_FEATURE = "data-split-feature"
PATH_FOLDER_SPLIT_LABEL = "data-split-label"

SPLIT_NAMES = ("train", "dev", "test")

DEV_SIZE = 0.20
TEST_SIZE = 0.20
TRAIN_SIZE = 1.0 - DEV_SIZE - TEST_SIZE

RANDOM_STATE = 42

COLLECTION_DAYS = (
    "2018-02-14",
    "2018-02-15",
    "2018-02-16",
    "2018-02-20",
    "2018-02-21",
    "2018-02-22",
    "2018-02-23",
    "2018-02-28",
    "2018-03-01",
    "2018-03-02",
)
```

Kode Program 5.16 Pengaturan pembagian dataset

Fungsi *create_feature_and_label_folders* pada Kode Program 5.17 membuat folder terpisah untuk fitur dan label beserta subfolder *train*, *dev*, dan *test* (Langkah 2).

```python
def create_feature_and_label_folders(
    features_folder_name: str, labels_folder_name: str, split_names: tuple[str, ...] = SPLIT_NAMES
) -> None:
    create_split_folders(features_folder_name, split_names)
    create_split_folders(labels_folder_name, split_names)
```

Kode Program 5.17 Pembuatan folder fitur dan label

Fungsi *split_rows* pada Kode Program 5.18 mengimplementasikan Langkah 4. Fungsi ini terlebih dahulu memeriksa bahwa jumlah ukuran data validasi dan data uji berada di antara 0 dan 1, dan menghentikan proses dengan galat apabila syarat tersebut tidak terpenuhi. Pembagian pertama dengan *train_test_split* memisahkan data latih dari data sisihan (*held-out*) menggunakan *stratify* terhadap kolom label. Pembagian kedua membagi data sisihan menjadi data validasi dan data uji dengan proporsi ukuran data uji terhadap ukuran data sisihan (0,20/0,40 = 0,50).

```python
def split_rows(
    df: pd.DataFrame,
    dev_size: float = DEV_SIZE,
    test_size: float = TEST_SIZE,
    label_column: str = LABEL_COLUMN,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    held_out_size = dev_size + test_size
    if not 0.0 < held_out_size < 1.0:
        raise ValueError("dev_size + test_size must leave rows for training, so it must be < 1.0")

    train_rows, held_out_rows = train_test_split(
        df,
        test_size=held_out_size,
        random_state=random_state,
        stratify=df[label_column],
    )
    dev_rows, test_rows = train_test_split(
        held_out_rows,
        test_size=test_size / held_out_size,
        random_state=random_state,
        stratify=held_out_rows[label_column],
    )
    return {"train": train_rows, "dev": dev_rows, "test": test_rows}
```

Kode Program 5.18 Pembagian baris data secara berstrata

Fungsi *write_split_to_parquet* pada Kode Program 5.19 memisahkan kolom label dari kolom fitur (Langkah 5), kemudian menulis keduanya ke folder fitur dan folder label untuk himpunan yang bersangkutan dengan nama berkas yang sama (Langkah 6).

```python
def write_split_to_parquet(
    rows: pd.DataFrame,
    split_name: str,
    file_name: str,
    features_folder_name: str,
    labels_folder_name: str,
    label_column: str = LABEL_COLUMN,
) -> None:
    features = rows.drop(columns=[label_column])
    labels = rows[[label_column]]

    write_dataframe_to_parquet(
        features, get_split_folder(features_folder_name, split_name), file_name
    )
    write_dataframe_to_parquet(
        labels, get_split_folder(labels_folder_name, split_name), file_name
    )
```

Kode Program 5.19 Penulisan satu himpunan hasil pembagian

Fungsi *split_file* pada Kode Program 5.20 membaca satu berkas (Langkah 3), membaginya menjadi tiga himpunan dengan *split_rows*, dan menulis setiap himpunan dengan *write_split_to_parquet*.

```python
def split_file(
    source_folder_name: str,
    file_name: str,
    features_folder_name: str,
    labels_folder_name: str,
    dev_size: float = DEV_SIZE,
    test_size: float = TEST_SIZE,
    label_column: str = LABEL_COLUMN,
    random_state: int = RANDOM_STATE,
) -> None:
    df = read_parquet_file(source_folder_name, file_name)
    rows_per_split = split_rows(df, dev_size, test_size, label_column, random_state)

    for split_name, rows in rows_per_split.items():
        write_split_to_parquet(
            rows, split_name, file_name, features_folder_name, labels_folder_name, label_column
        )
```

Kode Program 5.20 Pembagian satu berkas

Fungsi *split_one_day* pada Kode Program 5.21 mengimplementasikan Langkah 1 untuk satu hari. Fungsi ini memilih berkas Parquet yang namanya memuat tanggal hari tersebut, menghentikan proses dengan galat apabila tidak ada berkas yang ditemukan, membuat folder tujuan, dan menjalankan *split_file* untuk setiap berkas secara paralel.

```python
def split_one_day(
    source_folder_name: str,
    features_folder_name: str,
    labels_folder_name: str,
    day: str,
    dev_size: float = DEV_SIZE,
    test_size: float = TEST_SIZE,
    label_column: str = LABEL_COLUMN,
    random_state: int = RANDOM_STATE,
) -> None:
    file_names = filter_file_names(
        get_all_file_names_in_folder(source_folder_name, "parquet"), day
    )
    if not file_names:
        raise ValueError(f"No Parquet files found for {day} in {source_folder_name}/.")

    create_feature_and_label_folders(features_folder_name, labels_folder_name)

    run_tasks_in_parallel(
        delayed(split_file)(
            source_folder_name,
            file_name,
            features_folder_name,
            labels_folder_name,
            dev_size,
            test_size,
            label_column,
            random_state,
        )
        for file_name in file_names
    )

    print(f"Split {len(file_names)} files of {day} into {', '.join(SPLIT_NAMES)}.")
```

Kode Program 5.21 Pembagian seluruh berkas pada satu hari pengambilan data

Kode Program 5.22 menjalankan pembagian secara berurutan untuk setiap hari pada *COLLECTION_DAYS*, dari folder *data-no-inf* ke folder fitur dan folder label.

```python
for day in COLLECTION_DAYS:
    split_one_day(
        PATH_FOLDER_NO_INF, PATH_FOLDER_SPLIT_FEATURE, PATH_FOLDER_SPLIT_LABEL, day
    )
```

Kode Program 5.22 Menjalankan pembagian untuk kesepuluh hari pengambilan data

Kode Program 5.23 mengimplementasikan verifikasi hasil pembagian. Fungsi *count_labels_in_file* dan *count_labels_in_folder* menghitung jumlah baris pada setiap kelas dari berkas label, dijumlahkan dari seluruh berkas secara paralel. Fungsi *summarize_splits* menyusun tabel jumlah baris setiap kelas pada ketiga himpunan beserta persentase setiap himpunan terhadap total kelas tersebut, dan mencetak proporsi target 60%, 20%, dan 20% sebagai pembanding. Sel terakhir menjalankannya pada folder label hasil pembagian.

```python
def count_labels_in_file(folder_name: str, file_name: str, label_column: str) -> pd.Series:
    labels = read_parquet_file(folder_name, file_name)
    return labels[label_column].value_counts()

def count_labels_in_folder(folder_name: str, label_column: str) -> pd.Series:
    file_names = get_all_file_names_in_folder(folder_name, "parquet")

    counts_per_file = run_tasks_in_parallel(
        delayed(count_labels_in_file)(folder_name, file_name, label_column)
        for file_name in file_names
    )

    return pd.concat(counts_per_file).groupby(level=0).sum()

def summarize_splits(labels_folder_name: str = PATH_FOLDER_SPLIT_LABEL) -> pd.DataFrame:
    counts_per_split = {
        split_name: count_labels_in_folder(
            get_split_folder(labels_folder_name, split_name), LABEL_COLUMN
        )
        for split_name in SPLIT_NAMES
    }

    table = pd.DataFrame(counts_per_split).fillna(0).astype(int)
    table["total"] = table[list(SPLIT_NAMES)].sum(axis=1)
    for split_name in SPLIT_NAMES:
        table[f"{split_name} %"] = (table[split_name] / table["total"] * 100).round(1)

    print(
        f"{table['total'].sum():,} rows over {len(table)} classes. "
        f"Target share: {TRAIN_SIZE:.0%} train / {DEV_SIZE:.0%} dev / {TEST_SIZE:.0%} test."
    )
    return table.sort_values("total", ascending=False)

summarize_splits(PATH_FOLDER_SPLIT_LABEL)
```

Kode Program 5.23 Verifikasi proporsi kelas pada setiap himpunan
