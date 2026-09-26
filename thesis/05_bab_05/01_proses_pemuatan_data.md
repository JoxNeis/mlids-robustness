## **5.1 Implementasi Pemuatan dan Transformasi Data ke Format Parquet**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.1, yaitu konversi dataset CSE-CIC-IDS2018 dari format CSV ke format Parquet. Kode program disusun mengikuti Langkah 1 hingga 8 pada algoritma di Subbab 4.1. Fungsi bantu dan pengaturan disajikan terlebih dahulu, dilanjutkan dengan pembentukan skema kolom, pembersihan satu *chunk*, konversi satu berkas, konversi seluruh berkas secara paralel, dan pelaksanaan konversi.

Kode Program 5.2 memuat empat fungsi bantu yang digunakan oleh hampir seluruh tahap pada bab ini. Fungsi *run_tasks_in_parallel* menjalankan sekumpulan tugas secara paralel menggunakan pustaka *joblib* dengan *backend* *loky*, jumlah pekerja sama dengan seluruh inti CPU (*n_jobs=-1*), dan jumlah *thread* internal setiap pekerja dibatasi menjadi satu (*inner_max_num_threads=1*), sebagaimana dirancang pada Subbab 4.1. Fungsi *get_all_file_names_in_folder* mengembalikan nama berkas dengan ekstensi tertentu yang telah diurutkan secara alfabetis sehingga urutan pemrosesan bersifat deterministik (Langkah 1). Fungsi *read_parquet_file* dan *write_dataframe_to_parquet* membaca dan menulis berkas Parquet menggunakan mesin *PyArrow*, dengan kompresi Snappy dan tanpa menyimpan indeks baris (Langkah 8).

```python
def run_tasks_in_parallel(tasks) -> Any:
    with parallel_config(backend="loky", inner_max_num_threads=1,verbose=0):
        return Parallel(n_jobs=-1, verbose=0)(tasks)

def get_all_file_names_in_folder(folder:str,file_extension:str):
    files = []
    for file in os.listdir(folder):
        if len(folder) > 0:
            if file.endswith(f".{file_extension}"):
                files.append(file)
    return sorted(files)

def read_parquet_file(source_folder_name: str, file_name: str) -> pd.DataFrame:
    file_path = os.path.join(source_folder_name, file_name)
    pd.set_option("io.parquet.engine", "pyarrow")
    return pd.read_parquet(file_path)

def write_dataframe_to_parquet(
    df: pd.DataFrame, target_folder_name: str, file_name: str
) -> str:
    output_path = os.path.join(target_folder_name, file_name)
    df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    return output_path
```

Kode Program 5.2 Fungsi bantu pemrosesan paralel serta pembacaan dan penulisan berkas

Kode Program 5.3 mendefinisikan pengaturan tahap konversi. Konstanta *PATH_FOLDER_CSV* dan *PATH_FOLDER_RAW* menetapkan folder sumber dan folder tujuan, *CHUNK_SIZE* menetapkan ukuran *chunk* sebesar 100.000 baris, *LABEL_COLUMN* menetapkan nama kolom label, dan *FEATURE_DTYPE* menetapkan tipe data fitur *float32*. Himpunan *COLUMNS_TO_DROP* memuat nama kolom yang tidak digunakan, yaitu *Flow ID*, alamat IP sumber dan tujuan, port sumber, dan *Timestamp*. Nama kolom tersebut ditulis dalam huruf kecil karena dibandingkan setelah nama kolom dinormalisasi, dan dalam dua ragam penulisan, misalnya *src ip* dan *source ip*, untuk mengantisipasi perbedaan penamaan kolom antarberkas.

```python
PATH_FOLDER_CSV = "cse-cic-ids2018"
PATH_FOLDER_RAW = "data-raw"

CHUNK_SIZE = 100_000

LABEL_COLUMN = "label"
FEATURE_DTYPE = "float32"

COLUMNS_TO_DROP = {
    "flow id",
    "src ip",
    "source ip",
    "src port",
    "source port",
    "dst ip",
    "destination ip",
    "timestamp",
}
```

Kode Program 5.3 Pengaturan tahap konversi CSV ke Parquet

Kode Program 5.4 mengimplementasikan Langkah 2. Fungsi *normalize_column_names* menghapus spasi di awal dan akhir nama kolom serta mengubahnya menjadi huruf kecil, *exclude_unwanted_columns* menyisihkan nama kolom yang termasuk *COLUMNS_TO_DROP*, dan *read_csv_column_names* membaca hanya baris judul berkas (*nrows=0*) sehingga biayanya rendah. Fungsi *create_column_schema* menelusuri seluruh berkas CSV secara berurutan dan menambahkan setiap nama kolom yang belum tercatat ke dalam skema, sehingga urutan kemunculannya dipertahankan. Sel terakhir membentuk skema satu kali dan menampilkan jumlah kolom beserta contoh namanya.

```python
def normalize_column_names(column_names) -> list[str]:
    columns = []
    for column in column_names:
        columns.append(str(column).strip().lower())
    return columns

def exclude_unwanted_columns(
    column_names: list[str], columns_to_drop: set[str] = COLUMNS_TO_DROP
) -> list[str]:
    remaining_columns = []
    for column in column_names:
        if column not in columns_to_drop:
            remaining_columns.append(column)
    return remaining_columns

def read_csv_column_names(source_folder_name: str, file_name: str) -> list[str]:
    file_path = os.path.join(source_folder_name, file_name)
    header_only = pd.read_csv(file_path, nrows=0)
    return normalize_column_names(header_only.columns)

def create_column_schema(
    source_folder_name: str, columns_to_drop: set[str] = COLUMNS_TO_DROP
) -> list[str]:
    column_schema: list[str] = []
    for file_name in get_all_file_names_in_folder(source_folder_name, "csv"):
        file_columns = read_csv_column_names(source_folder_name, file_name)
        for column_name in exclude_unwanted_columns(file_columns, columns_to_drop):
            if column_name not in column_schema:
                column_schema.append(column_name)
    return column_schema

column_schema = create_column_schema(PATH_FOLDER_CSV)
print(f"{len(column_schema)} columns: {column_schema[:3]} ... {column_schema[-2:]}")
```

Kode Program 5.4 Pembentukan skema kolom terpadu

Kode Program 5.5 mengimplementasikan Langkah 4 hingga 7 yang diterapkan pada satu *chunk*. Fungsi *standardize_column_names* menormalisasi nama kolom (Langkah 4). Fungsi *align_to_column_schema* menggunakan *DataFrame.reindex* sehingga kolom di luar skema dihapus dan kolom pada skema yang tidak dijumpai diisi *NaN* (Langkah 5). Fungsi *cast_features_to_numeric* mengonversi seluruh kolom selain label menjadi bilangan dengan *pandas.to_numeric* berparameter *errors="coerce"*, kemudian mengubahnya menjadi *float32* (Langkah 6). Fungsi *drop_invalid_label_rows* menghapus baris yang labelnya kosong atau berupa teks "label", yaitu sisa baris judul, setelah spasi dihapus dan huruf diubah menjadi huruf kecil (Langkah 7). Fungsi *clean_chunk* merangkai keempat fungsi tersebut sesuai urutannya.

```python
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = normalize_column_names(df.columns)
    return df

def align_to_column_schema(df: pd.DataFrame, column_schema: list[str]) -> pd.DataFrame:
    return df.reindex(columns=column_schema)

def cast_features_to_numeric(
    df: pd.DataFrame, label_column: str = LABEL_COLUMN, datatype: Any = FEATURE_DTYPE
) -> pd.DataFrame:
    feature_columns = [name for name in df.columns if name != label_column]
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce").astype(datatype)
    return df

def drop_invalid_label_rows(df: pd.DataFrame, label_column: str = LABEL_COLUMN) -> pd.DataFrame:
    label_values = df[label_column].astype("string").str.strip().str.lower()
    is_repeated_header = label_values == label_column
    return df[~is_repeated_header].dropna(subset=[label_column])

def clean_chunk(df: pd.DataFrame, column_schema: list[str]) -> pd.DataFrame:
    df = standardize_column_names(df)
    df = align_to_column_schema(df, column_schema)
    df = cast_features_to_numeric(df)
    df = drop_invalid_label_rows(df)
    return df
```

Kode Program 5.5 Pembersihan satu chunk data

Fungsi *build_parquet_file_name* pada Kode Program 5.6 membentuk nama berkas keluaran dari nama berkas CSV asal yang diikuti nomor *chunk* lima digit. Dengan cara ini, urutan leksikografis berkas sama dengan urutan *chunk* dan tanggal pengambilan data pada nama berkas asal tetap dapat ditelusuri (Langkah 8).

```python
def build_parquet_file_name(csv_file_name: str, chunk_number: int) -> str:
    base_name = os.path.splitext(csv_file_name)[0]
    return f"{base_name}_{chunk_number:05d}.parquet"
```

Kode Program 5.6 Penamaan berkas Parquet hasil konversi

Kode Program 5.7 mengimplementasikan Langkah 3 dan 8 untuk satu berkas CSV. Fungsi *read_csv_in_chunks* membaca berkas menggunakan *pandas.read_csv* dengan parameter *chunksize* dan *low_memory=False*. Fungsi *convert_csv_file_to_parquet* mengulang, untuk setiap *chunk*, pembersihan dengan *clean_chunk* dan penulisan ke berkas Parquet yang bernomor sesuai *chunk*, kemudian mengembalikan jumlah berkas yang ditulis.

```python
def read_csv_in_chunks(source_folder_name: str, file_name: str, chunk_size: int = CHUNK_SIZE):
    file_path = os.path.join(source_folder_name, file_name)
    return pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)

def convert_csv_file_to_parquet(
    source_folder_name: str,
    target_folder_name: str,
    file_name: str,
    column_schema: list[str],
    chunk_size: int = CHUNK_SIZE,
) -> int:
    written_files = 0
    for chunk_number, chunk in enumerate(
        read_csv_in_chunks(source_folder_name, file_name, chunk_size), start=1
    ):
        cleaned_chunk = clean_chunk(chunk, column_schema)
        write_dataframe_to_parquet(
            cleaned_chunk,
            target_folder_name,
            build_parquet_file_name(file_name, chunk_number),
        )
        written_files += 1
    return written_files
```

Kode Program 5.7 Konversi satu berkas CSV

Fungsi *convert_all_csv_files_to_parquet* pada Kode Program 5.8 menyatukan seluruh langkah. Fungsi ini mendaftar berkas CSV (Langkah 1), membentuk skema kolom satu kali sebelum pemrosesan paralel dimulai (Langkah 2), membuat folder tujuan, dan menjalankan *convert_csv_file_to_parquet* untuk setiap berkas CSV secara paralel melalui *run_tasks_in_parallel*. Setiap berkas CSV ditangani oleh satu pekerja, sedangkan *chunk* di dalam satu berkas diproses secara berurutan. Jumlah berkas CSV dan jumlah berkas Parquet yang dihasilkan dicetak di akhir.

```python
def convert_all_csv_files_to_parquet(
    source_folder_name: str,
    target_folder_name: str,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    csv_file_names = get_all_file_names_in_folder(source_folder_name, "csv")
    column_schema = create_column_schema(source_folder_name)
    os.makedirs(target_folder_name, exist_ok=True)

    written_per_file: list[int] = run_tasks_in_parallel(
        delayed(convert_csv_file_to_parquet)(
            source_folder_name,
            target_folder_name,
            file_name,
            column_schema,
            chunk_size,
        )
        for file_name in csv_file_names
    )

    print(
        f"Converted {len(csv_file_names)} CSV files "
        f"into {sum(written_per_file)} Parquet files in {target_folder_name}/."
    )
```

Kode Program 5.8 Konversi seluruh berkas CSV secara paralel

Kode Program 5.9 menjalankan seluruh tahap konversi dari folder *cse-cic-ids2018* ke folder *data-raw*.

```python
convert_all_csv_files_to_parquet(PATH_FOLDER_CSV, PATH_FOLDER_RAW)
```

Kode Program 5.9 Menjalankan konversi CSV ke Parquet
