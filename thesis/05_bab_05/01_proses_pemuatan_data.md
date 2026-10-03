## **5.1 Implementasi Pemuatan dan Transformasi Data ke Format Parquet**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.1, yaitu konversi dataset CSE-CIC-IDS2018 dari format CSV ke format Parquet. Kode program disusun mengikuti langkah pada algoritma di Subbab 4.1, dimulai dari pengaturan, normalisasi nama kolom, penyusunan skema acuan, penerapan skema dan konversi tipe data, pembersihan baris tidak valid, penulisan berkas Parquet, hingga konversi seluruh berkas secara paralel.

Kode Program 5.9 menetapkan pengaturan tahap konversi. Konstanta *CHUNK_SIZE* menetapkan ukuran *chunk* sebesar 100.000 baris. Himpunan *COLUMNS_TO_DROP* memuat nama kolom yang tidak digunakan, yaitu *Flow ID*, alamat IP sumber dan tujuan, port sumber, dan *Timestamp*. Nama kolom tersebut ditulis dalam dua ragam penulisan, misalnya *src ip* dan *source ip*, untuk mengantisipasi perbedaan penamaan kolom antarberkas.

```python
CHUNK_SIZE = 100_000

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

Kode Program 5.9 Pengaturan tahap konversi CSV ke Parquet

Kode Program 5.10 mengimplementasikan Langkah 3, yaitu normalisasi nama kolom. Fungsi *normalize_column_name* menghapus spasi di awal dan akhir nama kolom, mengubah seluruh huruf menjadi huruf kecil, mengganti setiap rangkaian karakter selain huruf dan angka dengan garis bawah, dan menghapus garis bawah di awal dan akhir nama. Dengan normalisasi ini, misalnya, kolom *Flow Byts/s* menjadi *flow_byts_s*. Fungsi *normalize_column_names* menerapkan normalisasi tersebut pada daftar nama kolom.

```python
def normalize_column_name(column_name: Any) -> str:
    column_name = str(column_name).strip().lower()
    column_name = re.sub(r"[^0-9a-z]+", "_", column_name)
    return column_name.strip("_")

def normalize_column_names(column_names: list) -> list[str]:
    normalized_column_names = []
    for column_name in column_names:
        normalized_column_names.append(normalize_column_name(column_name))
    return normalized_column_names
```

Kode Program 5.10 Normalisasi nama kolom

Kode Program 5.11 mengimplementasikan Langkah 2 dan Langkah 4, yaitu penyusunan skema acuan dan penghapusan kolom yang tidak relevan. Fungsi *create_schema_columns* hanya membaca baris judul setiap berkas CSV (*nrows=0*) sehingga biayanya rendah, menormalisasi nama kolomnya, menggabungkan seluruh nama kolom dari semua berkas, kemudian mengurangkan nama kolom pada *COLUMNS_TO_DROP* yang juga telah dinormalisasi. Fungsi *create_schema* menyusun skema berupa tabel nama kolom dan tipe datanya, yaitu teks untuk kolom label dan *float32* untuk seluruh kolom lainnya.

```python
def create_schema_columns(folder_path: str, columns_to_drop: set[str] = COLUMNS_TO_DROP) -> list[str]:
    columns = set()
    for file_path in get_file_paths_in_folder(folder_path, "csv"):
        raw_columns = pd.read_csv(file_path, nrows=0).columns.tolist()
        columns.update(normalize_column_names(raw_columns))
    columns -= set(normalize_column_names(list(columns_to_drop)))
    return sorted(columns)

def create_schema(folder_path: str) -> pd.DataFrame:
    columns = create_schema_columns(folder_path)

    dtypes = []
    for column in columns:
        if column == LABEL_COLUMN:
            dtypes.append("str")
        else:
            dtypes.append(NUMERIC_DATA_TYPE)

    return pd.DataFrame({"column": columns, "dtype": dtypes})
```

Kode Program 5.11 Penyusunan skema kolom acuan

Kode Program 5.12 mengimplementasikan Langkah 5, yaitu konversi tipe data. Fungsi *apply_schema_to_dataframe* menormalisasi nama kolom sebuah *chunk*, menyusun ulang kolomnya sesuai skema dengan *DataFrame.reindex*, sehingga kolom di luar skema terhapus dan kolom skema yang tidak dijumpai diisi nilai kosong. Setiap kolom numerik dikonversi menggunakan *pandas.to_numeric* dengan parameter *errors="coerce"*, sehingga nilai yang tidak dapat dikonversi menjadi nilai kosong dan proses tidak terhenti, kemudian direpresentasikan dalam tipe *float32*. Kolom non-numerik, yaitu label, dikonversi ke tipe pada skema, dan peringatan ditampilkan apabila konversi gagal.

```python
def apply_schema_to_dataframe(
    dataframe: pd.DataFrame,
    schema: pd.DataFrame,
    numeric_data_type: str = NUMERIC_DATA_TYPE,
) -> pd.DataFrame:
    dataframe = dataframe.rename(columns=normalize_column_name)
    dataframe = dataframe.reindex(columns=schema["column"].tolist())

    for column, dtype in zip(schema["column"], schema["dtype"]):
        if not isinstance(dtype, str) or dtype == "":
            continue

        if dtype.startswith(("int", "float")):
            numeric_values = pd.to_numeric(dataframe[column], errors="coerce")
            dataframe[column] = numeric_values.astype(numeric_data_type)
        else:
            try:
                dataframe[column] = dataframe[column].astype(dtype)
            except (ValueError, TypeError) as error:
                warnings.warn(f"Could not cast column '{column}' to {dtype}: {error}")

    return dataframe
```

Kode Program 5.12 Penerapan skema dan konversi tipe data

Kode Program 5.13 mengimplementasikan Langkah 6. Fungsi *remove_duplicate_header* menghapus baris yang seluruh nilainya sama dengan nama kolom, yaitu baris judul yang tersisip berulang di tengah berkas CSV. Fungsi *clean_chunk* merangkai penghapusan baris judul tersebut dan penerapan skema pada satu *chunk*. Baris judul dihapus sebelum skema diterapkan karena pemeriksaannya membandingkan nilai dengan nama kolom asli pada berkas.

```python
def remove_duplicate_header(dataframe: pd.DataFrame) -> pd.DataFrame:
    is_header_row = dataframe.eq(dataframe.columns.tolist(), axis=1).all(axis=1)
    return dataframe[~is_header_row].reset_index(drop=True)

def clean_chunk(dataframe: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    dataframe = remove_duplicate_header(dataframe)
    dataframe = apply_schema_to_dataframe(dataframe, schema)
    return dataframe
```

Kode Program 5.13 Pembersihan satu chunk

Fungsi *write_chunk_to_parquet* pada Kode Program 5.14 menulis satu *chunk* ke folder tujuan sebagai berkas Parquet. Nama berkas terdiri atas nama berkas CSV asal yang diikuti nomor *chunk* empat digit, misalnya *2018-02-14-Wednesday_TrafficForML_CICFlowMeter_part0001.parquet*, sehingga tanggal pengambilan data tetap dapat ditelusuri dari nama berkas. Penulisan menggunakan *DataFrame.to_parquet* dengan mesin *PyArrow* dan kompresi Snappy bawaan, tanpa indeks baris.

```python
def write_chunk_to_parquet(
    chunk: pd.DataFrame,
    chunk_index: int,
    csv_file_path: str,
    output_folder_path: str,
) -> str:
    os.makedirs(output_folder_path, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    output_path = os.path.join(output_folder_path, f"{base_name}_part{chunk_index:04d}.parquet")
    chunk.to_parquet(output_path, index=False)
    return output_path
```

Kode Program 5.14 Penulisan satu chunk ke berkas Parquet

Kode Program 5.15 mengimplementasikan Langkah 1 untuk satu berkas CSV. Fungsi *convert_csv_to_parquet* membaca berkas menggunakan *pandas.read_csv* dengan parameter *chunksize* dan *low_memory=False*, sehingga berkas dibaca per 100.000 baris dan tipe data setiap *chunk* ditentukan dari seluruh baris pada *chunk* tersebut. Setiap *chunk* dibersihkan dengan *clean_chunk* kemudian ditulis dengan *write_chunk_to_parquet*, dengan nomor *chunk* yang dimulai dari satu. Status penulisan setiap *chunk* ditampilkan, dan daftar berkas Parquet yang dihasilkan dikembalikan.

```python
def convert_csv_to_parquet(
    csv_file_path: str,
    schema: pd.DataFrame,
    output_folder_path: str = PATH_FOLDER_PARQUET_DATASET,
    chunk_size: int = CHUNK_SIZE,
) -> list[str]:
    file_name = os.path.basename(csv_file_path)
    parquet_file_paths = []

    with pd.read_csv(csv_file_path, chunksize=chunk_size, low_memory=False) as chunks:
        for chunk_index, chunk in enumerate(chunks):
            chunk_index_interface = chunk_index + 1
            print_status(f"{file_name}: writing chunk {chunk_index_interface}")
            chunk = clean_chunk(chunk, schema)
            parquet_file_path = write_chunk_to_parquet(chunk, chunk_index_interface, csv_file_path, output_folder_path)
            parquet_file_paths.append(parquet_file_path)

    end_status(f"{file_name}: {len(parquet_file_paths)} chunks written")
    return parquet_file_paths
```

Kode Program 5.15 Konversi satu berkas CSV

Fungsi *convert_all_csv_to_parquet* pada Kode Program 5.16 menyatukan seluruh langkah. Fungsi ini mendaftar berkas CSV pada folder *cse-cic-ids2018*, membentuk skema satu kali sebelum pemrosesan paralel dimulai sehingga seluruh berkas Parquet memiliki susunan kolom yang identik, kemudian menjalankan *convert_csv_to_parquet* untuk setiap berkas CSV secara paralel dengan seluruh inti CPU (*n_jobs=-1*). Setiap berkas CSV ditangani oleh satu proses, sedangkan *chunk* di dalam satu berkas diproses secara berurutan. Jumlah berkas CSV dan berkas Parquet yang dihasilkan ditampilkan di akhir.

```python
def convert_all_csv_to_parquet(
    csv_folder_path: str = PATH_FOLDER_ORIGINAL_DATASET,
    n_jobs: int = -1,
) -> None:
    csv_file_paths = get_file_paths_in_folder(csv_folder_path, "csv")
    schema = create_schema(csv_folder_path)

    tasks = create_tasks_list(convert_csv_to_parquet, csv_file_paths, schema)
    results = run_tasks_in_parallel(tasks, n_jobs, "Converting CSV files to parquet")

    parquet_file_paths = []
    for result in results:
        parquet_file_paths.extend(result)

    end_status(f"Finished converting {len(csv_file_paths)} CSV files into {len(parquet_file_paths)} parquet files")
```

Kode Program 5.16 Konversi seluruh berkas CSV secara paralel

Kode Program 5.17 menjalankan seluruh tahap konversi dari folder *cse-cic-ids2018* ke folder *data-pipeline/01-raw-parquet*.

```python
convert_all_csv_to_parquet()
```

Kode Program 5.17 Menjalankan konversi CSV ke Parquet
