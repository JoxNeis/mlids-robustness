## **5.5 Implementasi Pengodean Label (*Label Encoding*)**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.5, yaitu pengodean label kelas menjadi bilangan bulat. Kode program disusun mengikuti Langkah 1 hingga 7 pada algoritma di Subbab 4.5, dilanjutkan dengan kode untuk menerjemahkan kode kembali menjadi nama kelas dan memverifikasi hasil pengodean.

Kode Program 5.33 mendefinisikan lokasi berkas pemetaan kelas (*PATH_LABEL_ENCODER*), folder tujuan *data-encoded-label*, dan fungsi *load_label_classes* yang membaca daftar nama kelas dari berkas JSON. Fungsi *load_label_classes* digunakan kembali untuk menerjemahkan kode menjadi nama kelas pada tahap evaluasi.

```python
def load_label_classes(input_file: str) -> list[str]:
    with open(input_file) as file:
        return json.load(file)

PATH_LABEL_ENCODER = os.path.join(PATH_CACHE, "label-encoder.json")

PATH_FOLDER_ENCODED_LABEL = "data-encoded-label"
```

Kode Program 5.33 Pemuat daftar kelas dan pengaturan pengodean label

Kode Program 5.34 mengimplementasikan Langkah 1 hingga 3. Fungsi *get_label_classes* memindai berkas label data latih secara *lazy* dengan *polars.scan_parquet* dan mengambil nilai unik pada kolom label dengan mesin *streaming*. Fungsi *create_label_encoder* melatih *LabelEncoder* pada daftar nama kelas tersebut sehingga setiap kelas memperoleh kode berdasarkan urutan leksikografisnya.

```python
def get_label_classes(
    labels_folder_name: str, split_name: str = FIT_SPLIT, label_column: str = LABEL_COLUMN
) -> list[str]:
    file_paths = get_split_file_paths(labels_folder_name, split_name)

    distinct_labels = (
        pl.scan_parquet(file_paths)
        .select(pl.col(label_column).unique())
        .collect(engine="streaming")
    )
    return distinct_labels[label_column].to_list()

def create_label_encoder(class_names: list[str]) -> LabelEncoder:
    return LabelEncoder().fit(np.array(class_names))
```

Kode Program 5.34 Penentuan himpunan kelas dan pembentukan pengode label

Fungsi *save_label_classes* pada Kode Program 5.35 menyimpan daftar nama kelas, dengan posisi yang sama dengan kodenya, sebagai berkas JSON (Langkah 4). Fungsi *load_label_encoder* membentuk kembali pengode yang identik dari berkas tersebut.

```python
def save_label_classes(class_names: list[str], output_file: str) -> str:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as file:
        json.dump(class_names, file, indent=4)
    return output_file

def load_label_encoder(input_file: str) -> LabelEncoder:
    return create_label_encoder(load_label_classes(input_file))
```

Kode Program 5.35 Penyimpanan dan pemuatan pemetaan kelas

Fungsi *create_and_save_label_encoder* pada Kode Program 5.36 merangkai penentuan kelas, pembentukan pengode, dan penyimpanan pemetaan, kemudian mencetak jumlah kelas yang ditemukan. Sel kedua menjalankannya dan menampilkan tabel pemetaan antara kode dan nama kelas.

```python
def create_and_save_label_encoder(
    labels_folder_name: str, output_file: str, split_name: str = FIT_SPLIT
) -> LabelEncoder:
    class_names = get_label_classes(labels_folder_name, split_name)
    encoder = create_label_encoder(class_names)
    save_label_classes([str(class_name) for class_name in encoder.classes_], output_file)

    print(
        f"Found {len(encoder.classes_)} classes in the {split_name} split "
        f"and saved them to {output_file}."
    )
    return encoder

label_encoder = create_and_save_label_encoder(PATH_FOLDER_SPLIT_LABEL, PATH_LABEL_ENCODER)
pd.Series(label_encoder.classes_, name="class").rename_axis("id").to_frame()
```

Kode Program 5.36 Pembentukan dan penyimpanan pengode label dari data latih

Kode Program 5.37 mengimplementasikan Langkah 5 hingga 7 untuk satu berkas. Fungsi *encode_labels* mengubah kolom label menjadi kode dengan *LabelEncoder.transform* dan menyusunnya menjadi *DataFrame* satu kolom. Fungsi *encode_label_file* membaca berkas label, memanggil fungsi tersebut, dan menulis hasilnya ke folder tujuan.

```python
def encode_labels(
    labels: pd.DataFrame, encoder: LabelEncoder, label_column: str = LABEL_COLUMN
) -> pd.DataFrame:
    return pd.DataFrame({label_column: encoder.transform(labels[label_column])})

def encode_label_file(
    source_folder_name: str, target_folder_name: str, file_name: str, encoder: LabelEncoder
) -> str:
    labels = read_parquet_file(source_folder_name, file_name)
    encoded_labels = encode_labels(labels, encoder)
    return write_dataframe_to_parquet(encoded_labels, target_folder_name, file_name)
```

Kode Program 5.37 Pengodean label pada satu berkas

Fungsi *encode_all_splits* pada Kode Program 5.38 memuat pengode dari berkas JSON dan menerapkannya pada ketiga himpunan label melalui *process_all_splits*.

```python
def encode_all_splits(
    source_folder_name: str,
    target_folder_name: str,
    encoder_file: str = PATH_LABEL_ENCODER,
    split_names: tuple[str, ...] = SPLIT_NAMES,
) -> None:
    encoder = load_label_encoder(encoder_file)
    process_all_splits(
        source_folder_name, target_folder_name, split_names, encode_label_file, encoder
    )
```

Kode Program 5.38 Pengodean label pada seluruh himpunan

Kode Program 5.39 menjalankan pengodean dari folder *data-split-label* ke folder *data-encoded-label*.

```python
encode_all_splits(PATH_FOLDER_SPLIT_LABEL, PATH_FOLDER_ENCODED_LABEL)
```

Kode Program 5.39 Menjalankan pengodean label

Fungsi *decode_labels* pada Kode Program 5.40 menerjemahkan kode kelas kembali menjadi nama kelas dengan *LabelEncoder.inverse_transform*, dan digunakan pada verifikasi hasil pengodean.

```python
def decode_labels(encoded, encoder_file: str = PATH_LABEL_ENCODER) -> np.ndarray:
    encoder = load_label_encoder(encoder_file)
    return encoder.inverse_transform(np.asarray(encoded))
```

Kode Program 5.40 Penerjemahan kode kembali menjadi nama kelas

Fungsi *summarize_encoded_labels* pada Kode Program 5.41 menghitung jumlah baris pada setiap kelas di setiap himpunan dari folder *data-encoded-label* menggunakan *count_labels_in_folder* (Kode Program 5.23), menerjemahkan kode kelas menjadi nama kelas, dan menyusun tabel beserta total setiap kelas. Sel kedua menjalankannya.

```python
def summarize_encoded_labels(
    folder_name: str,
    encoder_file: str = PATH_LABEL_ENCODER,
    split_names: tuple[str, ...] = SPLIT_NAMES,
) -> pd.DataFrame:
    counts_per_split = {
        split_name: count_labels_in_folder(
            get_split_folder(folder_name, split_name), LABEL_COLUMN
        )
        for split_name in split_names
    }

    table = pd.DataFrame(counts_per_split).fillna(0).astype(int).sort_index()
    table.insert(0, "class", decode_labels(table.index, encoder_file))
    table["total"] = table[list(split_names)].sum(axis=1)

    print(f"{table['total'].sum():,} rows over {len(table)} classes in {folder_name}/.")
    return table

summarize_encoded_labels(PATH_FOLDER_ENCODED_LABEL)
```

Kode Program 5.41 Verifikasi jumlah baris pada setiap kelas setelah pengodean
