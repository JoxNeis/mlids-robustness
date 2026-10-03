## **5.6 Implementasi Pengodean Label**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.6, yaitu pembentukan *LabelEncoder* dari daftar kelas dan pengodean label pada data latih dan data uji. Kode program disusun mengikuti Langkah 1 hingga 4 pada algoritma di Subbab 4.6.

Kode Program 5.49 memuat fungsi penyimpanan dan pemuatan *LabelEncoder*. Fungsi *load_label_encoder* dan *dump_label_encoder* memuat dan menyimpan objek *LabelEncoder* pada berkas *cache/label-encoder-transformer.pkl* menggunakan fungsi bantu *load* dan *dump* (pendahuluan Bab 5).

```python
def load_label_encoder(label_encoder_path:str = PATH_LABEL_ENCODER_TRANSFORMER) -> LabelEncoder:
    return load(label_encoder_path)

def dump_label_encoder(label_encoder:LabelEncoder,label_encoder_path:str = PATH_LABEL_ENCODER_TRANSFORMER) -> None:
    dump(label_encoder,label_encoder_path)
```

Kode Program 5.49 Penyimpanan dan pemuatan LabelEncoder

Fungsi *create_label_encoding* pada Kode Program 5.50 mengimplementasikan Langkah 1 hingga 3. Fungsi ini memuat daftar kelas dari berkas *cache/original-classes.json* hasil EDA (Subbab 5.2), melatih *LabelEncoder* pada daftar tersebut sehingga setiap nama kelas memperoleh kode sesuai urutan alfabetisnya, kemudian menyimpannya. Sel kedua menjalankan pembentukan *LabelEncoder*.

```python
def create_label_encoding()->LabelEncoder:
    encoder = LabelEncoder()
    classes_list = load_classes_list()
    encoder.fit(classes_list)
    dump_label_encoder(encoder)
    return encoder

create_label_encoding()
```

Kode Program 5.50 Pembentukan LabelEncoder dari daftar kelas

Kode Program 5.51 mengimplementasikan Langkah 4. Fungsi *encode_polar_labels* memuat *LabelEncoder*, mengambil daftar kelasnya (*classes_*), dan mengganti setiap nama kelas pada kolom label dengan posisinya pada daftar tersebut menggunakan *replace_strict*, yang menghasilkan kode yang sama dengan *LabelEncoder.transform*. Fungsi *replace_strict* menghentikan proses dengan galat apabila ditemukan nama kelas yang tidak terdapat pada daftar, dan seluruh penggantian dilakukan secara *lazy*. Fungsi *encode_train_and_test_labels* membaca berkas *labels.parquet* pada folder data latih dan data uji secara *lazy*, menerapkan *encode_polar_labels*, dan mengalirkan hasilnya ke berkas *encoded-labels.parquet* pada folder yang sama menggunakan *sink_parquet*. Sel terakhir menjalankan pengodean.

```python
def encode_polar_labels(dataset: pl.LazyFrame, column: str = LABEL_COLUMN) -> pl.LazyFrame:
    classes = load_label_encoder().classes_.tolist()
    codes = list(range(len(classes)))
    return dataset.with_columns(pl.col(column).replace_strict(classes, codes, return_dtype=pl.Int64))

def encode_train_and_test_labels(output_name:str=ENCODED_LABELS_FILE_NAME):
    for folder in (PATH_FOLDER_SPLITTED_DATASET_TRAIN, PATH_FOLDER_SPLITTED_DATASET_TEST):
        print_status(f"Encoding labels in {folder}")
        labels = pl.scan_parquet(os.path.join(folder, LABELS_FILE_NAME))
        encode_polar_labels(labels).sink_parquet(os.path.join(folder, output_name))
    end_status("Encoded labels in the train and test splits")

encode_train_and_test_labels()
```

Kode Program 5.51 Pengodean label data latih dan data uji
