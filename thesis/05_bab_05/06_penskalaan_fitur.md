## **5.6 Implementasi Penskalaan Fitur (*Feature Scaling*)**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.6, yaitu penskalaan fitur dengan standardisasi. Kode program disusun mengikuti Langkah 1 hingga 6 pada algoritma di Subbab 4.6, dilanjutkan dengan kode untuk memverifikasi rata-rata dan simpangan baku fitur sebelum dan sesudah penskalaan.

Kode Program 5.42 mendefinisikan lokasi berkas statistik penskalaan (*PATH_SCALER*) dan folder tujuan *data-scaled*.

```python
PATH_SCALER = os.path.join(PATH_CACHE, "standard-scaler.json")

PATH_FOLDER_SCALED = "data-scaled"
```

Kode Program 5.42 Pengaturan tahap penskalaan

Fungsi *fit_scaler* pada Kode Program 5.43 mengimplementasikan Langkah 1 dan 2. Fungsi ini mendaftar berkas data latih, membentuk *StandardScaler*, dan memperbarui statistiknya berkas demi berkas menggunakan *partial_fit*, dengan melewati berkas kosong dan menghentikan proses dengan galat apabila tidak ada satu baris pun yang dapat digunakan. Jumlah baris dan berkas yang digunakan dicetak di akhir.

```python
def fit_scaler(source_folder_name: str, split_name: str = FIT_SPLIT) -> StandardScaler:
    split_folder = get_split_folder(source_folder_name, split_name)
    file_names = get_all_file_names_in_folder(split_folder, "parquet")

    scaler = StandardScaler()
    rows_seen = 0
    for file_name in file_names:
        df = read_parquet_file(split_folder, file_name)
        if df.empty:
            continue
        scaler.partial_fit(df)
        rows_seen += len(df)

    if rows_seen == 0:
        raise ValueError(f"No rows to fit a scaler on in {split_folder}/.")

    print(f"Fitted on {rows_seen:,} rows from {len(file_names)} files in {split_folder}/.")
    return scaler
```

Kode Program 5.43 Perhitungan rata-rata dan simpangan baku secara inkremental

Fungsi *save_scaler_statistics* pada Kode Program 5.44 menyimpan rata-rata dan pembagi (*scale*) setiap fitur ke berkas JSON dalam dua kamus berkunci nama fitur, setelah memastikan bahwa *scaler* telah dilatih (Langkah 3). Fungsi *load_scaler_statistics* memuatnya kembali. Statistik ini juga digunakan pada simulasi gangguan (Subbab 5.15).

```python
def save_scaler_statistics(scaler: StandardScaler, output_file: str) -> str:
    if scaler.feature_names_in_ is None or scaler.mean_ is None or scaler.scale_ is None:
        raise ValueError("Scaler must be fitted before saving its statistics.")

    column_names = [str(name) for name in scaler.feature_names_in_]
    statistics = {
        "mean": {name: float(mean) for name, mean in zip(column_names, scaler.mean_)},
        "scale": {name: float(scale) for name, scale in zip(column_names, scaler.scale_)},
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as file:
        json.dump(statistics, file, indent=4)
    return output_file

def load_scaler_statistics(input_file: str) -> dict[str, dict[str, float]]:
    with open(input_file) as file:
        return json.load(file)
```

Kode Program 5.44 Penyimpanan dan pemuatan statistik penskalaan

Fungsi *create_and_save_scaler* pada Kode Program 5.45 merangkai pembelajaran statistik dan penyimpanannya, kemudian mencetak jumlah statistik yang tersimpan. Sel kedua menjalankannya pada folder *data-imputed* dan menampilkan tabel rata-rata dan pembagi setiap fitur.

```python
def create_and_save_scaler(
    source_folder_name: str, output_file: str, split_name: str = FIT_SPLIT
) -> StandardScaler:
    scaler = fit_scaler(source_folder_name, split_name)
    save_scaler_statistics(scaler, output_file)
    if scaler.mean_ is None:
        raise RuntimeError("Scaler was not fitted correctly.")
    print(f"Saved {len(scaler.mean_)} means and divisors to {output_file}.")
    return scaler

scaler = create_and_save_scaler(PATH_FOLDER_IMPUTED, PATH_SCALER)
pd.DataFrame(load_scaler_statistics(PATH_SCALER))
```

Kode Program 5.45 Pembentukan dan penyimpanan statistik penskalaan dari data latih

Kode Program 5.46 mengimplementasikan Langkah 4 hingga 6 untuk satu berkas. Fungsi *scale_features* memeriksa bahwa himpunan kolom berkas sama dengan kolom yang dipelajari oleh *scaler*, dan menghentikan proses dengan galat apabila berbeda. Selanjutnya fungsi ini menerapkan Persamaan 4.1 melalui operasi vektor pada *pandas* dan mengonversi hasilnya menjadi *float32*. Fungsi *scale_file* membaca berkas, memanggil fungsi tersebut, dan menulis hasilnya ke folder tujuan.

```python
def scale_features(df: pd.DataFrame, statistics: dict[str, dict[str, float]]) -> pd.DataFrame:
    means = pd.Series(statistics["mean"])
    scales = pd.Series(statistics["scale"])

    if set(df.columns) != set(means.index):
        raise ValueError("this file's columns are not the columns the scaler was fitted on")

    return ((df - means) / scales).astype(FEATURE_DTYPE)

def scale_file(
    source_folder_name: str,
    target_folder_name: str,
    file_name: str,
    statistics: dict[str, dict[str, float]],
) -> str:
    df = read_parquet_file(source_folder_name, file_name)
    scaled_features = scale_features(df, statistics)
    return write_dataframe_to_parquet(scaled_features, target_folder_name, file_name)
```

Kode Program 5.46 Standardisasi fitur pada satu berkas

Fungsi *scale_all_splits* pada Kode Program 5.47 memuat statistik dari berkas JSON dan menerapkannya pada himpunan yang ditentukan melalui *process_all_splits*, yang bawaannya adalah ketiga himpunan.

```python
def scale_all_splits(
    source_folder_name: str,
    target_folder_name: str,
    statistics_file: str = PATH_SCALER,
    split_names: tuple[str, ...] = SPLIT_NAMES,
) -> None:
    statistics = load_scaler_statistics(statistics_file)
    process_all_splits(
        source_folder_name, target_folder_name, split_names, scale_file, statistics
    )
```

Kode Program 5.47 Standardisasi fitur pada seluruh himpunan

Kode Program 5.48 menjalankan penskalaan dari folder *data-imputed* ke folder *data-scaled*.

```python
scale_all_splits(PATH_FOLDER_IMPUTED, PATH_FOLDER_SCALED)
```

Kode Program 5.48 Menjalankan penskalaan

Kode Program 5.49 mengimplementasikan verifikasi hasil penskalaan. Fungsi *measure_feature_statistics* menghitung rata-rata dan simpangan baku setiap kolom dari seluruh berkas satu himpunan menggunakan *Polars* dalam mode *streaming*, dan digunakan kembali pada verifikasi proyeksi (Subbab 5.7). Fungsi *summarize_scaling* meringkasnya pada setiap himpunan menjadi nilai mutlak rata-rata terbesar serta simpangan baku terkecil dan terbesar beserta nama fiturnya. Dua pemanggilan terakhir memeriksa folder *data-imputed* (sebelum penskalaan) dan folder *data-scaled* (setelah penskalaan).

```python
def measure_feature_statistics(folder_name: str, split_name: str) -> pd.DataFrame:
    file_paths = get_split_file_paths(folder_name, split_name)

    frame = pl.scan_parquet(file_paths)
    column_names = frame.collect_schema().names()
    measured = frame.select(
        pl.col(column_names).mean().name.prefix("mean_"),
        pl.col(column_names).std().name.prefix("std_"),
    ).collect(engine="streaming").row(0, named=True)

    return pd.DataFrame(
        {
            "mean": {name: measured[f"mean_{name}"] for name in column_names},
            "std": {name: measured[f"std_{name}"] for name in column_names},
        }
    )

def summarize_scaling(
    folder_name: str, split_names: tuple[str, ...] = SPLIT_NAMES
) -> pd.DataFrame:
    summary = {}
    for split_name in split_names:
        statistics = measure_feature_statistics(folder_name, split_name)
        summary[split_name] = {
            "columns": len(statistics),
            "largest |mean|": statistics["mean"].abs().max(),
            "at column": statistics["mean"].abs().idxmax(),
            "smallest std": statistics["std"].min(),
            "largest std": statistics["std"].max(),
            "at column ": statistics["std"].idxmax(),
        }

    print(f"Measured {folder_name}/.")
    return pd.DataFrame(summary).T

summarize_scaling(PATH_FOLDER_IMPUTED)

summarize_scaling(PATH_FOLDER_SCALED)
```

Kode Program 5.49 Verifikasi rata-rata dan simpangan baku sebelum dan sesudah penskalaan
