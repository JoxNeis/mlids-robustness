## **5.7 Implementasi Reduksi Dimensi dengan *Incremental Principal Component Analysis* (IPCA)**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.7, yaitu reduksi dimensi dengan *Incremental PCA*. Kode program disusun mengikuti Langkah 1 hingga 7 pada algoritma di Subbab 4.7, dilanjutkan dengan kode untuk memverifikasi hasil proyeksi.

Kode Program 5.50 mendefinisikan lokasi berkas proyeksi (*PATH_IPCA*), folder tujuan *data-ipca*, jumlah komponen maksimum 40 (*MAX_COMPONENTS*), dan ambang varians kumulatif 0,95 (*VARIANCE_TARGET*).

```python
PATH_IPCA = os.path.join(PATH_CACHE, "ipca.npz")

PATH_FOLDER_IPCA = "data-ipca"

MAX_COMPONENTS = 40

VARIANCE_TARGET = 0.95
```

Kode Program 5.50 Pengaturan tahap reduksi dimensi

Fungsi *fit_ipca* pada Kode Program 5.51 mengimplementasikan Langkah 1 dan 2. Fungsi ini mendaftar berkas data latih terskala, membentuk *IncrementalPCA*, dan memperbarui komponen berkas demi berkas dengan *partial_fit*. Berkas dengan jumlah baris kurang dari jumlah komponen dilewati, dan proses dihentikan dengan galat apabila tidak ada berkas yang memenuhi syarat.

```python
def fit_ipca(
    source_folder_name: str,
    split_name: str = FIT_SPLIT,
    n_components: int = MAX_COMPONENTS,
) -> IncrementalPCA:
    split_folder = get_split_folder(source_folder_name, split_name)
    file_names = get_all_file_names_in_folder(split_folder, "parquet")

    ipca = IncrementalPCA(n_components=n_components)
    rows_seen = 0
    for file_name in file_names:
        df = read_parquet_file(split_folder, file_name)
        if len(df) < n_components:
            continue
        ipca.partial_fit(df)
        rows_seen += len(df)

    if rows_seen == 0:
        raise ValueError(
            f"No file in {split_folder}/ has the {n_components} rows a fit of this size needs."
        )

    print(f"Fitted {n_components} components on {rows_seen:,} rows from {split_folder}/.")
    return ipca
```

Kode Program 5.51 Pelatihan IPCA secara inkremental

Fungsi *save_ipca* pada Kode Program 5.52 menyimpan komponen, rata-rata, varians terjelaskan, rasio varians terjelaskan, dan urutan nama fitur ke berkas NumPy (*.npz*) menggunakan *numpy.savez* (Langkah 3). Fungsi *load_ipca* memuatnya kembali sebagai kamus larik.

```python
def save_ipca(ipca: IncrementalPCA, output_file: str) -> str:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez(
        output_file,
        components=ipca.components_,
        mean=ipca.mean_,
        explained_variance=ipca.explained_variance_,
        explained_variance_ratio=ipca.explained_variance_ratio_,
        columns=np.array([str(name) for name in ipca.feature_names_in_]),
    )
    return output_file

def load_ipca(input_file: str) -> dict[str, np.ndarray]:
    with np.load(input_file) as stored:
        return {name: stored[name] for name in stored.files}
```

Kode Program 5.52 Penyimpanan dan pemuatan proyeksi

Fungsi *create_and_save_ipca* pada Kode Program 5.53 merangkai pelatihan IPCA dan penyimpanan proyeksinya, kemudian mencetak jumlah komponen dan fitur yang tersimpan. Sel kedua menjalankannya pada folder *data-scaled* dengan 40 komponen.

```python
def create_and_save_ipca(
    source_folder_name: str,
    output_file: str,
    split_name: str = FIT_SPLIT,
    n_components: int = MAX_COMPONENTS,
) -> IncrementalPCA:
    ipca = fit_ipca(source_folder_name, split_name, n_components)
    save_ipca(ipca, output_file)

    print(
        f"Saved {n_components} components of {ipca.n_features_in_} features to {output_file}."
    )
    return ipca

ipca = create_and_save_ipca(PATH_FOLDER_SCALED, PATH_IPCA)
```

Kode Program 5.53 Pembentukan dan penyimpanan proyeksi dari data latih

Kode Program 5.54 mengimplementasikan Langkah 4. Fungsi *get_variance_curve* menjumlahkan rasio varians terjelaskan secara kumulatif. Fungsi *choose_component_count* mengembalikan jumlah komponen terkecil yang varians kumulatifnya mencapai ambang, sesuai Persamaan 4.2, atau menghentikan proses dengan galat apabila ambang tidak tercapai.

```python
def get_variance_curve(input_file: str = PATH_IPCA) -> pd.Series:
    explained_variance_ratio = load_ipca(input_file)["explained_variance_ratio"]

    return pd.Series(
        np.cumsum(explained_variance_ratio),
        index=range(1, len(explained_variance_ratio) + 1),
        name="cumulative variance",
    )

def choose_component_count(
    variance_curve: pd.Series, variance_target: float = VARIANCE_TARGET
) -> int:
    reaching_target = variance_curve[variance_curve >= variance_target]
    if reaching_target.empty:
        raise ValueError(
            f"{len(variance_curve)} components reach only {variance_curve.iloc[-1]:.1%}; "
            f"raise MAX_COMPONENTS or lower the {variance_target:.0%} target."
        )
    return int(reaching_target.index[0])
```

Kode Program 5.54 Kurva varians kumulatif dan pemilihan jumlah komponen

Fungsi *plot_variance_curve* pada Kode Program 5.55 menggambar kurva varians kumulatif terhadap jumlah komponen menggunakan *Matplotlib*, dilengkapi garis ambang serta titik jumlah komponen terpilih beserta anotasinya, untuk pemeriksaan visual.

```python
def plot_variance_curve(
    variance_curve: pd.Series, variance_target: float = VARIANCE_TARGET
) -> None:
    surface, grid, ink, muted, series = "#fcfcfb", "#e8e7e3", "#0b0b0b", "#52514e", "#2a78d6"
    chosen = choose_component_count(variance_curve, variance_target)

    figure, axes = plt.subplots(figsize=(8, 4.5))
    figure.patch.set_facecolor(surface)
    axes.set_facecolor(surface)

    axes.plot(variance_curve.index, variance_curve * 100, color=series, linewidth=2)
    axes.axhline(variance_target * 100, color=muted, linewidth=1, linestyle=(0, (4, 3)))
    axes.text(1, variance_target * 100 + 2, f"{variance_target:.0%} target", color=muted, fontsize=9)

    axes.plot(
        [chosen], [variance_curve[chosen] * 100], marker="o", markersize=8,
        color=series, markeredgecolor=surface, markeredgewidth=2, zorder=3,
    )
    axes.annotate(
        f"{chosen} components → {variance_curve[chosen]:.1%}",
        (chosen, variance_curve[chosen] * 100),
        xytext=(10, -18), textcoords="offset points", color=ink, fontsize=9,
    )

    axes.set_title(
        "Variance retained by the first n principal components", color=ink, fontsize=11, loc="left", pad=12
    )
    axes.set_xlabel("Principal components", color=muted, fontsize=9)
    axes.set_ylabel("Cumulative explained variance (%)", color=muted, fontsize=9)
    axes.set_xlim(0, variance_curve.index[-1])
    axes.set_ylim(0, 100)
    axes.set_xticks(range(0, variance_curve.index[-1] + 1, 5))
    axes.grid(axis="y", color=grid, linewidth=1)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(grid)
    axes.tick_params(colors=muted, labelsize=9, length=0)

    figure.tight_layout()
    plt.show()
```

Kode Program 5.55 Visualisasi kurva varians kumulatif

Kode Program 5.56 membaca kurva varians kumulatif dari berkas proyeksi dan menggambarkannya. Selanjutnya *IPCA_COMPONENTS* ditetapkan, yaitu jumlah komponen yang digunakan pada seluruh tahap berikutnya, dan jumlah tersebut dicetak bersama varians kumulatif yang dicapainya.

```python
variance_curve = get_variance_curve(PATH_IPCA)
plot_variance_curve(variance_curve)

IPCA_COMPONENTS = choose_component_count(variance_curve)
print(
    f"{IPCA_COMPONENTS} of {len(variance_curve)} components keep "
    f"{variance_curve[IPCA_COMPONENTS]:.2%} of the training variance."
)
```

Kode Program 5.56 Penetapan jumlah komponen yang digunakan

Kode Program 5.57 mengimplementasikan Langkah 5 hingga 7 untuk satu berkas. Fungsi *project_features* memeriksa bahwa kolom berkas sama dengan kolom yang dipelajari IPCA, baik nama maupun urutannya, dan menghentikan proses dengan galat apabila berbeda. Selanjutnya fungsi ini menerapkan Persamaan 4.3 menggunakan *N* komponen pertama melalui perkalian matriks, dan menamai kolom hasil *pc1* hingga *pcN* dengan tipe *float32*. Fungsi *project_file* membaca berkas, memanggil fungsi tersebut, dan menulis hasilnya ke folder tujuan.

```python
def project_features(
    df: pd.DataFrame, projection: dict[str, np.ndarray], n_components: int
) -> pd.DataFrame:
    if list(df.columns) != list(projection["columns"]):
        raise ValueError(
            "this file's columns are not, in order, the columns the projection was fitted on"
        )

    axes = projection["components"][:n_components]
    projected = (df.to_numpy() - projection["mean"]) @ axes.T

    column_names = [f"pc{number}" for number in range(1, n_components + 1)]
    return pd.DataFrame(projected, columns=column_names).astype(FEATURE_DTYPE)

def project_file(
    source_folder_name: str,
    target_folder_name: str,
    file_name: str,
    projection: dict[str, np.ndarray],
    n_components: int,
) -> str:
    df = read_parquet_file(source_folder_name, file_name)
    projected_features = project_features(df, projection, n_components)
    return write_dataframe_to_parquet(projected_features, target_folder_name, file_name)
```

Kode Program 5.57 Proyeksi fitur pada satu berkas

Fungsi *project_all_splits* pada Kode Program 5.58 memuat proyeksi dari berkas NumPy dan menerapkannya pada himpunan yang ditentukan melalui *process_all_splits*, yang bawaannya adalah ketiga himpunan.

```python
def project_all_splits(
    source_folder_name: str,
    target_folder_name: str,
    n_components: int,
    projection_file: str = PATH_IPCA,
    split_names: tuple[str, ...] = SPLIT_NAMES,
) -> None:
    projection = load_ipca(projection_file)
    process_all_splits(
        source_folder_name, target_folder_name, split_names, project_file, projection, n_components
    )
```

Kode Program 5.58 Proyeksi fitur pada seluruh himpunan

Kode Program 5.59 menjalankan proyeksi dari folder *data-scaled* ke folder *data-ipca* dengan *IPCA_COMPONENTS* komponen.

```python
project_all_splits(PATH_FOLDER_SCALED, PATH_FOLDER_IPCA, IPCA_COMPONENTS)
```

Kode Program 5.59 Menjalankan proyeksi

Fungsi *summarize_projection* pada Kode Program 5.60 mengukur varians dan rata-rata setiap komponen pada setiap himpunan menggunakan *measure_feature_statistics* (Kode Program 5.49), kemudian membandingkan total varians yang terukur dengan varians yang diprediksi oleh pelatihan IPCA pada *N* komponen pertama, beserta nilai mutlak rata-rata terbesar. Sel kedua menjalankannya pada folder *data-ipca*.

```python
def summarize_projection(
    folder_name: str,
    projection_file: str = PATH_IPCA,
    split_names: tuple[str, ...] = SPLIT_NAMES,
) -> pd.DataFrame:
    fitted_variance = load_ipca(projection_file)["explained_variance"]

    summary = {}
    for split_name in split_names:
        measured = measure_feature_statistics(folder_name, split_name)
        summary[split_name] = {
            "components": len(measured),
            "variance carried": (measured["std"] ** 2).sum(),
            "variance predicted": fitted_variance[: len(measured)].sum(),
            "largest |mean|": measured["mean"].abs().max(),
        }

    return pd.DataFrame(summary).T

summarize_projection(PATH_FOLDER_IPCA)
```

Kode Program 5.60 Verifikasi varians dan rata-rata komponen hasil proyeksi
