## **5.8 Implementasi Penyeimbangan Kelas dengan *Synthetic Minority Oversampling Technique* (SMOTE)**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.8, yaitu penyeimbangan kelas pada data latih dengan SMOTE. Kode program disusun mengikuti Langkah 1 hingga 6 pada algoritma di Subbab 4.8, dilanjutkan dengan kode untuk memverifikasi perubahan jumlah baris pada setiap kelas.

Kode Program 5.61 mendefinisikan folder tujuan *data-smote* dan jumlah tetangga SMOTE sebesar 5 (*SMOTE_K_NEIGHBORS*).

```python
PATH_FOLDER_SMOTE = "data-smote"

SMOTE_K_NEIGHBORS = 5
```

Kode Program 5.61 Pengaturan tahap penyeimbangan kelas

Kode Program 5.62 mengimplementasikan Langkah 3 hingga 5. Fungsi *choose_k_neighbors* menurunkan jumlah tetangga menjadi satu kurang dari jumlah sampel kelas terkecil pada berkas, dengan batas bawah 1 (Langkah 4). Fungsi *resample_rows* meneruskan fitur dan label tanpa perubahan apabila berkas hanya memuat satu kelas (Langkah 3). Selain itu, fungsi ini menjalankan *SMOTE* dari pustaka *imbalanced-learn* dengan jumlah tetangga tersebut dan *random state* 42 (Langkah 5), kemudian mengonversi fitur hasilnya menjadi *float32*.

```python
def choose_k_neighbors(labels: pd.Series, k_neighbors: int = SMOTE_K_NEIGHBORS) -> int:
    smallest_class_count = int(labels.value_counts().min())
    return max(1, min(k_neighbors, smallest_class_count - 1))

def resample_rows(
    features: pd.DataFrame,
    labels: pd.Series,
    k_neighbors: int = SMOTE_K_NEIGHBORS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    if labels.nunique() < 2:
        return features, labels

    smote = SMOTE(
        k_neighbors=choose_k_neighbors(labels, k_neighbors), random_state=random_state
    )
    result = smote.fit_resample(features, labels)
    resampled_features = cast(pd.DataFrame, result[0])
    resampled_labels = cast(pd.Series, result[1])
    return (
        cast(pd.DataFrame, resampled_features).astype(FEATURE_DTYPE),
        cast(pd.Series, resampled_labels),
    )
```

Kode Program 5.62 Penyeimbangan kelas pada satu berkas

Fungsi *resample_file* pada Kode Program 5.63 membaca fitur dan label berkas yang bersangkutan (Langkah 2), memanggil *resample_rows*, menggabungkan fitur hasil *resampling* dengan kolom label, dan menulis hasilnya ke satu berkas Parquet (Langkah 6). Label disimpan pada berkas yang sama dengan fitur karena jumlah baris telah berubah.

```python
def resample_file(
    features_folder_name: str,
    labels_folder_name: str,
    target_folder_name: str,
    file_name: str,
    k_neighbors: int = SMOTE_K_NEIGHBORS,
    random_state: int = RANDOM_STATE,
) -> str:
    features = read_parquet_file(features_folder_name, file_name)
    labels = read_parquet_file(labels_folder_name, file_name)[LABEL_COLUMN]

    resampled_features, resampled_labels = resample_rows(
        features, labels, k_neighbors, random_state
    )
    resampled = resampled_features.assign(**{LABEL_COLUMN: resampled_labels.to_numpy()})

    return write_dataframe_to_parquet(resampled, target_folder_name, file_name)
```

Kode Program 5.63 Penyeimbangan kelas dan penulisan satu berkas

Fungsi *resample_training_split* pada Kode Program 5.64 mengimplementasikan Langkah 1. Fungsi ini hanya memproses himpunan *train*, membuat folder tujuan, dan menjalankan *resample_file* untuk setiap berkas secara paralel, kemudian mencetak jumlah berkas yang diproses.

```python
def resample_training_split(
    features_folder_name: str,
    labels_folder_name: str,
    target_folder_name: str,
    split_name: str = FIT_SPLIT,
    k_neighbors: int = SMOTE_K_NEIGHBORS,
    random_state: int = RANDOM_STATE,
) -> None:
    features_folder = get_split_folder(features_folder_name, split_name)
    labels_folder = get_split_folder(labels_folder_name, split_name)
    target_folder = get_split_folder(target_folder_name, split_name)
    create_split_folders(target_folder_name, (split_name,))

    file_names = get_all_file_names_in_folder(features_folder, "parquet")
    written_files = run_tasks_in_parallel(
        delayed(resample_file)(
            features_folder,
            labels_folder,
            target_folder,
            file_name,
            k_neighbors,
            random_state,
        )
        for file_name in file_names
    )

    print(f"Resampled {len(written_files)} {split_name} files into {target_folder}/.")
```

Kode Program 5.64 Penyeimbangan kelas pada seluruh berkas data latih

Kode Program 5.65 menjalankan penyeimbangan kelas dengan fitur dari folder *data-ipca*, label dari folder *data-encoded-label*, dan keluaran ke folder *data-smote*.

```python
resample_training_split(PATH_FOLDER_IPCA, PATH_FOLDER_ENCODED_LABEL, PATH_FOLDER_SMOTE)
```

Kode Program 5.65 Menjalankan penyeimbangan kelas

Fungsi *compare_class_counts* pada Kode Program 5.66 menghitung jumlah baris setiap kelas pada data latih sebelum SMOTE, dari folder *data-encoded-label*, dan sesudahnya, dari folder *data-smote*. Selanjutnya fungsi ini menyusun tabel yang memuat jumlah baris sintetis dan persentase setiap kelas setelah SMOTE. Sel kedua menjalankannya.

```python
def compare_class_counts(
    labels_folder_name: str,
    resampled_folder_name: str,
    encoder_file: str = PATH_LABEL_ENCODER,
    split_name: str = FIT_SPLIT,
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    before = count_labels_in_folder(
        get_split_folder(labels_folder_name, split_name), label_column
    )
    after = count_labels_in_folder(
        get_split_folder(resampled_folder_name, split_name), label_column
    )

    class_names = load_label_classes(encoder_file)
    table = pd.DataFrame({"before": before, "after": after}).fillna(0).astype(int).sort_index()
    table.insert(0, "class", [class_names[class_id] for class_id in table.index])
    table["synthetic"] = table["after"] - table["before"]
    table["share after %"] = (table["after"] / table["after"].sum() * 100).round(1)

    print(
        f"{table['before'].sum():,} rows in, {table['after'].sum():,} rows out, "
        f"{table['synthetic'].sum():,} of them synthetic."
    )
    return table

class_counts = compare_class_counts(PATH_FOLDER_ENCODED_LABEL, PATH_FOLDER_SMOTE)
class_counts
```

Kode Program 5.66 Perbandingan jumlah baris setiap kelas sebelum dan sesudah SMOTE

Fungsi *plot_class_counts* pada Kode Program 5.67 menggambar jumlah baris setiap kelas sebelum dan sesudah SMOTE pada skala logaritmik menggunakan *Matplotlib*. Sel kedua menjalankannya pada tabel perbandingan.

```python
def plot_class_counts(comparison: pd.DataFrame) -> None:
    surface, grid, ink, muted = "#fcfcfb", "#e8e7e3", "#0b0b0b", "#52514e"
    before_colour, after_colour = "#2a78d6", "#eb6834"

    ordered = comparison.sort_values("before")
    positions = np.arange(len(ordered))

    figure, axes = plt.subplots(figsize=(9, 6))
    figure.patch.set_facecolor(surface)
    axes.set_facecolor(surface)

    axes.hlines(positions, ordered["before"], ordered["after"], color=grid, linewidth=2, zorder=1)
    for column, colour, label, layer in (
        ("before", before_colour, "before", 2),
        ("after", after_colour, "after", 3),
    ):
        axes.plot(
            ordered[column], positions, "o", linestyle="none", markersize=8, color=colour,
            markeredgecolor=surface, markeredgewidth=2, label=label, zorder=layer,
        )

    axes.set_xscale("log")
    axes.set_yticks(positions)
    axes.set_yticklabels(ordered["class"], fontsize=9)
    axes.set_xlabel("Training rows (log scale)", color=muted, fontsize=9)
    axes.set_title(
        "Training rows per class, before and after SMOTE", color=ink, fontsize=11, loc="left", pad=12
    )
    axes.grid(axis="x", color=grid, linewidth=1)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(grid)
    axes.tick_params(colors=muted, labelsize=9, length=0)

    legend = axes.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(muted)

    figure.tight_layout()
    plt.show()

plot_class_counts(class_counts)
```

Kode Program 5.67 Visualisasi jumlah baris setiap kelas sebelum dan sesudah SMOTE
