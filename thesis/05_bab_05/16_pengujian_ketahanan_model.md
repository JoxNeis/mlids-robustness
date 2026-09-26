## **5.16 Implementasi Pengujian dan Pengukuran Ketahanan Model**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.16, yaitu pengujian kelima model pada dua belas skenario gangguan beserta pengukuran penurunan kinerjanya. Kode program disusun mengikuti Langkah 1 hingga 8 pada algoritma di Subbab 4.16, dilanjutkan dengan kode untuk menyajikan hasil pengujian. Pengujian menggunakan model terlatih dan fungsi prediksi pada Subbab 5.10 hingga 5.14.

Kode Program 5.125 mendefinisikan folder penyimpanan skor pengujian (*PATH_FOLDER_NOISE_SCORES*) dan nama skenario acuan data bersih (*NOISE_BASELINE*, yaitu *clean*). Fungsi *get_noise_split_name* menentukan subfolder prediksi suatu skenario, yaitu subfolder *test* untuk data bersih dan subfolder *noise* diikuti nama skenario untuk skenario gangguan. Fungsi *get_model_file_path* membentuk lokasi berkas model.

```python
PATH_FOLDER_NOISE_SCORES = os.path.join(PATH_FOLDER_SEARCH_RESULT, "noise-scores")
NOISE_BASELINE = "clean"


def get_noise_split_name(scenario: str) -> str:
    return NOISE_SPLIT if scenario == NOISE_BASELINE else os.path.join("noise", scenario)


def get_model_file_path(model_name: str, model_file: str) -> str:
    return os.path.join(PATH_FOLDER_MODEL, MODEL_SUBFOLDERS[model_name], model_file)
```

Kode Program 5.125 Lokasi skor pengujian dan penamaan folder prediksi skenario

Kode Program 5.126 mengimplementasikan bagian himpunan kelas pada Langkah 1. Fungsi *get_noise_labels* menetapkan satu kali himpunan kelas yang muncul pada label data uji dan menyimpannya pada *cache*, sehingga seluruh pengujian dinilai terhadap himpunan kelas yang sama. Kalimat yang dicetak di akhir menyatakan jumlah kelas tersebut.

```python
_NOISE_LABEL_CACHE: list[int] = []


def get_noise_labels(true_labels=None) -> list[int]:
    if not _NOISE_LABEL_CACHE:
        true = to_labels(test_y if true_labels is None else true_labels)
        _NOISE_LABEL_CACHE.extend(sorted({int(label) for label in np.unique(true)}))
    return _NOISE_LABEL_CACHE


print(f"every pass scored over the same {len(get_noise_labels())} classes")
```

Kode Program 5.126 Penetapan himpunan kelas evaluasi

Kode Program 5.127 mengimplementasikan Langkah 5 dan 6. Fungsi *get_noise_score_path*, *save_noise_score*, dan *load_noise_score* membentuk lokasi berkas skor untuk suatu pasangan model dan skenario, menyimpan skor sebagai berkas JSON melalui berkas sementara, dan memuatnya kembali. Fungsi *score_noise_pass* memuat prediksi tersimpan apabila tidak diberikan, memeriksa bahwa jumlahnya sama dengan jumlah label, menghitung metrik terhadap himpunan kelas yang sama, dan menyimpan catatan yang memuat model, jenis gangguan, intensitas, metrik, jumlah baris, dan waktu penilaian.

```python
def get_noise_score_path(
    model_name: str, scenario: str, folder_name: str = PATH_FOLDER_NOISE_SCORES
) -> str:
    return os.path.join(folder_name, f"{model_name.replace(' ', '-')}__{scenario}.json")


def save_noise_score(record: dict, folder_name: str = PATH_FOLDER_NOISE_SCORES) -> str:
    os.makedirs(folder_name, exist_ok=True)
    file_path = get_noise_score_path(record["model"], record["scenario"], folder_name)
    temporary_path = file_path + ".tmp"
    with open(temporary_path, "w") as file:
        json.dump(record, file, indent=2)
    os.replace(temporary_path, file_path)
    return file_path


def load_noise_score(
    model_name: str, scenario: str, folder_name: str = PATH_FOLDER_NOISE_SCORES
) -> dict | None:
    file_path = get_noise_score_path(model_name, scenario, folder_name)
    if not os.path.exists(file_path):
        return None
    with open(file_path) as file:
        return json.load(file)

def score_noise_pass(
    model_name: str,
    scenario: str,
    noise_type: str,
    intensity: float,
    pred: np.ndarray | None = None,
    true_labels=None,
    labels: list[int] | None = None,
    folder_name: str = PATH_FOLDER_NOISE_SCORES,
) -> dict:
    true = to_labels(test_y if true_labels is None else true_labels)
    labels = get_noise_labels() if labels is None else labels

    if pred is None:
        pred = load_prediction_chunks(
            get_prediction_folder(MODEL_SUBFOLDERS[model_name], get_noise_split_name(scenario))
        )

    if len(pred) != len(true):
        raise ValueError(
            f"{model_name} / {scenario}: {len(pred):,} predictions against {len(true):,} labels"
        )

    scores = evaluate(pred=pred, true=true, labels=labels).iloc[0]
    record = {
        "model": model_name,
        "noise": noise_type,
        "intensity": intensity,
        "scenario": scenario,
        **{name: float(value) for name, value in scores.items()},
        "rows": int(len(pred)),
        "scored_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"  stored {save_noise_score(record, folder_name)}")
    return record
```

Kode Program 5.127 Penyimpanan dan penghitungan skor satu pengujian

Kode Program 5.128 mengimplementasikan Langkah 2. Fungsi *load_noise_split* memuat data uji suatu skenario dari folder *data-noise-ipca* dan menyimpannya di memori, sehingga dapat digunakan bergantian oleh kelima model tanpa dimuat ulang. Fungsi *release_noise_split* melepasnya dan membebaskan memori sebelum skenario berikutnya dimuat.

```python
_NOISE_SPLIT_CACHE: dict[str, Any] = {}


def load_noise_split(
    scenario: str,
    features_folder_name: str = PATH_FOLDER_NOISE_IPCA,
    split_name: str = NOISE_SPLIT,
) -> pl.DataFrame:
    key = os.path.join(scenario, split_name)
    if _NOISE_SPLIT_CACHE.get("key") == key:
        print(f"  using the '{key}' split already in memory")
        return cast(pl.DataFrame, _NOISE_SPLIT_CACHE["frame"])

    release_noise_split()
    with step(f"loading '{key}'"):
        frame = load_split_frame(get_scenario_folder(features_folder_name, scenario), split_name)

    print(f"  {frame.height:,} rows x {frame.width} components")
    _NOISE_SPLIT_CACHE["key"] = key
    _NOISE_SPLIT_CACHE["frame"] = frame
    return frame


def release_noise_split() -> None:
    key = _NOISE_SPLIT_CACHE.pop("key", None)
    if _NOISE_SPLIT_CACHE.pop("frame", None) is not None:
        print(f"Released the '{key}' split")
    gc.collect()
    free_gpu_memory()
```

Kode Program 5.128 Pemuatan data uji skenario dengan penyimpanan sementara di memori

Kode Program 5.129 mengimplementasikan Langkah 3 hingga 5. Fungsi *get_noise_predictors* memetakan nama model ke fungsi prediksi dan nama berkas modelnya. Fungsi *predict_noise_pass* memeriksa bahwa model tersedia dan melewatinya apabila tidak, memuat data uji skenario, memprediksinya per potongan dengan fungsi prediksi model, dan menghitung skornya.

```python
def get_noise_predictors() -> dict[str, tuple]:
    functions = {
        "KNN": (knn_predict, "knn"),
        "SVM": (svm_predict, "svm"),
        "Random Forest": (rf_predict, "rf"),
        "Softmax": (logreg_predict, "logreg"),
        "XGBoost": (xgb_predict, "xgb"),
    }
    return {
        name: (predict, globals().get(f"{prefix}_model_name"))
        for name, (predict, prefix) in functions.items()
    }

def predict_noise_pass(
    model_name: str,
    noise_type: str,
    intensity: float,
    score: bool = True,
    features_folder_name: str = PATH_FOLDER_NOISE_IPCA,
    split_name: str = NOISE_SPLIT,
) -> dict | None:
    predictors = get_noise_predictors()
    if model_name not in predictors:
        raise ValueError(f"unknown model {model_name!r}; expected one of {tuple(predictors)}")

    scenario = get_noise_scenario_name(noise_type, intensity)
    predict, model_file = predictors[model_name]

    if model_file is None:
        print(f"{model_name} / {scenario}: no trained model to load, skipped.")
        return None

    model_path = get_model_file_path(model_name, model_file)
    if not os.path.exists(model_path):
        print(f"{model_name} / {scenario}: nothing saved at {model_path}, skipped.")
        return None

    print(f"=== {model_name} / {scenario} ({noise_type} at {intensity:.0%}) ===")
    x = load_noise_split(scenario, features_folder_name, split_name)
    pred = predict(model_file, x, get_noise_split_name(scenario))

    if not score:
        return None
    return score_noise_pass(model_name, scenario, noise_type, intensity, pred=pred)
```

Kode Program 5.129 Pengujian satu model pada satu skenario

Fungsi *predict_noise_scenario* pada Kode Program 5.130 menjalankan *predict_noise_pass* untuk setiap model pada satu skenario. Fungsi *predict_all_noise_scenarios* menjalankannya untuk seluruh skenario, melepas data uji skenario di akhir, dan menggabungkan hasilnya menjadi satu tabel.

```python
def predict_noise_scenario(
    noise_type: str, intensity: float, models: list[str] | None = None, **keywords
) -> pd.DataFrame:
    models = list(get_noise_predictors()) if models is None else models
    records = [
        predict_noise_pass(model_name, noise_type, intensity, **keywords)
        for model_name in models
    ]
    return pd.DataFrame([record for record in records if record is not None])


def predict_all_noise_scenarios(
    scenarios: list[tuple[str, float, str]] | None = None,
    models: list[str] | None = None,
    **keywords,
) -> pd.DataFrame:
    scenarios = list_noise_scenarios() if scenarios is None else scenarios

    tables = [
        predict_noise_scenario(noise_type, intensity, models, **keywords)
        for noise_type, intensity, _ in scenarios
    ]
    release_noise_split()
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
```

Kode Program 5.130 Pengujian seluruh model pada satu skenario dan pada seluruh skenario

Kode Program 5.131 menyediakan tabel status pengujian. Fungsi *count_scenario_rows* menghitung jumlah baris data uji suatu skenario. Fungsi *noise_prediction_status* menyusun, untuk setiap pasangan model dan skenario, jumlah potongan prediksi yang telah tersimpan terhadap jumlah yang diperlukan, penanda apakah prediksi telah lengkap, dan penanda apakah skor telah tersimpan, sehingga pengujian yang belum selesai dapat diidentifikasi dan dilanjutkan. Sel kedua menjalankannya.

```python
def count_scenario_rows(
    scenario: str,
    features_folder_name: str = PATH_FOLDER_NOISE_IPCA,
    split_name: str = NOISE_SPLIT,
) -> int:
    file_paths = get_split_file_paths(get_scenario_folder(features_folder_name, scenario), split_name)
    return int(pl.scan_parquet(file_paths).select(pl.len()).collect().item())


def noise_prediction_status(
    scenarios: list[tuple[str, float, str]] | None = None,
    models: list[str] | None = None,
    chunk_size: int = PREDICT_CHUNK_SIZE,
    features_folder_name: str = PATH_FOLDER_NOISE_IPCA,
    split_name: str = NOISE_SPLIT,
    scores_folder_name: str = PATH_FOLDER_NOISE_SCORES,
    only_unfinished: bool = False,
) -> pd.DataFrame:
    scenarios = list_noise_scenarios() if scenarios is None else scenarios
    models = list(get_noise_predictors()) if models is None else models

    rows = []
    for _, _, scenario in scenarios:
        expected = math.ceil(
            count_scenario_rows(scenario, features_folder_name, split_name) / chunk_size
        )
        for model_name in models:
            folder = get_prediction_folder(
                MODEL_SUBFOLDERS[model_name], get_noise_split_name(scenario)
            )
            written = (
                len(get_all_file_names_in_folder(folder, "npy")) if os.path.isdir(folder) else 0
            )
            rows.append(
                {
                    "model": model_name,
                    "scenario": scenario,
                    "chunks": written,
                    "of": expected,
                    "percent": round(100 * written / expected, 1) if expected else 0.0,
                    "predicted": written >= expected,
                    "scored": os.path.exists(
                        get_noise_score_path(model_name, scenario, scores_folder_name)
                    ),
                }
            )

    status = pd.DataFrame(rows)
    print(
        f"{int(status['predicted'].sum())}/{len(status)} passes predicted,"
        f" {int(status['scored'].sum())} scored"
    )
    return status[~status["predicted"]].reset_index(drop=True) if only_unfinished else status

noise_prediction_status()
```

Kode Program 5.131 Pemantauan kemajuan pengujian

Kode Program 5.132 menjalankan pengujian untuk setiap jenis gangguan pada intensitas 10%, 20%, dan 30% (Langkah 2 hingga 6). Setelah seluruh skenario selesai, data uji terakhir dilepaskan dari memori dan pengujian yang belum selesai ditampilkan.

```python
predict_noise_scenario("gaussian", 0.10)

predict_noise_scenario("gaussian", 0.20)

predict_noise_scenario("gaussian", 0.30)

predict_noise_scenario("uniform", 0.10)

predict_noise_scenario("uniform", 0.20)

predict_noise_scenario("uniform", 0.30)

predict_noise_scenario("multiplicative", 0.10)

predict_noise_scenario("multiplicative", 0.20)

predict_noise_scenario("multiplicative", 0.30)

predict_noise_scenario("missing", 0.10)

predict_noise_scenario("missing", 0.20)

predict_noise_scenario("missing", 0.30)

release_noise_split()
noise_prediction_status(only_unfinished=True)
```

Kode Program 5.132 Menjalankan pengujian pada seluruh skenario gangguan

Kode Program 5.133 mengimplementasikan bagian pengumpulan skor pada Langkah 7. Fungsi *collect_noise_results* mengumpulkan skor seluruh model pada data bersih dan seluruh skenario, mengutamakan skor yang telah tersimpan dan menghitungnya dari prediksi tersimpan apabila skor belum ada. Fungsi *pivot_noise_scores* menyusun tabel F1-*score* dengan model pada baris dan skenario pada kolom. Sel terakhir menyimpan tabel hasil ke *hyperparameter-search/noise-robustness.csv*.

```python
def collect_noise_results(
    scenarios: list[tuple[str, float, str]] | None = None,
    models: list[str] | None = None,
    include_baseline: bool = True,
    true_labels=None,
    scores_folder_name: str = PATH_FOLDER_NOISE_SCORES,
    prefer_stored: bool = True,
) -> pd.DataFrame:
    scenarios = list_noise_scenarios() if scenarios is None else scenarios
    models = list(MODEL_SUBFOLDERS) if models is None else models
    true = to_labels(test_y if true_labels is None else true_labels)

    if include_baseline:
        scenarios = [(NOISE_BASELINE, 0.0, NOISE_BASELINE)] + list(scenarios)

    rows = []
    for noise_type, intensity, scenario in scenarios:
        for model_name in models:
            if prefer_stored:
                stored = load_noise_score(model_name, scenario, scores_folder_name)
                if stored is not None:
                    rows.append(stored)
                    continue

            folder = get_prediction_folder(
                MODEL_SUBFOLDERS[model_name], get_noise_split_name(scenario)
            )
            try:
                pred = load_prediction_chunks(folder)
            except FileNotFoundError:
                print(f"{model_name} / {scenario}: no cached predictions, skipped.")
                continue

            if len(pred) != len(true):
                print(
                    f"{model_name} / {scenario}: part-way through"
                    f" ({len(pred):,} of {len(true):,} rows), skipped."
                )
                continue

            print(f"scoring {model_name} / {scenario}")
            rows.append(
                score_noise_pass(
                    model_name, scenario, noise_type, intensity,
                    pred=pred, true_labels=true, folder_name=scores_folder_name,
                )
            )

    return pd.DataFrame(rows)

noise_comparison = collect_noise_results()
noise_comparison

def pivot_noise_scores(
    comparison: pd.DataFrame, score_column: str = SCORE_COLUMN, include_baseline: bool = True
) -> pd.DataFrame:
    order = [scenario for _, _, scenario in list_noise_scenarios()]
    if include_baseline:
        order = [NOISE_BASELINE] + order

    table = comparison.pivot_table(index="model", columns="scenario", values=score_column)
    return table.reindex(
        index=[name for name in MODEL_SUBFOLDERS if name in table.index],
        columns=[scenario for scenario in order if scenario in table.columns],
    )

pivot_noise_scores(noise_comparison).round(4)

save_search_results(noise_comparison, "noise-robustness.csv", PATH_FOLDER_SEARCH_RESULT)
```

Kode Program 5.133 Pengumpulan skor pengujian dan penyusunan tabel hasil

Fungsi *measure_noise_degradation* pada Kode Program 5.134 mengimplementasikan Langkah 7. Fungsi ini mengambil skor data bersih setiap model sebagai acuan dan menghentikan proses dengan galat apabila acuan suatu model tidak tersedia. Selanjutnya fungsi ini menghitung penurunan mutlak (*drop*, Persamaan 4.8) dan retensi (*retention*, Persamaan 4.9) untuk setiap skenario. Sel kedua menjalankannya.

```python
def measure_noise_degradation(
    comparison: pd.DataFrame, score_column: str = SCORE_COLUMN
) -> pd.DataFrame:
    baseline = (
        comparison[comparison["scenario"] == NOISE_BASELINE]
        .set_index("model")[score_column]
        .rename("clean")
    )
    missing = set(comparison["model"]).difference(baseline.index)
    if missing:
        raise ValueError(f"no clean baseline for {sorted(missing)} — run 18.1 first")

    degraded = comparison[comparison["scenario"] != NOISE_BASELINE].copy()
    degraded = degraded.join(baseline, on="model")
    degraded["drop"] = degraded["clean"] - degraded[score_column]
    degraded["retention"] = degraded[score_column] / degraded["clean"]

    return degraded[
        ["model", "noise", "intensity", "scenario", "clean", score_column, "drop", "retention"]
    ].reset_index(drop=True)

noise_degradation = measure_noise_degradation(noise_comparison)
noise_degradation
```

Kode Program 5.134 Penghitungan penurunan dan retensi

Kode Program 5.135 menetapkan warna dan penanda setiap model, dan fungsi *plot_noise_degradation* menggambar, untuk setiap jenis gangguan, garis F1-*score* setiap model terhadap persentase baris terganggu dengan skor data bersih sebagai titik pada 0%. Sel terakhir menjalankannya.

```python
NOISE_MODEL_COLOURS = {
    "KNN": "#2a78d6",
    "SVM": "#eb6834",
    "Random Forest": "#1baf7a",
    "Softmax": "#eda100",
    "XGBoost": "#e87ba4",
}
NOISE_MODEL_MARKERS = {
    "KNN": "o",
    "SVM": "s",
    "Random Forest": "^",
    "Softmax": "D",
    "XGBoost": "v",
}

def plot_noise_degradation(
    comparison: pd.DataFrame,
    score_column: str = SCORE_COLUMN,
    noise_types: tuple[str, ...] = NOISE_TYPES,
) -> None:
    surface, grid, ink, muted = "#fcfcfb", "#e8e7e3", "#0b0b0b", "#52514e"
    baseline = comparison[comparison["scenario"] == NOISE_BASELINE].set_index("model")[score_column]
    models = [name for name in NOISE_MODEL_COLOURS if name in set(comparison["model"])]

    figure, axes_row = plt.subplots(
        1, len(noise_types), figsize=(4.4 * len(noise_types), 4.4), sharey=True
    )
    figure.patch.set_facecolor(surface)

    for axes, noise_type in zip(np.atleast_1d(axes_row), noise_types):
        panel = comparison[comparison["noise"] == noise_type]
        axes.set_facecolor(surface)

        for model_name in models:
            rows = panel[panel["model"] == model_name].sort_values("intensity")
            axes.plot(
                [0.0] + (rows["intensity"] * 100).tolist(),
                [baseline.get(model_name, np.nan)] + rows[score_column].tolist(),
                label=model_name,
                color=NOISE_MODEL_COLOURS[model_name],
                marker=NOISE_MODEL_MARKERS[model_name],
                markersize=8,
                markeredgecolor=surface,
                markeredgewidth=1.5,
                linewidth=2,
            )

        axes.set_title(noise_type.capitalize(), color=ink, fontsize=11, loc="left", pad=12)
        axes.set_xlabel("Corrupted records (%)", color=muted, fontsize=9)
        axes.set_xticks([0] + [intensity * 100 for intensity in NOISE_INTENSITIES])
        axes.grid(axis="y", color=grid, linewidth=1)
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(grid)
        axes.tick_params(colors=muted, labelsize=9, length=0)

    np.atleast_1d(axes_row)[0].set_ylabel(
        score_column.replace("_", " "), color=muted, fontsize=9
    )

    handles, names = np.atleast_1d(axes_row)[0].get_legend_handles_labels()
    figure.legend(
        handles, names, loc="lower center", ncol=len(names), frameon=False,
        labelcolor=ink, fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    plt.show()

plot_noise_degradation(noise_comparison)
```

Kode Program 5.135 Visualisasi F1-score terhadap intensitas gangguan

Fungsi *plot_noise_retention* pada Kode Program 5.136 menggambar retensi setiap pasangan model dan skenario sebagai peta panas dengan nilai pada setiap sel dan batang warna. Sel kedua menjalankannya.

```python
def plot_noise_retention(degradation: pd.DataFrame) -> None:
    surface, ink, muted = "#fcfcfb", "#0b0b0b", "#52514e"
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    colours = matplotlib.colors.LinearSegmentedColormap.from_list("retention", ramp)

    order = [scenario for _, _, scenario in list_noise_scenarios()]
    table = degradation.pivot_table(index="model", columns="scenario", values="retention")
    table = table.reindex(
        index=[name for name in NOISE_MODEL_COLOURS if name in table.index],
        columns=[scenario for scenario in order if scenario in table.columns],
    )

    values = table.to_numpy(dtype=float)
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    span = max(high - low, 1e-9)

    figure, axes = plt.subplots(
        figsize=(0.95 * table.shape[1] + 3, 0.55 * table.shape[0] + 2.4)
    )
    figure.patch.set_facecolor(surface)
    image = axes.imshow(values, cmap=colours, vmin=low, vmax=high, aspect="auto")

    axes.set_xticks(range(table.shape[1]))
    axes.set_xticklabels([str(c) for c in table.columns], fontsize=9, rotation=45, ha="right")
    axes.set_yticks(range(table.shape[0]))
    axes.set_yticklabels([str(i) for i in table.index], fontsize=9)
    axes.set_title(
        "Share of the clean score kept under feature noise",
        color=ink, fontsize=11, loc="left", pad=12,
    )
    axes.tick_params(colors=muted, length=0)
    for side in ("top", "right", "bottom", "left"):
        axes.spines[side].set_visible(False)

    axes.set_xticks(np.arange(-0.5, table.shape[1], 1), minor=True)
    axes.set_yticks(np.arange(-0.5, table.shape[0], 1), minor=True)
    axes.grid(which="minor", color=surface, linewidth=2)
    axes.tick_params(which="minor", length=0)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isnan(value):
                continue
            axes.text(
                column, row, f"{value:.2f}",
                ha="center", va="center", fontsize=9,
                color=surface if (value - low) / span > 0.55 else ink,
            )

    bar = figure.colorbar(image, ax=axes)
    bar.set_label("retention (noisy / clean)", color=muted, fontsize=9)
    bar.ax.tick_params(colors=muted, labelsize=9, length=0)
    bar.outline.set_visible(False)

    figure.tight_layout()
    plt.show()

plot_noise_retention(noise_degradation)
```

Kode Program 5.136 Visualisasi retensi F1-score setiap model dan skenario

Kode Program 5.137 mengimplementasikan Langkah 8. Fungsi *rank_noise_robustness* mengelompokkan hasil menurut model, menghitung skor data bersih, rata-rata retensi (Persamaan 4.10), retensi terburuk beserta skenarionya, rata-rata penurunan, dan penurunan terbesar, kemudian mengurutkan model menurut rata-rata retensi dan memberinya peringkat. Sel kedua menjalankannya dan menyimpan hasilnya ke *hyperparameter-search/noise-robustness-ranking.csv*.

```python
def rank_noise_robustness(degradation: pd.DataFrame) -> pd.DataFrame:
    ranking = degradation.groupby("model").agg(
        clean=("clean", "first"),
        mean_retention=("retention", "mean"),
        worst_retention=("retention", "min"),
        mean_drop=("drop", "mean"),
        worst_drop=("drop", "max"),
    )
    worst = degradation.loc[degradation.groupby("model")["retention"].idxmin()]
    ranking["worst_scenario"] = worst.set_index("model")["scenario"]

    ranking = ranking.sort_values("mean_retention", ascending=False)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    return ranking

noise_ranking = rank_noise_robustness(noise_degradation)
save_search_results(
    noise_ranking.reset_index(), "noise-robustness-ranking.csv", PATH_FOLDER_SEARCH_RESULT
)
noise_ranking.round(4)
```

Kode Program 5.137 Penyusunan peringkat ketahanan model

Fungsi *compare_noise_per_class_f1* pada Kode Program 5.138 menyusun tabel F1-*score* setiap kelas untuk setiap model pada suatu skenario dengan memanfaatkan *compare_per_class_f1* (Kode Program 5.79). Sel kedua menjalankannya pada skenario *missing-30* dan menggambarkannya dengan *plot_per_class_f1*.

```python
def compare_noise_per_class_f1(
    scenario: str,
    true_labels,
    subfolders: dict[str, str] | None = None,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    return compare_per_class_f1(
        MODEL_SUBFOLDERS if subfolders is None else subfolders,
        true_labels,
        get_noise_split_name(scenario),
        CLASS_NAMES if class_names is None else class_names,
    )

noise_per_class_f1 = compare_noise_per_class_f1("missing-30", test_y)
plot_per_class_f1(noise_per_class_f1, "missing-30")
```

Kode Program 5.138 F1-score per kelas pada satu skenario gangguan
